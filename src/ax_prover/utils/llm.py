"""LLM factory functions."""

import os

from anthropic import transform_schema
from langchain.chat_models import init_chat_model
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel

from ..config import LLMConfig

_PROVIDER_API_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google_genai": "GOOGLE_API_KEY",
}


def create_llm(config: LLMConfig) -> BaseChatModel:
    """Create an LLM instance from configuration."""
    provider = config.model.split(":")[0] if ":" in config.model else None
    key_env = _PROVIDER_API_KEY_ENV.get(provider)
    if key_env and not os.environ.get(key_env):
        raise OSError(f"{key_env} is not set. Check your .env.secrets file.")

    return init_chat_model(
        config.model,
        **config.provider_config,
    )


async def agentic_loop(
    client: "LLMClient",
    messages: list[BaseMessage],
    tools: list[BaseTool],
    output_schema: type[BaseModel] | None = None,
    max_tool_iterations: int = 5,
) -> tuple[AIMessage, list[BaseMessage]]:
    """Invoke an LLM with tools, executing tool calls in a loop until the model stops calling tools.

    Returns:
        A tuple of (final_response, all_new_messages) where all_new_messages includes every
        intermediate AI message, tool result, and the final response.
    """
    response = await client.ainvoke(messages, tools=tools, output_schema=output_schema)
    all_new_messages: list[BaseMessage] = [response]

    tool_node = ToolNode(tools)

    for iteration in range(max_tool_iterations):
        if not response.tool_calls:
            break

        result = await tool_node.ainvoke({"messages": messages + all_new_messages})
        tool_messages = result["messages"]
        all_new_messages += tool_messages

        is_last_iteration = iteration == max_tool_iterations - 1
        invoke_messages = messages + all_new_messages
        if is_last_iteration:
            # Prevent hallucinated tool calls with this message
            invoke_messages = invoke_messages + [
                HumanMessage(content="NO MORE TOOL CALLS ALLOWED.")
            ]
            response = await client.ainvoke(invoke_messages, output_schema=output_schema)
        else:
            response = await client.ainvoke(
                invoke_messages, tools=tools, output_schema=output_schema
            )

        all_new_messages.append(response)

    return response, all_new_messages


def get_reasoning(response: AIMessage) -> str:
    """Extract the reasoning from an LLM response."""
    reasoning = "\n\n".join(
        [msg.get("reasoning", "") for msg in response.content_blocks if msg["type"] == "reasoning"]
    )
    return reasoning


_UNSET = object()


class LLMClient:
    """Dynamically create a Runnable to invoke LLMs with structured output, tool calling and retry.

    Retry is applied by default using the config's retry_config. Pass retry_config=None
    to disable, or pass a custom dict to override.

    Usage:
        client = LLMClient(config)

        # Plain call (uses default retry from config)
        response = await client.ainvoke(messages)

        # With tools only
        response = await client.ainvoke(messages, tools=my_tools)

        # With structured output only
        response = await client.ainvoke(messages, output_schema=MyModel)


        # With tools and structured output
        response = await client.ainvoke(messages, tools=my_tools, output_schema=MyModel)

        # Override retry
        response = await client.ainvoke(messages, retry_config={"stop_after_attempt": 3})

        # Disable retry
        response = await client.ainvoke(messages, retry_config=None)
    """

    def __init__(self, config: LLMConfig):
        """Initialize the LLMClient with a configuration."""
        self._base_llm: BaseChatModel = create_llm(config)
        self._retry_config: dict | None = config.retry_config or None

    @property
    def profile(self) -> dict:
        """Model metadata (max_input_tokens, max_output_tokens, capabilities, etc.)."""
        return getattr(self._base_llm, "profile", {})

    async def ainvoke(
        self,
        messages: LanguageModelInput,
        tools: list[BaseTool] | None = None,
        output_schema: BaseModel | None = None,
        retry_config: dict | None | object = _UNSET,
    ) -> AIMessage:
        """Invoke with optional tools, structured output, and retry."""
        effective_retry = self._retry_config if retry_config is _UNSET else retry_config
        runnable = self._get_runnable(
            tools=tools, output_schema=output_schema, retry_config=effective_retry
        )
        return await runnable.ainvoke(messages)

    def _get_runnable(
        self,
        tools: list[BaseTool] | None = None,
        output_schema: BaseModel | None = None,
        retry_config: dict | None = None,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Build a retryable Runnable that always returns AIMessage.

        Layers are applied in order: bind_tools → bind structured output → with_retry.
        """
        model: Runnable = self._base_llm

        if tools:
            # OpenAI requires strict=True when combining tools with structured output
            # other providers default to None (omit the field)
            strict = True if isinstance(self._base_llm, ChatOpenAI) and output_schema else None
            model = self._base_llm.bind_tools(tools, strict=strict)

        if output_schema:
            model = model.bind(**self._structured_output_bind_kwargs(output_schema))

        if retry_config:
            model = model.with_retry(**retry_config)

        return model

    def _structured_output_bind_kwargs(self, schema: BaseModel) -> dict:
        """Return provider-specific kwargs that constrain the output to a JSON schema.

        These kwargs are passed via bind() so the response stays as an AIMessage, unlike
        `with_structured_output`, which prevents the use of any tools and forces the output
        to be an instance of the schema.
        """
        # LANGCHAIN ALLOW ME TO DO STRUCTURED OUTPUT WITH TOOL BINDINGS PLSSS

        if isinstance(self._base_llm, ChatAnthropic):
            model_name = getattr(self._base_llm, "model", "")  # Need to check 4.5 or 4.6+
            return _anthropic_structured_kwargs(model_name, schema)

        if isinstance(self._base_llm, ChatGoogleGenerativeAI):
            return _google_structured_kwargs(schema.model_json_schema())

        if isinstance(self._base_llm, ChatOpenAI):
            return _openai_structured_kwargs(schema)

        raise NotImplementedError(
            f"Structured output bind kwargs not implemented for {type(self._base_llm).__name__}."
        )


def _anthropic_structured_kwargs(model_name: str, schema: BaseModel) -> dict:
    json_schema = schema.model_json_schema()
    json_schema = transform_schema(json_schema)

    is_46 = "4-6" in model_name or "4.6" in model_name

    schema_payload = {"type": "json_schema", "schema": json_schema}

    if is_46:
        # Claude 4.6+: output_config.format (output_format is deprecated)
        return {"output_config": {"format": schema_payload}}
    else:
        # Claude 4.5 and earlier: output_format
        return {"output_format": schema_payload}


def _google_structured_kwargs(schema: BaseModel) -> dict:
    json_schema = schema.model_json_schema()

    return {
        "response_mime_type": "application/json",
        "response_json_schema": json_schema,
    }


def _openai_structured_kwargs(schema: BaseModel) -> dict:
    return {"response_format": schema}
