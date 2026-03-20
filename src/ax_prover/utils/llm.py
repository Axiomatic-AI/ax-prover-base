"""LLM factory functions."""

import os
from collections.abc import Awaitable, Callable

from anthropic import transform_schema
from langchain.chat_models import init_chat_model
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import Runnable
from langchain_core.runnables.retry import RunnableRetry
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


async def ainvoke_retry_with_structured_output(
    messages: LanguageModelInput, llm: RunnableRetry, schema: BaseModel
):
    "Invoke an LLM with retry enforcing native structured output for each provider."
    chat_model = _get_base_chat_model_from_binding(llm)

    # LANGCHAIN PLS ALLOW ME TO DO STRUCTURED OUTPUT WITH TOOL BINDINGS
    if isinstance(chat_model, ChatAnthropic):
        return await llm.ainvoke(
            messages, output_format={"type": "json_schema", "schema": transform_schema(schema)}
        )

    if isinstance(chat_model, ChatGoogleGenerativeAI):
        return await llm.ainvoke(
            messages,
            response_mime_type="application/json",
            response_json_schema=schema.model_json_schema(),
        )

    if isinstance(chat_model, ChatOpenAI):
        # For Qwen we will have to check the chat_model.openai_api_base (may be None for openai)
        return await llm.ainvoke(messages, response_format=schema)

    return await chat_model.with_structured_output(schema).with_retry().ainvoke(messages)


def _get_base_chat_model_from_binding(llm: RunnableRetry) -> BaseChatModel:
    """
    Iteratively find the unwrapped base chat model in a potentially nested binding of Runnable/Bound objects.
    Stops at the lowest-level model (e.g., ChatAnthropic, ChatGoogleGenerativeAI, etc.).
    """
    chat_model = llm

    while hasattr(chat_model, "bound"):
        if chat_model is chat_model.bound:
            break  # Prevent infinite recursion if .bound returns self
        chat_model = chat_model.bound

    return chat_model


async def run_tools_and_respond(
    response: AIMessage,
    tools: list[BaseTool],
    messages: list[BaseMessage],
    invoke_function_async: Callable[[LanguageModelInput, RunnableRetry], Awaitable[BaseMessage]],
    llm: RunnableRetry,
    extra_message: str | None = None,
) -> list[BaseMessage]:
    """Execute pending tool calls from a given response and invoke the llm to get the next response
    using the provided function.

    Args:
        response: The response from the LLM that may contain tool calls
        tools: The tools to execute
        messages: The history of messages in the conversation prior to the response
        invoke_function_async: The function to invoke an LLM with a given messages and LLM
        llm: The LLM to call to get the next response after the tool calls have been executed
        extra_message: An optional extra message to add to the messages

    Returns:
        A list of messages including the new response after the tool calls have been executed"""

    tool_node = ToolNode(tools)
    result = await tool_node.ainvoke({"messages": messages + [response]})
    new_messages = result["messages"]

    invoke_messages = messages + [response] + new_messages
    if extra_message:
        invoke_messages = invoke_messages + [HumanMessage(content=extra_message)]

    new_response = await invoke_function_async(invoke_messages, llm)
    return new_messages + [new_response]


def get_reasoning(response: AIMessage) -> str:
    """Extract the reasoning from an LLM response."""
    reasoning = "\n\n".join(
        [msg.get("reasoning", "") for msg in response.content_blocks if msg["type"] == "reasoning"]
    )
    return reasoning


class LLMClient:
    """Dynamically create a Runnable to invoke LLMs with structured output, tool calling and retry.

    Usage:
        client = LLMClient(config)

        # Plain call
        response = await client.ainvoke(messages)

        # With tools only
        response = await client.ainvoke(messages, tools=my_tools)

        # With structured output only
        response = await client.ainvoke(messages, output_schema=MyModel)

        # With retry only
        response = await client.ainvoke(messages, retry_config=retry_config)

        # With tools and structured output
        response = await client.ainvoke(messages, tools=my_tools, output_schema=MyModel)

        # With tools, structured output and retry
        response = await client.ainvoke(messages, tools=my_tools, output_schema=MyModel, retry_config=retry_config)
    """

    def __init__(self, config: LLMConfig):
        """Initialize the LLMClient with a configuration."""
        self._base_llm: BaseChatModel = create_llm(config)

    async def ainvoke(
        self,
        messages: LanguageModelInput,
        tools: list[BaseTool] | None = None,
        output_schema: BaseModel | None = None,
        retry_config: dict | None = None,
    ) -> AIMessage:
        """Invoke with optional tools, structured output, and retry."""
        runnable = self._get_runnable(
            tools=tools, output_schema=output_schema, retry_config=retry_config
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
            model = self._base_llm.bind_tools(tools)

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
        json_schema = schema.model_json_schema()

        if isinstance(self._base_llm, ChatAnthropic):
            model_name = getattr(self._base_llm, "model", "")  # Need to check 4.5 or 4.6+
            return _anthropic_structured_kwargs(model_name, json_schema)

        if isinstance(self._base_llm, ChatGoogleGenerativeAI):
            return _google_structured_kwargs(json_schema)

        if isinstance(self._base_llm, ChatOpenAI):
            return _openai_structured_kwargs(json_schema)

        raise NotImplementedError(
            f"Structured output bind kwargs not implemented for {type(self._base_llm).__name__}."
        )


def _anthropic_structured_kwargs(model_name: str, json_schema: dict) -> dict:
    is_46 = "4-6" in model_name or "4.6" in model_name

    schema_payload = {"type": "json_schema", "schema": transform_schema(json_schema)}

    if is_46:
        # Claude 4.6+: output_config.format (output_format is deprecated)
        return {"output_config": {"format": schema_payload}}
    else:
        # Claude 4.5 and earlier: output_format
        return {"output_format": schema_payload}


def _openai_structured_kwargs(json_schema: dict) -> dict:
    return {
        "response_format": {
            "type": "json_schema",
            "strict": True,
            "schema": json_schema,
        }
    }


def _google_structured_kwargs(json_schema: dict) -> dict:
    return {
        "response_mime_type": "application/json",
        "response_json_schema": json_schema,
    }
