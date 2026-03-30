"""LLM factory functions."""

import json
import os
import re
import uuid

import httpx
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
from .google_auth import VertexAIAuth
from .logging import get_logger

logger = get_logger(__name__)

_PROVIDER_API_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google_genai": "GOOGLE_API_KEY",
}

_VERTEX_AI_HOST = ".prediction.vertexai.goog"

_PROVIDER_MODEL_CLASS = {
    "openai": ChatOpenAI,
    "google_genai": ChatGoogleGenerativeAI,
    "anthropic": ChatAnthropic,
}


def _filter_for_model(provider_config: dict, model_class: type) -> dict:
    """Keep only keys that are known fields or aliases of the target LangChain model class."""
    known = set(model_class.model_fields.keys())
    for field in model_class.model_fields.values():
        if field.alias:
            known.add(field.alias)
    return {k: v for k, v in provider_config.items() if k in known}


class _VLLMVertexChatModel(ChatOpenAI):
    """ChatOpenAI that captures vLLM reasoning from model_extra into additional_kwargs.

    vLLM stores thinking tokens in choice.message.model_extra["reasoning"], but LangChain's
    _create_chat_result converts via response.model_dump() and only copies known fields.
    This subclass rescues the reasoning before it's lost so get_reasoning() works for Qwen.
    """

    def _create_chat_result(self, response, generation_info=None):
        result = super()._create_chat_result(response, generation_info)
        try:
            message = response.choices[0].message
            extra = getattr(message, "model_extra", None) or {}
            reasoning = extra.get("reasoning") or extra.get("reasoning_content")
            if reasoning:
                result.generations[0].message.additional_kwargs["reasoning"] = reasoning
        except (AttributeError, IndexError):
            pass
        return result


def create_llm(config: LLMConfig) -> BaseChatModel:
    """Create an LLM instance from configuration."""
    provider = config.model.split(":")[0] if ":" in config.model else None
    provider_config = dict(config.provider_config)
    base_url = provider_config.get("base_url", "")

    if _VERTEX_AI_HOST in base_url:
        # Auth via auto-refreshing Google OAuth2; api_key satisfies the OpenAI SDK check
        provider_config["http_async_client"] = httpx.AsyncClient(auth=VertexAIAuth())
        provider_config.setdefault("api_key", "vertex-ai")
        # Use subclass to capture vLLM reasoning from model_extra into additional_kwargs
        model_name = (
            config.model.split(":", 1)[1] if ":" in config.model else config.model
        )
        return _VLLMVertexChatModel(model=model_name, **_filter_for_model(provider_config, _VLLMVertexChatModel))
    else:
        key_env = _PROVIDER_API_KEY_ENV.get(provider)
        if key_env and not os.environ.get(key_env):
            raise OSError(f"{key_env} is not set. Check your .env.secrets file.")

    if provider in _PROVIDER_MODEL_CLASS:
        provider_config = _filter_for_model(provider_config, _PROVIDER_MODEL_CLASS[provider])
    return init_chat_model(config.model, **provider_config)


def _rescue_qwen_tool_calls(response: AIMessage) -> AIMessage:
    """Reconstruct tool_calls from Qwen3's native XML format when vLLM's hermes parser drops them.

    Qwen3 outputs:
        <tool_call>
        <function=FNAME>
        <parameter=PNAME>VALUE</parameter>
        </function>
        </tool_call>

    vLLM's hermes parser expects JSON inside <tool_call>, sees XML, raises JSONDecodeError,
    and returns the raw XML as text content with tool_calls=[]. This function parses the XML
    and returns a new AIMessage with proper tool_calls so the loop can execute them.
    """
    content = response.content if isinstance(response.content, str) else ""
    if not content or "<tool_call>" not in content:
        return response

    tool_calls = []
    for block in re.findall(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL):
        fn_match = re.search(r"<function=(\w+)>", block)
        if not fn_match:
            continue
        args = {
            m.group(1): m.group(2).strip()
            for m in re.finditer(
                r"<parameter=(\w+)>(.*?)</parameter>", block, re.DOTALL
            )
        }
        tool_calls.append(
            {"name": fn_match.group(1), "args": args, "id": uuid.uuid4().hex[:8]}
        )

    if not tool_calls:
        logger.debug(
            "_rescue_qwen_tool_calls: <tool_call> found in content but no parseable calls"
        )
        return response

    names = [tc["name"] for tc in tool_calls]
    logger.debug(
        f"_rescue_qwen_tool_calls: rescued {len(tool_calls)} tool call(s) from XML: {names}"
    )
    return AIMessage(content="", tool_calls=tool_calls)


async def agentic_loop(
    client: "LLMClient",
    messages: list[BaseMessage],
    tools: list[BaseTool],
    output_schema: type[BaseModel] | None = None,
    max_tool_iterations: int = 5,
) -> AIMessage:
    """Invoke an LLM with tools, executing tool calls in a loop until the model stops calling tools.

    Returns:
        A tuple of (final_response, all_new_messages) where all_new_messages includes every
        intermediate AI message, tool result, and the final response.

    For Qwen: delegates to _agentic_loop_qwen which strictly separates the tool phase
    from the final structured output call (detected via client.is_qwen).
    """
    if client.is_qwen:
        return await _agentic_loop_qwen(
            client, messages, tools, output_schema, max_tool_iterations
        )

    response = await client.ainvoke(messages, tools=tools, output_schema=output_schema)
    new_messages: list[BaseMessage] = [response]

    tool_node = ToolNode(tools)

    for iteration in range(max_tool_iterations):
        if not response.tool_calls:
            break

        result = await tool_node.ainvoke({"messages": messages + new_messages})
        tool_messages = result["messages"]
        new_messages += tool_messages

        is_last_iteration = iteration == max_tool_iterations - 1
        invoke_messages = messages + new_messages
        if is_last_iteration:
            # Prevent hallucinated tool calls with this message
            invoke_messages = invoke_messages + [
                HumanMessage(content="NO MORE TOOL CALLS ALLOWED.")
            ]
            response = await client.ainvoke(
                invoke_messages, output_schema=output_schema
            )
        else:
            response = await client.ainvoke(
                invoke_messages, tools=tools, output_schema=output_schema
            )

        new_messages.append(response)

    return response


async def _agentic_loop_qwen(
    client: "LLMClient",
    messages: list[BaseMessage],
    tools: list[BaseTool],
    output_schema: type[BaseModel] | None,
    max_tool_iterations: int,
) -> AIMessage:
    """Qwen loop: same structure as agentic_loop but strips output_schema from tool-phase calls.

    Qwen ignores tools when response_format is also set, so the two are always separated:
    tool iterations use tools only, and structured output is extracted in a final dedicated call.
    The no-tools case inserts an extra thinking call before the structured output extraction.
    """
    if not tools:
        thinking = await client.ainvoke(messages)
        if not output_schema:
            return thinking
        final = _rescue_qwen_json_in_thinking(
            await client.ainvoke(
                list(messages) + [thinking], output_schema=output_schema
            )
        )
        _attach_reasoning(final, [get_reasoning(thinking)])
        return final

    response = await client.ainvoke(messages, tools=tools)
    new_messages: list[BaseMessage] = [response]
    accumulated_reasoning: list[str] = []
    if r := get_reasoning(response):
        accumulated_reasoning.append(r)

    tool_node = ToolNode(tools)

    for iteration in range(max_tool_iterations):
        if not response.tool_calls:
            response = _rescue_qwen_tool_calls(response)
            new_messages[-1] = response
        if not response.tool_calls:
            break

        logger.debug(f"Qwen: tool calls {[tc['name'] for tc in response.tool_calls]}")
        result = await tool_node.ainvoke({"messages": messages + new_messages})
        new_messages += result["messages"]

        is_last_iteration = iteration == max_tool_iterations - 1
        invoke_messages = messages + new_messages
        if is_last_iteration:
            invoke_messages = invoke_messages + [
                HumanMessage(content="NO MORE TOOL CALLS ALLOWED.")
            ]
            response = await client.ainvoke(invoke_messages)
        else:
            response = await client.ainvoke(invoke_messages, tools=tools)

        new_messages.append(response)
        if r := get_reasoning(response):
            accumulated_reasoning.append(r)

    if not output_schema:
        return response

    final = _rescue_qwen_json_in_thinking(
        await client.ainvoke(messages + new_messages, output_schema=output_schema)
    )
    _attach_reasoning(final, accumulated_reasoning)
    return final


def _attach_reasoning(response: AIMessage, parts: list[str]) -> None:
    """Attach accumulated reasoning from thinking-phase calls to the final response."""
    if parts and not response.additional_kwargs.get("reasoning"):
        response.additional_kwargs["reasoning"] = "\n\n---\n\n".join(parts)


def _rescue_qwen_json_in_thinking(response: AIMessage) -> AIMessage:
    """Rescue JSON that Qwen placed inside <think> when thinking + response_format are both active.

    vLLM extracts <think>...</think> into model_extra["reasoning"]; our subclass moves it to
    additional_kwargs["reasoning"]. When the entire response is JSON (no actual thinking text),
    move it to content so callers can parse it normally.
    """
    if response.content:
        return response
    reasoning = response.additional_kwargs.get("reasoning", "")
    if not reasoning:
        return response
    try:
        json.loads(reasoning)
        response.content = reasoning
        del response.additional_kwargs["reasoning"]
    except (json.JSONDecodeError, ValueError):
        pass
    return response


def get_reasoning(response: AIMessage) -> str:
    """Extract the reasoning from an LLM response."""
    # OpenAI-compatible with a reasoning parser (e.g. vLLM --reasoning-parser qwen3)
    # Field name varies by deployment: vLLM uses "reasoning", some others use "reasoning_content"
    kwargs = response.additional_kwargs
    reasoning = kwargs.get("reasoning") or kwargs.get("reasoning_content") or ""
    if reasoning:
        return reasoning

    # Claude: reasoning blocks live in content_blocks (langchain_anthropic extension)
    content_blocks = getattr(response, "content_blocks", None)
    if content_blocks:
        return "\n\n".join(
            msg.get("reasoning", "")
            for msg in content_blocks
            if msg["type"] == "reasoning"
        )

    # vLLM fallback: for tool-call responses the reasoning parser may leave <think>...</think>
    # embedded in content instead of extracting it to model_extra["reasoning"]
    if isinstance(response.content, str):
        m = re.search(r"<think>(.*?)</think>", response.content, re.DOTALL)
        if m:
            return m.group(1).strip()

    return ""


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
    """

    def __init__(self, config: LLMConfig):
        """Initialize the LLMClient with a configuration."""
        self._base_llm: BaseChatModel = create_llm(config)
        self._retry_config: dict = config.retry_config

    @property
    def profile(self) -> dict:
        """Model metadata (max_input_tokens, max_output_tokens, capabilities, etc.)."""
        return getattr(self._base_llm, "profile", None) or {}

    @property
    def is_qwen(self) -> bool:
        """True if the model requires phase-separated tool + structured output calls."""
        if not isinstance(self._base_llm, ChatOpenAI):
            return False
        model_name = getattr(self._base_llm, "model_name", "") or ""
        return "qwen" in model_name.lower()

    async def ainvoke(
        self,
        messages: LanguageModelInput,
        tools: list[BaseTool] | None = None,
        output_schema: type[BaseModel] | None = None,
        retry_config: dict | None = None,
    ) -> AIMessage:
        """Invoke with optional tools, structured output, and retry."""
        effective_retry = retry_config or self._retry_config
        runnable = self._get_runnable(
            tools=tools, output_schema=output_schema, retry_config=effective_retry
        )
        return await runnable.ainvoke(messages)

    def _get_runnable(
        self,
        tools: list[BaseTool] | None = None,
        output_schema: type[BaseModel] | None = None,
        retry_config: dict | None = None,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Build a retryable Runnable that always returns AIMessage.

        Layers are applied in order: bind_tools → bind structured output → with_retry.
        """
        model: Runnable = self._base_llm

        if tools:
            # OpenAI requires strict=True when combining tools with structured output
            # other providers default to None (omit the field)
            strict = (
                True
                if isinstance(self._base_llm, ChatOpenAI) and output_schema
                else None
            )
            model = self._base_llm.bind_tools(tools, strict=strict)

        if output_schema:
            model = model.bind(**self._structured_output_bind_kwargs(output_schema))

        if retry_config:
            model = model.with_retry(**retry_config)

        return model

    def _structured_output_bind_kwargs(self, schema: type[BaseModel]) -> dict:
        """Return provider-specific kwargs that constrain the output to a JSON schema.

        These kwargs are passed via bind() so the response stays as an AIMessage, unlike
        `with_structured_output`, which prevents the use of any tools and forces the output
        to be an instance of the schema.
        """
        # LANGCHAIN PLSSS ALLOW ME TO DO STRUCTURED OUTPUT WITH TOOL BINDINGS
        # LOOK AT WHAT IT NEED TO DO C'MONNNNN

        if isinstance(self._base_llm, ChatAnthropic):
            # Need to check 4.5 or 4.6+
            model_name = getattr(self._base_llm, "model", "")
            return _anthropic_structured_kwargs(model_name, schema)

        if isinstance(self._base_llm, ChatGoogleGenerativeAI):
            return _google_structured_kwargs(schema)

        if isinstance(self._base_llm, ChatOpenAI):
            kwargs = _openai_structured_kwargs(schema)
            if self.is_qwen:
                # Qwen: thinking + response_format → JSON goes inside <think>, content = "".
                # Disable thinking in the same bind call (single RunnableBinding, as confirmed
                # by gcp/test_qwen_vertex.py). Thinking already happened in the prior call.
                kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
            return kwargs

        raise NotImplementedError(
            f"Structured output bind kwargs not implemented for {type(self._base_llm).__name__}."
        )


def _anthropic_structured_kwargs(model_name: str, schema: type[BaseModel]) -> dict:
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


def _google_structured_kwargs(schema: type[BaseModel]) -> dict:
    json_schema = schema.model_json_schema()

    return {
        "response_mime_type": "application/json",
        "response_json_schema": json_schema,
    }


def _openai_structured_kwargs(schema: type[BaseModel]) -> dict:
    json_schema = schema.model_json_schema()
    return {
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": schema.__name__, "schema": json_schema},
        }
    }
