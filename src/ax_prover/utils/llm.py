"""LLM factory functions."""

import os

import anthropic
import openai
from anthropic import transform_schema
from langchain.chat_models import init_chat_model
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel

from ..config import LLMConfig
from .logging import get_logger

logger = get_logger(__name__)

_PROVIDER_API_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google_genai": "GOOGLE_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

OPENROUTER_PREFIX = "openrouter:"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


#: A stuck upstream request must not park the whole run: with the SDK defaults (600s,
#: 2 retries) one hung call burns ~30 minutes silently, which stalls a blueprint run
#: outright once the frontier narrows to a single node.
#:
#: The bound is deliberately loose. Measured throughput on this model is a flat 60-70
#: output tokens/sec at every duration, so a 600s call is not a stall, it is 41k tokens of
#: reasoning on a hard lemma. Timing those out kills exactly the work most likely to close
#: a straggler. Output volume, not wall clock, is what actually needs capping; that belongs
#: in a model's `max_tokens`, not here.
REQUEST_TIMEOUT_SECONDS = 900.0
MAX_REQUEST_RETRIES = 1


class TransientProviderError(Exception):
    """A retryable LLM-call failure, wrapping whatever shape the provider produced.

    Retry policy is a deny-list, not an allow-list. An allow-list of typed transient
    exceptions was tried first and lost twice to OpenRouter's failure zoo: a 504
    "Provider timed out" arrived as a bare ValueError inside an HTTP 200 body, and a
    malformed response with `choices: null` crashed langchain-openai with a TypeError -
    each killed a run because the type was not on the list. Now anything that is not
    provably permanent is wrapped in this type and retried; only statuses that will fail
    identically on every attempt (auth, bad request, the data-policy routing 404 that
    once retried invisibly for 90 minutes) fail fast, as the original exception.
    """


#: HTTP statuses that fail identically on every retry. 408/429 are absent on purpose.
_PERMANENT_STATUSES = {400, 401, 403, 404, 405, 409, 413, 422}


def _permanent_status(error: BaseException) -> int | None:
    """The error's HTTP status if it is provably permanent, else None.

    Reads `status_code` from typed openai/anthropic APIStatusError exceptions, and the
    embedded `code` from the ValueError langchain-openai raises for an error payload
    inside an HTTP 200 body.
    """
    status = getattr(error, "status_code", None)
    if status is None and isinstance(error, ValueError) and error.args:
        payload = error.args[0]
        if isinstance(payload, dict):
            status = payload.get("code")
    try:
        status = int(status)
    except (TypeError, ValueError):
        return None
    return status if status in _PERMANENT_STATUSES else None


RETRYABLE_LLM_EXCEPTIONS: tuple[type[BaseException], ...] = (TransientProviderError,)


def create_llm(config: LLMConfig) -> BaseChatModel:
    """Create an LLM instance from configuration."""
    provider = config.model.split(":")[0] if ":" in config.model else None
    key_env = _PROVIDER_API_KEY_ENV.get(provider)
    if key_env and not os.environ.get(key_env):
        raise OSError(f"{key_env} is not set. Check your .env.secrets file.")

    if config.model.startswith(OPENROUTER_PREFIX):
        return _create_openrouter_llm(config)

    return init_chat_model(
        config.model,
        **config.provider_config,
    )


def _create_openrouter_llm(config: LLMConfig) -> BaseChatModel:
    """Create a chat model backed by OpenRouter's OpenAI-compatible endpoint.

    `extra_body` passes through verbatim and lands at the request body root, which is how
    OpenRouter's own controls are reached: `reasoning.effort`, `provider.only`, and a
    top-level `session_id`. Set them in the YAML rather than here, so routing is explicit
    and reviewable per model.

    Note that `reasoning_effort` as a `ChatOpenAI` kwarg is not a reliable way to reach
    OpenRouter's unified reasoning control; prefer `extra_body.reasoning.effort`.
    """
    provider_config = dict(config.provider_config)
    base_url = provider_config.pop("base_url", DEFAULT_OPENROUTER_BASE_URL)
    # Defaults, not overrides: a model that declares its own timeout in YAML keeps it.
    provider_config.setdefault("timeout", REQUEST_TIMEOUT_SECONDS)
    provider_config.setdefault("max_retries", MAX_REQUEST_RETRIES)

    return ChatOpenAI(
        model=config.model.removeprefix(OPENROUTER_PREFIX),
        base_url=base_url,
        api_key=os.environ["OPENROUTER_API_KEY"],
        **provider_config,
    )


#: Finish reasons that mean the response was cut off rather than completed. A truncated
#: response is not an exception, so `with_retry` never sees it; it arrives as a normal
#: `AIMessage` holding partial content, which then fails to parse or fails to compile.
#: Detecting it here is the only place every call site passes through.
TRUNCATION_FINISH_REASONS = frozenset({"length", "max_tokens", "MAX_TOKENS"})


def was_truncated(response: AIMessage) -> bool:
    """True when the model stopped because it hit the output token limit."""
    metadata = response.response_metadata or {}
    reason = metadata.get("finish_reason") or metadata.get("stop_reason")
    return str(reason) in TRUNCATION_FINISH_REASONS


def _warn_if_truncated(response: AIMessage) -> None:
    """Log a warning when a response was cut off at the output limit.

    Truncation is otherwise silent and expensive: the caller sees unparseable or
    uncompilable output and retries without knowing the cause.
    """
    if not was_truncated(response):
        return

    usage = response.usage_metadata or {}
    logger.warning(
        "LLM response truncated at the output token limit "
        f"({usage.get('output_tokens', 'unknown')} output tokens). "
        "Raise `max_tokens` for this model if its work legitimately needs more room, "
        "or expect the caller to retry with a more concise instruction."
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
            response = await client.ainvoke(invoke_messages, output_schema=output_schema)
        else:
            response = await client.ainvoke(
                invoke_messages, tools=tools, output_schema=output_schema
            )

        new_messages.append(response)

    return response


def get_reasoning(response: AIMessage) -> str:
    """Extract the reasoning from an LLM response."""
    reasoning = "\n\n".join(
        [msg.get("reasoning", "") for msg in response.content_blocks if msg["type"] == "reasoning"]
    )
    return reasoning


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
        return getattr(self._base_llm, "profile", {})

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
        response = await runnable.ainvoke(messages)
        _warn_if_truncated(response)
        return response

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
            strict = True if isinstance(self._base_llm, ChatOpenAI) and output_schema else None
            model = self._base_llm.bind_tools(tools, strict=strict)

        if output_schema:
            model = model.bind(**self._structured_output_bind_kwargs(output_schema))

        if retry_config:
            inner = model

            async def classify_errors(messages, config=None):
                try:
                    return await inner.ainvoke(messages, config=config)
                except Exception as e:
                    if _permanent_status(e) is not None:
                        raise
                    raise TransientProviderError(f"{type(e).__name__}: {e}") from e

            model = RunnableLambda(classify_errors).with_retry(
                retry_if_exception_type=retry_config.get(
                    "retry_if_exception_type", RETRYABLE_LLM_EXCEPTIONS
                ),
                **{k: v for k, v in retry_config.items() if k != "retry_if_exception_type"},
            )

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
            model_name = getattr(self._base_llm, "model", "")  # Need to check 4.5 or 4.6+
            return _anthropic_structured_kwargs(model_name, schema)

        if isinstance(self._base_llm, ChatGoogleGenerativeAI):
            return _google_structured_kwargs(schema.model_json_schema())

        if isinstance(self._base_llm, ChatOpenAI):
            return _openai_structured_kwargs(schema)

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
    return {"response_format": schema}
