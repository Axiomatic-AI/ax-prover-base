"""LLM factory functions."""

from anthropic import transform_schema
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables.retry import RunnableRetry
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from ..config import LLMConfig


def create_llm(config: LLMConfig) -> BaseChatModel:
    """
    Create an LLM instance from configuration.

    Uses LangChain's init_chat_model for automatic provider detection.

    Args:
        config: LLM configuration

    Returns:
        Configured LLM instance

    Example:
        >>> config = LLMConfig(model="anthropic:claude-haiku-4-5-20251001")
        >>> llm = create_llm(config)
    """
    return init_chat_model(
        config.model,
        **config.provider_config,
    )


async def ainvoke_retry_with_structured_output(
    messages: LanguageModelInput, llm: RunnableRetry, schema: BaseModel
):
    "Invoke an LLM with retry enforcing native structured output for each provider."
    chat_model = _get_base_chat_model_from_binding(llm)

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


def get_reasoning(response: AIMessage) -> str:
    """Extract the reasoning from an LLM response."""
    reasoning = "\n\n".join(
        [msg.get("reasoning", "") for msg in response.content_blocks if msg["type"] == "reasoning"]
    )
    return reasoning
