"""LLM factory functions."""

import os
from collections.abc import Awaitable, Callable

from anthropic import transform_schema
from langchain.chat_models import init_chat_model
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
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
