"""Test Qwen3 Vertex AI endpoint: thinking × tools × structured output combinations.

Usage:
    python gcp/test_qwen_vertex.py

Requires: gcloud auth application-default login (or running on GCP with a service account)
"""

import asyncio

import google.auth
import google.auth.transport.requests
import httpx
from langchain_core.messages import AIMessage as LCAIMessage
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI

BASE_URL = (
    "https://346462710981984256.asia-southeast1-136811191949.prediction.vertexai.goog"
    "/v1beta1/projects/136811191949/locations/asia-southeast1/endpoints/346462710981984256"
)
MODEL = "Qwen/Qwen3.5-35B-A3B"

# Simple question — used for cases that don't need tool calls
SIMPLE_MESSAGES = [{"role": "user", "content": "What is 2+2? Be concise."}]

# Forces a tool call: the model can't answer without searching
TOOL_MESSAGES = [{"role": "user", "content": "What is the exact Lean 4 type signature of `Nat.add_comm`? You MUST use the search tool to look it up."}]

# System prompt instructs tool-first behavior
TOOL_MESSAGES_WITH_SYSTEM = [
    {"role": "system", "content": "Always use the available search tools before answering questions. Never answer from memory."},
    {"role": "user", "content": "What is the exact Lean 4 type signature of `Nat.add_comm`?"},
]

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_lean",
        "description": "Search the Lean 4 / Mathlib library for theorem statements and type signatures.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The theorem or lemma to look up"}},
            "required": ["query"],
        },
    },
}

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "Answer",
        "schema": {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        },
    },
}

THINK_ON = {"chat_template_kwargs": {"enable_thinking": True}, "thinking_token_budget": 1024}
THINK_OFF = {"chat_template_kwargs": {"enable_thinking": False}}

CASES = [
    # (name, messages, extra_kwargs)
    # Baselines
    ("thinking only",                          SIMPLE_MESSAGES, dict(extra_body=THINK_ON)),
    ("structured only",                        SIMPLE_MESSAGES, dict(response_format=RESPONSE_FORMAT, extra_body=THINK_OFF)),
    ("tools only (forced)",                    TOOL_MESSAGES,   dict(tools=[SEARCH_TOOL], tool_choice="required", extra_body=THINK_OFF)),
    ("thinking + structured",                  SIMPLE_MESSAGES, dict(response_format=RESPONSE_FORMAT, extra_body=THINK_ON)),
    # Natural tool use (no tool_choice="required") — does the model call tools spontaneously?
    ("tools + structured (natural)",           TOOL_MESSAGES,   dict(tools=[SEARCH_TOOL], response_format=RESPONSE_FORMAT, extra_body=THINK_OFF)),
    ("thinking + tools + structured (natural)",TOOL_MESSAGES,   dict(tools=[SEARCH_TOOL], response_format=RESPONSE_FORMAT, extra_body=THINK_ON)),
    # Forced tool use — confirms tools work when needed
    ("tools + structured (forced)",            TOOL_MESSAGES,             dict(tools=[SEARCH_TOOL], tool_choice="required", response_format=RESPONSE_FORMAT, extra_body=THINK_OFF)),
    ("thinking + tools + structured (forced)", TOOL_MESSAGES,             dict(tools=[SEARCH_TOOL], tool_choice="required", response_format=RESPONSE_FORMAT, extra_body=THINK_ON)),
    # System prompt instructs tool-first — does it work without tool_choice="required"?
    ("system prompt tool-first + structured",  TOOL_MESSAGES_WITH_SYSTEM, dict(tools=[SEARCH_TOOL], response_format=RESPONSE_FORMAT, extra_body=THINK_OFF)),
    ("system prompt tool-first + think + structured", TOOL_MESSAGES_WITH_SYSTEM, dict(tools=[SEARCH_TOOL], response_format=RESPONSE_FORMAT, extra_body=THINK_ON)),
]


class _QwenVLLMChatModel(ChatOpenAI):
    """ChatOpenAI that captures vLLM reasoning from model_extra into additional_kwargs.

    vLLM stores thinking in choice.message.model_extra["reasoning"], but LangChain's
    _create_chat_result converts via response.model_dump() and only copies known fields.
    This subclass grabs the reasoning before it's lost.
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


class _VertexAIAuth(httpx.Auth):
    """Auto-refreshing Google OAuth2 auth. Runs after OpenAI SDK sets its headers, overriding them."""

    def __init__(self) -> None:
        self._credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )

    def auth_flow(self, request: httpx.Request):  # type: ignore[override]
        if not self._credentials.valid:
            self._credentials.refresh(google.auth.transport.requests.Request())
        request.headers["Authorization"] = f"Bearer {self._credentials.token}"
        yield request


async def run() -> None:
    auth = _VertexAIAuth()
    http = httpx.AsyncClient(auth=auth)

    client = AsyncOpenAI(
        api_key="vertex-ai",
        base_url=BASE_URL,
        http_client=http,
    )

    for name, messages, kwargs in CASES:
        print(f"\n{'=' * 60}")
        print(f"[{name}]")
        try:
            resp = await client.chat.completions.create(
                model=MODEL, messages=messages, max_tokens=2048, **kwargs
            )
            choice = resp.choices[0]
            # vLLM stores reasoning in model_extra under key "reasoning" (not "reasoning_content")
            extra = choice.message.model_extra or {}
            reasoning = extra.get("reasoning") or extra.get("reasoning_content")
            content_preview = (choice.message.content or "")[:300]
            reasoning_preview = (reasoning or "")[:300]
            print(f"  content:    {content_preview!r}")
            print(f"  reasoning:  {reasoning_preview!r}" if reasoning else "  reasoning:  (none)")
            if choice.message.tool_calls:
                for tc in choice.message.tool_calls:
                    print(f"  tool_call:  {tc.function.name}({tc.function.arguments})")
            else:
                print(f"  tool_calls: (none)")
            print(f"  finish:     {choice.finish_reason}")
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")


async def run_langchain() -> None:
    """Test langchain path: reasoning capture and two-phase thinking + structured output."""
    http = httpx.AsyncClient(auth=_VertexAIAuth())

    model = _QwenVLLMChatModel(
        model=MODEL,
        base_url=BASE_URL,
        api_key="vertex-ai",
        http_async_client=http,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": True},
            "thinking_token_budget": 1024,
        },
        max_tokens=2048,
        temperature=None,
    )
    messages = [HumanMessage(content="What is 2+2? Explain briefly.")]

    print(f"\n{'=' * 60}")
    print("[langchain: thinking capture via _QwenVLLMChatModel]")
    try:
        response = await model.ainvoke(messages)
        reasoning = response.additional_kwargs.get("reasoning", "")
        print(f"  content:   {response.content[:200]!r}")
        if reasoning:
            print(f"  reasoning: {reasoning[:200]!r}")
        else:
            print("  reasoning: (EMPTY — subclass not capturing model_extra!)")
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")

    print(f"\n{'=' * 60}")
    print("[langchain: two-phase thinking + structured output]")
    try:
        # Phase 1: thinking-enabled call (no response_format)
        thinking_response = await model.ainvoke(messages)
        reasoning = thinking_response.additional_kwargs.get("reasoning", "")
        print(f"  phase-1 content:   {thinking_response.content[:150]!r}")
        print(f"  phase-1 reasoning: {reasoning[:150]!r}" if reasoning else "  phase-1 reasoning: (EMPTY)")

        # Phase 2: structured output call (thinking disabled, thinking call in context)
        model_structured = model.bind(
            response_format=RESPONSE_FORMAT,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        messages_with_thinking = messages + [thinking_response]
        structured_response = await model_structured.ainvoke(messages_with_thinking)
        print(f"  phase-2 content:   {structured_response.content[:200]!r}")
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")


if __name__ == "__main__":
    asyncio.run(run())
    asyncio.run(run_langchain())
