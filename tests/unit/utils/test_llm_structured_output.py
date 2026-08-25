"""Regression tests for OpenAI-compatible manual structured output."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from ax_prover.config import LLMConfig, StructuredOutputMode
from ax_prover.models.files import Location
from ax_prover.models.messages import StructuredOutputParsingFailedFeedback
from ax_prover.models.proving import ProverAgentState, ProverResult, TargetItem
from ax_prover.prover.agent import ProverAgent
from ax_prover.utils.llm import LLMClient, _openai_structured_kwargs, agentic_loop


def test_native_pydantic_remains_the_default_openai_format() -> None:
    assert LLMConfig(model="openai:test").structured_output_mode == "native_pydantic"
    assert _openai_structured_kwargs(ProverResult) == {"response_format": ProverResult}


def test_manual_openai_format_is_strict_json_schema() -> None:
    response_format = _openai_structured_kwargs(
        ProverResult, StructuredOutputMode.JSON_SCHEMA_MANUAL
    )["response_format"]

    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "ProverResult"
    assert response_format["json_schema"]["strict"] is True
    schema = response_format["json_schema"]["schema"]
    assert schema["required"] == ["imports", "opens", "updated_theorem"]
    assert schema["additionalProperties"] is False


def test_manual_mode_rejects_non_openai_provider() -> None:
    client = LLMClient.__new__(LLMClient)
    client._base_llm = ChatAnthropic(
        model="claude-sonnet-4-5-20250929",
        api_key="test-key",
    )
    client._structured_output_mode = StructuredOutputMode.JSON_SCHEMA_MANUAL

    with pytest.raises(ValueError, match="requires an OpenAI-compatible"):
        client._structured_output_bind_kwargs(ProverResult)


def test_content_with_tool_call_survives_until_tool_execution() -> None:
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        if len(requests) == 1:
            message = {
                "role": "assistant",
                "content": "I will search for the relevant Lean declaration first.",
                "tool_calls": [
                    {
                        "id": "call-search-1",
                        "type": "function",
                        "function": {
                            "name": "search_lean_test",
                            "arguments": '{"query":"Nat.add_zero"}',
                        },
                    }
                ],
            }
            finish_reason = "tool_calls"
        else:
            message = {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "imports": [],
                        "opens": [],
                        "updated_theorem": "theorem add_zero_test (n : Nat) : n + 0 = n := by simp",
                    }
                ),
            }
            finish_reason = "stop"
        return httpx.Response(
            200,
            json={
                "id": f"chatcmpl-{len(requests)}",
                "object": "chat.completion",
                "created": 1,
                "model": "local-test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": finish_reason,
                        "logprobs": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )

    async def run() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            model = ChatOpenAI(
                model="local-test-model",
                api_key="test-key",
                base_url="http://local.test/v1",
                http_async_client=http_client,
                max_retries=0,
            )
            client = LLMClient.__new__(LLMClient)
            client._base_llm = model
            client._retry_config = {}
            client._structured_output_mode = StructuredOutputMode.JSON_SCHEMA_MANUAL
            client._trace_http_client = None

            @tool
            def search_lean_test(query: str) -> str:
                """Search for a Lean declaration."""
                assert query == "Nat.add_zero"
                return "#check Nat.add_zero"

            class TestToolNode:
                def __init__(self, tools):
                    self.tools = {available.name: available for available in tools}

                async def ainvoke(self, state):
                    call = state["messages"][-1].tool_calls[0]
                    result = self.tools[call["name"]].invoke(call["args"])
                    return {
                        "messages": [
                            ToolMessage(
                                content=result,
                                name=call["name"],
                                tool_call_id=call["id"],
                            )
                        ]
                    }

            with patch("ax_prover.utils.llm.ToolNode", TestToolNode):
                response = await agentic_loop(
                    client,
                    [HumanMessage(content="Prove add_zero_test")],
                    tools=[search_lean_test],
                    output_schema=ProverResult,
                    max_tool_iterations=1,
                )

        result = ProverResult.model_validate_json(response.text)
        assert result.updated_theorem.startswith("theorem add_zero_test")

    asyncio.run(run())

    assert len(requests) == 2
    assert requests[0]["response_format"]["type"] == "json_schema"
    second_messages = requests[1]["messages"]
    assistant = next(message for message in second_messages if message["role"] == "assistant")
    assert assistant["content"].startswith("I will search")
    assert assistant["tool_calls"][0]["function"]["name"] == "search_lean_test"
    tool_result = next(message for message in second_messages if message["role"] == "tool")
    assert tool_result["content"] == "#check Nat.add_zero"


def test_invalid_final_json_becomes_graph_feedback_not_transport_error(tmp_path) -> None:
    source = tmp_path / "Test" / "Module.lean"
    source.parent.mkdir()
    source.write_text("theorem thm : True := by\n  sorry\n", encoding="utf-8")
    with patch.object(ProverAgent, "__init__", lambda self, *args, **kwargs: None):
        agent = ProverAgent.__new__(ProverAgent)
    agent.logger = MagicMock()
    agent.config = SimpleNamespace(
        max_iterations=50,
        user_comments=None,
        reasoning_trace=SimpleNamespace(
            run_id="run",
            problem_uuid="problem",
        ),
    )
    agent.runtime = SimpleNamespace(
        base_folder=str(tmp_path),
        config=SimpleNamespace(max_tool_calling_iterations=1),
    )
    agent.llm_client = MagicMock()
    agent.proposer_tools = []
    agent._trace_call_index = 0
    agent.reasoning_trace_writer = MagicMock()
    agent.reasoning_trace_writer.record_proposer.return_value = {
        "call_id": "run:problem:proposer:1:1",
        "alignment_status": "aligned",
        "reasoning": "attempt",
    }
    state = ProverAgentState(
        item=TargetItem(
            location=Location(name="thm", module_path="Test.Module"),
            original_source=source.read_text(encoding="utf-8"),
        )
    )

    with patch(
        "ax_prover.prover.agent.agentic_loop",
        new=AsyncMock(return_value=AIMessage(content="not valid final JSON")),
    ):
        update = asyncio.run(agent._proposer_node(state, {}))

    feedback = update["messages"][0]
    assert isinstance(feedback, StructuredOutputParsingFailedFeedback)
    agent.reasoning_trace_writer.record_transport_failure.assert_not_called()
    agent.reasoning_trace_writer.record_lean_check.assert_called_once_with(
        call_id="run:problem:proposer:1:1",
        problem_uuid="problem",
        theorem_name="thm",
        iteration=1,
        outcome="not_run_structured_output_parse_failed",
        success=False,
        feedback_type="structured_output_parsing_failed",
        diagnostics=feedback.error_message,
        duration_seconds=0.0,
    )
