"""Diagnostic: structured-output via tools on the configured Anthropic model.

Reproduces the proposer's forced-final structured call after a real tool round
and dumps every content block + stop_reason so we can see WHERE the JSON went.
Run from repo root with the venv active:  python diag_structured.py
"""

import asyncio

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from ax_prover.config import LLMConfig
from ax_prover.models.proving import ProverResult
from ax_prover.utils.llm import LLMClient

load_dotenv(".env.secrets")


@tool
def search_lean_local(query: str) -> str:
    """Search the local Lean library. ALWAYS call before proving."""
    return "Found: theorem Nat.add_comm (a b : Nat) : a + b = b + a"


def dump(label, resp):
    print(f"\n--- {label} ---")
    print("  .text repr:", repr(resp.text)[:300])
    print("  content type:", type(resp.content).__name__)
    if isinstance(resp.content, list):
        for i, b in enumerate(resp.content):
            if isinstance(b, dict):
                keys = list(b.keys())
                text = repr(b.get("text"))[:120] if "text" in b else ""
                print(f"  block[{i}] type={b.get('type')} keys={keys} text={text}")
            else:
                print(f"  block[{i}] {type(b).__name__}: {repr(b)[:120]}")
    rm = getattr(resp, "response_metadata", None)
    if rm:
        print("  response_metadata keys:", list(rm.keys()))
        print("  stop_reason:", rm.get("stop_reason"))


async def main():
    client = LLMClient(
        LLMConfig(
            model="anthropic:claude-opus-4-8",
            provider_config={
                "temperature": 1.0,
                "max_tokens": None,
                "effort": "high",
                "thinking": {"type": "adaptive"},
            },
            retry_config={},
        )
    )
    sys = SystemMessage(
        content="You are a Lean 4 prover. Call search_lean_local once, then output the structured ProverResult."
    )
    human = HumanMessage(content="Prove theorem t : 1 + 1 = 2 := by sorry. Search 'add' first.")

    first = await client.ainvoke(
        [sys, human], tools=[search_lean_local], output_schema=ProverResult
    )
    print("first.tool_calls:", [tc["name"] for tc in first.tool_calls])
    dump("FIRST", first)
    if not first.tool_calls:
        return

    tc = first.tool_calls[0]
    tmsg = ToolMessage(content=search_lean_local.invoke(tc["args"]), tool_call_id=tc["id"])
    final = await client.ainvoke(
        [sys, human, first, tmsg, HumanMessage(content="NO MORE TOOL CALLS ALLOWED.")],
        output_schema=ProverResult,
    )
    dump("FINAL", final)
    try:
        ProverResult.model_validate_json(final.text)
        print("\n>>> PARSE OK")
    except Exception as e:
        print("\n>>> PARSE FAIL:", str(e)[:160])


asyncio.run(main())
