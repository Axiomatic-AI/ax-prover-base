"""Tests for ProverAgent input-token budget resolution and build-output trimming.

Covers the unprofiled-provider fallback (DeepSeek/qwen have no langchain profile) and
the token→character conversion used when trimming oversized build output.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from ax_prover.prover.agent import (
    CHARS_PER_TOKEN,
    DEFAULT_MAX_INPUT_TOKENS,
    ProverAgent,
)


def _agent_with(max_input_tokens=None, profile=None) -> ProverAgent:
    """A ProverAgent shell (no heavy __init__) wired for budget-resolution tests."""
    agent = ProverAgent.__new__(ProverAgent)
    agent.logger = MagicMock()
    agent.config = SimpleNamespace(prover_llm=SimpleNamespace(max_input_tokens=max_input_tokens))
    agent.llm_client = SimpleNamespace(profile=profile or {})
    return agent


class TestResolveMaxInputTokens:
    def test_falls_back_to_default_for_unprofiled_provider(self):
        # No config value and an empty profile (DeepSeek/qwen) -> conservative default.
        agent = _agent_with(max_input_tokens=None, profile={})
        assert agent._resolve_max_input_tokens() == DEFAULT_MAX_INPUT_TOKENS

    def test_prefers_config_value(self):
        agent = _agent_with(max_input_tokens=64_000, profile={"max_input_tokens": 999_999})
        assert agent._resolve_max_input_tokens() == 64_000

    def test_uses_profile_when_config_unset(self):
        agent = _agent_with(max_input_tokens=None, profile={"max_input_tokens": 200_000})
        assert agent._resolve_max_input_tokens() == 200_000


class TestBuildErrorProcessing:
    def _agent(self, max_input_tokens: int) -> ProverAgent:
        agent = ProverAgent.__new__(ProverAgent)
        agent.max_input_tokens = max_input_tokens
        return agent

    def test_keeps_message_within_budget(self):
        agent = self._agent(max_input_tokens=100)  # char budget = 200
        message = "short error"
        assert agent._build_error_processing(message) == message

    def test_uses_char_budget_not_token_budget(self):
        # 150 chars exceeds the raw token count (100) but fits the char budget
        # (100 * CHARS_PER_TOKEN = 200), so it must be kept intact. This is the fix:
        # the old 1-char-per-token bound would have truncated it.
        agent = self._agent(max_input_tokens=100)
        message = "e" * 150
        assert agent._build_error_processing(message) == message

    def test_truncates_message_beyond_budget(self):
        agent = self._agent(max_input_tokens=100)  # char budget = 200
        message = "x" * 500
        out = agent._build_error_processing(message)

        assert len(out) < len(message)  # truncated
        assert "build output too long" in out  # separator inserted
        assert out.startswith("x") and out.endswith("x")  # head and tail preserved
        assert len(out) <= int(agent.max_input_tokens * CHARS_PER_TOKEN)  # respects budget
