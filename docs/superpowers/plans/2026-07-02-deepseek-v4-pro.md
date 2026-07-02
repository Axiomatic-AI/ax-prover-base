# DeepSeek V4 Pro Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the ability to run the prover reliably with DeepSeek's `deepseek-v4-pro` model via its OpenAI-compatible API, including reasoning-effort selection.

**Architecture:** Route DeepSeek through `model_provider: "openai"` (the proven `qwen` pattern). Because that makes `LLMClient._base_llm` a `ChatOpenAI` instance indistinguishable from real OpenAI, add a `_is_deepseek_model` detector and branch two behaviors: structured output uses `json_object` mode (DeepSeek rejects `json_schema` strict) with the schema injected into the prompt, and tool binding omits `strict`. Reasoning effort is a native `ChatOpenAI` field, so it is config-only.

**Tech Stack:** Python, LangChain (`langchain_openai`), OmegaConf YAML configs, Pydantic, pytest.

## Global Constraints

- DeepSeek endpoint: OpenAI-compatible, base URL `https://api.deepseek.com`, auth via `DEEPSEEK_API_KEY`.
- Model id: `deepseek-v4-pro` (also `deepseek-v4-flash` exists).
- `response_format: json_schema` (strict) is **unavailable** on DeepSeek — must use `{"type": "json_object"}`.
- `json_object` mode requires the literal word **"JSON"** in the prompt.
- Valid `reasoning_effort` values: `low`, `medium`, `high`, `max`, `xhigh` (default `high`).
- Structured output is extracted by parsing `response.text` as JSON (`model_validate_json`). Do not change that contract.
- Keep docstrings concise; do not add defensive try/except without a specific handling strategy (per repo CLAUDE.md).
- All unit tests live under `tests/unit/`; run with `.venv/bin/pytest tests/unit`.

---

### Task 1: DeepSeek detection helper

**Files:**
- Modify: `src/ax_prover/utils/llm.py` (add module-level `_is_deepseek_model`)
- Test: `tests/unit/utils/test_llm.py` (create)

**Interfaces:**
- Produces: `_is_deepseek_model(llm: BaseChatModel) -> bool` — returns True when `llm` is a `ChatOpenAI` whose model name starts with `deepseek` or whose base URL contains `deepseek`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/utils/test_llm.py`:

```python
"""Tests for LLM factory helpers and DeepSeek-specific structured-output handling."""

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from ax_prover.utils.llm import _is_deepseek_model


class SamplePerson(BaseModel):
    name: str
    age: int


def test_is_deepseek_true_by_model_name():
    llm = ChatOpenAI(model="deepseek-v4-pro", api_key="test-key", base_url="https://api.deepseek.com")
    assert _is_deepseek_model(llm) is True


def test_is_deepseek_true_by_base_url():
    llm = ChatOpenAI(model="some-proxy-model", api_key="test-key", base_url="https://api.deepseek.com/v1")
    assert _is_deepseek_model(llm) is True


def test_is_deepseek_false_for_plain_openai():
    llm = ChatOpenAI(model="gpt-4o", api_key="test-key")
    assert _is_deepseek_model(llm) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/utils/test_llm.py -v`
Expected: FAIL with `ImportError: cannot import name '_is_deepseek_model'`

- [ ] **Step 3: Write minimal implementation**

In `src/ax_prover/utils/llm.py`, add after the `_PROVIDER_API_KEY_ENV` dict (near top, before `create_llm`):

```python
def _is_deepseek_model(llm: BaseChatModel) -> bool:
    """True when this ChatOpenAI instance is backed by the DeepSeek endpoint.

    DeepSeek is routed via model_provider="openai", so it is a ChatOpenAI
    instance and cannot be told apart from real OpenAI by isinstance alone.
    """
    if not isinstance(llm, ChatOpenAI):
        return False
    model = (getattr(llm, "model_name", "") or "").lower()
    base_url = str(getattr(llm, "openai_api_base", "") or "").lower()
    return model.startswith("deepseek") or "deepseek" in base_url
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/utils/test_llm.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/ax_prover/utils/llm.py tests/unit/utils/test_llm.py
git commit -m "feat: detect DeepSeek-backed ChatOpenAI in LLMClient"
```

---

### Task 2: DeepSeek structured-output uses json_object

**Files:**
- Modify: `src/ax_prover/utils/llm.py:181-203` (`LLMClient._structured_output_bind_kwargs`)
- Test: `tests/unit/utils/test_llm.py`

**Interfaces:**
- Consumes: `_is_deepseek_model` (Task 1).
- Produces: for a DeepSeek client, `LLMClient._structured_output_bind_kwargs(schema)` returns `{"response_format": {"type": "json_object"}}` (never a pydantic model / `json_schema`).

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/utils/test_llm.py`:

```python
from ax_prover.config import LLMConfig
from ax_prover.utils.llm import LLMClient


def _deepseek_client() -> LLMClient:
    config = LLMConfig(
        model="deepseek-v4-pro",
        provider_config={
            "model_provider": "openai",
            "base_url": "https://api.deepseek.com",
            "api_key": "test-key",
            "temperature": None,
            "max_tokens": None,
            "reasoning_effort": "high",
        },
    )
    return LLMClient(config)


def test_structured_kwargs_deepseek_uses_json_object():
    client = _deepseek_client()
    kwargs = client._structured_output_bind_kwargs(SamplePerson)
    assert kwargs == {"response_format": {"type": "json_object"}}


def test_reasoning_effort_reaches_chat_openai():
    # Confirms the config's reasoning_effort flows through init_chat_model to ChatOpenAI.
    client = _deepseek_client()
    assert getattr(client._base_llm, "reasoning_effort", None) == "high"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/utils/test_llm.py::test_structured_kwargs_deepseek_uses_json_object -v`
Expected: FAIL — the current `ChatOpenAI` branch returns `{"response_format": SamplePerson}`, not json_object.

- [ ] **Step 3: Write minimal implementation**

In `src/ax_prover/utils/llm.py`, add a DeepSeek branch at the **top** of `_structured_output_bind_kwargs`, before the `ChatAnthropic` check:

```python
    def _structured_output_bind_kwargs(self, schema: type[BaseModel]) -> dict:
        """Return provider-specific kwargs that constrain the output to a JSON schema.

        These kwargs are passed via bind() so the response stays as an AIMessage, unlike
        `with_structured_output`, which prevents the use of any tools and forces the output
        to be an instance of the schema.
        """
        # DeepSeek rejects json_schema strict mode; use json_object and inject the
        # schema into the prompt via _maybe_inject_schema (see ainvoke).
        if _is_deepseek_model(self._base_llm):
            return {"response_format": {"type": "json_object"}}

        if isinstance(self._base_llm, ChatAnthropic):
            model_name = getattr(self._base_llm, "model", "")  # Need to check 4.5 or 4.6+
            return _anthropic_structured_kwargs(model_name, schema)
        # ... rest unchanged
```

Leave the existing `ChatAnthropic` / `ChatGoogleGenerativeAI` / `ChatOpenAI` branches untouched below the new block.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/utils/test_llm.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ax_prover/utils/llm.py tests/unit/utils/test_llm.py
git commit -m "feat: DeepSeek structured output via json_object mode"
```

---

### Task 3: Disable strict tool binding for DeepSeek

**Files:**
- Modify: `src/ax_prover/utils/llm.py:155-179` (`LLMClient._get_runnable`; extract strict decision into `_use_strict_tools`)
- Test: `tests/unit/utils/test_llm.py`

**Interfaces:**
- Consumes: `_is_deepseek_model` (Task 1).
- Produces: `LLMClient._use_strict_tools(output_schema) -> bool` — True only for real OpenAI (`ChatOpenAI` and not DeepSeek) when an `output_schema` is present.

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/utils/test_llm.py`:

```python
import os

import pytest


def _openai_client(monkeypatch) -> LLMClient:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    return LLMClient(LLMConfig(model="openai:gpt-4o", provider_config={"api_key": "test-key"}))


def test_deepseek_does_not_use_strict_tools():
    client = _deepseek_client()
    assert client._use_strict_tools(SamplePerson) is False


def test_openai_uses_strict_tools_with_schema(monkeypatch):
    client = _openai_client(monkeypatch)
    assert client._use_strict_tools(SamplePerson) is True


def test_openai_no_strict_without_schema(monkeypatch):
    client = _openai_client(monkeypatch)
    assert client._use_strict_tools(None) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/utils/test_llm.py -k use_strict -v`
Expected: FAIL with `AttributeError: 'LLMClient' object has no attribute '_use_strict_tools'`

- [ ] **Step 3: Write minimal implementation**

In `src/ax_prover/utils/llm.py`, add the method and use it in `_get_runnable`:

```python
    def _use_strict_tools(self, output_schema: type[BaseModel] | None) -> bool:
        """Strict tool schemas are OpenAI-only; DeepSeek rejects them like strict json_schema."""
        return bool(
            isinstance(self._base_llm, ChatOpenAI)
            and not _is_deepseek_model(self._base_llm)
            and output_schema
        )
```

Then in `_get_runnable`, replace the strict line:

```python
        if tools:
            # OpenAI requires strict=True when combining tools with structured output;
            # DeepSeek and other providers omit the field.
            strict = True if self._use_strict_tools(output_schema) else None
            model = self._base_llm.bind_tools(tools, strict=strict)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/utils/test_llm.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ax_prover/utils/llm.py tests/unit/utils/test_llm.py
git commit -m "feat: omit strict tool schema for DeepSeek"
```

---

### Task 4: Inject JSON schema into the prompt for DeepSeek

**Files:**
- Modify: `src/ax_prover/utils/llm.py` (add `import json`; add `LLMClient._maybe_inject_schema`; call it in `ainvoke`)
- Test: `tests/unit/utils/test_llm.py`

**Interfaces:**
- Consumes: `_is_deepseek_model` (Task 1).
- Produces: `LLMClient._maybe_inject_schema(messages, output_schema) -> LanguageModelInput` — when the model is DeepSeek, `output_schema` is not None, and `messages` is a list, returns a new list with one appended `HumanMessage` containing the JSON schema and a JSON instruction; otherwise returns `messages` unchanged. `ainvoke` calls it before building the runnable.

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/utils/test_llm.py`:

```python
from langchain_core.messages import HumanMessage, SystemMessage


def test_schema_injection_appends_json_instruction_for_deepseek():
    client = _deepseek_client()
    messages = [SystemMessage(content="sys"), HumanMessage(content="prove it")]
    out = client._maybe_inject_schema(messages, SamplePerson)

    assert len(out) == 3
    assert isinstance(out[-1], HumanMessage)
    assert "JSON" in out[-1].content
    assert "properties" in out[-1].content  # the schema itself is embedded
    # original list is not mutated
    assert len(messages) == 2


def test_no_schema_injection_without_schema():
    client = _deepseek_client()
    messages = [HumanMessage(content="hi")]
    assert client._maybe_inject_schema(messages, None) == messages


def test_no_schema_injection_for_non_deepseek(monkeypatch):
    client = _openai_client(monkeypatch)
    messages = [HumanMessage(content="hi")]
    assert client._maybe_inject_schema(messages, SamplePerson) == messages
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/utils/test_llm.py -k schema_injection -v`
Expected: FAIL with `AttributeError: 'LLMClient' object has no attribute '_maybe_inject_schema'`

- [ ] **Step 3: Write minimal implementation**

In `src/ax_prover/utils/llm.py`, add `import json` at the top (with the other stdlib imports), then add the method and wire it into `ainvoke`:

```python
    def _maybe_inject_schema(
        self,
        messages: LanguageModelInput,
        output_schema: type[BaseModel] | None,
    ) -> LanguageModelInput:
        """Append a JSON-schema instruction for DeepSeek's json_object mode.

        json_object mode neither enforces nor communicates the schema and requires
        the literal word "JSON" in the prompt, so the schema is injected explicitly.
        The instruction permits tool use first so the proposer's search tools still work.
        Non-DeepSeek providers pass the schema natively and are left unchanged.
        """
        if output_schema is None or not _is_deepseek_model(self._base_llm):
            return messages
        if not isinstance(messages, list):
            return messages

        schema_json = json.dumps(output_schema.model_json_schema(), indent=2)
        instruction = (
            "You may use the available tools as needed. When you give your final "
            "answer, respond with a single JSON object and no other text, no markdown "
            "code fences. The JSON object must conform to this JSON schema:\n"
            f"{schema_json}"
        )
        return list(messages) + [HumanMessage(content=instruction)]
```

Update `ainvoke` to inject before building the runnable:

```python
    async def ainvoke(
        self,
        messages: LanguageModelInput,
        tools: list[BaseTool] | None = None,
        output_schema: type[BaseModel] | None = None,
        retry_config: dict | None = None,
    ) -> AIMessage:
        """Invoke with optional tools, structured output, and retry."""
        effective_retry = retry_config or self._retry_config
        messages = self._maybe_inject_schema(messages, output_schema)
        runnable = self._get_runnable(
            tools=tools, output_schema=output_schema, retry_config=effective_retry
        )
        return await runnable.ainvoke(messages)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/utils/test_llm.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ax_prover/utils/llm.py tests/unit/utils/test_llm.py
git commit -m "feat: inject JSON schema into prompt for DeepSeek json_object mode"
```

---

### Task 5: Make get_reasoning fall back to DeepSeek reasoning_content

**Files:**
- Modify: `src/ax_prover/utils/llm.py:85-90` (`get_reasoning`)
- Test: `tests/unit/utils/test_llm.py`

**Interfaces:**
- Produces: `get_reasoning(response)` returns reasoning from content blocks; if none, falls back to `response.additional_kwargs["reasoning_content"]` (where `langchain_openai` places DeepSeek's separate reasoning field).

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/utils/test_llm.py`:

```python
from langchain_core.messages import AIMessage

from ax_prover.utils.llm import get_reasoning


def test_get_reasoning_falls_back_to_deepseek_reasoning_content():
    response = AIMessage(
        content="final answer",
        additional_kwargs={"reasoning_content": "step-by-step thinking"},
    )
    assert get_reasoning(response) == "step-by-step thinking"


def test_get_reasoning_empty_when_no_reasoning():
    response = AIMessage(content="hi")
    assert get_reasoning(response) == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/utils/test_llm.py -k get_reasoning -v`
Expected: FAIL — `test_get_reasoning_falls_back_to_deepseek_reasoning_content` returns `""` instead of the reasoning_content.

- [ ] **Step 3: Write minimal implementation**

Replace `get_reasoning` in `src/ax_prover/utils/llm.py`:

```python
def get_reasoning(response: AIMessage) -> str:
    """Extract reasoning from an LLM response.

    Prefers native reasoning content blocks; falls back to DeepSeek's separate
    `reasoning_content` field, which langchain_openai surfaces in additional_kwargs.
    """
    reasoning = "\n\n".join(
        [msg.get("reasoning", "") for msg in response.content_blocks if msg["type"] == "reasoning"]
    )
    if not reasoning:
        reasoning = response.additional_kwargs.get("reasoning_content", "") or ""
    return reasoning
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/utils/test_llm.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ax_prover/utils/llm.py tests/unit/utils/test_llm.py
git commit -m "feat: get_reasoning falls back to DeepSeek reasoning_content"
```

---

### Task 6: Add DeepSeek LLM config entries

**Files:**
- Modify: `configs/llms.yaml` (add `deepseek_v4_pro` entry)
- Modify: `src/ax_prover/configs/llms.yaml` (add the same entry to the bundled copy)
- Test: `tests/unit/utils/test_llm.py`

**Interfaces:**
- Produces: `llm_configs.deepseek_v4_pro` resolvable via OmegaConf, with `model: "deepseek-v4-pro"`, `provider_config.model_provider: "openai"`, `provider_config.base_url: "https://api.deepseek.com"`, `provider_config.api_key: ${oc.env:DEEPSEEK_API_KEY}`, `provider_config.reasoning_effort: "high"`.

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/utils/test_llm.py`:

```python
from pathlib import Path

from omegaconf import OmegaConf

_REPO_ROOT = Path(__file__).resolve().parents[3]


def test_deepseek_llms_config_entry(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "dummy-key")
    cfg = OmegaConf.load(_REPO_ROOT / "configs" / "llms.yaml")
    ds = cfg.llm_configs.deepseek_v4_pro
    assert ds.model == "deepseek-v4-pro"
    assert ds.provider_config.model_provider == "openai"
    assert ds.provider_config.base_url == "https://api.deepseek.com"
    assert ds.provider_config.reasoning_effort == "high"
    # api_key resolves from DEEPSEEK_API_KEY via the oc.env resolver
    assert ds.provider_config.api_key == "dummy-key"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/utils/test_llm.py::test_deepseek_llms_config_entry -v`
Expected: FAIL with `omegaconf.errors.ConfigAttributeError: Key 'deepseek_v4_pro' not in 'llm_configs'`

- [ ] **Step 3: Write minimal implementation**

Append to the `llm_configs:` block in **both** `configs/llms.yaml` and `src/ax_prover/configs/llms.yaml`:

```yaml
  # DeepSeek models (OpenAI-compatible endpoint)
  deepseek_v4_pro:
    model: "deepseek-v4-pro"
    provider_config:
      model_provider: "openai"
      base_url: "https://api.deepseek.com"
      api_key: ${oc.env:DEEPSEEK_API_KEY}
      temperature: null
      max_tokens: null
      reasoning_effort: "high"  # low | medium | high | max | xhigh
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/utils/test_llm.py::test_deepseek_llms_config_entry -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add configs/llms.yaml src/ax_prover/configs/llms.yaml tests/unit/utils/test_llm.py
git commit -m "feat: add deepseek_v4_pro LLM config entry"
```

---

### Task 7: Add DeepSeek run config

**Files:**
- Create: `configs/deepseek_local.yaml`
- Test: `tests/unit/utils/test_llm.py`

**Interfaces:**
- Consumes: `deepseek_v4_pro` entry (Task 6), `configs/tools.yaml`, `configs/default.yaml`.
- Produces: a run config selecting DeepSeek with the full local tool set and `restrict_to_proof_body: true`, mirroring `configs/opus48_local.yaml`.

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/utils/test_llm.py`:

```python
def test_deepseek_local_run_config_selects_deepseek(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "dummy-key")
    from ax_prover.utils.config import _load_yaml_with_imports

    cfg = _load_yaml_with_imports(str(_REPO_ROOT / "configs" / "deepseek_local.yaml"))
    assert cfg.prover.prover_llm.model == "deepseek-v4-pro"
    assert cfg.prover.restrict_to_proof_body is True
    assert "search_lean_local" in cfg.prover.proposer_tools
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/utils/test_llm.py::test_deepseek_local_run_config_selects_deepseek -v`
Expected: FAIL with a file-not-found / load error for `configs/deepseek_local.yaml`.

- [ ] **Step 3: Write minimal implementation**

Create `configs/deepseek_local.yaml` (mirror of `configs/opus48_local.yaml`):

```yaml
# DeepSeek V4 Pro with the default tool set (public leansearch.net + web + local search).
# Uses DeepSeek's OpenAI-compatible endpoint; requires DEEPSEEK_API_KEY.
import:
  - llms.yaml
  - tools.yaml
  - default.yaml

prover:
  prover_llm: ${llm_configs.deepseek_v4_pro}
  # AI4Math competition rules: only the sorry placeholders may change.
  restrict_to_proof_body: true
  proposer_tools:
    search_lean: ${tool_configs.search_lean_search}
    search_web: ${tool_configs.search_web}
    search_lean_local: ${tool_configs.search_lean_local}
    search_cslib: ${tool_configs.search_cslib}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/utils/test_llm.py::test_deepseek_local_run_config_selects_deepseek -v`
Expected: PASS

If `_load_yaml_with_imports` requires a different call signature, inspect `src/ax_prover/utils/config.py` and match it (the import mechanism is the same one `opus48_local.yaml` relies on).

- [ ] **Step 5: Commit**

```bash
git add configs/deepseek_local.yaml tests/unit/utils/test_llm.py
git commit -m "feat: add deepseek_local run config"
```

---

### Task 8: Live end-to-end + parallelism smoke test (opt-in)

**Files:**
- Test: `tests/unit/utils/test_llm.py` (add a network-gated test)

**Interfaces:**
- Consumes: `deepseek_v4_pro` config, `LLMClient.ainvoke` with `output_schema`.
- Produces: a test, skipped unless `DEEPSEEK_API_KEY` is available, that runs multiple structured-output calls concurrently against the live endpoint and asserts each parses — confirming endpoint wiring, json_object structured output, and parallel safety without needing a Lean project.

- [ ] **Step 1: Write the test**

Add to `tests/unit/utils/test_llm.py`:

```python
import asyncio

from ax_prover.utils.config import load_env_secrets


def _deepseek_key_available() -> bool:
    load_env_secrets()  # loads .env.secrets into os.environ if present
    return bool(os.environ.get("DEEPSEEK_API_KEY"))


@pytest.mark.skipif(not _deepseek_key_available(), reason="DEEPSEEK_API_KEY not available")
def test_live_deepseek_structured_output_parallel():
    load_env_secrets()
    config = LLMConfig(
        model="deepseek-v4-pro",
        provider_config={
            "model_provider": "openai",
            "base_url": "https://api.deepseek.com",
            "api_key": os.environ["DEEPSEEK_API_KEY"],
            "reasoning_effort": "high",
        },
        retry_config={"stop_after_attempt": 3},
    )
    client = LLMClient(config)

    async def one(person_desc: str) -> SamplePerson:
        response = await client.ainvoke(
            [HumanMessage(content=f"Extract the person: {person_desc}")],
            output_schema=SamplePerson,
        )
        return SamplePerson.model_validate_json(response.text)

    async def run_all():
        return await asyncio.gather(
            one("Alice is 30 years old"),
            one("Bob is 42 years old"),
            one("Carol is 25 years old"),
        )

    results = asyncio.run(run_all())
    assert len(results) == 3
    assert all(isinstance(r, SamplePerson) for r in results)
    assert {r.name for r in results} == {"Alice", "Bob", "Carol"}
```

- [ ] **Step 2: Run the test**

Run: `.venv/bin/pytest tests/unit/utils/test_llm.py::test_live_deepseek_structured_output_parallel -v`
Expected: PASS if `DEEPSEEK_API_KEY` is set (in shell env or `.env.secrets`); otherwise SKIPPED. This is the primary reliability check: three structured-output calls succeed concurrently and each parses into `SamplePerson`.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/utils/test_llm.py
git commit -m "test: live parallel structured-output smoke test for DeepSeek"
```

- [ ] **Step 4: Full-run verification (manual, on a machine with the Lean project)**

Document the command for a full prove/experiment run (not automated here because it needs the Lean repo):

```bash
# Single theorem:
ax-prover --config configs/deepseek_local.yaml prove <Module.Path:theorem> --folder <lean-project>

# Concurrent batch (confirms real parallelism):
ax-prover --config configs/deepseek_local.yaml experiment <dataset> --folder <lean-project> --max-concurrency 5
```

Confirm: proposer produces a parsed proposal, reviewer produces a parsed `ReviewDecision`, and concurrent items run without shared-state errors. Record the outcome in the PR description.

---

### Task 9: Full suite + lint gate

**Files:** none (verification only)

- [ ] **Step 1: Run the full unit suite**

Run: `.venv/bin/pytest tests/unit -q`
Expected: all pass (the live test SKIPs without a key).

- [ ] **Step 2: Lint / format**

Run: `ruff check --fix . && ruff format .`
Expected: clean.

- [ ] **Step 3: Commit any formatting changes**

```bash
git add -A
git commit -m "chore: ruff format for DeepSeek support" || echo "nothing to format"
```

---

## Self-Review

**Spec coverage:**
- Endpoint decision (OpenAI-compatible) → Tasks 6, 7 (config). ✅
- DeepSeek detection trap → Task 1. ✅
- Structured output json_object → Task 2. ✅
- Strict tool binding disabled → Task 3. ✅
- Schema injection with "JSON" literal → Task 4. ✅
- Reasoning-effort selection (config-only) → Task 6 (`reasoning_effort: high`) + verified by `test_deepseek_llms_config_entry` and reaching `ChatOpenAI`. ✅
- reasoning_content (minor) → Task 5. ✅
- Validation/retry reuse (no new machinery) → covered by existing `model_validate_json` path; no task needed (unchanged contract). ✅
- Parallelism → Task 8 concurrent smoke test + documented batch run. ✅
- Verification (concurrent, not single prove) → Task 8. ✅

**Placeholder scan:** No TBD/TODO; every code step has complete code. Task 7 Step 4 notes a possible signature check for `_load_yaml_with_imports` — this is a real fallback instruction, not a placeholder.

**Type consistency:** `_is_deepseek_model` (module fn), `_use_strict_tools`, `_maybe_inject_schema`, `get_reasoning`, `_deepseek_client`, `_openai_client`, `SamplePerson` used consistently across tasks. `_deepseek_client` defined in Task 2 and reused in Tasks 3, 4; `_openai_client` defined in Task 3 and reused in Task 4; `_REPO_ROOT` defined in Task 6 and reused in Task 7.
