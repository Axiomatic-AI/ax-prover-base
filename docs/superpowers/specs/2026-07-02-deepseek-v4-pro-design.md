# DeepSeek V4 Pro Support — Design

**Date:** 2026-07-02
**Status:** Approved for planning
**Branch:** `adding_deepseek`

## Goal

Add the ability to run the prover with DeepSeek's `deepseek-v4-pro` model, reliably. "Reliably" means the proposer, reviewer, memory, and summary nodes all produce structured output that parses correctly, and tool calling (search tools) works end-to-end.

## Endpoint decision

Route DeepSeek through its **native OpenAI-compatible API** (`model_provider: "openai"`), not the Anthropic-compatible endpoint.

Rationale:
- Proven precedent: the existing `qwen` config already routes a non-native model through an OpenAI-compatible endpoint and works.
- The Anthropic structured-output path (`_anthropic_structured_kwargs`) parses the model name for a Claude `4.6`-style version to select the output-format shape; a `deepseek-*` name silently falls back to the deprecated `output_format` — quiet unreliability.
- DeepSeek does not honor Anthropic betas / `thinking` / `effort` params.

Base URL: `https://api.deepseek.com`. Auth: `DEEPSEEK_API_KEY`.

## Findings from live API probes (2026-07-02)

Probed the real endpoint with a valid key:

| Capability | Result |
|---|---|
| `GET /models` | serves `deepseek-v4-pro` and `deepseek-v4-flash` |
| Tool calling | ✅ returns well-formed `tool_calls` with JSON arguments |
| `response_format: {"type":"json_object"}` | ✅ returns valid JSON in message `content` |
| `response_format: {"type":"json_schema", strict:true}` | ❌ HTTP 400 `"This response_format type is unavailable now"` |
| `json_object` + `tools` in one call | ✅ coexist; model still emits tool calls |
| Assistant `reasoning_content` echoed back in message history | ✅ no error |
| `reasoning_effort` | ✅ accepts exactly `low`, `medium`, `high`, `max`, `xhigh` (invalid → 400 listing valid set) |
| `thinking: {"type": "disabled"}` | ✅ turns reasoning off (0 reasoning tokens) |

Key structural fact about the codebase: the prover extracts structured output by **parsing the response text as JSON** — `ProverResult.model_validate_json(response.text)` at `src/ax_prover/prover/agent.py:283`, and the reviewer (`agent.py:407`), memory (`memory.py:181`), and summary (`agent.py:495`) calls follow the same content-parsing pattern. Extraction does **not** depend on langchain's `parsed` object. Therefore any path that puts schema-matching JSON into `content` works — which `json_object` mode does.

## The core trap

Routing via `model_provider: "openai"` makes `LLMClient._base_llm` a `ChatOpenAI` instance, **indistinguishable from real OpenAI by `isinstance`**. Two OpenAI-only behaviors then fire and break against DeepSeek:

1. `_structured_output_bind_kwargs` → `_openai_structured_kwargs` returns `{"response_format": <pydantic model>}`, which langchain serializes to `json_schema` strict mode → **HTTP 400**.
2. `_get_runnable` sets `strict=True` on tool binding whenever an output schema is present (`llm.py:170`) → DeepSeek is expected to reject strict tool schemas the same way it rejects strict `response_format`.

The design must **detect that a `ChatOpenAI` instance is actually DeepSeek** and branch both behaviors.

## Approach A (selected)

`json_object` mode + auto-injected schema + the existing validate/retry loop.

### 1. Configuration

Add to `configs/llms.yaml` (and mirror into `src/ax_prover/configs/llms.yaml` if that copy is used at runtime):

```yaml
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

- Model string has no `provider:` prefix, so `create_llm`'s `_PROVIDER_API_KEY_ENV` check is skipped (same as `qwen`) — it will not demand `OPENAI_API_KEY`.
- `api_key` is passed explicitly from `DEEPSEEK_API_KEY` via OmegaConf's `oc.env` resolver.

Add a run config `configs/deepseek_local.yaml` mirroring `configs/opus48_local.yaml` (same tool set: `search_lean`, `search_web`, `search_lean_local`, `search_cslib`; `restrict_to_proof_body: true`), but with `prover_llm: ${llm_configs.deepseek_v4_pro}`.

### 2. DeepSeek detection in `LLMClient`

Add a helper that identifies a DeepSeek-backed `ChatOpenAI`:

```python
def _is_deepseek(self) -> bool:
    if not isinstance(self._base_llm, ChatOpenAI):
        return False
    model = (getattr(self._base_llm, "model_name", "") or "").lower()
    base_url = str(getattr(self._base_llm, "openai_api_base", "") or "").lower()
    return model.startswith("deepseek") or "deepseek.com" in base_url
```

(Exact attribute names to be confirmed during implementation; detection may key on either the model name or the base URL.)

### 3. Structured output branch

In `_structured_output_bind_kwargs`, add a DeepSeek branch **before** the generic `ChatOpenAI` branch:

- Return `{"response_format": {"type": "json_object"}}` — never a pydantic model, never `json_schema`.

Schema communication: because `json_object` mode does not enforce or communicate the schema, `ainvoke` must inject, when the model is DeepSeek and an `output_schema` is provided:
- the pydantic JSON schema (`schema.model_json_schema()`), and
- an instruction to respond with a single JSON object matching it. The instruction must contain the literal word **"JSON"** (DeepSeek requires it for `json_object` mode).

This injection happens in `LLMClient.ainvoke` (a system or trailing message appended to `messages`) so call sites in `agent.py` / `memory.py` stay unchanged. Injection applies only for DeepSeek + `output_schema`; other providers keep passing the schema natively.

### 4. Tool binding branch

In `_get_runnable`, do not set `strict=True` for DeepSeek. Change the guard so `strict` is `True` only for real OpenAI (i.e. `isinstance(ChatOpenAI)` **and not** `_is_deepseek()`), leaving DeepSeek at `strict=None`.

### 5. Validation / retry

No new machinery. When `json_object` output fails to validate against the schema, the existing `model_validate_json` + `StructuredOutputParsingFailedFeedback` path (`agent.py:282–287`) already feeds the failure back for another attempt. Confirm the reviewer/memory/summary call sites have equivalent handling; if any assume a guaranteed-valid pydantic, add matching parse-failure handling.

### 6. Thinking-mode / effort selection

DeepSeek exposes reasoning effort through the OpenAI-standard `reasoning_effort` parameter, which `langchain_openai.ChatOpenAI` forwards natively. This is therefore **config-only** — no `LLMClient` change.

- Valid values (confirmed against the live API): `low`, `medium`, `high`, `max`, `xhigh`. These map 1:1 to the Opus 4.8 effort slider, so effort intent transfers directly across configs.
- Set `reasoning_effort` in `deepseek_v4_pro.provider_config` (default `high`, matching `opus48_local.yaml`).
- To disable reasoning entirely, `thinking: {"type": "disabled"}` works, but a theorem prover wants reasoning on, so this is not exposed by default.
- Validation: unit test asserting the `reasoning_effort` value flows into the constructed `ChatOpenAI` (and that the config accepts the documented values).

### 7. Reasoning content (minor)

`get_reasoning` reads `response.content_blocks` for `type == "reasoning"`. DeepSeek returns reasoning in a separate `reasoning_content` field. During implementation, verify whether `langchain_openai` surfaces `reasoning_content` as a reasoning content block. If it does not, adapt `get_reasoning` to also read the DeepSeek location (e.g. `additional_kwargs`). Impact is limited to the lab-notebook / previous-attempt context and logging, not proof correctness — so this is a nice-to-have, not a blocker.

## Alternatives considered (rejected)

- **B — forced-tool-call structured output**: bind a synthetic `submit_result` tool as the schema and force `tool_choice`. Stronger schema enforcement, but collides with the proposer's real search tools and forces restructuring `agentic_loop`. Too invasive.
- **C — schema-in-prompt with no `response_format`**: simplest, but drops the `json_object` guarantee, producing more malformed-JSON retries. Strictly worse than A.

## Parallelism

Running many proofs in parallel (the `experiment` command's `--max-concurrency`, with Lean builds gated by `lean_semaphore`) is **orthogonal to the model choice** and requires no DeepSeek-specific work:

- Each proving task builds its own `ProverAgent` → its own `LLMClient` → its own `ChatOpenAI`, so there is no shared mutable LLM state across parallel proofs.
- DeepSeek calls are ordinary async HTTP requests via the OpenAI async client; concurrency is bounded by `--max-concurrency`, exactly as with the Anthropic/OpenAI providers today.
- DeepSeek rate limits (429s) are absorbed by the existing `retry_config` (exponential backoff with jitter), the same mechanism already relied on for other providers.

Expected to work as-is; the end-to-end verification run will use the `experiment` path (or at least a concurrent smoke check) rather than a single synchronous prove to confirm.

## Verification

1. Unit tests for the new `LLMClient` branches: DeepSeek detection true/false; structured-output kwargs return `json_object` (not a pydantic/`json_schema`); tool binding uses `strict=None` for DeepSeek and `strict=True` for real OpenAI; schema-injection adds a JSON instruction for DeepSeek only.
2. End-to-end: prove one simple theorem with `configs/deepseek_local.yaml` and confirm the proposer and reviewer structured outputs parse and a proof is produced.

## Out of scope

- DeepSeek reasoning-effort / "thinking mode" parameters (the API returned reasoning by default; effort mapping can be a later enhancement).
- Prompt-cache payload reordering optimizations.
- Any multi-agent / orchestration changes.
