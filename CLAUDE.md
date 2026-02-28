# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ax-prover is a minimal LangGraph-based agent for automated Lean 4 theorem proving. It uses off-the-shelf LLMs (no fine-tuning) with iterative proof refinement, a memory system, and library search tools to prove theorems.

The agent runs a 4-node loop: Proposer → Compiler → Reviewer → Memory, iterating until the proof is complete or the iteration budget is exhausted.

## Development Commands

### Installation & Setup

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install with dev dependencies
pip install -e ".[dev]"

# Setup pre-commit hooks (runs ruff formatting on commit)
pre-commit install

# Manual formatting
ruff format .
ruff check --fix .
```

### Testing

```bash
# Run unit tests (normal development workflow)
.venv/bin/pytest tests/unit

# Run specific test file
.venv/bin/pytest tests/unit/utils/test_files.py

# Run single test function
.venv/bin/pytest tests/unit/utils/test_files.py::test_function_name
```

**Note:** In normal development workflows, run `tests/unit` rather than `tests/regression`. Regression tests are typically run in CI or before major releases.

### Running the Agent

```bash
# Prove a specific theorem by location (module path)
ax-prover prove MyModule.Path:theorem_name

# Prove a specific theorem by file path
ax-prover prove MyProject/Algebra/Ring.lean:theorem_name

# Prove the theorem at a specific line
ax-prover prove MyProject/Algebra/Ring.lean#L42

# Prove all unproven theorems in a file
ax-prover prove MyProject/Algebra/Ring.lean

# Skip lake build (if repo is already built)
ax-prover prove MyModule:theorem_name --skip-build

# Force re-proving
ax-prover prove MyModule:theorem_name --overwrite

# Run batch experiments on a LangSmith dataset
ax-prover experiment dataset_name --max-concurrency 8
```

**Note on `--skip-build` flag:**
- By default, `prove` runs `lake exe cache get && lake build` before starting
- Use `--skip-build` when the repo is already built and up-to-date
- The build step is defined in `src/ax_prover/utils/build.py:build_lean_repo()`

## Architecture

### Prover Agent Loop (`src/ax_prover/prover/agent.py`)

The agent uses a 4-node iterative LangGraph workflow:

1. **Proposer** — A ReAct-style LLM agent that writes Lean 4 proof code. Can optionally use tools (LeanSearch, web search) to find relevant Mathlib lemmas before proposing.
2. **Compiler (Builder)** — Applies the proposed code via `TemporaryProposal`, builds with `lake env lean`, and extracts goal states at `sorry` locations using `lean_interact`. Returns `BuildSuccessFeedback` or `BuildFailedFeedback`.
3. **Reviewer** — Verifies statement preservation and proof validity (no `sorry`, no cheating tactics like `native_decide`). Returns `ReviewApprovedFeedback` or `ReviewRejectedFeedback`.
4. **Memory** (`src/ax_prover/prover/memory.py`) — Summarizes lessons from failed attempts into a concise context ("lab notebook") to prevent repeating mistakes. Default strategy: `ExperienceProcessor` (self-reflection).

Loop: Proposer → Builder → (Reviewer if build succeeds) → Memory → back to Proposer. Terminates on review approval, max iterations, or build timeout.

### Key Abstractions

**State Models** (`src/ax_prover/models/`):
- `ProverAgentState` (`proving.py`): Main state for the prover workflow — messages, item, metrics, iteration tracking
- `TargetItem` (`proving.py`): A theorem to prove — title, location, proven status
- `Location` (`files.py`): Where code lives — `Module.Path:function_name` or `path/to/file.lean:function_name`
- `Declaration` (`declaration.py`): A parsed Lean declaration with name, type, body, and line info

**Messages** (`src/ax_prover/models/messages.py`):
- `ProposalMessage`: Code proposals with reasoning, imports, opens, and updated theorem
- `FeedbackMessage`: Base class for feedback — `BuildSuccessFeedback`, `BuildFailedFeedback`, `ReviewApprovedFeedback`, `ReviewRejectedFeedback`, `SorriesGoalStateFeedback`, etc.

**Configuration** (`src/ax_prover/config.py`):
- `Config`: Root config with `ProverConfig` and `ToolsConfig`
- `ProverConfig`: LLM config, tools list, max iterations, memory config
- OmegaConf-based: supports YAML files, CLI overrides, config merging

### Tools

**Lean Search** (`src/ax_prover/tools/lean_search.py`):
- Searches Lean 4/Mathlib theorems via LeanSearch API
- Default: `https://leansearch.net` (public, no setup)

**Web Search** (`src/ax_prover/tools/web_search.py`):
- Tavily API for finding proof strategies online

**Lean Build** (`src/ax_prover/utils/build.py`):
- `build_lean_repo()`: Runs `lake exe cache get && lake build`
- `check_lean_file()`: Compiles a single file with `lake env lean`
- `TemporaryProposal`: Context manager that applies code changes to a temp file, tests compilation, and can commit permanently

### Commands

- `prove` (`src/ax_prover/commands/prove.py`): Prove theorems by location or all unproven in a file
- `experiment` (`src/ax_prover/commands/experiment.py`): Run batch experiments on LangSmith datasets with evaluation metrics
- `configure` (`src/ax_prover/commands/configure.py`): Interactive setup for API keys (writes to platform config dir via `platformdirs`)

### LangSmith Integration

Agent runs include LangSmith metadata for tracing:
- Git hash and dirty status
- Repository metadata
- Run names follow pattern: `prove:<theorem_name>`

## Important Patterns

### Safe Code Application

Always use `TemporaryProposal` for testing code changes:

```python
with TemporaryProposal(base_folder, location, proposal) as applier:
    if not applier.success:
        return  # Handle error

    success, output = check_lean_file(base_folder, applier.location.path)

    if success:
        applier.apply_permanently()
```

### Location Handling

Locations support both file paths and module paths:

```python
from ax_prover.models.files import Location

# Both formats work
loc = Location.from_string("MyProject.Path:theorem")
loc = Location.from_string("MyProject/Path.lean:theorem")
```

## Code Quality Guidelines

### Documentation and Docstrings

**Keep docstrings concise.** Only write longer, detailed docstrings when there is important functionality to describe.

**Docstring Guidelines:**
- **Single-line docstrings** are preferred for simple, self-explanatory functions
- **Multi-line docstrings** should only be used when:
  - The function has complex behavior that isn't obvious from the name and type hints
  - There are important edge cases, side effects, or constraints to document
  - The function takes multiple parameters that need explanation
  - The return value or exceptions need clarification

**Examples:**

Good (simple function, single line):
```python
def _format_relative_time(timestamp_str: str) -> str:
    """Format timestamp as relative time (e.g., '5 minutes ago')."""
```

Good (complex function with unintuitive logic, multi-line docstring justified):
```python
def _assign_hierarchical_numbers(checkpoints: list[dict], parent_map: dict[str, str | None]) -> None:
    """
    Assign hierarchical numbers to checkpoints, detecting branches from restorations.

    This function mutates the checkpoints list in-place, adding 'hierarchical_number'
    and 'parent_number' keys. Branch detection uses two heuristics:
    1. Step number decreases (restoration to earlier checkpoint)
    2. Step number repeats with >5min time gap (restoration to same checkpoint)

    Branch numbering: main sequence gets "1", "2", "3", etc.
    Branches get "7.1", "7.2" where 7 is the parent checkpoint number.

    Args:
        checkpoints: List sorted by timestamp (oldest first). Modified in-place.
        parent_map: Checkpoint ID -> parent ID mapping (may be incomplete/unreliable)

    Note: Relies on timestamp ordering. Will produce incorrect numbers if checkpoints
          are not sorted chronologically.
    """
    # implementation
```

Bad (over-documented simple function):
```python
def _print_success(message: str) -> None:
    """
    Print a success message to the console.

    Args:
        message: The message string to print

    Returns:
        None
    """
```

### Inline Comments

**Write self-documenting code first.** Only add inline comments for "why" not "what":

Good (explains why):
```python
# Narrow exception handling to only I/O errors - this metadata is optional
try:
    custom_meta = self._load_json(custom_meta_path)
except (OSError, IOError, json.JSONDecodeError) as e:
    logger.debug(f"Failed to load metadata: {e}")
```

Bad (states the obvious):
```python
# Increment counter
counter += 1
```

### Error Handling and Defensive Programming

**Do not be overly defensive.** Write clear, explicit code that fails fast when given invalid inputs.

- **Do not wrap code in try-except without good reason.** Only catch exceptions when you have a specific handling strategy.
- **Do not swallow exceptions silently.** Always handle meaningfully, re-raise, or log.
- **Validate inputs and raise exceptions for invalid arguments.** Do not return garbage values.
- **Fail fast** with clear error messages.

## Configuration

### API Keys

Set up API keys interactively with `ax-prover configure`, or via environment variables.

Secrets cascade (first found wins, shell env always takes priority):
1. CWD `.env.secrets`
2. `--folder` `.env.secrets`
3. `<platformdirs config>/.env.secrets` (written by `ax-prover configure`)
4. Package root `.env.secrets` (editable installs)

Required: at least one of `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`
Optional: `TAVILY_API_KEY` (web search), `LANGSMITH_API_KEY` (tracing)

### YAML Config

LLM model is configured via YAML config files (no hardcoded default).
Default config is bundled in the package at `src/ax_prover/configs/default.yaml`.
Config resolution: CWD > `--folder` > bundled package configs > package root `configs/`.

## Repository Structure

```
src/ax_prover/
├── commands/        # CLI command implementations (prove, experiment, configure)
├── configs/         # Bundled default YAML configs and secrets template
├── models/          # Pydantic state models (proving, messages, files, declaration)
├── prover/          # Prover agent (agent, memory, prompts)
├── tools/           # LangChain tools (lean_search, web_search)
├── utils/           # Utilities (build, files, git, llm, lean_interact, lean_parsing, logging)
├── config.py        # OmegaConf configuration dataclasses
├── evaluators.py    # LangSmith experiment evaluators
└── main.py          # CLI entry point

configs/             # Experimental/ablation configs (not shipped in package)
tests/               # Pytest tests mirroring src structure
```

## Lean 4 Integration

The system expects a Lean 4 project with:
- `lakefile.lean` or `lakefile.toml` in base folder
- Lake commands available on PATH: `lake`, `lake exe cache get`

Before proving, the system automatically runs `build_lean_repo()` to ensure dependencies are ready (unless `--skip-build` is set).
