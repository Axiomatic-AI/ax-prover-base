"""Full `prove --blueprint` run against a real Lean project with a scripted model.

Only `LLMClient.ainvoke` is replaced. Everything else - CLI wiring, the architect loop,
extraction, frontier scheduling, the proof store, assembly, the final build, the Comparator
gate, and the atomic source edit - runs for real, so this catches wiring mistakes the
stage-stubbed unit tests cannot.
"""

import json

import pytest
from langchain_core.messages import AIMessage

from ax_prover.blueprint.generation import ArchitectProposal
from ax_prover.blueprint.node_prover import NodeProposal, NodeTriage
from ax_prover.blueprint.tools import ProofBodyInput
from ax_prover.commands.prove import BlueprintOverrides, prove
from ax_prover.config import BlueprintConfig, Config, LLMConfig
from ax_prover.utils.llm import LLMClient

TARGET = "Blueprint:blueprint_target"

HELPERS = """/--
```ax-blueprint
{"version": 1, "id": "double_eq_two_mul", "parents": []}
```

## Statement

Doubling equals multiplication by two.
-/
theorem double_eq_two_mul (n : Nat) : double n = 2 * n := by
  sorry

/--
```ax-blueprint
{"version": 1, "id": "add_zero_double", "parents": ["double_eq_two_mul"]}
```

## Statement

Adding zero to a doubling leaves it unchanged.
-/
theorem add_zero_double (n : Nat) : double n + 0 = double n := by
  sorry
"""


class ScriptedModel:
    """Answers each role from the statement it is shown, never calling a real provider."""

    def __init__(self, namespace_hint: str = "AxProverGenerated_blueprint_target"):
        self.namespace_hint = namespace_hint
        self.calls: list[str] = []

    def _proof_body(self, prompt: str) -> str:
        namespace = self._namespace(prompt)
        if "double_eq_two_mul" in prompt and "add_zero_double" not in prompt:
            return "by\n  simp [double, Nat.two_mul]"
        if "theorem add_zero_double" in prompt:
            return "by\n  simp"
        return f"by\n  rw [{namespace}.add_zero_double, {namespace}.double_eq_two_mul]"

    def _namespace(self, prompt: str) -> str:
        for token in prompt.replace("(", " ").replace(")", " ").split():
            if token.startswith(self.namespace_hint):
                return token.split(".")[0]
        return self.namespace_hint

    async def ainvoke(self, messages, tools=None, output_schema=None, retry_config=None):
        prompt = "\n".join(str(getattr(m, "content", m)) for m in messages)
        self.calls.append(output_schema.__name__ if output_schema else "none")

        if output_schema is ArchitectProposal:
            payload = ArchitectProposal(
                reasoning="split into two arithmetic steps",
                helpers=HELPERS,
                target_parents=["add_zero_double", "double_eq_two_mul"],
                target_proof_plan="Rewrite with both helpers.",
            )
        elif output_schema is NodeProposal:
            payload = NodeProposal(reasoning="routine", proof_body=self._proof_body(prompt))
        elif output_schema is NodeTriage:
            payload = NodeTriage(outcome="PROOF_TOO_HARD", detail="not reached in this scenario")
        else:
            payload = None

        content = json.dumps(payload.model_dump()) if payload is not None else "{}"
        return AIMessage(content=content, usage_metadata=None)


@pytest.fixture
def scripted_model(monkeypatch):
    model = ScriptedModel()
    monkeypatch.setattr(LLMClient, "__init__", lambda self, config: None)
    monkeypatch.setattr(LLMClient, "ainvoke", model.ainvoke)
    return model


def blueprint_config(tmp_path) -> Config:
    config = Config()
    config.blueprint = BlueprintConfig(
        enabled=True,
        llm=LLMConfig(model="openrouter:deepseek/deepseek-v4-flash-0731", retry_config={}),
        checkpoint_dir=str(tmp_path / "checkpoints"),
        artifacts_dir=str(tmp_path / "artifacts"),
        max_refinement_rounds=1,
        max_node_agents=2,
    )
    return config


async def test_prove_blueprint_solves_the_fixture_and_edits_the_source_once(
    lean_blueprint_project, tmp_path, scripted_model
):
    from pathlib import Path

    source_path = Path(lean_blueprint_project) / "Blueprint.lean"
    original = source_path.read_text(encoding="utf-8")
    output_file = tmp_path / "result.json"

    exit_code = await prove(
        lean_blueprint_project,
        TARGET,
        blueprint_config(tmp_path),
        output_file=str(output_file),
        blueprint=BlueprintOverrides(enabled=True),
    )

    assert exit_code == 0, output_file.read_text(encoding="utf-8")

    updated = source_path.read_text(encoding="utf-8")
    assert updated != original
    assert "sorry" not in updated
    assert "ax-blueprint" not in updated
    assert "The user's own docstring, which must survive assembly untouched." in updated
    assert "theorem after_target : True := trivial" in updated
    assert "double_eq_two_mul" in updated

    result = json.loads(output_file.read_text(encoding="utf-8"))[TARGET]
    assert result["success"] is True
    # macOS has no landrun, so a successful build reports comparator_pending.
    assert result["details"]["status"] in ("solved", "comparator_pending")
    assert result["details"]["graph_size"] == 3
    assert result["details"]["source_modified"] is True

    assert not list(Path(lean_blueprint_project).rglob("tmp_bp_*.lean"))


async def test_a_bad_architect_fails_without_touching_the_source(
    lean_blueprint_project, tmp_path, monkeypatch
):
    from pathlib import Path

    class BrokenArchitect(ScriptedModel):
        async def ainvoke(self, messages, tools=None, output_schema=None, retry_config=None):
            if output_schema is ArchitectProposal:
                payload = ArchitectProposal(
                    helpers="theorem no_metadata : True := by sorry", target_parents=[]
                )
                return AIMessage(content=json.dumps(payload.model_dump()), usage_metadata=None)
            return await super().ainvoke(messages, tools, output_schema, retry_config)

    model = BrokenArchitect()
    monkeypatch.setattr(LLMClient, "__init__", lambda self, config: None)
    monkeypatch.setattr(LLMClient, "ainvoke", model.ainvoke)

    source_path = Path(lean_blueprint_project) / "Blueprint.lean"
    original = source_path.read_text(encoding="utf-8")

    config = blueprint_config(tmp_path)
    config.blueprint.architect.max_attempts = 1
    output_file = tmp_path / "failed.json"

    exit_code = await prove(
        lean_blueprint_project,
        TARGET,
        config,
        output_file=str(output_file),
        blueprint=BlueprintOverrides(enabled=True),
    )

    assert exit_code == 1
    assert source_path.read_text(encoding="utf-8") == original

    result = json.loads(output_file.read_text(encoding="utf-8"))[TARGET]
    assert result["success"] is False
    assert "ax-blueprint" in result["error"]


async def test_an_interrupted_run_resumes_without_reproving_solved_nodes(
    lean_blueprint_project, tmp_path, scripted_model
):

    from ax_prover.blueprint.proof_store import ProofStore

    config = blueprint_config(tmp_path)

    await prove(
        lean_blueprint_project,
        TARGET,
        config,
        blueprint=BlueprintOverrides(enabled=True),
    )
    store = ProofStore.open(config.blueprint.checkpoint_dir, TARGET, resume=True)
    assert store.state.helpers, "the skeleton should be checkpointed"
    assert len(store.solved_proofs()) == 3

    proposals_before = scripted_model.calls.count("NodeProposal")

    exit_code = await prove(
        lean_blueprint_project,
        TARGET,
        config,
        blueprint=BlueprintOverrides(enabled=True, resume=True),
    )

    assert exit_code == 0
    assert "ArchitectProposal" not in scripted_model.calls[proposals_before:]
    assert scripted_model.calls.count("NodeProposal") == proposals_before


class ToolCallingModel(ScriptedModel):
    """Calls `lean_compile` before answering, exercising real tool dispatch."""

    def __init__(self):
        super().__init__()
        self.tool_calls = 0
        self._probed: set[str] = set()

    async def ainvoke(self, messages, tools=None, output_schema=None, retry_config=None):
        prompt = "\n".join(str(getattr(m, "content", m)) for m in messages)

        # Tool phase: tools bound, no schema requested (see `roles.run_turn`). The node
        # prover's lean_compile takes a proof body; the architect's takes helper source.
        node_compile = any(
            getattr(tool, "args_schema", None) is ProofBodyInput for tool in tools or []
        )
        if node_compile and output_schema is None:
            body = self._proof_body(prompt)
            if body not in self._probed:
                self._probed.add(body)
                self.tool_calls += 1
                self.calls.append("tool:lean_compile")
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "lean_compile",
                            "args": {"proof_body": body},
                            "id": str(self.tool_calls),
                        }
                    ],
                    usage_metadata=None,
                )
            return AIMessage(content="It compiles; I am done.", usage_metadata=None)

        return await super().ainvoke(messages, tools, output_schema, retry_config)


async def test_a_model_that_calls_lean_compile_still_solves_the_target(
    lean_blueprint_project, tmp_path, monkeypatch
):
    """Tool dispatch runs outside a LangGraph graph, so it must work standalone."""
    from pathlib import Path

    model = ToolCallingModel()
    monkeypatch.setattr(LLMClient, "__init__", lambda self, config: None)
    monkeypatch.setattr(LLMClient, "ainvoke", model.ainvoke)

    source_path = Path(lean_blueprint_project) / "Blueprint.lean"

    exit_code = await prove(
        lean_blueprint_project,
        TARGET,
        blueprint_config(tmp_path),
        blueprint=BlueprintOverrides(enabled=True),
    )

    assert exit_code == 0
    assert model.tool_calls == 3, "every node prover should have compiled through the tool"
    assert "sorry" not in source_path.read_text(encoding="utf-8")
