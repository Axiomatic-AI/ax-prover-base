"""End-to-end orchestration: source safety, refinement loop, and result reporting."""

from types import SimpleNamespace

import pytest

from ax_prover.blueprint import orchestrator as orchestrator_module
from ax_prover.blueprint.comparator import ComparatorReport
from ax_prover.blueprint.generation import SkeletonCandidate
from ax_prover.blueprint.models import (
    BlueprintError,
    ComparatorStatus,
    NodeDiagnosis,
    NodeOutcome,
    RunStatus,
)
from ax_prover.blueprint.orchestrator import BlueprintOptions, BlueprintOrchestrator
from ax_prover.blueprint.scheduler import ScheduleReport
from ax_prover.config import BlueprintConfig, LeanConfig, LLMConfig
from ax_prover.models.files import Location
from ax_prover.models.proving import TargetItem

from .conftest import (
    TARGET_DECLARATION_TEXT,
    TARGET_FILE_SOURCE,
    build_declaration,
    make_blueprint,
    make_node,
)

DEEPSEEK = "openrouter:deepseek/deepseek-v4-flash-0731"


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")


@pytest.fixture
def project(tmp_path):
    """A tiny project directory holding the target file."""
    (tmp_path / "Mod.lean").write_text(TARGET_FILE_SOURCE, encoding="utf-8")
    return tmp_path


@pytest.fixture
def item():
    declaration = build_declaration(
        TARGET_FILE_SOURCE,
        TARGET_DECLARATION_TEXT,
        name="my_target",
        signature="(n : Nat) : n + 0 = n",
        type_pp="n + 0 = n",
    )
    return TargetItem(
        location=Location(name="my_target", module_path="Mod"),
        original_source=TARGET_DECLARATION_TEXT,
        original_declarations=[declaration],
    )


@pytest.fixture
def config(tmp_path):
    return BlueprintConfig(
        enabled=True,
        llm=LLMConfig(model=DEEPSEEK, retry_config={}),
        checkpoint_dir=str(tmp_path / "checkpoints"),
        artifacts_dir=str(tmp_path / "artifacts"),
        max_refinement_rounds=1,
    )


@pytest.fixture
def runtime(project):
    import asyncio

    return SimpleNamespace(
        base_folder=str(project),
        config=SimpleNamespace(lean=LeanConfig()),
        lean_semaphore=asyncio.Semaphore(1),
        lean_interact_server=None,
    )


@pytest.fixture
def blueprint():
    return make_blueprint(
        make_node("helper", signature=": True"),
        make_node("target", ("helper",), is_target=True, lean_name="my_target"),
    )


def patch_pipeline(
    monkeypatch,
    blueprint,
    *,
    schedule_reports=None,
    complete=True,
    compiles=True,
    comparator=None,
    refine=None,
    proofs=None,
):
    """Stub every stage around the orchestrator so its control flow can be tested."""
    candidate = SkeletonCandidate(
        blueprint=blueprint,
        helpers="theorem helper : True := by sorry",
        target_parents=("helper",),
        target_proof_plan="plan",
    )
    reports = list(schedule_reports or [ScheduleReport(rounds=1, solved=["helper", "target"])])
    proofs = proofs or {"helper": "trivial", "target": "by simp"}

    async def fake_generate(*args, **kwargs):
        return candidate

    async def fake_schedule(
        workspace, bp, store, client, role, search_tool=None, max_node_agents=4
    ):
        report = reports.pop(0) if len(reports) > 1 else reports[0]
        for node_id, body in proofs.items():
            if node_id in store.records and node_id in report.solved:
                store.mark_solved(node_id, body, attempts=1)
        return report

    async def fake_compile(source, label="scratch"):
        return SimpleNamespace(success=compiles, output="" if compiles else "boom", source=source)

    async def fake_comparator(*args, **kwargs):
        return comparator or ComparatorReport(status=ComparatorStatus.PENDING, detail="no landrun")

    monkeypatch.setattr(orchestrator_module, "generate_blueprint", fake_generate)
    monkeypatch.setattr(orchestrator_module, "run_schedule", fake_schedule)
    monkeypatch.setattr(orchestrator_module, "run_comparator", fake_comparator)
    monkeypatch.setattr(orchestrator_module, "is_complete", lambda bp, store: complete)
    if refine is not None:
        monkeypatch.setattr(orchestrator_module, "refine_blueprint", refine)

    return candidate, fake_compile


async def run(config, runtime, item, monkeypatch, fake_compile, options=None):
    orchestrator = BlueprintOrchestrator(config, runtime)
    monkeypatch.setattr(
        "ax_prover.blueprint.workspace.BlueprintWorkspace.compile_source",
        staticmethod(fake_compile),
    )
    return await orchestrator.prove(item, options or BlueprintOptions())


async def test_a_successful_run_edits_the_source_exactly_once(
    config, runtime, item, blueprint, project, monkeypatch
):
    _, fake_compile = patch_pipeline(monkeypatch, blueprint)

    result = await run(config, runtime, item, monkeypatch, fake_compile)

    assert result.status is RunStatus.COMPARATOR_PENDING
    assert result.source_modified
    assert result.comparator_status is ComparatorStatus.PENDING

    source = (project / "Mod.lean").read_text(encoding="utf-8")
    assert "theorem my_target (n : Nat) : n + 0 = n := by simp" in source
    assert "sorry" not in source
    assert "ax-blueprint" not in source
    assert source.count("theorem my_target") == 1


async def test_comparator_pass_reports_solved(config, runtime, item, blueprint, monkeypatch):
    _, fake_compile = patch_pipeline(
        monkeypatch, blueprint, comparator=ComparatorReport(status=ComparatorStatus.PASSED)
    )

    result = await run(config, runtime, item, monkeypatch, fake_compile)

    assert result.status is RunStatus.SOLVED


async def test_comparator_rejection_leaves_the_source_untouched(
    config, runtime, item, blueprint, project, monkeypatch
):
    _, fake_compile = patch_pipeline(
        monkeypatch,
        blueprint,
        comparator=ComparatorReport(status=ComparatorStatus.REJECTED, detail="bad axiom"),
    )

    result = await run(config, runtime, item, monkeypatch, fake_compile)

    assert result.status is RunStatus.FAILED
    assert not result.source_modified
    assert (project / "Mod.lean").read_text(encoding="utf-8") == TARGET_FILE_SOURCE


async def test_require_comparator_fails_when_it_is_unavailable(
    config, runtime, item, blueprint, project, monkeypatch
):
    config.require_comparator = True
    _, fake_compile = patch_pipeline(monkeypatch, blueprint)

    result = await run(config, runtime, item, monkeypatch, fake_compile)

    assert result.status is RunStatus.FAILED
    assert "require-comparator" in result.error
    assert (project / "Mod.lean").read_text(encoding="utf-8") == TARGET_FILE_SOURCE


async def test_a_failing_final_build_leaves_the_source_untouched(
    config, runtime, item, blueprint, project, monkeypatch
):
    _, fake_compile = patch_pipeline(monkeypatch, blueprint, compiles=False)

    result = await run(config, runtime, item, monkeypatch, fake_compile)

    assert result.status is RunStatus.FAILED
    assert "does not compile" in result.error
    assert (project / "Mod.lean").read_text(encoding="utf-8") == TARGET_FILE_SOURCE


async def test_a_sorry_in_a_stored_proof_is_caught_before_committing(
    config, runtime, item, blueprint, project, monkeypatch
):
    _, fake_compile = patch_pipeline(
        monkeypatch, blueprint, proofs={"helper": "trivial", "target": "by sorry"}
    )

    result = await run(config, runtime, item, monkeypatch, fake_compile)

    assert result.status is RunStatus.FAILED
    assert "sorry" in result.error
    assert (project / "Mod.lean").read_text(encoding="utf-8") == TARGET_FILE_SOURCE


async def test_an_infrastructure_error_leaves_the_source_untouched(
    config, runtime, item, blueprint, project, monkeypatch
):
    _, fake_compile = patch_pipeline(
        monkeypatch,
        blueprint,
        schedule_reports=[ScheduleReport(rounds=1, infrastructure_error="the REPL died")],
        complete=False,
    )

    result = await run(config, runtime, item, monkeypatch, fake_compile)

    assert result.status is RunStatus.INFRASTRUCTURE_ERROR
    assert "the REPL died" in result.error
    assert (project / "Mod.lean").read_text(encoding="utf-8") == TARGET_FILE_SOURCE


async def test_exhausting_the_refinement_budget_fails_without_editing(
    config, runtime, item, blueprint, project, monkeypatch
):
    async def refine(*args, **kwargs):
        return SkeletonCandidate(
            blueprint=blueprint,
            helpers="theorem helper : True := by sorry",
            target_parents=("helper",),
            target_proof_plan="plan",
        )

    _, fake_compile = patch_pipeline(
        monkeypatch,
        blueprint,
        schedule_reports=[ScheduleReport(rounds=1, failed=["helper"])],
        complete=False,
        refine=refine,
    )

    result = await run(config, runtime, item, monkeypatch, fake_compile)

    assert result.status is RunStatus.FAILED
    assert "refinement budget exhausted" in result.error
    assert result.refinement_rounds == 1
    assert (project / "Mod.lean").read_text(encoding="utf-8") == TARGET_FILE_SOURCE


async def test_a_failing_refiner_is_reported_not_raised(
    config, runtime, item, blueprint, project, monkeypatch
):
    async def refine(*args, **kwargs):
        raise BlueprintError("the refiner gave up")

    _, fake_compile = patch_pipeline(
        monkeypatch,
        blueprint,
        schedule_reports=[ScheduleReport(rounds=1, failed=["helper"])],
        complete=False,
        refine=refine,
    )

    result = await run(config, runtime, item, monkeypatch, fake_compile)

    assert result.status is RunStatus.FAILED
    assert "the refiner gave up" in result.error
    assert (project / "Mod.lean").read_text(encoding="utf-8") == TARGET_FILE_SOURCE


async def test_a_concurrent_source_edit_aborts_the_commit(
    config, runtime, item, blueprint, project, monkeypatch
):
    _, fake_compile = patch_pipeline(monkeypatch, blueprint)
    orchestrator = BlueprintOrchestrator(config, runtime)
    monkeypatch.setattr(
        "ax_prover.blueprint.workspace.BlueprintWorkspace.compile_source",
        staticmethod(fake_compile),
    )

    original_render = orchestrator_module.render_assembly

    def render_then_edit(workspace, bp, proofs):
        assembled = original_render(workspace, bp, proofs)
        workspace.file_path.write_text("-- a teammate edited this\n", encoding="utf-8")
        return assembled

    monkeypatch.setattr(orchestrator_module, "render_assembly", render_then_edit)

    result = await orchestrator.prove(item, BlueprintOptions())

    assert result.status is RunStatus.FAILED
    assert "refusing to overwrite" in result.error
    assert (project / "Mod.lean").read_text(encoding="utf-8") == "-- a teammate edited this\n"


async def test_a_missing_target_declaration_is_reported(config, runtime, monkeypatch, blueprint):
    _, fake_compile = patch_pipeline(monkeypatch, blueprint)
    empty = TargetItem(location=Location(name="ghost", module_path="Mod"))

    result = await run(config, runtime, empty, monkeypatch, fake_compile)

    assert result.status is RunStatus.FAILED
    assert "not found" in result.error


async def test_resume_rebuilds_the_checkpointed_skeleton_without_the_architect(
    config, runtime, item, blueprint, monkeypatch
):
    _, fake_compile = patch_pipeline(monkeypatch, blueprint)

    async def refuse_to_generate(*args, **kwargs):
        raise AssertionError("the architect must not run on resume")

    async def fake_build(*args, **kwargs):
        return blueprint

    # First run stores the skeleton and one solved proof, then fails on the final build.
    first = await run(config, runtime, item, monkeypatch, fake_compile)
    assert first.source_modified

    monkeypatch.setattr(orchestrator_module, "generate_blueprint", refuse_to_generate)
    monkeypatch.setattr(orchestrator_module, "build_blueprint", fake_build)

    second = await run(
        config, runtime, item, monkeypatch, fake_compile, BlueprintOptions(resume=True)
    )

    assert second.status in (RunStatus.SOLVED, RunStatus.COMPARATOR_PENDING, RunStatus.FAILED)


async def test_restart_discards_the_checkpoint(
    config, runtime, item, blueprint, monkeypatch, tmp_path
):
    _, fake_compile = patch_pipeline(monkeypatch, blueprint)
    await run(config, runtime, item, monkeypatch, fake_compile)

    checkpoints = list((tmp_path / "checkpoints").glob("*.json"))
    assert checkpoints

    generated: list[bool] = []

    async def fake_generate(*args, **kwargs):
        generated.append(True)
        return SkeletonCandidate(
            blueprint=blueprint,
            helpers="theorem helper : True := by sorry",
            target_parents=("helper",),
            target_proof_plan="plan",
        )

    monkeypatch.setattr(orchestrator_module, "generate_blueprint", fake_generate)
    await run(config, runtime, item, monkeypatch, fake_compile, BlueprintOptions(restart=True))

    assert generated == [True]


async def test_the_result_records_graph_and_reuse_metrics(
    config, runtime, item, blueprint, monkeypatch
):
    _, fake_compile = patch_pipeline(monkeypatch, blueprint)

    result = await run(config, runtime, item, monkeypatch, fake_compile)

    assert result.graph_size == 2
    assert result.namespace.startswith("AxProverGenerated_my_target_")
    assert {record.node_id for record in result.node_records} == {"helper", "target"}
    assert result.target == "Mod:my_target"


async def test_a_diagnosis_survives_into_the_result(config, runtime, item, blueprint, monkeypatch):
    async def refine(*args, **kwargs):
        return SkeletonCandidate(
            blueprint=blueprint,
            helpers="theorem helper : True := by sorry",
            target_parents=("helper",),
            target_proof_plan="plan",
        )

    _, fake_compile = patch_pipeline(
        monkeypatch,
        blueprint,
        schedule_reports=[ScheduleReport(rounds=1, failed=["helper"])],
        complete=False,
        refine=refine,
    )
    orchestrator = BlueprintOrchestrator(config, runtime)
    monkeypatch.setattr(
        "ax_prover.blueprint.workspace.BlueprintWorkspace.compile_source",
        staticmethod(fake_compile),
    )

    original_schedule = orchestrator_module.run_schedule

    async def schedule_with_diagnosis(workspace, bp, store, *args, **kwargs):
        report = await original_schedule(workspace, bp, store, *args, **kwargs)
        store.mark_failed(
            "helper",
            NodeDiagnosis(outcome=NodeOutcome.STATEMENT_WRONG, detail="false as stated"),
            attempts=4,
        )
        return report

    monkeypatch.setattr(orchestrator_module, "run_schedule", schedule_with_diagnosis)

    result = await orchestrator.prove(item, BlueprintOptions())

    diagnoses = [r.diagnosis for r in result.node_records if r.diagnosis]
    assert any(d.detail == "false as stated" for d in diagnoses)


async def test_a_run_emits_one_traced_span(config, runtime, item, blueprint, monkeypatch):
    """`prove --blueprint` had no tracing root, so single-target runs emitted nothing."""
    spans: list[dict] = []

    class FakeSpan:
        def __init__(self, kwargs):
            self.kwargs = kwargs
            self.outputs = None

        def end(self, outputs=None):
            self.outputs = outputs

    class fake_trace:
        def __init__(self, **kwargs):
            self.span = FakeSpan(kwargs)

        def __enter__(self):
            spans.append(self.kwargs_and_span())
            return self.span

        def kwargs_and_span(self):
            return {"kwargs": self.span.kwargs, "span": self.span}

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(orchestrator_module, "trace", fake_trace)
    _, fake_compile = patch_pipeline(monkeypatch, blueprint)

    result = await run(config, runtime, item, monkeypatch, fake_compile)

    assert len(spans) == 1
    kwargs = spans[0]["kwargs"]
    assert kwargs["name"] == "blueprint:my_target"
    assert kwargs["run_type"] == "chain"
    assert kwargs["inputs"]["target"] == "Mod:my_target"
    meta = kwargs["metadata"]
    assert meta["mode"] == "blueprint"
    assert meta["prover_model"].startswith("openrouter:")
    assert "git_hash" in meta and "max_refinement_rounds" in meta
    # The outcome is attached, so a trace shows the verdict without opening the checkpoint.
    assert spans[0]["span"].outputs["status"] == result.status.value


async def test_a_failed_run_still_closes_its_span(config, runtime, monkeypatch, blueprint):
    """A failure must not leave the span open, or the trace never resolves."""
    spans: list = []

    class FakeSpan:
        def __init__(self):
            self.outputs = None

        def end(self, outputs=None):
            self.outputs = outputs

    class fake_trace:
        def __init__(self, **kwargs):
            self.span = FakeSpan()

        def __enter__(self):
            spans.append(self.span)
            return self.span

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(orchestrator_module, "trace", fake_trace)
    _, fake_compile = patch_pipeline(monkeypatch, blueprint)
    empty = TargetItem(location=Location(name="ghost", module_path="Mod"))

    result = await run(config, runtime, empty, monkeypatch, fake_compile)

    assert result.status is RunStatus.FAILED
    assert len(spans) == 1
    assert spans[0].outputs is not None, "the span must be closed even on failure"
    assert spans[0].outputs["status"] == "failed"
