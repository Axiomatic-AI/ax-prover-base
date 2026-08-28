"""Comparator availability, module rendering, and platform policy."""

import json
import sys

import pytest

from ax_prover.blueprint.comparator import (
    render_challenge,
    render_solution,
    run_comparator,
    unavailable_reason,
)
from ax_prover.blueprint.models import ComparatorStatus
from ax_prover.config import ComparatorConfig

from .conftest import make_blueprint, make_node

CONFIG = ComparatorConfig()


@pytest.fixture
def blueprint():
    return make_blueprint(
        make_node("helper"),
        make_node("target", ("helper",), is_target=True, lean_name="my_target"),
    )


def test_macos_reports_comparator_unavailable(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")

    reason = unavailable_reason(CONFIG)

    assert reason is not None
    assert "requires Linux" in reason


def test_missing_binary_is_reported_on_linux(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr("shutil.which", lambda name: None)

    assert "not found on PATH" in unavailable_reason(CONFIG)


def test_missing_dependency_is_reported_on_linux(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr("shutil.which", lambda name: None if name == "landrun" else "/usr/bin/x")

    assert "landrun" in unavailable_reason(CONFIG)


def test_available_when_every_binary_is_present(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")

    assert unavailable_reason(CONFIG) is None


def test_challenge_and_solution_share_the_trusted_prefix(workspace):
    challenge = render_challenge(workspace)
    solution = render_solution(
        workspace, "namespace G\ntheorem h : True := trivial\nend G", "by simp"
    )

    assert "import Mathlib.Tactic" in challenge
    assert "import Mathlib.Tactic" in solution
    assert challenge.count("theorem my_target") == 1
    assert solution.count("theorem my_target") == 1
    assert ":= sorry" in challenge
    assert ":= by simp" in solution
    assert "theorem h : True := trivial" in solution


async def test_run_comparator_reports_pending_when_unavailable(workspace, blueprint, monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")

    report = await run_comparator(workspace, blueprint, CONFIG, "", "by simp")

    assert report.status is ComparatorStatus.PENDING
    assert "requires Linux" in report.detail


async def test_run_comparator_passes_and_cleans_up(workspace, blueprint, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
    seen = {}

    async def fake_subprocess(command, cwd, timeout):
        seen["command"] = command
        seen["config"] = json.loads(open(command[-1]).read())
        return 0, "all good", ""

    monkeypatch.setattr("ax_prover.blueprint.comparator.run_lean_subprocess", fake_subprocess)

    report = await run_comparator(workspace, blueprint, CONFIG, "", "by simp")

    assert report.status is ComparatorStatus.PASSED
    assert seen["command"][:3] == ["lake", "env", "comparator"]
    assert seen["config"]["theorem_names"] == ["my_target"]
    assert seen["config"]["permitted_axioms"] == ["propext", "Quot.sound", "Classical.choice"]
    assert not list(workspace.file_path.parent.glob("AxProverComparator*"))


async def test_run_comparator_reports_a_rejection(workspace, blueprint, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")

    async def fake_subprocess(command, cwd, timeout):
        return 1, "", "axiom not permitted: myAxiom"

    monkeypatch.setattr("ax_prover.blueprint.comparator.run_lean_subprocess", fake_subprocess)

    report = await run_comparator(workspace, blueprint, CONFIG, "", "by simp")

    assert report.status is ComparatorStatus.REJECTED
    assert "myAxiom" in report.output


async def test_an_invocation_failure_is_pending_not_a_rejection(workspace, blueprint, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")

    async def fake_subprocess(command, cwd, timeout):
        raise OSError("lake is missing")

    monkeypatch.setattr("ax_prover.blueprint.comparator.run_lean_subprocess", fake_subprocess)

    report = await run_comparator(workspace, blueprint, CONFIG, "", "by simp")

    assert report.status is ComparatorStatus.PENDING
    assert "lake is missing" in report.detail
