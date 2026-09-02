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
from ax_prover.blueprint.provision import (
    ProvisionError,
    read_project_toolchain,
    select_comparator_tag,
)
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


def test_missing_landrun_is_reported_on_linux(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr("shutil.which", lambda name: None)
    monkeypatch.delenv("COMPARATOR_LANDRUN", raising=False)

    assert "landrun" in unavailable_reason(CONFIG)


def test_landrun_env_override_counts_as_available(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr("shutil.which", lambda name: None)
    monkeypatch.setenv("COMPARATOR_LANDRUN", "/opt/landrun")

    assert unavailable_reason(CONFIG) is None


def test_available_when_landrun_is_present(monkeypatch):
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

    async def fake_subprocess(command, cwd, timeout, env=None):
        seen["command"] = command
        seen["config"] = json.loads(open(command[-1]).read())
        return 0, "all good", ""

    monkeypatch.setattr("ax_prover.blueprint.comparator.run_lean_subprocess", fake_subprocess)

    report = await run_comparator(workspace, blueprint, CONFIG, "", "by simp")

    assert report.status is ComparatorStatus.PASSED
    assert seen["command"][:3] == ["lake", "env", "/usr/bin/comparator"]
    # The workspace fixture's target sits at the project root, so no dotted prefix.
    assert seen["config"]["challenge_module"] == "AxProverComparatorChallenge"
    assert seen["config"]["theorem_names"] == ["my_target"]
    assert seen["config"]["permitted_axioms"] == ["propext", "Quot.sound", "Classical.choice"]
    assert not list(workspace.file_path.parent.glob("AxProverComparator*"))


async def test_run_comparator_reports_a_rejection(workspace, blueprint, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")

    async def fake_subprocess(command, cwd, timeout, env=None):
        return 1, "", "axiom not permitted: myAxiom"

    monkeypatch.setattr("ax_prover.blueprint.comparator.run_lean_subprocess", fake_subprocess)

    report = await run_comparator(workspace, blueprint, CONFIG, "", "by simp")

    assert report.status is ComparatorStatus.REJECTED
    assert "myAxiom" in report.output


async def test_an_invocation_failure_is_pending_not_a_rejection(workspace, blueprint, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")

    async def fake_subprocess(command, cwd, timeout, env=None):
        raise OSError("lake is missing")

    monkeypatch.setattr("ax_prover.blueprint.comparator.run_lean_subprocess", fake_subprocess)

    report = await run_comparator(workspace, blueprint, CONFIG, "", "by simp")

    assert report.status is ComparatorStatus.PENDING
    assert "lake is missing" in report.detail


async def test_missing_binaries_are_provisioned(workspace, blueprint, monkeypatch, tmp_path):
    """Off PATH, comparator and a toolchain-matched lean4export are built on demand."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(
        "shutil.which", lambda name: "/usr/bin/landrun" if name == "landrun" else None
    )
    (tmp_path / "lean-toolchain").write_text("leanprover/lean4:v4.24.0\n")
    monkeypatch.setattr(workspace, "base_folder", str(tmp_path))
    seen = {}

    async def fake_comparator(version):
        seen["comparator_version"] = version
        return tmp_path / "bin" / "comparator"

    async def fake_lean4export(version):
        seen["version"] = version
        return tmp_path / "bin" / "lean4export"

    monkeypatch.setattr("ax_prover.blueprint.comparator.ensure_comparator", fake_comparator)
    monkeypatch.setattr("ax_prover.blueprint.comparator.ensure_lean4export", fake_lean4export)

    async def fake_subprocess(command, cwd, timeout, env=None):
        seen["command"] = command
        seen["env"] = env
        return 0, "ok", ""

    monkeypatch.setattr("ax_prover.blueprint.comparator.run_lean_subprocess", fake_subprocess)

    report = await run_comparator(workspace, blueprint, CONFIG, "", "by simp")

    assert report.status is ComparatorStatus.PASSED
    assert seen["version"] == "v4.24.0"
    assert seen["comparator_version"] == "v4.24.0"
    assert seen["command"][2] == str(tmp_path / "bin" / "comparator")
    assert seen["env"]["COMPARATOR_LEAN4EXPORT"] == str(tmp_path / "bin" / "lean4export")


async def test_a_provision_failure_is_pending_not_a_crash(workspace, blueprint, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(
        "shutil.which", lambda name: "/usr/bin/landrun" if name == "landrun" else None
    )
    (workspace.file_path.parent / "lean-toolchain").write_text("leanprover/lean4:v4.24.0\n")

    async def fail(version):
        raise ProvisionError("no network")

    monkeypatch.setattr("ax_prover.blueprint.comparator.ensure_comparator", fail)

    report = await run_comparator(workspace, blueprint, CONFIG, "", "by simp")

    assert report.status is ComparatorStatus.PENDING
    assert "no network" in report.detail


def test_read_project_toolchain_parses_both_forms(tmp_path):
    (tmp_path / "lean-toolchain").write_text("leanprover/lean4:v4.24.0\n")
    assert read_project_toolchain(str(tmp_path)) == "v4.24.0"

    (tmp_path / "lean-toolchain").write_text("v4.19.0\n")
    assert read_project_toolchain(str(tmp_path)) == "v4.19.0"


def test_read_project_toolchain_rejects_a_missing_file(tmp_path):
    with pytest.raises(ProvisionError):
        read_project_toolchain(str(tmp_path))


async def test_modules_are_written_beside_a_nested_target(workspace, blueprint, monkeypatch):
    """A root-level module belongs to no Lake target; the modules must live in the
    target's own directory with dotted module names."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")

    nested = workspace.file_path.parent / "Bench" / "Sub"
    nested.mkdir(parents=True)
    new_path = nested / workspace.file_path.name
    new_path.write_text(workspace.original_source, encoding="utf-8")
    monkeypatch.setattr(workspace, "file_path", new_path)
    seen = {}

    async def fake_subprocess(command, cwd, timeout, env=None):
        seen["config"] = json.loads(open(command[-1]).read())
        seen["config_path"] = command[-1]
        return 0, "ok", ""

    monkeypatch.setattr("ax_prover.blueprint.comparator.run_lean_subprocess", fake_subprocess)

    report = await run_comparator(workspace, blueprint, CONFIG, "", "by simp")

    assert report.status is ComparatorStatus.PASSED
    assert seen["config"]["challenge_module"] == "Bench.Sub.AxProverComparatorChallenge"
    assert seen["config"]["solution_module"] == "Bench.Sub.AxProverComparatorSolution"
    assert str(nested) in seen["config_path"]
    assert not list(nested.glob("AxProverComparator*"))


@pytest.mark.parametrize(
    ("toolchain", "expected"),
    [
        ("v4.24.0", "v4.25.0-rc2"),  # older than every release: closest era wins
        ("v4.29.1", "v4.29.0"),
        ("v4.27.0", "v4.27.0"),
        ("v4.99.0", "v4.34.0-rc2"),
    ],
)
def test_comparator_tag_selection(toolchain, expected):
    tags = [
        "v4.34.0-rc2",
        "v4.33.0",
        "v4.31.0",
        "v4.29.0",
        "v4.28.0",
        "v4.27.0",
        "v4.25.0-rc2",
        "nanoda",
    ]
    assert select_comparator_tag(tags, toolchain) == expected


def test_comparator_tag_selection_rejects_junk():
    with pytest.raises(ProvisionError):
        select_comparator_tag(["nanoda", "pre-v25"], "v4.24.0")
    with pytest.raises(ProvisionError):
        select_comparator_tag(["v4.27.0"], "nightly-2026-01-01")


async def test_a_landrun_failure_is_pending_not_a_rejection(workspace, blueprint, monkeypatch):
    """A sandbox that cannot exec lean4export is infrastructure, not a proof verdict."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")

    async def fake_subprocess(command, cwd, timeout, env=None):
        return 1, "", '[landrun:error] Failed to find binary: exec: "lean4export"'

    monkeypatch.setattr("ax_prover.blueprint.comparator.run_lean_subprocess", fake_subprocess)

    report = await run_comparator(workspace, blueprint, CONFIG, "", "by simp")

    assert report.status is ComparatorStatus.PENDING
    assert "infrastructure" in report.detail


def test_lean4export_shim_restores_the_separator(tmp_path):
    """landrun swallows the bare `--` in the wrapped command, so comparator's decl names
    arrive as module arguments; the shim reinserts the separator."""
    import subprocess

    from ax_prover.blueprint.provision import write_lean4export_shim

    real = tmp_path / "real"
    real.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n", encoding="utf-8")
    real.chmod(0o755)
    shim = write_lean4export_shim(real, tmp_path / "shim")

    stripped = subprocess.run(
        [str(shim), "Mod", "thm1", "thm2"], capture_output=True, text=True
    ).stdout.splitlines()
    intact = subprocess.run(
        [str(shim), "Mod", "--", "thm1"], capture_output=True, text=True
    ).stdout.splitlines()

    assert stripped == ["Mod", "--", "thm1", "thm2"]
    assert intact == ["Mod", "--", "thm1"], "an already-present separator must not double"
