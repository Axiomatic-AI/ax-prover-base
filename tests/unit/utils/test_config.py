"""Tests for configuration utilities: resolve_config_path, _load_yaml_with_imports, merge_configs, load_env_secrets."""

import os

import pytest
from omegaconf import OmegaConf

from ax_prover.config import Config
from ax_prover.utils.config import (
    _PACKAGE_ROOT,
    _load_yaml_with_imports,
    load_env_secrets,
    merge_configs,
    resolve_config_path,
)


@pytest.fixture
def config_dir(tmp_path):
    """Create a temporary config directory with test YAML files."""
    # Base config
    base = tmp_path / "base.yaml"
    base.write_text("prover:\n  max_iterations: 10\n")

    # Config with relative import
    sub = tmp_path / "sub"
    sub.mkdir()
    child = sub / "child.yaml"
    child.write_text("import:\n  - ../base.yaml\n\nprover:\n  max_iterations: 50\n")

    # Config with sibling import
    sibling = sub / "sibling.yaml"
    sibling.write_text("prover:\n  max_iterations: 99\n")
    nested = sub / "nested.yaml"
    nested.write_text("import:\n  - sibling.yaml\n\nprover:\n  max_iterations: 200\n")

    return tmp_path


class TestResolveConfigPath:
    """Tests for resolve_config_path function."""

    def test_absolute_path_exists(self, config_dir):
        """Absolute path that exists is returned as-is."""
        path = config_dir / "base.yaml"
        result = resolve_config_path(path)
        assert result == path

    def test_absolute_path_not_found(self, tmp_path):
        """Absolute path that doesn't exist raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            resolve_config_path(tmp_path / "nonexistent.yaml")

    def test_relative_to_cwd(self, config_dir, monkeypatch):
        """Relative path found in CWD."""
        monkeypatch.chdir(config_dir)
        result = resolve_config_path("base.yaml")
        assert result == (config_dir / "base.yaml").resolve()

    def test_relative_to_folder(self, config_dir, monkeypatch, tmp_path):
        """Relative path found via --folder fallback."""
        # CWD is a different tmp dir — file won't be found there
        other = tmp_path / "other"
        other.mkdir()
        monkeypatch.chdir(other)

        result = resolve_config_path("base.yaml", folder=config_dir)
        assert result == (config_dir / "base.yaml").resolve()

    def test_relative_to_package_root(self, monkeypatch, tmp_path):
        """Relative path found via package root fallback."""
        # CWD is a different tmp dir
        monkeypatch.chdir(tmp_path)

        # configs/default.yaml should exist at package root
        result = resolve_config_path("configs/default.yaml")
        assert result == (_PACKAGE_ROOT / "configs/default.yaml").resolve()

    def test_not_found_anywhere(self, monkeypatch, tmp_path):
        """FileNotFoundError when file doesn't exist in any search path."""
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError, match="Searched in"):
            resolve_config_path("totally_missing.yaml", folder=tmp_path / "nope")

    def test_priority_cwd_over_folder(self, config_dir, monkeypatch):
        """CWD takes priority over --folder."""
        # Put a different file in a "folder" location
        folder = config_dir / "folder_dir"
        folder.mkdir()
        (folder / "base.yaml").write_text("prover:\n  max_iterations: 999\n")

        monkeypatch.chdir(config_dir)
        result = resolve_config_path("base.yaml", folder=folder)
        # Should resolve to CWD version, not folder version
        assert result == (config_dir / "base.yaml").resolve()


class TestLoadYamlWithImports:
    """Tests for _load_yaml_with_imports function."""

    def test_file_without_imports(self, config_dir):
        """Loading a file with no imports returns a single DictConfig."""
        result = _load_yaml_with_imports(config_dir / "base.yaml")
        assert len(result) == 1
        assert result[0].prover.max_iterations == 10

    def test_relative_parent_import(self, config_dir):
        """Import using ../ resolves relative to the importing file."""
        result = _load_yaml_with_imports(config_dir / "sub" / "child.yaml")
        assert len(result) == 2
        # First is the imported base (lower precedence)
        assert result[0].prover.max_iterations == 10
        # Second is the child (higher precedence)
        assert result[1].prover.max_iterations == 50

    def test_sibling_import(self, config_dir):
        """Import of a sibling file resolves relative to the importing file."""
        result = _load_yaml_with_imports(config_dir / "sub" / "nested.yaml")
        assert len(result) == 2
        assert result[0].prover.max_iterations == 99  # sibling
        assert result[1].prover.max_iterations == 200  # nested

    def test_merged_precedence(self, config_dir):
        """When merged, the importing file's values take precedence."""
        configs = _load_yaml_with_imports(config_dir / "sub" / "child.yaml")
        merged = OmegaConf.merge(*configs)
        assert merged.prover.max_iterations == 50


class TestMergeConfigs:
    """Tests for merge_configs with folder parameter."""

    def test_merge_with_folder(self, config_dir, monkeypatch, tmp_path):
        """merge_configs resolves config paths using folder fallback."""
        other = tmp_path / "other"
        other.mkdir()
        monkeypatch.chdir(other)

        result = merge_configs([Config(), "base.yaml"], folder=config_dir)
        assert result.prover.max_iterations == 10

    def test_merge_with_relative_imports(self, config_dir, monkeypatch, tmp_path):
        """merge_configs handles YAML files with relative imports."""
        other = tmp_path / "other"
        other.mkdir()
        monkeypatch.chdir(other)

        result = merge_configs([Config(), "sub/child.yaml"], folder=config_dir)
        assert result.prover.max_iterations == 50

    def test_merge_backwards_compatible(self, monkeypatch):
        """merge_configs still works without folder parameter."""
        monkeypatch.chdir(_PACKAGE_ROOT)
        result = merge_configs([Config(), "configs/default.yaml"])
        assert result.prover.max_iterations == 50


class TestLoadEnvSecrets:
    """Tests for load_env_secrets function."""

    def test_loads_from_cwd(self, tmp_path, monkeypatch):
        """Loads .env.secrets from CWD."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".env.secrets").write_text("TEST_AX_SECRET_CWD=from_cwd\n")
        monkeypatch.delenv("TEST_AX_SECRET_CWD", raising=False)

        load_env_secrets()

        assert os.environ["TEST_AX_SECRET_CWD"] == "from_cwd"
        monkeypatch.delenv("TEST_AX_SECRET_CWD")

    def test_loads_from_folder(self, tmp_path, monkeypatch):
        """Loads .env.secrets from --folder when not in CWD."""
        cwd = tmp_path / "cwd"
        cwd.mkdir()
        folder = tmp_path / "project"
        folder.mkdir()
        (folder / ".env.secrets").write_text("TEST_AX_SECRET_FOLDER=from_folder\n")
        monkeypatch.chdir(cwd)
        monkeypatch.delenv("TEST_AX_SECRET_FOLDER", raising=False)

        load_env_secrets(folder=folder)

        assert os.environ["TEST_AX_SECRET_FOLDER"] == "from_folder"
        monkeypatch.delenv("TEST_AX_SECRET_FOLDER")

    def test_cwd_takes_priority_over_folder(self, tmp_path, monkeypatch):
        """CWD .env.secrets takes priority over --folder."""
        folder = tmp_path / "project"
        folder.mkdir()
        (tmp_path / ".env.secrets").write_text("TEST_AX_SECRET_PRIO=from_cwd\n")
        (folder / ".env.secrets").write_text("TEST_AX_SECRET_PRIO=from_folder\n")
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("TEST_AX_SECRET_PRIO", raising=False)

        load_env_secrets(folder=folder)

        assert os.environ["TEST_AX_SECRET_PRIO"] == "from_cwd"
        monkeypatch.delenv("TEST_AX_SECRET_PRIO")

    def test_folder_fills_gaps_from_cwd(self, tmp_path, monkeypatch):
        """Folder .env.secrets fills in variables missing from CWD."""
        folder = tmp_path / "project"
        folder.mkdir()
        (tmp_path / ".env.secrets").write_text("TEST_AX_A=from_cwd\n")
        (folder / ".env.secrets").write_text("TEST_AX_A=from_folder\nTEST_AX_B=from_folder\n")
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("TEST_AX_A", raising=False)
        monkeypatch.delenv("TEST_AX_B", raising=False)

        load_env_secrets(folder=folder)

        assert os.environ["TEST_AX_A"] == "from_cwd"
        assert os.environ["TEST_AX_B"] == "from_folder"
        monkeypatch.delenv("TEST_AX_A")
        monkeypatch.delenv("TEST_AX_B")

    def test_no_file_is_silent(self, tmp_path, monkeypatch):
        """No error when .env.secrets doesn't exist anywhere."""
        monkeypatch.chdir(tmp_path)
        load_env_secrets()  # Should not raise
