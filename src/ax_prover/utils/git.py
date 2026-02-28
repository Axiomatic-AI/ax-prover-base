"""Git utilities for repository information and version tracking."""

import subprocess
from importlib.metadata import version


def get_repo_metadata(base_folder: str) -> dict:
    """Get all Git repository metadata as a dictionary."""
    return {
        "repo_url": _get_git_repo_url(base_folder),
        "branch": _get_git_branch(base_folder),
        "commit": _get_git_commit_hash(base_folder),
        "dirty": _is_git_repo_dirty(base_folder),
        "user_email": _get_git_user_email(base_folder),
    }


def _get_git_repo_url(base_folder: str = ".") -> str | None:
    """Get the git repository remote origin URL, converting SSH to HTTPS format."""
    try:
        result = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            cwd=base_folder,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        url = result.stdout.strip()

        # Convert SSH URLs to HTTPS format for consistency
        if url.startswith("git@github.com:"):
            url = url.replace("git@github.com:", "https://github.com/")
        if url.endswith(".git"):
            url = url[:-4]

        return url
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _get_git_branch(base_folder: str = ".") -> str | None:
    """Get the current git branch name."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=base_folder,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _get_git_commit_hash(base_folder: str = ".") -> str | None:
    """Get the current git commit hash (short form)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=base_folder,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _is_git_repo_dirty(base_folder: str = ".") -> bool:
    """Check if the git repository has uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=base_folder,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        return bool(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _get_git_user_email(base_folder: str = ".") -> str | None:
    """Get the git user email from config."""
    try:
        result = subprocess.run(
            ["git", "config", "user.email"],
            cwd=base_folder,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _get_version_string() -> str:
    """Get version string from package metadata."""
    try:
        return version("ax-prover")
    except Exception:
        return "0.0.0+unknown"


def get_git_hash() -> str:
    """Get git commit hash from version string."""
    try:
        ver = _get_version_string()
        if "+g" in ver:
            hash_part = ver.split("+g")[1].split(".")[0]
            return hash_part
        return "unknown"
    except Exception:
        return "unknown"


def is_git_dirty() -> bool:
    """Check if build was from dirty git state."""
    try:
        ver = _get_version_string()
        # setuptools-scm adds '.dYYYYMMDD' when dirty
        return ".d" in ver.split("+")[-1] if "+" in ver else False
    except Exception:
        return True
