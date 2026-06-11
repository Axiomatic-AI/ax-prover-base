"""File reading, listing, and editing utilities for Lean projects."""

import json
import re
from pathlib import Path

from ..models.output import ProverOutput
from .logging import get_logger

logger = get_logger(__name__)


def read_file(base_folder: str, file_path: str) -> str:
    """Read a file's content.

    Args:
        base_folder: Base folder path
        file_path: Path to file relative to base_folder

    Returns:
        File content or directory listing if path is a directory
    """
    try:
        full_path = Path(base_folder) / file_path
        if not full_path.exists():
            return ""

        if full_path.is_dir():
            files = sorted([f.name for f in full_path.iterdir()])
            return f"[Directory: {file_path}]\nContents:\n" + "\n".join(f"  - {f}" for f in files)

        return full_path.read_text(encoding="utf-8")
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return f"Error reading file: {e}"


def replace_in_file(
    file_path: Path,
    original: str,
    replacement: str,
) -> bool:
    """Edit an existing function or add a new one.

    Args:
        file_path: Path to the file to edit
        original_text: Original text to replace
        new_text: New text to replace the original text

    Returns:
        True if successful, False otherwise
    """
    if not file_path.exists():
        logger.warning(f"File {file_path} does not exist")
        return False

    content = file_path.read_text(encoding="utf-8")

    original_text = original.strip()
    new_text = replacement.strip()

    if original_text not in content:
        logger.warning(f"Original text '{original_text}' not found in file {file_path}")
        return False

    # Preserve docstring in case they were removed by the new text
    doc_match = re.match(r"/--[\s\S]*?-/\s*", original_text)
    if doc_match and not re.match(r"/--", new_text):
        replacement = doc_match.group() + new_text

    new_content = content.replace(original_text, new_text, 1)
    file_path.write_text(new_content, encoding="utf-8")
    return True


def edit_imports(base_folder: str, file_path: str, new_imports: list[str]) -> bool:
    """Edit imports in a Lean file by merging with new imports."""
    full_path = Path(base_folder) / file_path

    try:
        full_path.parent.mkdir(parents=True, exist_ok=True)
        normalized_new = []
        for imp in new_imports:
            imp = imp.strip()
            if imp.startswith("import "):
                normalized_new.append(imp[7:].strip())
            else:
                normalized_new.append(imp)

        existing = _get_imports(base_folder, file_path)
        merged = sorted(set(normalized_new) | set(existing))
        merged_text = "\n".join(f"import {imp}" for imp in merged)

        content = full_path.read_text(encoding="utf-8") if full_path.exists() else ""

        if existing:
            pattern = r"(^import\s+.*\n)+"
            new_content = re.sub(pattern, merged_text + "\n", content, count=1, flags=re.MULTILINE)
        else:
            new_content = merged_text + "\n\n" + content if content else merged_text + "\n"

        full_path.write_text(new_content, encoding="utf-8")
        return True
    except Exception as e:
        logger.error(f"Error editing imports in {file_path}: {e}")
        return False


def _get_imports(base_folder: str, file_path: str) -> list[str]:
    """Get all import module paths from a Lean file (without 'import' keyword)."""
    full_path = Path(base_folder) / file_path
    if not full_path.exists():
        return []

    try:
        content = full_path.read_text(encoding="utf-8")
        imports = []
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("import "):
                imports.append(stripped[7:].strip())
            elif imports and stripped and not stripped.startswith(("--", "/-")):
                break
        return imports
    except Exception as e:
        logger.error(f"Error reading imports from {file_path}: {e}")
        return []


def edit_opens(base_folder: str, file_path: str, new_opens: list[str]) -> bool:
    """Edit namespace opens in a Lean file by merging with new opens."""
    full_path = Path(base_folder) / file_path

    try:
        full_path.parent.mkdir(parents=True, exist_ok=True)
        normalized_new = []
        for ns in new_opens:
            ns = ns.strip()
            if ns.startswith("open "):
                namespaces_part = ns[5:].strip()
                for n in namespaces_part.split():
                    n = n.strip()
                    if n:
                        normalized_new.append(n)
            else:
                normalized_new.append(ns)

        existing = _get_opens(base_folder, file_path)
        merged = sorted(set(normalized_new) | set(existing))
        merged_text = "\n".join(f"open {ns}" for ns in merged)

        content = full_path.read_text(encoding="utf-8") if full_path.exists() else ""

        lines = content.splitlines(keepends=True) if content else []
        last_import_idx = -1
        last_open_idx = -1

        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("import "):
                last_import_idx = i
            elif stripped.startswith("open "):
                last_open_idx = i

        if existing:
            first_open_idx = -1
            for i, line in enumerate(lines):
                if line.strip().startswith("open "):
                    if first_open_idx == -1:
                        first_open_idx = i
                    last_open_idx = i
                elif first_open_idx != -1 and line.strip() and not line.strip().startswith("open "):
                    break

            new_lines = lines[:first_open_idx] + [merged_text + "\n"] + lines[last_open_idx + 1 :]
            new_content = "".join(new_lines)
        else:
            if last_import_idx >= 0:
                insert_idx = last_import_idx + 1
                new_lines = lines[:insert_idx] + [merged_text + "\n"] + lines[insert_idx:]
                new_content = "".join(new_lines)
            else:
                new_content = merged_text + "\n\n" + content if content else merged_text + "\n"

        full_path.write_text(new_content, encoding="utf-8")
        return True
    except Exception as e:
        logger.error(f"Error editing opens in {file_path}: {e}")
        return False


def _get_opens(base_folder: str, file_path: str) -> list[str]:
    """Get all namespace opens from a Lean file (without 'open' keyword)."""
    full_path = Path(base_folder) / file_path
    if not full_path.exists():
        return []

    try:
        content = full_path.read_text(encoding="utf-8")
        opens = []
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("open "):
                namespaces_part = stripped[5:].strip()
                for ns in namespaces_part.split():
                    ns = ns.strip()
                    if ns and not ns.startswith(("--", "/-")):
                        opens.append(ns)
        return opens
    except Exception as e:
        logger.error(f"Error reading opens from {file_path}: {e}")
        return []


def write_json_output(outputs: dict[str, ProverOutput], output_file: str) -> None:
    """Write a dict of ProverOutput results to a JSON file."""
    json_dict = {key: output.model_dump() for key, output in outputs.items()}
    Path(output_file).write_text(json.dumps(json_dict, indent=2))
    logger.info(f"JSON output written to: {output_file}")
