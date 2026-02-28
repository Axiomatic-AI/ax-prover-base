"""Utilities for parsing Lean code structure and declarations."""

from __future__ import annotations

import os
import re
from enum import Enum
from pathlib import Path

from ..models.declaration import Declaration, DeclarationType
from ..models.files import Location
from .logging import get_logger

logger = get_logger(__name__)

# Lean keywords for declarations
LEAN_KEYWORDS = [d.value for d in DeclarationType]


def count_sorries(content: str, context_lines: int = 1) -> tuple[int, list[tuple[int, str]]]:
    """Count 'sorry' and 'admit' statements in Lean code with context.

    Args:
        content: The Lean file content
        context_lines: Number of lines to show before and after

    Returns:
        Tuple of (count, locations) where locations is a list of (line_num, formatted_context)
    """
    sorry_locations = []
    lines = content.splitlines()

    for i, line in enumerate(lines):
        for match in re.finditer(r"\b(sorry|admit)\b", line):
            line_num = i + 1
            col = match.start()

            start = max(0, i - context_lines)
            end = min(len(lines), i + context_lines + 1)

            context = []
            for j in range(start, end):
                context.append(f"  {lines[j]}")

                if j == i:
                    context.append("  " + " " * col + "^^^^^")

            sorry_locations.append((line_num, "\n".join(context)))

    return len(sorry_locations), sorry_locations


def strip_comments(src: str) -> str:
    """
    Remove Lean comments from src.
    Handles nested block comments '/- ... -/' and '--' line comments.
    Leaves string literals intact.
    """

    class ParsingState(Enum):
        Out = 1
        LineComment = 2
        BlockComment = 3
        StringLiteral = 4

    state = ParsingState.Out
    i = 0
    depth = 0
    out = []
    n = len(src)

    while i < n:
        c = src[i]
        c2 = src[i : i + 2]

        if state == ParsingState.Out:
            if c == '"':
                state = ParsingState.StringLiteral
            if c2 == "--":
                state = ParsingState.LineComment
                out.append("  ")
                i += 2
            elif c2 == "/-":
                state = ParsingState.BlockComment
                depth = 1
                out.append("  ")
                i += 2
            else:
                out.append(c)
                i += 1

        elif state == ParsingState.LineComment:
            if c == "\n":
                state = ParsingState.Out
                out.append("\n")
            else:
                out.append(" ")  # preserve byte count
            i += 1

        elif state == ParsingState.BlockComment:
            if c2 == "/-":
                depth += 1
                out.append("  ")
                i += 2
            elif c2 == "-/":
                depth -= 1
                out.append("  ")
                i += 2
                if depth == 0:
                    state = ParsingState.Out
            else:
                out.append(" " if c != "\n" else "\n")
                i += 1

        elif state == ParsingState.StringLiteral:
            if c == '"':
                state = ParsingState.Out
            out.append(c)
            i += 1

    return "".join(out)


def extract_function_from_content(content: str, function_name: str) -> str | None:
    """Extract a function/theorem/lemma definition from Lean code.

    Args:
        content: Lean code content as string
        function_name: Name of the function/theorem/lemma to extract

    Returns:
        The complete definition block including doc comments, or None
    """
    keywords_pattern = "|".join(LEAN_KEYWORDS)
    pattern = rf"^(\s*)({keywords_pattern})\s+{re.escape(function_name)}\b"

    match = re.search(pattern, content, re.MULTILINE)
    if not match:
        return None

    start_pos = match.start()
    start_indent = len(match.group(1))

    # Look backwards for Lean4 doc comment (/-- ... -/)
    before_def = content[:start_pos]
    all_doc_comments = list(re.finditer(r"/--[\s\S]*?-/", before_def))

    # Check doc comments in reverse order to find the closest one
    for doc_match in reversed(all_doc_comments):
        between = content[doc_match.end() : start_pos]
        # If no definition keyword between comment and target, use it
        if not re.search(rf"\b(?:{keywords_pattern})\s+\w+", between):
            start_pos = doc_match.start()
            break

    # Find next definition, doc comment, structural keyword, or top-level comment
    # at same or lower indentation
    end_pattern = rf"^[ \t]{{0,{start_indent}}}(/--|--|{keywords_pattern}(?:\s+|\b))"

    remaining_content = content[match.end() :]
    end_match = re.search(end_pattern, remaining_content, re.MULTILINE)

    if end_match:
        end_pos = match.end() + end_match.start()
    else:
        end_pos = len(content)

    return content[start_pos:end_pos].strip()


def get_function_from_location(base_folder: str, location: Location) -> str | None:
    """Get a function/theorem/lemma definition using a Location object.

    Args:
        base_folder: Base folder path
        location: Location object with import path (dot notation), name, and is_external flag

    Returns:
        The complete definition block, or None if not found
    """
    if location.is_external:
        # Resolve external library path (already in dot notation)
        full_path = _resolve_external_path(base_folder, location.module_path)
        if not full_path:
            logger.warning(f"This path does not exist: {location.module_path}.")
            return None
    else:
        # Local project file - use the path property which converts to file path
        full_path = Path(base_folder) / location.path

    if not full_path.exists():
        return None

    try:
        content = full_path.read_text(encoding="utf-8")
        return extract_function_from_content(content, location.name)
    except Exception as e:
        logger.error(f"Error in get_function_from_location: {e}")
        return None


def normalize_location(location_str: str) -> str:
    """Normalize location string to module path format.

    Converts file paths to module paths: "path/to/file.lean:func" -> "path.to.file:func"
    """
    if ".lean:" in location_str:
        file_part, func_part = location_str.rsplit(":", 1)
        module_part = file_part.replace("/", ".").removesuffix(".lean")
        return f"{module_part}:{func_part}"
    return location_str


def get_unproven(base_folder: str, file_path: str) -> list[str]:
    """Get all function/theorem/lemma names that contain 'sorry' in their body.

    Args:
        base_folder: Base folder path
        file_path: Path to file relative to base_folder

    Returns:
        List of function names that contain 'sorry' in their implementation
    """

    all_defs = list_all_declarations_in_path_as_text(base_folder, file_path, show_statements=False)

    if not all_defs:
        return []

    module_path = file_path.replace("/", ".").removesuffix(".lean")
    unproven_functions = []

    for line in all_defs.strip().split("\n"):
        if not line:
            continue

        func_name = extract_theorem_name(line)
        if not func_name:
            continue

        location = Location(module_path=module_path, name=func_name, is_external=False)
        func_body = get_function_from_location(base_folder, location)
        if func_body and re.search(r"\bsorry\b", func_body):
            unproven_functions.append(func_name)

    return unproven_functions


def _resolve_external_path(base_folder: str, import_path: str) -> Path | None:
    """Resolve an external library import path to a file path.

    Args:
        base_folder: Base folder path
        import_path: Import path like "Mathlib.Algebra.Group.Defs"

    Returns:
        Full path to the file, or None if not found
    """
    packages_dir = Path(base_folder) / ".lake" / "packages"

    # Build case-insensitive package directory map
    package_dir_map = {
        d.lower(): d for d in os.listdir(packages_dir) if (packages_dir / d).is_dir()
    }

    # Split import path
    # E.g., "Mathlib.Algebra.Group.Defs" -> ["Mathlib", "Algebra", "Group", "Defs"]
    parts = import_path.split(".")
    if not parts:
        return None

    package_name = parts[0]
    dir_name = package_dir_map.get(package_name.lower())
    if not dir_name:
        return None

    # Build file path: package_dir/part1/part2/.../partN
    # For "Mathlib.Algebra.Group.Defs" -> ".lake/packages/mathlib/Mathlib/Algebra/Group/Defs.lean"
    file_path = packages_dir / dir_name / "/".join(parts)

    if not str(file_path).endswith(".lean"):
        file_path = Path(str(file_path) + ".lean")

    return file_path if file_path.exists() else None


def extract_theorem_name(theorem_statement: str) -> str | None:
    """Extract theorem name from a theorem statement.

    Args:
        theorem_statement: A Lean theorem/lemma/def/etc statement

    Returns:
        The theorem name, or None if not found

    Example:
        >>> extract_theorem_name("theorem foo : P := sorry")
        'foo'
        >>> extract_theorem_name("lemma bar (n : Nat) : n > 0 := by sorry")
        'bar'
        >>> extract_theorem_name("theorem Polynomial.not_isPrincipalIdealRing : ¬IsPrincipalIdealRing R[X] := sorry")
        'Polynomial.not_isPrincipalIdealRing'
    """
    theorem_statement = strip_comments(theorem_statement)

    keywords_pattern = "|".join(re.escape(kw) for kw in LEAN_KEYWORDS)
    match = re.search(rf"\b(?:{keywords_pattern})\s+([\w.]+)", theorem_statement)
    if match:
        return match.group(1)
    return None


def list_all_declarations_in_lean_code(raw_code: str) -> list[Declaration]:
    """
    List all theorems, definitions, lemmas, axioms, and other Lean constructs; in a given string of code.

    Args:
        raw_code: Raw code to search in

    Returns:
        List of declarations
    """

    declarations = []
    declaration = None
    code = strip_comments(raw_code)

    for line in code.split("\n"):
        line_keywords = line.strip().split()
        if len(line_keywords) >= 2 and line_keywords[0] in list(DeclarationType):
            # Extract just the name, splitting on punctuation that can follow it
            name = re.split(r"[:({[\[]", line_keywords[1])[0]
            content = line_keywords[2:]
            if declaration is not None:
                declarations.append(declaration)
            declaration = Declaration(
                declaration_type=line_keywords[0],
                name=name,
                content=" ".join(content),
            )
        elif declaration is not None:
            declaration.content += "\n" + line

    if declaration is not None:
        declarations.append(declaration)

    return declarations


def _list_all_declarations_in_path(
    base_folder: str = ".", path: str = ""
) -> list[tuple[Path, Declaration]]:
    """
    List all theorems, definitions, lemmas, axioms, and other Lean constructs; in a given path.

    Args:
        base_folder: Base folder to search in
        path: Path to subfolder or file to search in

    Returns:
        List of tuples (file_path, declaration)
    """

    if path:
        full_path = Path(base_folder) / path
    else:
        full_path = Path(base_folder)

    file_list = None
    if full_path.is_dir():
        file_list = list(
            filter(lambda p: p.is_file() and p.suffix == ".lean", full_path.rglob("*"))
        )
    else:
        assert full_path.suffix == ".lean"
        file_list = [full_path]

    declarations = []
    for file_path in file_list:
        for declaration in list_all_declarations_in_lean_code(file_path.read_text()):
            declarations.append((file_path, declaration))

    return declarations


def list_all_declarations_in_path_as_text(
    base_folder: str = ".", path: str = "", show_statements: bool = False
) -> str:
    """
    List all theorems, definitions, lemmas, axioms, and other Lean constructs as text; in a given path.

    Args:
        base_folder: Base folder to search in
        path: Path to subfolder or file to search in
        show_statements: If True, show full statements

    Returns:
        Text (string) containing all paths and declarations
    """
    declarations = _list_all_declarations_in_path(base_folder, path)
    if show_statements:
        return "\n".join(f"{decl_path}:{str(decl)}" for decl_path, decl in declarations)
    else:
        return "\n".join(
            f"{decl_path}:{decl.declaration_type.value} {decl.name}"
            for decl_path, decl in declarations
        )


def find_declaration_by_name(declarations: list[Declaration], name: str) -> Declaration | None:
    for declaration in declarations:
        if declaration.name == name:
            return declaration
    return None


def find_declaration_at_line(content: str, line_number: int) -> str | None:
    """Find the declaration name containing the given line number.

    Args:
        content: Lean code content as string
        line_number: 1-indexed line number to search for

    Returns:
        The name of the declaration containing the line, or None if not found
    """
    if line_number < 1:
        return None

    # strip_comments preserves newlines, so line numbers remain valid
    stripped = strip_comments(content)
    lines = stripped.split("\n")

    if line_number > len(lines):
        return None

    keywords_pattern = "|".join(LEAN_KEYWORDS)
    pattern = rf"^(\s*)({keywords_pattern})\s+([\w.]+)"

    declarations: list[tuple[str, int, int]] = []

    for i, line in enumerate(lines):
        match = re.match(pattern, line)
        if match:
            name = match.group(3)
            # Split on punctuation that can follow the name
            name = re.split(r"[:({[\[]", name)[0]
            start_line = i + 1  # Convert to 1-indexed

            # Close previous declaration at same or lower indent
            if declarations:
                prev_name, prev_start, _ = declarations[-1]
                declarations[-1] = (prev_name, prev_start, i + 1)  # end is exclusive, 1-indexed

            declarations.append((name, start_line, len(lines) + 1))

    for name, start, end in declarations:
        if start <= line_number < end:
            return name

    return None
