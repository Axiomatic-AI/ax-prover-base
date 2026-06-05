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

# Declaration types that introduce a real, keepable named declaration (the kind that
# `TemporaryProposal` would strip if it is not the target). Structural/non-declaration
# entries (Import, Namespace, Section, End, Open, Notation, syntax/macro/elab, ...) are
# excluded: they are not standalone declarations and stripping them is not the concern.
STRIPPABLE_DECLARATION_TYPES = frozenset(
    {
        DeclarationType.Definition,
        DeclarationType.Theorem,
        DeclarationType.Lemma,
        DeclarationType.Instance,
        DeclarationType.Structure,
        DeclarationType.Class,
        DeclarationType.Inductive,
        DeclarationType.Axiom,
        DeclarationType.Abbrev,
        DeclarationType.NoncomputableDef,
        DeclarationType.NoncomputableAbbrev,
    }
)

# Assertion that a declaration name has ended. A name token runs until whitespace or
# one of these delimiters (matching list_all_declarations_in_lean_code's name pattern).
# Use this after a name instead of `\b`: `\b` treats `.` as a boundary, so "Treap" would
# wrongly match the earlier "Treap.insert" declaration.
# A universe binder `.{u}` (valid Lean 4 syntax, e.g. `theorem foo.{u} ...`) may directly
# follow the name, so allow a literal `.{` as a valid boundary too — but still reject a
# qualified-name continuation like `foo.bar` (where `foo` is a prefix of another decl).
DECL_NAME_END = r"(?:(?=\.\{)|(?![^\s:({\[\]},]))"

# Search/suggestion tactics that emit "Try this" and must not appear in a final proof.
# These names are tactic-only — none are API methods ending in "?", so real code like
# List.find?, xs.head?, Array.get?, m.lookup? is NOT matched. Extend as needed.
SEARCH_TACTICS = ("apply", "exact", "rw", "simp", "simp_all", "aesop", "observe")
# Longer names first so "simp_all?" isn't shadowed by "simp".
SEARCH_TACTIC_PATTERN = rf"\b({'|'.join(sorted(SEARCH_TACTICS, key=len, reverse=True))})\?"

# Modifiers that may precede a declaration keyword (e.g. `private def`, `partial def`).
# `noncomputable` is intentionally absent: it is folded into the compound keywords
# `noncomputable def` / `noncomputable abbrev` enumerated in DeclarationType.
DECL_MODIFIERS = ("private", "protected", "partial", "unsafe", "nonrec", "scoped", "local")

# Optional, inline declaration prefix matched before the keyword: zero or more attribute
# lists (`@[simp]`, `@[simp, reducible]`) followed by zero or more modifier words. Uses
# `[ \t]` (not `\s`) so it never spans newlines; nested `]` inside `@[...]` is unsupported.
DECL_PREFIX = (
    r"(?:@\[[^\]]*\][ \t]*)*"
    rf"(?:(?:{'|'.join(DECL_MODIFIERS)})[ \t]+)*"
)

# Declaration keyword alternation, longest-first so compound keywords like
# `noncomputable def` win over the bare `def`.
_KEYWORDS_ALT = "|".join(re.escape(kw) for kw in sorted(LEAN_KEYWORDS, key=len, reverse=True))


def count_pattern(
    content: str,
    pattern: str,
    context_lines: int = 1,
) -> tuple[int, list[tuple[int, str]]]:
    """Count pattern matches in Lean code with context.

    Args:
        content: The Lean file content
        context_lines: Number of lines to show before and after
        pattern: Regex pattern to search for (default: sorry/admit)

    Returns:
        Tuple of (count, locations) where locations is a list of (line_num, formatted_context)
    """
    sorry_locations = []
    lines = content.splitlines()

    for i, line in enumerate(lines):
        for match in re.finditer(pattern, line):
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


def blank_string_literals(src: str) -> str:
    """Replace the CONTENTS of string literals with spaces, preserving the quotes.

    Same state machine as `strip_comments` but inverted: comments are left intact while
    the inside of `"..."` string literals is blanked (length and positions preserved,
    surrounding quotes kept). Used before pattern checks (e.g. search-tactic or `axiom`
    detection) so a tactic-like substring inside a string literal does not falsely match.
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
                out.append(c)
                i += 1
            elif c2 == "--":
                state = ParsingState.LineComment
                out.append(c2)
                i += 2
            elif c2 == "/-":
                state = ParsingState.BlockComment
                depth = 1
                out.append(c2)
                i += 2
            else:
                out.append(c)
                i += 1

        elif state == ParsingState.LineComment:
            if c == "\n":
                state = ParsingState.Out
            out.append(c)
            i += 1

        elif state == ParsingState.BlockComment:
            if c2 == "/-":
                depth += 1
                out.append(c2)
                i += 2
            elif c2 == "-/":
                depth -= 1
                out.append(c2)
                i += 2
                if depth == 0:
                    state = ParsingState.Out
            else:
                out.append(c)
                i += 1

        elif state == ParsingState.StringLiteral:
            if c == '"':
                state = ParsingState.Out
                out.append(c)
            else:
                out.append(" " if c != "\n" else "\n")
            i += 1

    return "".join(out)


def comment_spans(src: str) -> list[tuple[int, int]]:
    """Character spans (start, end) of every Lean comment region in `src`.

    Mirrors `strip_comments`' state machine — handles `--` line comments, nested
    `/- … -/` block comments, and leaves string literals intact — but records each
    comment's character range instead of deleting it. Used to filter out declaration
    keyword matches that fall inside a comment, so raw-content matching agrees with
    the comment-stripped enumeration in `list_all_declarations_in_lean_code`.
    """

    class ParsingState(Enum):
        Out = 1
        LineComment = 2
        BlockComment = 3
        StringLiteral = 4

    state = ParsingState.Out
    i = 0
    depth = 0
    n = len(src)
    spans: list[tuple[int, int]] = []
    comment_start = 0

    while i < n:
        c = src[i]
        c2 = src[i : i + 2]

        if state == ParsingState.Out:
            if c == '"':
                state = ParsingState.StringLiteral
                i += 1
            elif c2 == "--":
                state = ParsingState.LineComment
                comment_start = i
                i += 2
            elif c2 == "/-":
                state = ParsingState.BlockComment
                depth = 1
                comment_start = i
                i += 2
            else:
                i += 1

        elif state == ParsingState.LineComment:
            if c == "\n":
                spans.append((comment_start, i))
                state = ParsingState.Out
            i += 1

        elif state == ParsingState.BlockComment:
            if c2 == "/-":
                depth += 1
                i += 2
            elif c2 == "-/":
                depth -= 1
                i += 2
                if depth == 0:
                    spans.append((comment_start, i))
                    state = ParsingState.Out
            else:
                i += 1

        elif state == ParsingState.StringLiteral:
            if c == '"':
                state = ParsingState.Out
            i += 1

    # An unterminated line/block comment runs to end-of-input.
    if state in (ParsingState.LineComment, ParsingState.BlockComment):
        spans.append((comment_start, n))

    return spans


def _in_any_span(offset: int, spans: list[tuple[int, int]]) -> bool:
    """True if `offset` falls within any (start, end) span (end-exclusive)."""
    return any(start <= offset < end for start, end in spans)


def non_comment_matches(pattern: str, content: str) -> list[re.Match[str]]:
    """`re.finditer(pattern, content, re.MULTILINE)` matches not starting inside a comment.

    Keeps raw-content occurrence indexing aligned with the comment-stripped
    enumeration in `list_all_declarations_in_lean_code`, which ignores commented decls.
    """
    spans = comment_spans(content)
    return [
        m for m in re.finditer(pattern, content, re.MULTILINE) if not _in_any_span(m.start(), spans)
    ]


# A command prefix that binds to the following declaration ends with a standalone `in`
# keyword (e.g. `open Nat in`, `set_option foo true in`, `variable (x) in`). The negative
# lookbehind `(?<![\w.])` ensures we match the keyword `in`, not the tail of an identifier
# like `Fin` or `min`.
_IN_PREFIX_LINE = re.compile(r"(?<![\w.])in[ \t]*$")


def _extend_start_over_in_prefixes(content: str, start_pos: int) -> int:
    """Move start_pos back over contiguous preceding `... in` command-prefix lines.

    Stops at a blank line or any line that is not an `... in` prefix, so ordinary
    preceding declarations and comments are never absorbed.
    """
    # start_pos may sit on leading whitespace (the matcher's `\s*` can span blank lines),
    # so advance to the first non-blank line — the real start of the declaration block.
    while start_pos < len(content) and content[start_pos] in " \t\n":
        start_pos += 1

    line_start = content.rfind("\n", 0, start_pos) + 1
    while line_start > 0:
        # The line above the current block (without its trailing newline).
        prev_line_end = line_start - 1
        prev_line_start = content.rfind("\n", 0, prev_line_end) + 1
        prev_line = content[prev_line_start:prev_line_end]

        if not prev_line.strip() or not _IN_PREFIX_LINE.search(prev_line.rstrip()):
            break

        line_start = prev_line_start

    return line_start


def extract_function_from_content(
    content: str, function_name: str, occurrence: int = 0
) -> str | None:
    """Extract a function/theorem/lemma definition from Lean code.

    Args:
        content: Lean code content as string
        function_name: Name of the function/theorem/lemma to extract
        occurrence: 0-based index to disambiguate when several declarations share the
            same simple name in one file (e.g. `A.insert` and `B.insert` written as
            `def insert` inside different namespaces). 0 selects the first.

    Returns:
        The complete definition block including doc comments, or None
    """
    keywords_pattern = "|".join(LEAN_KEYWORDS)
    pattern = rf"^(\s*){DECL_PREFIX}(?:{_KEYWORDS_ALT})\s+{re.escape(function_name)}{DECL_NAME_END}"

    matches = non_comment_matches(pattern, content)
    if occurrence >= len(matches):
        return None
    match = matches[occurrence]

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

    # Include any contiguous command-prefix lines that bind to this declaration via a
    # trailing `in` (e.g. `open Nat in`, `set_option ... in`). Without them the bound
    # command is lost when the block is applied -> "unknown identifier". Walk backward
    # over preceding lines that end in a standalone ` in` token, stopping at a blank line
    # or any other line (ordinary decls, comments) so we never absorb unrelated code.
    start_pos = _extend_start_over_in_prefixes(content, start_pos)

    # Find next definition, doc comment, structural keyword, or top-level comment
    # at same or lower indentation. The next declaration may itself carry an
    # attribute/modifier prefix (e.g. `@[simp] def`), so allow DECL_PREFIX before it.
    end_pattern = rf"^[ \t]{{0,{start_indent}}}(/--|--|{DECL_PREFIX}(?:{_KEYWORDS_ALT})(?:\s+|\b))"

    remaining_content = content[match.end() :]
    end_match = re.search(end_pattern, remaining_content, re.MULTILINE)

    if end_match:
        end_pos = match.end() + end_match.start()
    else:
        end_pos = len(content)

    return content[start_pos:end_pos].strip()


def _iter_namespaced_declarations(content: str):
    """Yield (declaration, qualified_name, occurrence) for each named declaration.

    Tracks namespace/section scope to build each declaration's namespace-qualified name
    and counts per-simple-name occurrences (aligned with `extract_function_from_content`'s
    re.finditer ordering). This mirrors the namespace-aware enumeration in
    `tools.local_lean_search._iter_searchable`; it lives here so both the search tool and
    `utils.build` can resolve a target without `utils` importing from `tools` (which would
    create a circular / layering dependency).
    """
    namespace_stack: list[str | None] = []
    name_counts: dict[str, int] = {}
    for declaration in list_all_declarations_in_lean_code(content):
        occurrence = name_counts.get(declaration.name, 0)
        name_counts[declaration.name] = occurrence + 1
        declaration_type = declaration.declaration_type
        if declaration_type == DeclarationType.Namespace:
            namespace_stack.append(declaration.name)
            continue
        if declaration_type == DeclarationType.Section:
            namespace_stack.append(None)  # sections do not contribute to the name
            continue
        if declaration_type == DeclarationType.End:
            if namespace_stack:
                namespace_stack.pop()
            continue
        if declaration_type not in STRIPPABLE_DECLARATION_TYPES:
            continue
        prefix = ".".join(part for part in namespace_stack if part)
        if prefix and not declaration.name.startswith(f"{prefix}."):
            qualified = f"{prefix}.{declaration.name}"
        else:
            qualified = declaration.name
        yield declaration, qualified, occurrence


def resolve_target_occurrence(content: str, target_name: str) -> tuple[str, int] | None:
    """Resolve `target_name` to the `(simple_name, occurrence)` of its declaration in `content`.

    `target_name` may be namespace-qualified (`Treap.insert`) or simple (`insert`). The
    namespace-qualified name of each declaration is computed (tracking `namespace`/`section`
    scope), then a match is sought where the qualified name equals `target_name`, or — for a
    simple `target_name` — where the qualified name equals it or ends with `.<target_name>`.

    `occurrence` is the 0-based index among declarations sharing the SIMPLE name, suitable
    for passing to `extract_function_from_content`.

    Returns None when there is no match or the match is ambiguous (more than one distinct
    declaration matches), so callers can fall back to the conservative occurrence-0 default.
    """
    matches: list[tuple[str, int]] = []
    is_qualified = "." in target_name
    for declaration, qualified, occurrence in _iter_namespaced_declarations(content):
        if qualified == target_name:
            matches.append((declaration.name, occurrence))
        elif not is_qualified and qualified.endswith(f".{target_name}"):
            matches.append((declaration.name, occurrence))

    if len(matches) == 1:
        return matches[0]
    return None


def find_stripped_declaration_names(raw_code: str, target_name: str) -> list[str]:
    """Names of top-level declarations that would be stripped when applying a proposal.

    Only the target declaration survives `TemporaryProposal` application (it calls
    `extract_function_from_content` to keep a single declaration). Any other top-level
    `def`/`lemma`/`theorem`/etc. the proposer wrote is silently discarded, which breaks
    references to it. This returns those soon-to-be-stripped declaration names so the
    builder can warn the proposer.
    """
    declarations = list_all_declarations_in_lean_code(raw_code)
    return [
        d.name
        for d in declarations
        if d.name != target_name and d.declaration_type in STRIPPABLE_DECLARATION_TYPES
    ]


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
    # Match an optional attribute/modifier prefix, then the keyword (longest-first so
    # `noncomputable def` beats `def`), the name, and the rest of the line.
    declaration_pattern = re.compile(
        rf"^{DECL_PREFIX}({_KEYWORDS_ALT})\s+([^\s:({{[\]}},]+)\s*(.*)"
    )
    code = strip_comments(raw_code)

    for line in code.split("\n"):
        declaration_match = declaration_pattern.match(line.strip())
        # The regex's keyword group is built from LEAN_KEYWORDS, so a match already
        # guarantees a valid declaration keyword — no extra membership check needed.
        if declaration_match:
            if declaration is not None:
                declarations.append(declaration)
            declaration = Declaration(
                declaration_type=declaration_match.group(1),
                name=declaration_match.group(2),
                content=declaration_match.group(3),
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

    pattern = rf"^(\s*){DECL_PREFIX}({_KEYWORDS_ALT})\s+([\w.]+)"

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
