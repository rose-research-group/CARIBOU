"""
Message parsing and text processing utilities for agent communication.

This module handles:
- Detecting delegation commands in agent messages
- Detecting RAG queries in agent messages
- Extracting artifacts (notes, TODOs) from messages
- Counting and previewing code blocks
"""
from __future__ import annotations

import re
from typing import Iterator, List, Optional, Tuple

from caribou.core.io_helpers import extract_python_code_blocks


# --- Regex Patterns ---
_DELEG_RE = re.compile(r"delegate_to_([A-Za-z0-9_]+)")
_RAG_LINE_RE = re.compile(
    r"^[ \t]*query_rag_(?:<(?P<canonical>[^<>`\r\n]+)>|"
    r"(?P<bare>[^<>`\r\n]+?))[ \t]*$"
)
_MARKDOWN_FENCE_RE = re.compile(r"^[ \t]*(?P<marker>`{3,}|~{3,})")
_END_SESSION_RE = re.compile(r"^\s*end_session\s*$", re.MULTILINE)


def detect_delegation(msg: str) -> Optional[str]:
    """Return the *full* command name (e.g. 'delegate_to_coder') if present."""
    m = _DELEG_RE.search(msg)
    return f"delegate_to_{m.group(1)}" if m else None


def _lines_outside_fenced_code(msg: str) -> Iterator[str]:
    """Yield message lines that are not inside Markdown fenced code."""
    fence_character: Optional[str] = None
    fence_length = 0

    for line in msg.splitlines():
        fence_match = _MARKDOWN_FENCE_RE.match(line)
        if fence_character is None:
            if fence_match:
                marker = fence_match.group("marker")
                fence_character = marker[0]
                fence_length = len(marker)
                continue
            yield line
            continue

        if fence_match:
            marker = fence_match.group("marker")
            if (
                marker[0] == fence_character
                and len(marker) >= fence_length
                and not line[fence_match.end() :].strip()
            ):
                fence_character = None
                fence_length = 0


def detect_rag(msg: str) -> Optional[str]:
    """Return the topic from the first standalone RAG command outside code."""
    if not msg:
        return None

    for line in _lines_outside_fenced_code(msg):
        match = _RAG_LINE_RE.fullmatch(line)
        if not match:
            continue
        topic = match.group("canonical") or match.group("bare")
        if topic and topic.strip():
            return topic.strip() if match.group("bare") else topic
    return None


def detect_end_session(msg: str) -> bool:
    """Return True if the assistant requests to end the session as a standalone line."""
    if not msg:
        return False
    return bool(_END_SESSION_RE.search(msg))


def extract_labeled_block(msg: str, label: str) -> Optional[str]:
    """Return the (raw, unparsed) content of the first ```<label> fenced block,
    or None if there isn't one. Used for the session-brief block during the
    briefing phase (WS-5); this module has fenced-block helpers for code,
    notes/todos, and RAG, but nothing generic for an arbitrary label, so this
    is new rather than reused.
    """
    if not msg:
        return None
    pattern = re.compile(
        r"```" + re.escape(label) + r"[ \t]*\n([\s\S]*?)^[ \t]*```[ \t]*$",
        re.MULTILINE,
    )
    match = pattern.search(msg)
    if not match:
        return None
    content = match.group(1).strip()
    return content or None


def _extract_artifacts_from_msg(msg: str) -> Tuple[List[str], List[str]]:
    """Return (notes, todos) extracted from assistant content."""
    notes: List[str] = []
    todos: List[str] = []

    # Code fences for bulk capture
    fence_patterns = [
        (r"```notes\n([\s\S]*?)```", notes),
        (r"```todo\n([\s\S]*?)```", todos),
        (r"```todos\n([\s\S]*?)```", todos),
    ]
    for pattern, bucket in fence_patterns:
        for m in re.finditer(pattern, msg, flags=re.IGNORECASE):
            content = m.group(1).strip()
            if not content:
                continue
            lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
            for ln in lines:
                bucket.append(ln)

    for raw_line in msg.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        upper = line.upper()
        if upper.startswith("NOTE:"):
            notes.append(line[len("NOTE:"):].strip())
            continue
        if upper.startswith("TODO:"):
            todos.append(line[len("TODO:"):].strip())
            continue
        if line.startswith("- [ ]"):
            todos.append(line[len("- [ ]"):].strip())
            continue
        if line.startswith("- [x]") or line.startswith("- [X]"):
            todos.append(line[len("- [x]"):].strip())

    return notes, todos


def _count_code_blocks(msg: str) -> int:
    """Count fenced code blocks in an assistant message."""
    if not msg:
        return 0
    return len(extract_python_code_blocks(msg))


def _code_preview(code: str, max_chars: int = 200, max_lines: int = 4) -> str:
    """Return a short, meaningful preview of a code block."""
    lines = [ln.strip() for ln in code.splitlines() if ln.strip()]
    snippet = "\n".join(lines[:max_lines])
    if len(snippet) > max_chars:
        snippet = snippet[: max_chars - 3] + "..."
    return snippet or "(empty code block)"
