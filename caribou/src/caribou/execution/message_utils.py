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
from typing import List, Optional, Tuple


# --- Regex Patterns ---
_DELEG_RE = re.compile(r"delegate_to_([A-Za-z0-9_]+)")
# Matches both `query_rag_<topic>` (canonical) and `query_rag_topic` (LLMs often drop the brackets)
_RAG_RE = re.compile(r"query_rag_(?:<([^>]+)>|([A-Za-z0-9_.]+))")
_END_SESSION_RE = re.compile(r"^\s*end_session\s*$", re.MULTILINE)
_CODE_BLOCK_RE = re.compile(r"```(?:python|r|R)?[ \t]*\n[\s\S]*?\n```", re.MULTILINE)


def detect_delegation(msg: str) -> Optional[str]:
    """Return the *full* command name (e.g. 'delegate_to_coder') if present."""
    m = _DELEG_RE.search(msg)
    return f"delegate_to_{m.group(1)}" if m else None


def detect_rag(msg: str) -> Optional[str]:
    """Return the RAG topic if present (handles both `<topic>` and bare-word forms)."""
    m = _RAG_RE.search(msg)
    if m:
        return m.group(1) or m.group(2)
    return None


def detect_end_session(msg: str) -> bool:
    """Return True if the assistant requests to end the session as a standalone line."""
    if not msg:
        return False
    return bool(_END_SESSION_RE.search(msg))


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
    return len(_CODE_BLOCK_RE.findall(msg))


def _code_preview(code: str, max_chars: int = 200, max_lines: int = 4) -> str:
    """Return a short, meaningful preview of a code block."""
    lines = [ln.strip() for ln in code.splitlines() if ln.strip()]
    snippet = "\n".join(lines[:max_lines])
    if len(snippet) > max_chars:
        snippet = snippet[: max_chars - 3] + "..."
    return snippet or "(empty code block)"
