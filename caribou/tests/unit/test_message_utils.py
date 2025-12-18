"""
Unit tests for message parsing and routing utilities.

Tests delegation detection, RAG query detection, and artifact extraction.
"""
import pytest

from caribou.execution.message_utils import (
    detect_delegation,
    detect_rag,
    _extract_artifacts_from_msg,
    _count_code_blocks,
    _code_preview,
)


class TestDelegationDetection:
    """Test delegation command detection."""

    def test_detect_simple_delegation(self):
        """Test detecting simple delegation command."""
        msg = "I will delegate_to_coder to handle this task."
        result = detect_delegation(msg)
        assert result == "delegate_to_coder"

    def test_detect_delegation_with_underscores(self):
        """Test detecting delegation with underscores in agent name."""
        msg = "Let's delegate_to_my_special_agent for this."
        result = detect_delegation(msg)
        assert result == "delegate_to_my_special_agent"

    def test_detect_delegation_with_numbers(self):
        """Test detecting delegation with numbers in agent name."""
        msg = "Time to delegate_to_agent123 now."
        result = detect_delegation(msg)
        assert result == "delegate_to_agent123"

    def test_no_delegation(self):
        """Test when no delegation command present."""
        msg = "This is just a regular message without delegation."
        result = detect_delegation(msg)
        assert result is None

    def test_delegation_at_start(self):
        """Test delegation command at message start."""
        msg = "delegate_to_planner - we need to plan this better."
        result = detect_delegation(msg)
        assert result == "delegate_to_planner"

    def test_delegation_at_end(self):
        """Test delegation command at message end."""
        msg = "Let me handle this by using delegate_to_executor"
        result = detect_delegation(msg)
        assert result == "delegate_to_executor"

    def test_multiple_delegations(self):
        """Test that first delegation is found when multiple present."""
        msg = "First delegate_to_agent1 then delegate_to_agent2"
        result = detect_delegation(msg)
        assert result == "delegate_to_agent1"

    def test_delegation_case_sensitive(self):
        """Test that delegation is case-sensitive."""
        msg = "Should not match DELEGATE_TO_CODER or Delegate_To_Coder"
        result = detect_delegation(msg)
        assert result is None

    def test_delegation_with_special_chars_fails(self):
        """Test that special characters break the pattern."""
        msg = "This delegate_to_agent-name should not match"
        result = detect_delegation(msg)
        # Should match "delegate_to_agent" only (stops at hyphen)
        assert result == "delegate_to_agent"

    def test_empty_message(self):
        """Test with empty message."""
        result = detect_delegation("")
        assert result is None


class TestRAGDetection:
    """Test RAG query detection."""

    def test_detect_simple_rag_query(self):
        """Test detecting simple RAG query."""
        msg = "Let me query_rag_<search for documentation> to find info."
        result = detect_rag(msg)
        assert result == "search for documentation"

    def test_detect_rag_query_with_spaces(self):
        """Test RAG query with spaces."""
        msg = "I need to query_rag_<API authentication methods>"
        result = detect_rag(msg)
        assert result == "API authentication methods"

    def test_detect_rag_query_at_start(self):
        """Test RAG query at message start."""
        msg = "query_rag_<database schema> should help us here."
        result = detect_rag(msg)
        assert result == "database schema"

    def test_no_rag_query(self):
        """Test when no RAG query present."""
        msg = "This is just a regular message."
        result = detect_rag(msg)
        assert result is None

    def test_rag_query_empty_brackets(self):
        """Test RAG query with empty brackets won't match (requires at least one char)."""
        msg = "What about query_rag_<> this?"
        result = detect_rag(msg)
        # The [^>]+ pattern requires at least one character, so empty brackets don't match
        assert result is None

    def test_multiple_rag_queries(self):
        """Test that first RAG query is found when multiple present."""
        msg = "First query_rag_<query1> then query_rag_<query2>"
        result = detect_rag(msg)
        assert result == "query1"

    def test_rag_query_with_newlines(self):
        """Test RAG query can match content with newlines."""
        msg = "query_rag_<this has\nnewlines>"
        result = detect_rag(msg)
        # The [^>]+ pattern matches any char except >, including newlines
        assert result == "this has\nnewlines"

    def test_empty_message(self):
        """Test with empty message."""
        result = detect_rag("")
        assert result is None


class TestArtifactExtraction:
    """Test extraction of notes and TODOs from messages."""

    def test_extract_note_prefix(self):
        """Test extracting NOTE: prefix."""
        msg = "NOTE: This is an important observation."
        notes, todos = _extract_artifacts_from_msg(msg)
        assert len(notes) == 1
        assert notes[0] == "This is an important observation."
        assert len(todos) == 0

    def test_extract_todo_prefix(self):
        """Test extracting TODO: prefix."""
        msg = "TODO: Implement error handling."
        notes, todos = _extract_artifacts_from_msg(msg)
        assert len(notes) == 0
        assert len(todos) == 1
        assert todos[0] == "Implement error handling."

    def test_extract_checkbox_unchecked(self):
        """Test extracting unchecked checkbox."""
        msg = "- [ ] Complete the documentation"
        notes, todos = _extract_artifacts_from_msg(msg)
        assert len(todos) == 1
        assert todos[0] == "Complete the documentation"

    def test_extract_checkbox_checked(self):
        """Test extracting checked checkbox."""
        msg = "- [x] Write unit tests"
        notes, todos = _extract_artifacts_from_msg(msg)
        assert len(todos) == 1
        assert todos[0] == "Write unit tests"

    def test_extract_checkbox_checked_uppercase(self):
        """Test extracting checked checkbox with uppercase X."""
        msg = "- [X] Deploy to production"
        notes, todos = _extract_artifacts_from_msg(msg)
        assert len(todos) == 1
        assert todos[0] == "Deploy to production"

    def test_extract_notes_code_fence(self):
        """Test extracting notes from code fence."""
        msg = """
```notes
First note
Second note
Third note
```
        """
        notes, todos = _extract_artifacts_from_msg(msg)
        assert len(notes) == 3
        assert "First note" in notes
        assert "Second note" in notes
        assert "Third note" in notes

    def test_extract_todos_code_fence(self):
        """Test extracting TODOs from code fence."""
        msg = """
```todo
Task 1
Task 2
```
        """
        notes, todos = _extract_artifacts_from_msg(msg)
        assert len(todos) == 2
        assert "Task 1" in todos
        assert "Task 2" in todos

    def test_extract_todos_code_fence_plural(self):
        """Test extracting TODOs from code fence with plural."""
        msg = """
```todos
Task A
Task B
```
        """
        notes, todos = _extract_artifacts_from_msg(msg)
        assert len(todos) == 2
        assert "Task A" in todos
        assert "Task B" in todos

    def test_extract_mixed_artifacts(self):
        """Test extracting mixed notes and TODOs."""
        msg = """
NOTE: Configuration loaded successfully
TODO: Add validation
- [ ] Write tests
- [x] Update documentation

```notes
System initialized
```

```todo
Refactor code
```
        """
        notes, todos = _extract_artifacts_from_msg(msg)

        assert len(notes) == 2
        assert "Configuration loaded successfully" in notes
        assert "System initialized" in notes

        assert len(todos) == 4
        assert "Add validation" in todos
        assert "Write tests" in todos
        assert "Update documentation" in todos
        assert "Refactor code" in todos

    def test_extract_case_insensitive_note(self):
        """Test that NOTE: is case-insensitive."""
        msg = "note: Lowercase note"
        notes, todos = _extract_artifacts_from_msg(msg)
        assert len(notes) == 1
        assert notes[0] == "Lowercase note"

    def test_extract_case_insensitive_todo(self):
        """Test that TODO: is case-insensitive."""
        msg = "todo: Lowercase todo"
        notes, todos = _extract_artifacts_from_msg(msg)
        assert len(todos) == 1
        assert todos[0] == "Lowercase todo"

    def test_extract_empty_code_fence_ignored(self):
        """Test that empty code fences are ignored."""
        msg = """
```notes
```

```todo
```
        """
        notes, todos = _extract_artifacts_from_msg(msg)
        assert len(notes) == 0
        assert len(todos) == 0

    def test_extract_with_empty_lines(self):
        """Test that empty lines are skipped."""
        msg = """
NOTE: First note

NOTE: Second note


TODO: First task

TODO: Second task
        """
        notes, todos = _extract_artifacts_from_msg(msg)
        assert len(notes) == 2
        assert len(todos) == 2

    def test_no_artifacts(self):
        """Test message with no artifacts."""
        msg = "This is just a regular message with no notes or TODOs."
        notes, todos = _extract_artifacts_from_msg(msg)
        assert len(notes) == 0
        assert len(todos) == 0

    def test_empty_message(self):
        """Test with empty message."""
        notes, todos = _extract_artifacts_from_msg("")
        assert len(notes) == 0
        assert len(todos) == 0


class TestCodeBlockCounting:
    """Test code block counting."""

    def test_count_single_code_block(self):
        """Test counting single code block."""
        msg = """
Here is some code:
```python
print("Hello")
```
        """
        count = _count_code_blocks(msg)
        assert count == 1

    def test_count_multiple_code_blocks(self):
        """Test counting multiple code blocks."""
        msg = """
First block:
```python
x = 1
```

Second block:
```
y = 2
```

Third block:
```python
z = 3
```
        """
        count = _count_code_blocks(msg)
        assert count == 3

    def test_count_code_block_without_language(self):
        """Test counting code block without language specifier."""
        msg = """
```
generic code
```
        """
        count = _count_code_blocks(msg)
        assert count == 1

    def test_count_no_code_blocks(self):
        """Test counting when no code blocks present."""
        msg = "This message has no code blocks."
        count = _count_code_blocks(msg)
        assert count == 0

    def test_count_empty_message(self):
        """Test counting with empty message."""
        count = _count_code_blocks("")
        assert count == 0

    def test_count_none_message(self):
        """Test counting with None message."""
        count = _count_code_blocks(None)
        assert count == 0

    def test_count_inline_code_ignored(self):
        """Test that inline code is not counted."""
        msg = "This has `inline code` but no blocks."
        count = _count_code_blocks(msg)
        assert count == 0


class TestCodePreview:
    """Test code preview generation."""

    def test_preview_short_code(self):
        """Test preview of short code snippet."""
        code = "x = 1\ny = 2"
        preview = _code_preview(code)
        assert preview == "x = 1\ny = 2"

    def test_preview_truncate_long_code(self):
        """Test that long code is truncated."""
        code = "a" * 300
        preview = _code_preview(code, max_chars=200)
        assert len(preview) <= 203  # 200 + "..."
        assert preview.endswith("...")

    def test_preview_limit_lines(self):
        """Test that number of lines is limited."""
        code = "\n".join([f"line {i}" for i in range(10)])
        preview = _code_preview(code, max_lines=4)
        lines = preview.split("\n")
        assert len(lines) <= 4

    def test_preview_strips_empty_lines(self):
        """Test that empty lines are stripped."""
        code = "\n\nline 1\n\nline 2\n\n"
        preview = _code_preview(code)
        assert preview == "line 1\nline 2"

    def test_preview_empty_code(self):
        """Test preview of empty code."""
        preview = _code_preview("")
        assert preview == "(empty code block)"

    def test_preview_whitespace_only(self):
        """Test preview of whitespace-only code."""
        preview = _code_preview("   \n   \n   ")
        assert preview == "(empty code block)"

    def test_preview_default_params(self):
        """Test preview with default parameters."""
        code = "\n".join([f"line {i}" for i in range(10)])
        preview = _code_preview(code)
        # Should limit to 4 lines by default
        lines = [ln for ln in preview.split("\n") if ln.strip()]
        assert len(lines) <= 4


class TestEdgeCases:
    """Test edge cases across all utilities."""

    def test_unicode_in_delegation(self):
        """Test handling unicode in messages."""
        msg = "Let's delegate_to_agent with 🎯 emoji"
        result = detect_delegation(msg)
        assert result == "delegate_to_agent"

    def test_unicode_in_artifacts(self):
        """Test unicode in artifacts."""
        msg = "NOTE: Handle UTF-8 like café and 日本語"
        notes, todos = _extract_artifacts_from_msg(msg)
        assert len(notes) == 1
        assert "café" in notes[0]
        assert "日本語" in notes[0]

    def test_multiline_mixed_content(self):
        """Test complex multiline message."""
        msg = """
I think we should delegate_to_planner first.

NOTE: Current approach is working
TODO: Add error handling

```python
def foo():
    pass
```

Then query_rag_<find examples> for more info.

- [ ] Review code
- [x] Write tests
        """

        delegation = detect_delegation(msg)
        assert delegation == "delegate_to_planner"

        rag = detect_rag(msg)
        assert rag == "find examples"

        notes, todos = _extract_artifacts_from_msg(msg)
        assert len(notes) == 1
        assert len(todos) == 3

        count = _count_code_blocks(msg)
        assert count == 1
