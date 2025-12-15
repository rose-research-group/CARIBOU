from __future__ import annotations
import json
from typing import List, Dict

class MemoryManager:
    """
    Manages the agent's conversation history with episodic summarization
    to prevent context decay in long-running sessions.

    Context layout returned by `get_context()`:
      [pinned (early/system), pivotal_code, summarized_log, recent working_history]
    """

    def __init__(
        self,
        llm_client: object,
        model_name: str,
        initial_history: List[Dict[str, str]],
        working_history_size: int = 4,
        summarization_threshold: int = 20,
        chunk_size_to_summarize: int = 10
    ):
        self.llm_client = llm_client
        self.model_name = model_name
        self.config = {
            "working_history_size": int(working_history_size),
            "summarization_threshold": int(summarization_threshold),
            "chunk_size_to_summarize": int(chunk_size_to_summarize),
        }

        # --- Internal State ---
        self._full_history: List[Dict[str, str]] = list(initial_history)
        self._summarized_log: List[Dict[str, str]] = []  # list of {"role": "system", "content": "EPISODIC SUMMARY:\n..."}
        self._pivotal_code: List[Dict[str, str]] = []

        # Pin the first N early/system messages (robust to short histories)
        # Default behavior: try to keep (at least) first 2 system messages (global policy, agent prompt) + maybe 1 more
        pin_n = min(3, len(self._full_history))
        self._pinned_messages: List[Dict[str, str]] = [self._full_history[i] for i in range(pin_n)]

    # ------------------------------
    # Public mutation API
    # ------------------------------

    def add_message(self, role: str, content: str) -> None:
        """Adds a new message to the full, unabridged history."""
        self._full_history.append({"role": role, "content": content})

    def add_pivotal_code(self, code: str) -> None:
        """Adds a successfully executed code snippet to be preserved in context."""
        formatted_message = {
            "role": "system",
            "content": f"PIVOTAL CODE (Successfully Executed):\n```python\n{code}\n```"
        }
        self._pivotal_code.append(formatted_message)

    def update_system_prompt(self, new_prompt_content: str) -> None:
        """
        Replaces the agent-specific prompt while leaving the global policy intact.
        Tries to modify the second pinned message if it exists; otherwise inserts it.
        """
        if len(self._pinned_messages) >= 2:
            # Update in place (these dicts are same objects as in _full_history for early items)
            self._pinned_messages[1]["content"] = new_prompt_content
        else:
            new_prompt = {"role": "system", "content": new_prompt_content}
            self._pinned_messages.insert(1, new_prompt)
            # Also reflect in full history at index 1 if possible; else insert
            if len(self._full_history) >= 2:
                self._full_history[1] = new_prompt
            else:
                self._full_history.insert(1, new_prompt)

    # ------------------------------
    # Summarization implementation
    # ------------------------------

    def _summarize_chunk(self) -> None:
        """
        Summarize the next unsummarized chunk of the history.

        We compute a moving window that starts after pinned messages and after
        all previously summarized messages (measured in *messages*, not summaries).
        """
        chunk_size = self.config["chunk_size_to_summarize"]
        working_tail = self.config["working_history_size"]

        # How many messages have we already summarized (in terms of original messages)?
        summarized_message_count = len(self._summarized_log) * chunk_size

        # Establish the start/end for the next chunk
        start_index = len(self._pinned_messages) + summarized_message_count
        end_index = start_index + chunk_size

        # Do not encroach on the working tail
        max_end = max(0, len(self._full_history) - working_tail)
        if start_index >= max_end or end_index > max_end:
            # Not enough unsummarized messages yet to safely summarize another chunk.
            return

        chunk_to_summarize = self._full_history[start_index:end_index]
        if not chunk_to_summarize:
            return

        summary_prompt = (
            "You are a summarization AI. Condense the following conversation excerpt into a concise, factual paragraph. "
            "Focus on key decisions, executed code, and critical outcomes. This summary will serve as memory for another AI, "
            "so clarity and accuracy are essential."
        )

        try:
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": summary_prompt},
                    {"role": "user", "content": json.dumps(chunk_to_summarize)},
                ],
                temperature=0.0,
            )
            summary_text = response.choices[0].message.content
            self._summarized_log.append(
                {"role": "system", "content": f"EPISODIC SUMMARY:\n{summary_text}"}
            )
        except Exception as e:
            # Non-fatal: keep going with existing context
            print(f"Warning: Could not summarize context chunk: {e}")

    def _should_summarize(self) -> bool:
        """
        Decide whether to attempt another summarization pass.

        We compute how many unsummarized messages lie between the pinned head and the working tail.
        If that count exceeds the threshold, we summarize the next chunk.
        """
        chunk_size = self.config["chunk_size_to_summarize"]
        working_tail = self.config["working_history_size"]

        summarized_message_count = len(self._summarized_log) * chunk_size
        total = len(self._full_history)

        # Messages we must keep untouched on the head and tail
        head = len(self._pinned_messages)
        tail = working_tail

        # Unsummarized region length between head and tail:
        unsummarized_len = max(0, total - head - tail - summarized_message_count)
        return unsummarized_len > self.config["summarization_threshold"]

    # ------------------------------
    # Context assembly
    # ------------------------------

    def get_context(self) -> List[Dict[str, str]]:
        """
        Dynamically assembles the context to be sent to the LLM:
          pinned (early/system) + pivotal_code + episodic summaries + recent working history.

        It attempts a single summarization step per call when the threshold condition is met.
        """
        if self._should_summarize():
            self._summarize_chunk()

        # Take the most recent N full messages as "working history"
        working_history = self._full_history[-self.config["working_history_size"]:] if self._full_history else []

        context_to_send = (
            self._pinned_messages
            + self._pivotal_code
            + self._summarized_log
            + working_history
        )
        return context_to_send
