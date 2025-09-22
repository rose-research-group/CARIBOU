from __future__ import annotations
import json
from typing import List, Dict

class MemoryManager:
    """
    Manages the agent's conversation history with episodic summarization
    to prevent context decay in long-running sessions.
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
            "working_history_size": working_history_size,
            "summarization_threshold": summarization_threshold,
            "chunk_size_to_summarize": chunk_size_to_summarize
        }

        # --- Internal State ---
        self._full_history: List[Dict[str, str]] = list(initial_history)
        self._summarized_log: List[Dict[str, str]] = []
        self._pivotal_code: List[Dict[str, str]] = []
        self._pinned_messages: List[Dict[str, str]] = self._full_history[:3]

    def add_message(self, role: str, content: str):
        """Adds a new message to the full, unabridged history."""
        self._full_history.append({"role": role, "content": content})

    def add_pivotal_code(self, code: str):
        """Adds a successfully executed code snippet to be preserved in context."""
        formatted_message = {
            "role": "system",
            "content": f"PIVOTAL CODE (Successfully Executed):\n```python\n{code}\n```"
        }
        self._pivotal_code.append(formatted_message)

    # --- ADDED: Method to update the agent-specific system prompt ---
    def update_system_prompt(self, new_prompt_content: str):
        """
        Replaces the agent-specific prompt (at index 1) upon delegation,
        leaving the global policy (at index 0) intact.
        """
        # Ensure there are enough messages to have a separate agent prompt
        if len(self._pinned_messages) > 1:
            # The agent-specific prompt is now the second message
            self._pinned_messages[1]["content"] = new_prompt_content
            self._full_history[1]["content"] = new_prompt_content
        else:
            # Fallback for unexpected history structures
            new_prompt = {"role": "system", "content": new_prompt_content}
            self._pinned_messages.insert(1, new_prompt)
            self._full_history.insert(1, new_prompt)


    def _summarize_chunk(self):
        """Internal method to summarize a chunk of the history."""
        start_index = len(self._pinned_messages) + len(self._summarized_log)
        end_index = start_index + self.config["chunk_size_to_summarize"]
        
        if end_index > len(self._full_history) - self.config["working_history_size"]:
            return

        chunk_to_summarize = self._full_history[start_index:end_index]
        if not chunk_to_summarize:
            return

        summary_prompt = "You are a summarization AI. Condense the following conversation excerpt into a concise, factual paragraph. Focus on key decisions, executed code, and critical outcomes. This summary will serve as the memory for another AI, so clarity and accuracy are essential."
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": summary_prompt},
                    {"role": "user", "content": json.dumps(chunk_to_summarize)}
                ],
                temperature=0.2
            )
            summary_text = response.choices[0].message.content
            print(f"Generated summary: {summary_text}")
            self._summarized_log.append({"role": "system", "content": f"EPISODIC SUMMARY:\n{summary_text}"})
            print(self._summarized_log)
        except Exception as e:
            print(f"Warning: Could not summarize context chunk: {e}")

    def get_context(self) -> List[Dict[str, str]]:
        """
        Dynamically assembles the context to be sent to the LLM.
        """
        if len(self._full_history) > self.config["summarization_threshold"] + len(self._summarized_log):
            self._summarize_chunk()

        working_history = self._full_history[-self.config["working_history_size"]:]

        context_to_send = (
            self._pinned_messages +
            self._pivotal_code +
            self._summarized_log +
            working_history
        )
        return context_to_send