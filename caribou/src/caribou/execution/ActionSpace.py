from __future__ import annotations

import json
from typing import Dict, List, Optional


class AgentActionSpace:
    """
    Lightweight, serializable tracker for an agent's recent actions and
    available future actions. Intended to be surfaced to the LLM so it can
    reason about what it has done and what it can do next.
    """

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.past_actions: List[Dict] = []  # Each: {"type": "...", "description": "...", "status": "...", "meta": {...}}
        self.possible_actions: List[Dict] = []  # Each: {"name": "...", "detail": "..."}

    # ----------------------------
    # Mutation helpers
    # ----------------------------
    def set_possible_actions(self, actions: List[Dict]) -> None:
        self.possible_actions = list(actions) if actions else []

    def add_possible_action(self, action: Dict) -> None:
        self.possible_actions.append(action)

    def add_action(
        self,
        action_type: str,
        description: str,
        *,
        status: Optional[str] = None,
        meta: Optional[Dict] = None,
    ) -> None:
        entry = {
            "type": action_type,
            "description": description,
        }
        if status is not None:
            entry["status"] = status
        if meta:
            entry["meta"] = meta
        self.past_actions.append(entry)

    # ----------------------------
    # Serialization helpers
    # ----------------------------
    def to_dict(self, max_recent: int = 5) -> Dict:
        return {
            "agent": self.agent_name,
            "recent_actions": self.past_actions[-max_recent:],
            "possible_actions": self.possible_actions,
        }

    def to_message(self, max_recent: int = 5) -> str:
        """
        Returns a compact system message for the LLM.
        """
        payload = self.to_dict(max_recent=max_recent)
        return "ACTION SPACE UPDATE:\n" + json.dumps(payload, ensure_ascii=False)
