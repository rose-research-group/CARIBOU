"""
Agent management utilities for multi-agent workflows.

This module handles:
- Extracting possible actions from agents
- Applying agent switches and updating context
"""
from __future__ import annotations

from typing import Dict, List, Optional

from caribou.agents.AgentSystem import Agent
from caribou.execution.ActionSpace import AgentActionSpace
from caribou.execution.MemoryManager import MemoryManager


def _extract_possible_actions(agent: Agent) -> List[Dict[str, str]]:
    """Extract the list of possible actions an agent can perform."""
    actions: List[Dict[str, str]] = [
        {"name": "continue", "detail": "Continue reasoning and generate next step."}
    ]
    if getattr(agent, "is_rag_enabled", False):
        actions.append({
            "name": "query_rag_<topic>",
            "detail": "Retrieve context from knowledge base for a specific topic or function (replace <topic> accordingly).",
        })
    for cmd_name, cmd in getattr(agent, "commands", {}).items():
        detail = f"Delegate via command '{cmd_name}'"
        target = getattr(cmd, "target_agent", None)
        if target:
            detail += f" to agent '{target}'"
        actions.append({"name": cmd_name, "detail": detail})
    return actions


def _apply_agent_switch(
    *,
    new_agent_prompt: str,
    analysis_context: str,
    history: List[Dict[str, str]],
    memory_manager: Optional[MemoryManager],
    action_space: Optional[AgentActionSpace],
    new_agent: Agent,
) -> None:
    """
    Ensure both the raw history and the memory manager reflect the current agent prompt.
    Replace the second system message and append a short reminder so identity survives summarization.
    """
    updated_prompt = {"role": "system", "content": new_agent_prompt + "\n\n" + analysis_context}

    if len(history) >= 2 and history[1].get("role") == "system":
        history[1] = updated_prompt
    else:
        history.insert(1, updated_prompt)

    if memory_manager:
        memory_manager.update_system_prompt(updated_prompt["content"])
        reminder = {
            "role": "system",
            "content": "REMINDER: You are now following the above agent system prompt; stay in that role.",
        }
        history.append(reminder)
        memory_manager.add_message(reminder["role"], reminder["content"])

    if action_space:
        action_space.agent_name = new_agent.name
        action_space.set_possible_actions(_extract_possible_actions(new_agent))
        action_space.add_action(
            "agent_switch",
            f"Switched to agent '{new_agent.name}' via delegation.",
            status="ok",
            meta={"prompt_refreshed": True},
        )
        summary_msg = action_space.to_message()
        history.append({"role": "system", "content": summary_msg})
        if memory_manager:
            memory_manager.add_message("system", summary_msg)
