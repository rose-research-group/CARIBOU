import json
from typing import Dict, Optional
from pathlib import Path

from caribou.execution.session_brief import BriefPolicy
from caribou.execution.work_items import WorkItemPolicy, render_work_item_prompt

# Import the central CARIBOU_HOME path from our config module
from caribou.config import CARIBOU_HOME

# 1. The user-specific directory (for custom samples)
USER_CODE_SAMPLES_DIR = CARIBOU_HOME / "code_samples"

# 2. The package-internal directory (for default samples), found relative to this file
PACKAGE_CODE_SAMPLES_DIR = Path(__file__).resolve().parent.parent / "code_samples"


class Command:
    """Represents a command an agent can issue to a neighboring agent."""

    def __init__(self, name: str, target_agent: str, description: str):
        self.name = name
        self.target_agent = target_agent
        self.description = description

    def __repr__(self) -> str:
        return (
            f"Command(name='{self.name}', target='{self.target_agent}', "
            f"desc='{self.description[:30]}...')"
        )


class Agent:
    """Represents a single agent in the system."""

    def __init__(
        self,
        name: str,
        prompt: str,
        commands: Dict[str, Command],
        code_samples: Dict[str, str],
        is_rag_enabled: bool = False,
    ):
        self.name = name
        self.prompt = prompt
        self.commands = commands
        self.code_samples = code_samples
        self.is_rag_enabled = is_rag_enabled

    def __repr__(self) -> str:
        sample_keys = list(self.code_samples.keys())
        return f"Agent(name='{self.name}', commands={list(self.commands.keys())}, samples={sample_keys}, rag_enabled={self.is_rag_enabled})"

    def get_full_prompt(self, global_policy=None, work_item_policy=None) -> str:
        """Constructs the full prompt including the global policy and command descriptions."""
        full_prompt = ""
        if global_policy:
            full_prompt += f"**GLOBAL POLICY**: {global_policy}\n\n---\n\n"

        full_prompt += self.prompt

        if self.commands:
            full_prompt += "\n\nYou can use the following commands to delegate tasks:"
            for name, command in self.commands.items():
                full_prompt += f"\n- Command: `{name}`"
                full_prompt += f"\n  - Description: {command.description}"
                full_prompt += f"\n  - Target Agent: {command.target_agent}"
            full_prompt += "\n\n**YOU MUST USE THESE EXACT COMMANDS TO DELEGATE TASKS. NO OTHER FORMATTING OR COMMANDS ARE ALLOWED.**"

        full_prompt += (
            "\n\nTo end the session early when you believe the task is complete, "
            "emit the command on a standalone line: `end_session`. "
            "The end_session message must not include code blocks or other content."
        )

        if self.is_rag_enabled:
            full_prompt += "\n\nIf an error occurs, admit the error and query your specialized knowledge base for more context with the following command:"
            full_prompt += "\n- Command: `query_rag_<function>`"
            full_prompt += "\n  - Description: Retrieves relevant information about a specific <function> from your knowledge base. Replace <function> with a concise, descriptive search query (e.g., function names, task you are trying to complete)."
            full_prompt += "\n  - Example: `query_rag_<scvi model setup>`"
            full_prompt += (
                "\n\n**You can only delegate to one agent at a time. "
                "Never wrap delegation or RAG calls in backticks as if they are code.**"
            )

        if work_item_policy is not None:
            full_prompt += render_work_item_prompt(work_item_policy)

        if self.code_samples:
            full_prompt += (
                "\n\n## Code Samples (Reference ONLY — DO NOT IMPORT)\n"
                "**CRITICAL RULES:**\n"
                "1. These files DO NOT EXIST on disk inside the sandbox. You CANNOT `import` them or run them directly.\n"
                "2. They exist solely to show you the correct API, function signatures, and patterns.\n"
                "3. You MUST rewrite any logic you need from scratch inside your own ```python``` code block.\n"
                "4. Treat them like documentation — read, understand, then write your own implementation.\n"
            )
            for sample_name, sample_content in self.code_samples.items():
                full_prompt += f"\n### `{sample_name}` (reference only — rewrite, do not import)\n```python\n{sample_content}\n```\n"
        return full_prompt


class AgentSystem:
    """
    Loads and holds the entire agent system configuration from a JSON file,
    including the global policy and the network of agents.
    """

    def __init__(
        self,
        global_policy: str,
        agents: Dict[str, Agent],
        evaluator_agent_name: Optional[str] = None,
        work_item_policy: Optional[WorkItemPolicy] = None,
        brief_policy: Optional[BriefPolicy] = None,
    ):
        self.global_policy = global_policy
        self.agents = agents
        self.evaluator_agent_name = evaluator_agent_name
        self.work_item_policy = work_item_policy or WorkItemPolicy()
        self.brief_policy = brief_policy or BriefPolicy()

    @classmethod
    def load_from_json(cls, file_path: str) -> "AgentSystem":
        """
        Parses the JSON blueprint, reads code sample files from disk from both user
        and package locations, and builds the AgentSystem data structure.
        """
        print(f"Loading agent system from: {file_path}")
        with open(file_path, "r") as f:
            config = json.load(f)

        global_policy = config.get("global_policy", "")
        evaluator_agent_name = config.get("evaluator_agent") or None
        work_item_policy = WorkItemPolicy.from_dict(config.get("work_item_policy"))
        brief_policy = BriefPolicy.from_dict(config.get("brief_policy"))
        agents: Dict[str, Agent] = {}

        for agent_name, agent_data in config.get("agents", {}).items():
            commands: Dict[str, Command] = {}
            for cmd_name, cmd_data in agent_data.get("neighbors", {}).items():
                commands[cmd_name] = Command(
                    name=cmd_name,
                    target_agent=cmd_data["target_agent"],
                    description=cmd_data["description"],
                )

            loaded_samples: Dict[str, str] = {}
            sample_filenames = agent_data.get("code_samples", [])

            if sample_filenames:
                print(f"  Loading code samples for '{agent_name}'...")
                for filename in sample_filenames:
                    if (
                        not isinstance(filename, str)
                        or filename in {"", ".", ".."}
                        or "/" in filename
                        or "\\" in filename
                    ):
                        raise ValueError(
                            "code sample names must be path-safe filenames"
                        )
                    user_path = USER_CODE_SAMPLES_DIR / filename
                    package_path = PACKAGE_CODE_SAMPLES_DIR / filename

                    path_to_load = None
                    source_label = ""
                    if user_path.exists():
                        path_to_load = user_path
                        source_label = "User"
                    elif package_path.exists():
                        path_to_load = package_path
                        source_label = "Package"

                    if path_to_load:
                        try:
                            loaded_samples[filename] = path_to_load.read_text(
                                encoding="utf-8"
                            )
                            print(f"    ✅ Loaded {filename} (from {source_label})")
                        except Exception as e:
                            print(
                                f"    ❌ ERROR: Could not read code sample file {path_to_load}: {e}"
                            )
                    else:
                        print(
                            f"    ❌ WARNING: Code sample file '{filename}' not found in any location."
                        )

            rag_config = agent_data.get("rag", {})
            is_rag_enabled = rag_config.get("enabled", False)

            agent = Agent(
                name=agent_name,
                prompt=agent_data["prompt"],
                commands=commands,
                code_samples=loaded_samples,
                is_rag_enabled=is_rag_enabled,
            )
            agents[agent_name] = agent

        if evaluator_agent_name is not None and evaluator_agent_name not in agents:
            raise ValueError(
                f"evaluator_agent '{evaluator_agent_name}' does not match any agent "
                f"defined in {file_path}"
            )
        if work_item_policy.qc_mode == "required" and evaluator_agent_name is None:
            raise ValueError(
                "work_item_policy.qc_mode 'required' needs an evaluator_agent "
                "declared in the running blueprint"
            )

        print("Agent system loaded successfully.")
        return cls(
            global_policy,
            agents,
            evaluator_agent_name=evaluator_agent_name,
            work_item_policy=work_item_policy,
            brief_policy=brief_policy,
        )

    def get_agent(self, name: str) -> Optional[Agent]:
        """Retrieves an agent by its unique name."""
        return self.agents.get(name)

    def get_evaluator_agent(self) -> Optional[Agent]:
        """Returns the agent designated as this system's evaluator via the
        top-level 'evaluator_agent' blueprint field, if any. Used by the
        /evaluate command and its web-UI equivalent to review a run without
        requiring a separate evaluator blueprint."""
        if self.evaluator_agent_name is None:
            return None
        return self.agents.get(self.evaluator_agent_name)

    def get_all_agents(self) -> Dict[str, Agent]:
        """Returns a dictionary of all agents in the system."""
        return self.agents

    def get_instructions(self) -> str:
        """
        Generates a summary of the system's instructions, including the global policy.
        Not typically used in prompts anymore, but useful for debugging and logging.
        """
        instructions = (
            f"**GLOBAL POLICY FOR ALL AGENTS**: {self.global_policy}\n\n---\n\n"
        )
        instructions += "**SYSTEM AGENTS**:\n"
        for agent in self.agents.values():
            instructions += (
                f"\n- **Agent**: {agent.name}\n  - **Prompt**: {agent.prompt}\n"
            )
            if agent.commands:
                instructions += "  - **Commands**:\n"
                for cmd in agent.commands.values():
                    instructions += f"    - `{cmd.name}`: {cmd.description} (delegates to: {cmd.target_agent})\n"
        return instructions

    def __repr__(self) -> str:
        return f"AgentSystem(global_policy='{self.global_policy[:40]}...', agents={list(self.agents.keys())})"
