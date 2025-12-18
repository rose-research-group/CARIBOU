"""
Unit tests for Agent system and agent management.

Tests agent configuration, command handling, prompt generation,
and agent switching logic.
"""
import pytest
import json
import tempfile
from pathlib import Path
from typing import Dict
from unittest.mock import Mock, patch

from caribou.agents.AgentSystem import Agent, AgentSystem, Command
from caribou.execution.agent_management import _extract_possible_actions, _apply_agent_switch
from caribou.execution.ActionSpace import AgentActionSpace


class TestCommand:
    """Test Command class."""

    def test_command_creation(self):
        """Test creating a command."""
        cmd = Command(
            name="delegate_to_coder",
            target_agent="coder",
            description="Send task to coder"
        )

        assert cmd.name == "delegate_to_coder"
        assert cmd.target_agent == "coder"
        assert cmd.description == "Send task to coder"

    def test_command_repr(self):
        """Test command string representation."""
        cmd = Command(
            name="test_cmd",
            target_agent="test_agent",
            description="A" * 50  # Long description
        )

        repr_str = repr(cmd)
        assert "test_cmd" in repr_str
        assert "test_agent" in repr_str
        assert "..." in repr_str  # Truncated description


class TestAgent:
    """Test Agent class."""

    def test_agent_creation(self):
        """Test creating an agent."""
        commands = {
            "delegate_to_other": Command("delegate_to_other", "other", "Delegate to other")
        }
        samples = {"sample1.py": "print('hello')"}

        agent = Agent(
            name="test_agent",
            prompt="You are a test agent.",
            commands=commands,
            code_samples=samples,
            is_rag_enabled=True
        )

        assert agent.name == "test_agent"
        assert agent.prompt == "You are a test agent."
        assert len(agent.commands) == 1
        assert len(agent.code_samples) == 1
        assert agent.is_rag_enabled is True

    def test_agent_repr(self):
        """Test agent string representation."""
        commands = {"cmd1": Command("cmd1", "target", "desc")}
        samples = {"sample.py": "code"}

        agent = Agent(
            name="agent1",
            prompt="Prompt",
            commands=commands,
            code_samples=samples
        )

        repr_str = repr(agent)
        assert "agent1" in repr_str
        assert "cmd1" in repr_str
        assert "sample.py" in repr_str

    def test_get_full_prompt_basic(self):
        """Test basic prompt generation."""
        agent = Agent(
            name="basic",
            prompt="Basic prompt",
            commands={},
            code_samples={}
        )

        prompt = agent.get_full_prompt()
        assert "Basic prompt" in prompt
        assert "NOTE:" in prompt  # Notes instruction
        assert "TODO:" in prompt  # TODO instruction

    def test_get_full_prompt_with_global_policy(self):
        """Test prompt with global policy."""
        agent = Agent(
            name="agent",
            prompt="Agent prompt",
            commands={},
            code_samples={}
        )

        prompt = agent.get_full_prompt(global_policy="Be helpful")
        assert "**GLOBAL POLICY**: Be helpful" in prompt
        assert "Agent prompt" in prompt

    def test_get_full_prompt_with_commands(self):
        """Test prompt with commands."""
        commands = {
            "delegate_to_coder": Command(
                "delegate_to_coder",
                "coder",
                "Send coding tasks"
            ),
            "delegate_to_planner": Command(
                "delegate_to_planner",
                "planner",
                "Send planning tasks"
            )
        }

        agent = Agent(
            name="agent",
            prompt="Agent prompt",
            commands=commands,
            code_samples={}
        )

        prompt = agent.get_full_prompt()
        assert "delegate_to_coder" in prompt
        assert "delegate_to_planner" in prompt
        assert "Send coding tasks" in prompt
        assert "coder" in prompt
        assert "YOU MUST USE THESE EXACT COMMANDS" in prompt

    def test_get_full_prompt_with_rag(self):
        """Test prompt with RAG enabled."""
        agent = Agent(
            name="agent",
            prompt="Agent prompt",
            commands={},
            code_samples={},
            is_rag_enabled=True
        )

        prompt = agent.get_full_prompt()
        assert "query_rag_<function>" in prompt
        assert "knowledge base" in prompt
        assert "query_rag_<scvi model setup>" in prompt  # Example

    def test_get_full_prompt_with_code_samples(self):
        """Test prompt with code samples."""
        samples = {
            "example1.py": "code1",
            "example2.py": "code2"
        }

        agent = Agent(
            name="agent",
            prompt="Agent prompt",
            commands={},
            code_samples=samples
        )

        prompt = agent.get_full_prompt()
        assert "Code Samples Available" in prompt
        assert "example1.py" in prompt
        assert "example2.py" in prompt
        assert "MUST BE REWRITTEN TO BE USED" in prompt

    def test_get_full_prompt_complete(self):
        """Test prompt with all features."""
        commands = {
            "delegate_to_other": Command("delegate_to_other", "other", "Delegate")
        }
        samples = {"sample.py": "code"}

        agent = Agent(
            name="complete",
            prompt="Complete agent",
            commands=commands,
            code_samples=samples,
            is_rag_enabled=True
        )

        prompt = agent.get_full_prompt(global_policy="Global")

        # Check all sections present
        assert "**GLOBAL POLICY**: Global" in prompt
        assert "Complete agent" in prompt
        assert "delegate_to_other" in prompt
        assert "query_rag_<function>" in prompt
        assert "sample.py" in prompt
        assert "NOTE:" in prompt
        assert "TODO:" in prompt


class TestAgentSystem:
    """Test AgentSystem class."""

    def test_agent_system_creation(self):
        """Test creating an agent system."""
        agents = {
            "agent1": Agent("agent1", "Prompt 1", {}, {}),
            "agent2": Agent("agent2", "Prompt 2", {}, {})
        }

        system = AgentSystem(global_policy="Be helpful", agents=agents)

        assert system.global_policy == "Be helpful"
        assert len(system.agents) == 2
        assert "agent1" in system.agents
        assert "agent2" in system.agents

    def test_get_agent(self):
        """Test retrieving agent by name."""
        agent1 = Agent("agent1", "Prompt", {}, {})
        agents = {"agent1": agent1}

        system = AgentSystem(global_policy="Policy", agents=agents)

        retrieved = system.get_agent("agent1")
        assert retrieved is agent1

        not_found = system.get_agent("nonexistent")
        assert not_found is None

    def test_get_all_agents(self):
        """Test getting all agents."""
        agents = {
            "agent1": Agent("agent1", "P1", {}, {}),
            "agent2": Agent("agent2", "P2", {}, {})
        }

        system = AgentSystem(global_policy="Policy", agents=agents)

        all_agents = system.get_all_agents()
        assert len(all_agents) == 2
        assert "agent1" in all_agents
        assert "agent2" in all_agents

    def test_get_instructions(self):
        """Test generating system instructions."""
        commands = {
            "delegate_to_agent2": Command("delegate_to_agent2", "agent2", "Delegate")
        }
        agents = {
            "agent1": Agent("agent1", "Agent 1 prompt", commands, {})
        }

        system = AgentSystem(global_policy="Be accurate", agents=agents)

        instructions = system.get_instructions()
        assert "Be accurate" in instructions
        assert "agent1" in instructions
        assert "Agent 1 prompt" in instructions
        assert "delegate_to_agent2" in instructions

    def test_agent_system_repr(self):
        """Test agent system string representation."""
        agents = {
            "agent1": Agent("agent1", "P1", {}, {}),
            "agent2": Agent("agent2", "P2", {}, {})
        }

        system = AgentSystem(global_policy="A" * 50, agents=agents)

        repr_str = repr(system)
        assert "AgentSystem" in repr_str
        assert "agent1" in repr_str
        assert "agent2" in repr_str
        assert "..." in repr_str  # Truncated policy


class TestAgentSystemLoadFromJSON:
    """Test loading agent system from JSON."""

    def test_load_basic_config(self):
        """Test loading basic configuration."""
        config = {
            "global_policy": "Be helpful and accurate.",
            "agents": {
                "planner": {
                    "prompt": "You are a planning agent.",
                    "neighbors": {},
                    "code_samples": [],
                    "rag": {"enabled": False}
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            temp_path = f.name

        try:
            system = AgentSystem.load_from_json(temp_path)

            assert system.global_policy == "Be helpful and accurate."
            assert len(system.agents) == 1
            assert "planner" in system.agents

            planner = system.get_agent("planner")
            assert planner.name == "planner"
            assert planner.prompt == "You are a planning agent."
            assert planner.is_rag_enabled is False
        finally:
            Path(temp_path).unlink()

    def test_load_with_commands(self):
        """Test loading configuration with commands."""
        config = {
            "global_policy": "Global",
            "agents": {
                "planner": {
                    "prompt": "Planner prompt",
                    "neighbors": {
                        "delegate_to_coder": {
                            "target_agent": "coder",
                            "description": "Send to coder"
                        }
                    },
                    "code_samples": [],
                    "rag": {"enabled": False}
                },
                "coder": {
                    "prompt": "Coder prompt",
                    "neighbors": {},
                    "code_samples": [],
                    "rag": {"enabled": False}
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            temp_path = f.name

        try:
            system = AgentSystem.load_from_json(temp_path)

            planner = system.get_agent("planner")
            assert len(planner.commands) == 1
            assert "delegate_to_coder" in planner.commands

            cmd = planner.commands["delegate_to_coder"]
            assert cmd.target_agent == "coder"
            assert cmd.description == "Send to coder"
        finally:
            Path(temp_path).unlink()

    def test_load_with_rag_enabled(self):
        """Test loading configuration with RAG enabled."""
        config = {
            "global_policy": "Global",
            "agents": {
                "researcher": {
                    "prompt": "Research prompt",
                    "neighbors": {},
                    "code_samples": [],
                    "rag": {"enabled": True}
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            temp_path = f.name

        try:
            system = AgentSystem.load_from_json(temp_path)

            researcher = system.get_agent("researcher")
            assert researcher.is_rag_enabled is True
        finally:
            Path(temp_path).unlink()

    @patch('caribou.agents.AgentSystem.USER_CODE_SAMPLES_DIR')
    @patch('caribou.agents.AgentSystem.PACKAGE_CODE_SAMPLES_DIR')
    def test_load_with_code_samples(self, mock_package_dir, mock_user_dir):
        """Test loading configuration with code samples."""
        # Create temporary directories for code samples
        with tempfile.TemporaryDirectory() as temp_user_dir, \
             tempfile.TemporaryDirectory() as temp_package_dir:

            mock_user_dir.__truediv__ = lambda self, x: Path(temp_user_dir) / x
            mock_package_dir.__truediv__ = lambda self, x: Path(temp_package_dir) / x

            # Create a sample file
            sample_path = Path(temp_package_dir) / "sample.py"
            sample_path.write_text("print('hello')")

            config = {
                "global_policy": "Global",
                "agents": {
                    "coder": {
                        "prompt": "Coder prompt",
                        "neighbors": {},
                        "code_samples": ["sample.py"],
                        "rag": {"enabled": False}
                    }
                }
            }

            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(config, f)
                temp_config_path = f.name

            try:
                # Mock the path resolution
                with patch.object(Path, 'exists', return_value=True), \
                     patch.object(Path, 'read_text', return_value="print('hello')"):
                    system = AgentSystem.load_from_json(temp_config_path)

                    coder = system.get_agent("coder")
                    assert len(coder.code_samples) == 1
                    assert "sample.py" in coder.code_samples
            finally:
                Path(temp_config_path).unlink()


class TestExtractPossibleActions:
    """Test extracting possible actions from agents."""

    def test_extract_from_basic_agent(self):
        """Test extracting actions from agent with no special features."""
        agent = Agent(name="basic", prompt="Prompt", commands={}, code_samples={})

        actions = _extract_possible_actions(agent)

        # Should always have "continue" action
        assert len(actions) >= 1
        assert any(action["name"] == "continue" for action in actions)

    def test_extract_with_commands(self):
        """Test extracting actions including commands."""
        commands = {
            "delegate_to_coder": Command("delegate_to_coder", "coder", "Send to coder"),
            "delegate_to_planner": Command("delegate_to_planner", "planner", "Send to planner")
        }
        agent = Agent(name="agent", prompt="Prompt", commands=commands, code_samples={})

        actions = _extract_possible_actions(agent)

        assert len(actions) >= 3  # continue + 2 commands
        assert any(action["name"] == "delegate_to_coder" for action in actions)
        assert any(action["name"] == "delegate_to_planner" for action in actions)

    def test_extract_with_rag(self):
        """Test extracting actions with RAG enabled."""
        agent = Agent(
            name="agent",
            prompt="Prompt",
            commands={},
            code_samples={},
            is_rag_enabled=True
        )

        actions = _extract_possible_actions(agent)

        # Should include RAG action
        rag_actions = [a for a in actions if "query_rag" in a["name"]]
        assert len(rag_actions) == 1
        assert "knowledge base" in rag_actions[0]["detail"]

    def test_extract_all_features(self):
        """Test extracting actions with all features."""
        commands = {
            "delegate_to_other": Command("delegate_to_other", "other", "Delegate")
        }
        agent = Agent(
            name="agent",
            prompt="Prompt",
            commands=commands,
            code_samples={},
            is_rag_enabled=True
        )

        actions = _extract_possible_actions(agent)

        # Should have: continue + rag + command
        assert len(actions) >= 3
        action_names = [a["name"] for a in actions]
        assert "continue" in action_names
        assert "delegate_to_other" in action_names
        assert any("query_rag" in name for name in action_names)


class TestApplyAgentSwitch:
    """Test applying agent switches."""

    def test_apply_agent_switch_updates_history(self, mock_llm_client):
        """Test that agent switch updates history."""
        new_agent = Agent(
            name="new_agent",
            prompt="New agent prompt",
            commands={},
            code_samples={}
        )

        history = [
            {"role": "system", "content": "Global policy"},
            {"role": "system", "content": "Old agent prompt"},
            {"role": "user", "content": "Hello"}
        ]

        _apply_agent_switch(
            new_agent_prompt="New agent prompt",
            analysis_context="",
            history=history,
            memory_manager=None,
            action_space=None,
            new_agent=new_agent
        )

        # Second message should be updated
        assert history[1]["content"] == "New agent prompt\n\n"

    def test_apply_agent_switch_with_memory_manager(self, mock_llm_client):
        """Test agent switch with memory manager."""
        client = mock_llm_client()
        memory_manager = Mock()

        new_agent = Agent(
            name="new_agent",
            prompt="New prompt",
            commands={},
            code_samples={}
        )

        history = [
            {"role": "system", "content": "Global"},
            {"role": "system", "content": "Old prompt"}
        ]

        _apply_agent_switch(
            new_agent_prompt="New prompt",
            analysis_context="",
            history=history,
            memory_manager=memory_manager,
            action_space=None,
            new_agent=new_agent
        )

        # Should update memory manager
        memory_manager.update_system_prompt.assert_called_once()
        memory_manager.add_message.assert_called()

        # Should add reminder to history
        assert any("REMINDER" in msg.get("content", "") for msg in history)

    def test_apply_agent_switch_with_action_space(self):
        """Test agent switch with action space."""
        action_space = AgentActionSpace(agent_name="old_agent")

        new_agent = Agent(
            name="new_agent",
            prompt="New prompt",
            commands={
                "delegate_to_other": Command("delegate_to_other", "other", "Delegate")
            },
            code_samples={}
        )

        history = [
            {"role": "system", "content": "Global"},
            {"role": "system", "content": "Old prompt"}
        ]

        _apply_agent_switch(
            new_agent_prompt="New prompt",
            analysis_context="",
            history=history,
            memory_manager=None,
            action_space=action_space,
            new_agent=new_agent
        )

        # Should update action space
        assert action_space.agent_name == "new_agent"

        # Should add action space message to history
        action_space_msgs = [msg for msg in history if "ACTION SPACE UPDATE" in msg.get("content", "")]
        assert len(action_space_msgs) >= 1

    def test_apply_agent_switch_with_analysis_context(self):
        """Test agent switch includes analysis context."""
        new_agent = Agent(name="agent", prompt="Prompt", commands={}, code_samples={})

        history = [
            {"role": "system", "content": "Global"},
            {"role": "system", "content": "Old"}
        ]

        analysis_context = "\nAdditional context about the task."

        _apply_agent_switch(
            new_agent_prompt="New prompt",
            analysis_context=analysis_context,
            history=history,
            memory_manager=None,
            action_space=None,
            new_agent=new_agent
        )

        # Should include analysis context in updated prompt
        assert "Additional context about the task." in history[1]["content"]

    def test_apply_agent_switch_inserts_if_needed(self):
        """Test agent switch inserts prompt if history is short."""
        new_agent = Agent(name="agent", prompt="Prompt", commands={}, code_samples={})

        history = [{"role": "system", "content": "Global"}]

        _apply_agent_switch(
            new_agent_prompt="New prompt",
            analysis_context="",
            history=history,
            memory_manager=None,
            action_space=None,
            new_agent=new_agent
        )

        # Should insert at index 1
        assert len(history) == 2
        assert history[1]["content"] == "New prompt\n\n"
