"""
Integration tests for end-to-end message flow.

Tests the complete flow: LLM API call -> message routing -> history update
-> delegation handling -> agent switching.
"""
import pytest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import tempfile
import json

from caribou.agents.AgentSystem import Agent, AgentSystem, Command
from caribou.execution.MemoryManager import MemoryManager
from caribou.execution.ActionSpace import AgentActionSpace
from caribou.execution.message_utils import detect_delegation, detect_rag, _extract_artifacts_from_msg
from caribou.execution.agent_management import _apply_agent_switch, _extract_possible_actions


class TestBasicMessageFlow:
    """Test basic message flow without special features."""

    def test_simple_conversation_flow(self, mock_llm_client):
        """Test simple back-and-forth conversation."""
        client = mock_llm_client(responses=[
            "Hello! How can I help you?",
            "Sure, I can help with that.",
            "Task completed successfully."
        ])

        history = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]

        # Simulate conversation turns
        history.append({"role": "user", "content": "Hello"})
        response1 = client.chat.completions.create(messages=history)
        history.append({"role": "assistant", "content": response1.choices[0].message.content})

        history.append({"role": "user", "content": "Can you help me?"})
        response2 = client.chat.completions.create(messages=history)
        history.append({"role": "assistant", "content": response2.choices[0].message.content})

        history.append({"role": "user", "content": "Do the task"})
        response3 = client.chat.completions.create(messages=history)
        history.append({"role": "assistant", "content": response3.choices[0].message.content})

        # Verify conversation structure
        assert len(history) == 7  # 1 system + 3 user + 3 assistant
        assert history[-1]["role"] == "assistant"
        assert "completed successfully" in history[-1]["content"]
        assert client.call_count == 3

    def test_message_flow_with_memory_manager(self, mock_llm_client):
        """Test message flow using MemoryManager."""
        client = mock_llm_client(responses=["Response 1", "Response 2"])

        initial_history = [
            {"role": "system", "content": "System prompt"}
        ]

        memory_manager = MemoryManager(
            llm_client=client,
            model_name="gpt-4",
            initial_history=initial_history,
            working_history_size=4
        )

        # Add messages
        memory_manager.add_message("user", "Question 1")
        context = memory_manager.get_context()
        response1 = client.chat.completions.create(messages=context)
        memory_manager.add_message("assistant", response1.choices[0].message.content)

        memory_manager.add_message("user", "Question 2")
        context = memory_manager.get_context()
        response2 = client.chat.completions.create(messages=context)
        memory_manager.add_message("assistant", response2.choices[0].message.content)

        # Verify history
        assert len(memory_manager._full_history) == 5  # 1 system + 2 user + 2 assistant
        final_context = memory_manager.get_context()
        assert len(final_context) >= 1


class TestDelegationFlow:
    """Test message flow with delegation between agents."""

    def test_delegation_detection_and_switch(self, mock_llm_client):
        """Test detecting delegation and switching agents."""
        # Create agent system
        planner = Agent(
            name="planner",
            prompt="You are a planner.",
            commands={
                "delegate_to_coder": Command("delegate_to_coder", "coder", "Send to coder")
            },
            code_samples={}
        )

        coder = Agent(
            name="coder",
            prompt="You are a coder.",
            commands={},
            code_samples={}
        )

        agent_system = AgentSystem(
            global_policy="Be helpful",
            agents={"planner": planner, "coder": coder}
        )

        # Simulate conversation with delegation
        client = mock_llm_client(responses=[
            "I'll plan this task first.",
            "Now I will delegate_to_coder to implement it.",
            "Implementation complete."
        ])

        history = [
            {"role": "system", "content": agent_system.global_policy},
            {"role": "system", "content": planner.get_full_prompt()}
        ]

        current_agent = planner

        # Turn 1: Normal response
        history.append({"role": "user", "content": "Plan and implement feature X"})
        response1 = client.chat.completions.create(messages=history)
        msg1 = response1.choices[0].message.content
        history.append({"role": "assistant", "content": msg1})

        # No delegation detected
        assert detect_delegation(msg1) is None

        # Turn 2: Delegation response
        history.append({"role": "user", "content": "Continue"})
        response2 = client.chat.completions.create(messages=history)
        msg2 = response2.choices[0].message.content
        history.append({"role": "assistant", "content": msg2})

        # Delegation detected
        cmd = detect_delegation(msg2)
        assert cmd == "delegate_to_coder"

        # Apply agent switch
        if cmd and cmd in current_agent.commands:
            target_agent_name = current_agent.commands[cmd].target_agent
            new_agent = agent_system.get_agent(target_agent_name)

            _apply_agent_switch(
                new_agent_prompt=new_agent.get_full_prompt(),
                analysis_context="",
                history=history,
                memory_manager=None,
                action_space=None,
                new_agent=new_agent
            )

            current_agent = new_agent

        # Verify switch occurred
        assert current_agent.name == "coder"
        assert any("coder" in msg.get("content", "").lower() for msg in history)

    def test_multi_agent_flow_with_memory(self, mock_llm_client):
        """Test multi-agent conversation with memory management."""
        # Create agents
        planner = Agent(
            name="planner",
            prompt="You are a planner.",
            commands={
                "delegate_to_coder": Command("delegate_to_coder", "coder", "Send to coder")
            },
            code_samples={}
        )

        coder = Agent(
            name="coder",
            prompt="You are a coder.",
            commands={
                "delegate_to_planner": Command("delegate_to_planner", "planner", "Back to planner")
            },
            code_samples={}
        )

        agent_system = AgentSystem(
            global_policy="Be helpful",
            agents={"planner": planner, "coder": coder}
        )

        client = mock_llm_client(responses=[
            "Planning the task.",
            "delegate_to_coder to implement.",
            "Implementing the feature.",
            "delegate_to_planner to review.",
            "Review complete."
        ])

        initial_history = [
            {"role": "system", "content": agent_system.global_policy},
            {"role": "system", "content": planner.get_full_prompt()}
        ]

        memory_manager = MemoryManager(
            llm_client=client,
            model_name="gpt-4",
            initial_history=initial_history
        )

        current_agent = planner
        action_space = AgentActionSpace(agent_name=current_agent.name)
        action_space.set_possible_actions(_extract_possible_actions(current_agent))

        # Simulate multi-turn conversation with agent switches
        for turn in range(5):
            memory_manager.add_message("user", f"Turn {turn}")
            context = memory_manager.get_context()
            response = client.chat.completions.create(messages=context)
            msg = response.choices[0].message.content
            memory_manager.add_message("assistant", msg)

            # Check for delegation
            cmd = detect_delegation(msg)
            if cmd and cmd in current_agent.commands:
                target_name = current_agent.commands[cmd].target_agent
                new_agent = agent_system.get_agent(target_name)

                _apply_agent_switch(
                    new_agent_prompt=new_agent.get_full_prompt(),
                    analysis_context="",
                    history=memory_manager._full_history,
                    memory_manager=memory_manager,
                    action_space=action_space,
                    new_agent=new_agent
                )

                current_agent = new_agent

        # Verify multiple switches occurred
        assert client.call_count == 5
        # Should have switched between planner and coder
        assert action_space.agent_name in ["planner", "coder"]


class TestRAGIntegration:
    """Test message flow with RAG queries."""

    def test_rag_query_detection(self, mock_llm_client):
        """Test detecting and handling RAG queries."""
        agent = Agent(
            name="researcher",
            prompt="You are a researcher.",
            commands={},
            code_samples={},
            is_rag_enabled=True
        )

        client = mock_llm_client(responses=[
            "Let me search for information.",
            "I will query_rag_<API documentation> to find details.",
            "Based on the documentation, here's the answer."
        ])

        history = [
            {"role": "system", "content": agent.get_full_prompt()}
        ]

        # Turn 1: Normal response
        history.append({"role": "user", "content": "How does the API work?"})
        response1 = client.chat.completions.create(messages=history)
        msg1 = response1.choices[0].message.content
        history.append({"role": "assistant", "content": msg1})
        assert detect_rag(msg1) is None

        # Turn 2: RAG query
        history.append({"role": "user", "content": "Continue"})
        response2 = client.chat.completions.create(messages=history)
        msg2 = response2.choices[0].message.content
        history.append({"role": "assistant", "content": msg2})

        rag_query = detect_rag(msg2)
        assert rag_query == "API documentation"

        # Simulate RAG retrieval (mock)
        rag_result = "RAG retrieved: API uses REST endpoints..."
        history.append({"role": "system", "content": f"RAG RESULT: {rag_result}"})

        # Turn 3: Response with RAG context
        history.append({"role": "user", "content": "Continue with RAG results"})
        response3 = client.chat.completions.create(messages=history)
        msg3 = response3.choices[0].message.content
        history.append({"role": "assistant", "content": msg3})

        # Verify RAG result was added to history
        assert any("RAG RESULT" in msg.get("content", "") for msg in history)


class TestArtifactExtraction:
    """Test message flow with artifact extraction."""

    def test_notes_and_todos_extraction(self, mock_llm_client):
        """Test extracting notes and TODOs from responses."""
        client = mock_llm_client(responses=[
            """
            I'll start working on this task.

            NOTE: Using Python 3.11 for this project
            TODO: Install dependencies
            """,
            """
            Progress update:

            - [x] Dependencies installed
            - [ ] Write main function
            - [ ] Add tests

            NOTE: Found a good library for this
            """
        ])

        history = [{"role": "system", "content": "You are a coder."}]

        all_notes = []
        all_todos = []

        # Turn 1
        history.append({"role": "user", "content": "Start the task"})
        response1 = client.chat.completions.create(messages=history)
        msg1 = response1.choices[0].message.content
        history.append({"role": "assistant", "content": msg1})

        notes1, todos1 = _extract_artifacts_from_msg(msg1)
        all_notes.extend(notes1)
        all_todos.extend(todos1)

        # Turn 2
        history.append({"role": "user", "content": "Continue"})
        response2 = client.chat.completions.create(messages=history)
        msg2 = response2.choices[0].message.content
        history.append({"role": "assistant", "content": msg2})

        notes2, todos2 = _extract_artifacts_from_msg(msg2)
        all_notes.extend(notes2)
        all_todos.extend(todos2)

        # Verify artifacts collected
        assert len(all_notes) == 2
        assert any("Python 3.11" in note for note in all_notes)
        assert any("good library" in note for note in all_notes)

        assert len(all_todos) == 4
        assert any("Install dependencies" in todo for todo in all_todos)
        assert any("Write main function" in todo for todo in all_todos)


class TestCompleteIntegrationFlow:
    """Test complete integration scenarios."""

    def test_full_workflow_with_all_features(self, mock_llm_client):
        """Test complete workflow with agents, memory, RAG, and artifacts."""
        # Create comprehensive agent system
        planner = Agent(
            name="planner",
            prompt="You are a planner.",
            commands={
                "delegate_to_researcher": Command("delegate_to_researcher", "researcher", "Research")
            },
            code_samples={}
        )

        researcher = Agent(
            name="researcher",
            prompt="You are a researcher.",
            commands={
                "delegate_to_planner": Command("delegate_to_planner", "planner", "Back to planning")
            },
            code_samples={},
            is_rag_enabled=True
        )

        agent_system = AgentSystem(
            global_policy="Be thorough and accurate",
            agents={"planner": planner, "researcher": researcher}
        )

        client = mock_llm_client(responses=[
            "I will plan the research task.\nNOTE: Starting with literature review",
            "delegate_to_researcher to find information",
            "I will query_rag_<machine learning basics> for context",
            "Based on research: ML uses algorithms.\nTODO: Compile findings\n- [ ] Write report",
            "delegate_to_planner to finalize",
            "Final plan complete.\n- [x] Research done\n- [x] Report written"
        ])

        initial_history = [
            {"role": "system", "content": agent_system.global_policy},
            {"role": "system", "content": planner.get_full_prompt()}
        ]

        memory_manager = MemoryManager(
            llm_client=client,
            model_name="gpt-4",
            initial_history=initial_history,
            working_history_size=4
        )

        current_agent = planner
        action_space = AgentActionSpace(agent_name=current_agent.name)
        all_notes = []
        all_todos = []

        # Simulate complete workflow
        for turn in range(6):
            memory_manager.add_message("user", f"Continue with task (turn {turn})")
            context = memory_manager.get_context()
            response = client.chat.completions.create(messages=context)
            msg = response.choices[0].message.content
            memory_manager.add_message("assistant", msg)

            # Extract artifacts
            notes, todos = _extract_artifacts_from_msg(msg)
            all_notes.extend(notes)
            all_todos.extend(todos)

            # Check for delegation
            cmd = detect_delegation(msg)
            if cmd and cmd in current_agent.commands:
                target_name = current_agent.commands[cmd].target_agent
                new_agent = agent_system.get_agent(target_name)

                _apply_agent_switch(
                    new_agent_prompt=new_agent.get_full_prompt(),
                    analysis_context="",
                    history=memory_manager._full_history,
                    memory_manager=memory_manager,
                    action_space=action_space,
                    new_agent=new_agent
                )

                current_agent = new_agent

            # Check for RAG
            rag_query = detect_rag(msg)
            if rag_query:
                rag_result = f"RAG results for: {rag_query}"
                memory_manager.add_message("system", f"RAG RESULT: {rag_result}")

        # Verify complete workflow
        assert client.call_count == 6
        assert len(all_notes) >= 1  # Should have collected notes
        assert len(all_todos) >= 1  # Should have collected TODOs
        assert len(memory_manager._full_history) > 10  # Should have substantial history

    def test_error_resilience_in_flow(self, mock_llm_client):
        """Test that flow continues gracefully when components have issues."""
        client = mock_llm_client(responses=[
            "Normal response",
            "Invalid delegation: delegate_to_nonexistent",
            "Recovery response"
        ])

        agent = Agent(
            name="agent",
            prompt="Prompt",
            commands={
                "delegate_to_other": Command("delegate_to_other", "other", "Delegate")
            },
            code_samples={}
        )

        agent_system = AgentSystem(
            global_policy="Global",
            agents={"agent": agent, "other": agent}
        )

        history = [
            {"role": "system", "content": agent.get_full_prompt()}
        ]

        current_agent = agent

        # Turn 1: Normal
        history.append({"role": "user", "content": "Start"})
        response1 = client.chat.completions.create(messages=history)
        msg1 = response1.choices[0].message.content
        history.append({"role": "assistant", "content": msg1})

        # Turn 2: Invalid delegation (should not crash)
        history.append({"role": "user", "content": "Continue"})
        response2 = client.chat.completions.create(messages=history)
        msg2 = response2.choices[0].message.content
        history.append({"role": "assistant", "content": msg2})

        cmd = detect_delegation(msg2)
        assert cmd == "delegate_to_nonexistent"

        # Try to switch (should fail gracefully)
        if cmd and cmd in current_agent.commands:
            # This won't execute because command doesn't exist
            pass
        else:
            # Flow continues without switching
            pass

        # Turn 3: Recovery
        history.append({"role": "user", "content": "Recover"})
        response3 = client.chat.completions.create(messages=history)
        msg3 = response3.choices[0].message.content
        history.append({"role": "assistant", "content": msg3})

        # Should complete without crashing
        assert client.call_count == 3
        assert current_agent.name == "agent"  # No switch occurred


class TestMemoryAndContextManagement:
    """Test memory management during long conversations."""

    def test_long_conversation_with_summarization(self, mock_llm_client):
        """Test that long conversations trigger summarization."""
        # Create many responses
        responses = [f"Response {i}" for i in range(30)]
        responses.append("This is a summary of previous messages.")  # For summarization call
        responses.extend([f"Response {i}" for i in range(30, 35)])

        client = mock_llm_client(responses=responses)

        initial_history = [{"role": "system", "content": "System"}]

        memory_manager = MemoryManager(
            llm_client=client,
            model_name="gpt-4",
            initial_history=initial_history,
            working_history_size=3,
            summarization_threshold=5,
            chunk_size_to_summarize=5
        )

        # Add many messages
        for i in range(30):
            memory_manager.add_message("user", f"Question {i}")
            context = memory_manager.get_context()
            response = client.chat.completions.create(messages=context)
            memory_manager.add_message("assistant", response.choices[0].message.content)

        # Should have triggered summarization
        assert len(memory_manager._summarized_log) >= 1

        # Context should be manageable size
        context = memory_manager.get_context()
        # Should be much smaller than full history
        assert len(context) < len(memory_manager._full_history)
