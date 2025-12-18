# CARIBOU Test Suite

Comprehensive unit and integration tests for CARIBOU's LLM API integration, message routing, history management, and system prompt handling.

## Overview

This test suite ensures the reliability and correctness of:

- **LLM API Wrappers**: AnthropicClient and OllamaClient for OpenAI-compatible interfaces
- **Message Routing**: Delegation detection, RAG query detection, and artifact extraction
- **History Management**: MemoryManager with episodic summarization
- **Agent System**: Multi-agent configuration, prompt generation, and agent switching
- **Integration Flows**: End-to-end message flows with all features combined

## Test Structure

```
tests/
├── unit/                           # Unit tests for individual components
│   ├── test_anthropic_wrapper.py  # AnthropicClient tests
│   ├── test_ollama_wrapper.py     # OllamaClient tests
│   ├── test_message_utils.py      # Message parsing and routing tests
│   ├── test_memory_manager.py     # MemoryManager tests
│   └── test_agent_system.py       # Agent system and management tests
├── integration/                    # Integration tests
│   └── test_message_flow.py       # End-to-end flow tests
├── fixtures/                       # Test fixtures and data
├── conftest.py                     # Shared pytest fixtures
├── run_tests.sh                    # Test runner script
└── README.md                       # This file
```

## Prerequisites

Install test dependencies:

```bash
pip install pytest pytest-cov
```

Or install all CARIBOU dependencies (includes test deps):

```bash
pip install -r requirements.txt
```

## Running Tests

### Quick Start

Run all tests:

```bash
cd caribou/tests
./run_tests.sh
```

Or using pytest directly from project root:

```bash
pytest caribou/tests/
```

### Specific Test Categories

**Unit tests only:**
```bash
./run_tests.sh --unit
# or
pytest caribou/tests/unit/
```

**Integration tests only:**
```bash
./run_tests.sh --integration
# or
pytest caribou/tests/integration/
```

**Specific test file:**
```bash
pytest caribou/tests/unit/test_anthropic_wrapper.py
```

**Specific test class or function:**
```bash
pytest caribou/tests/unit/test_message_utils.py::TestDelegationDetection
pytest caribou/tests/unit/test_message_utils.py::TestDelegationDetection::test_detect_simple_delegation
```

### Test Options

**Verbose output:**
```bash
./run_tests.sh --verbose
# or
pytest caribou/tests/ -v
```

**Show print statements:**
```bash
pytest caribou/tests/ -s
```

**Run with coverage:**
```bash
./run_tests.sh --coverage
# or
pytest caribou/tests/ --cov=caribou --cov-report=html --cov-report=term
```

Coverage report will be in `htmlcov/index.html`

**Stop on first failure:**
```bash
pytest caribou/tests/ -x
```

**Run only failed tests from last run:**
```bash
pytest caribou/tests/ --lf
```

## Test Coverage

### Unit Tests

#### test_anthropic_wrapper.py
- ✅ Client initialization (default and custom params)
- ✅ OpenAI-compatible interface structure
- ✅ System message extraction and combination
- ✅ Role filtering (assistant, user, system)
- ✅ API call parameter handling
- ✅ Response formatting to OpenAI structure
- ✅ Multiple text block handling
- ✅ Edge cases (empty messages, missing content, etc.)

#### test_ollama_wrapper.py
- ✅ Client initialization with host variants
- ✅ OpenAI-compatible interface structure
- ✅ API call with temperature parameter
- ✅ ND-JSON response parsing
- ✅ Multi-line response handling
- ✅ Response structure validation
- ✅ Error handling (HTTP errors, invalid JSON, no message)

#### test_message_utils.py
- ✅ Delegation command detection (various formats)
- ✅ RAG query detection
- ✅ Artifact extraction (notes, TODOs, checkboxes, code fences)
- ✅ Code block counting
- ✅ Code preview generation
- ✅ Edge cases (unicode, multiline, empty content)

#### test_memory_manager.py
- ✅ Initialization with various parameters
- ✅ Message pinning strategy
- ✅ Adding messages and pivotal code
- ✅ System prompt updates
- ✅ Context assembly (pinned + pivotal + summaries + working)
- ✅ Summarization triggering logic
- ✅ Episodic summarization
- ✅ Multiple summarization rounds
- ✅ Error handling in summarization
- ✅ Context layout verification

#### test_agent_system.py
- ✅ Command and Agent class creation
- ✅ Agent prompt generation (basic, with commands, with RAG, with samples)
- ✅ AgentSystem creation and agent retrieval
- ✅ Loading from JSON configuration
- ✅ Code sample loading from disk
- ✅ Extracting possible actions from agents
- ✅ Agent switching logic
- ✅ Memory manager updates during switch
- ✅ Action space updates during switch

### Integration Tests

#### test_message_flow.py
- ✅ Simple conversation flow
- ✅ Message flow with MemoryManager
- ✅ Delegation detection and agent switching
- ✅ Multi-agent conversation with memory
- ✅ RAG query detection and handling
- ✅ Artifact extraction from responses
- ✅ Complete workflow with all features
- ✅ Error resilience
- ✅ Long conversations with summarization

## Writing New Tests

### Test File Naming

- Unit tests: `test_<module_name>.py` in `unit/`
- Integration tests: `test_<feature_name>.py` in `integration/`

### Test Class Naming

Use descriptive class names grouped by functionality:

```python
class TestComponentName:
    """Test ComponentName functionality."""

    def test_specific_behavior(self):
        """Test that specific behavior works correctly."""
        pass
```

### Using Fixtures

Common fixtures are defined in `conftest.py`:

```python
def test_with_mock_client(mock_llm_client):
    """Test using mock LLM client."""
    client = mock_llm_client(responses=["Response 1", "Response 2"])
    # Use client in test
```

Available fixtures:
- `mock_anthropic_response` - Create mock Anthropic responses
- `mock_openai_response` - Create mock OpenAI responses
- `sample_messages` - Pre-built message history
- `sample_agent_system` - Sample agent configuration
- `mock_llm_client` - Mock LLM client factory

### Mocking External APIs

Always mock external API calls in tests:

```python
from unittest.mock import Mock, patch

@patch("caribou.core.anthropic_wrapper.anthropic.Anthropic")
def test_api_call(mock_anthropic):
    mock_instance = Mock()
    mock_response = Mock(content=[...])
    mock_instance.messages.create.return_value = mock_response
    mock_anthropic.return_value = mock_instance

    # Test code here
```

## Continuous Integration

To run tests in CI/CD:

```bash
# Install dependencies
pip install -r requirements.txt
pip install pytest pytest-cov

# Run tests with coverage
pytest caribou/tests/ --cov=caribou --cov-report=xml --cov-report=term

# Fail if coverage is below threshold (optional)
pytest caribou/tests/ --cov=caribou --cov-fail-under=80
```

## Troubleshooting

### Import Errors

Make sure CARIBOU is installed or the path is set:

```bash
# From project root
pip install -e .
# or
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Tests Hanging

Some tests may hang if they're waiting for actual API calls. Ensure all external calls are mocked.

### Fixture Not Found

If pytest can't find a fixture, check:
1. Is it defined in `conftest.py`?
2. Is `conftest.py` in the correct location?
3. Are you importing fixtures correctly?

## Best Practices

1. **One assertion per test** (generally) - Makes failures easier to diagnose
2. **Test names should be descriptive** - `test_delegation_with_underscores_in_name` not `test_1`
3. **Mock external dependencies** - Never make real API calls in tests
4. **Use fixtures for common setup** - Keeps tests DRY
5. **Test edge cases** - Empty inputs, None values, extreme values
6. **Test error conditions** - Don't just test the happy path
7. **Keep tests fast** - Unit tests should run in milliseconds
8. **Make tests independent** - Each test should be runnable in isolation

## Contributing

When adding new features to CARIBOU:

1. Write tests first (TDD) or alongside the feature
2. Ensure all tests pass: `./run_tests.sh`
3. Check coverage: `./run_tests.sh --coverage`
4. Add new test cases for edge cases
5. Update this README if adding new test categories

## Questions?

For questions about the test suite, please open an issue on the CARIBOU GitHub repository.
