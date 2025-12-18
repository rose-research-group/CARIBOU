# CARIBOU Test Suite - Quick Start Guide

## Installation

1. **Activate your conda environment (if using conda):**

```bash
conda activate olaf  # or your environment name
```

2. **Install test dependencies:**

```bash
cd /data1/peerd/riffled/riffled/Olaf_project/CARIBOU
python -m pip install pytest pytest-cov
```

Or install all requirements:

```bash
python -m pip install -r requirements.txt
```

**Important:** Make sure pytest is installed in the same Python environment where `anthropic`, `openai`, and other CARIBOU dependencies are installed.

## Running Tests

### Option 1: Using the Test Runner Script (Recommended)

```bash
cd caribou/tests
./run_tests.sh
```

Available options:
- `./run_tests.sh --unit` - Run only unit tests
- `./run_tests.sh --integration` - Run only integration tests
- `./run_tests.sh --verbose` - Verbose output
- `./run_tests.sh --coverage` - Generate coverage report

### Option 2: Using pytest Directly

From the project root directory:

```bash
# Run all tests
pytest caribou/tests/

# Run specific test categories
pytest caribou/tests/unit/
pytest caribou/tests/integration/

# Run specific test file
pytest caribou/tests/unit/test_message_utils.py

# Run with verbose output
pytest caribou/tests/ -v

# Run with coverage
pytest caribou/tests/ --cov=caribou --cov-report=html
```

## What Gets Tested

✅ **LLM API Wrappers**
- AnthropicClient (OpenAI compatibility)
- OllamaClient (local models)

✅ **Message Routing**
- Delegation detection (`delegate_to_agent`)
- RAG query detection (`query_rag_<topic>`)
- Artifact extraction (notes, TODOs)

✅ **History Management**
- MemoryManager with episodic summarization
- Context assembly and compression

✅ **Agent System**
- Multi-agent configuration
- Prompt generation
- Agent switching

✅ **End-to-End Integration**
- Complete message flows
- Multi-agent conversations
- Error handling

## Verifying the Setup

Run a quick smoke test:

```bash
pytest caribou/tests/unit/test_message_utils.py::TestDelegationDetection::test_detect_simple_delegation -v
```

Expected output:
```
test_detect_simple_delegation PASSED
```

## Troubleshooting

**Import errors?**
The tests automatically add `caribou/src` to the Python path via `conftest.py`. If you still get import errors, verify the directory structure:

```
CARIBOU/
└── caribou/
    ├── src/
    │   └── caribou/
    │       ├── core/
    │       ├── execution/
    │       └── agents/
    └── tests/
        ├── conftest.py  # ← Should add src/ to path
        ├── unit/
        └── integration/
```

**Tests hanging?**
All external API calls are mocked - tests should run quickly (< 10 seconds total).

## Next Steps

- See [README.md](README.md) for detailed documentation
- Run with coverage to see what's tested: `./run_tests.sh --coverage`
- Open `htmlcov/index.html` to view the coverage report
