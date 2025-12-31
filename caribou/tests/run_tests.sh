#!/bin/bash

# Test runner script for CARIBOU test suite
# Usage: ./run_tests.sh [options]

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "======================================"
echo "CARIBOU Test Suite Runner"
echo "======================================"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

cd "$PROJECT_ROOT"

# Parse arguments
TEST_TYPE="all"
VERBOSE=""
COVERAGE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --unit)
            TEST_TYPE="unit"
            shift
            ;;
        --integration)
            TEST_TYPE="integration"
            shift
            ;;
        -v|--verbose)
            VERBOSE="-v"
            shift
            ;;
        --coverage)
            COVERAGE="--cov=caribou --cov-report=html --cov-report=term"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --unit          Run only unit tests"
            echo "  --integration   Run only integration tests"
            echo "  -v, --verbose   Verbose output"
            echo "  --coverage      Generate coverage report"
            echo "  -h, --help      Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Detect the correct Python and pytest to use
# Priority: python in current environment > python3 > python
if command -v python &> /dev/null; then
    PYTHON_CMD="python"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    echo -e "${RED}Error: No Python interpreter found${NC}"
    exit 1
fi

# Check if pytest is installed for the current Python
if ! $PYTHON_CMD -m pytest --version &> /dev/null; then
    echo -e "${RED}Error: pytest is not installed for $PYTHON_CMD${NC}"
    echo "Install with: $PYTHON_CMD -m pip install pytest pytest-cov"
    exit 1
fi

echo "Using Python: $($PYTHON_CMD --version)"
echo "Using pytest: $($PYTHON_CMD -m pytest --version | head -1)"
echo ""

# Run tests based on TEST_TYPE
case $TEST_TYPE in
    unit)
        echo -e "${YELLOW}Running unit tests...${NC}"
        echo ""
        $PYTHON_CMD -m pytest caribou/tests/unit/ $VERBOSE $COVERAGE
        ;;
    integration)
        echo -e "${YELLOW}Running integration tests...${NC}"
        echo ""
        $PYTHON_CMD -m pytest caribou/tests/integration/ $VERBOSE $COVERAGE
        ;;
    all)
        echo -e "${YELLOW}Running all tests...${NC}"
        echo ""
        $PYTHON_CMD -m pytest caribou/tests/ $VERBOSE $COVERAGE
        ;;
esac

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}======================================"
    echo "All tests passed! ✓"
    echo -e "======================================${NC}"
else
    echo ""
    echo -e "${RED}======================================"
    echo "Some tests failed! ✗"
    echo -e "======================================${NC}"
    exit 1
fi
