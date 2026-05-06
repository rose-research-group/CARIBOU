#!/bin/bash
# Universal CARIBOU comparison run launcher.
#
# Usage:
#   sbatch run_comparison.sh --dataset aba_hippocampus --llm chatgpt --mode full_system
#   sbatch run_comparison.sh --dataset tsp_large_intestine --llm deepseek --mode single_agent
#
# All datasets are defined in  ../datasets/<id>/config.json  and  prompt.txt.
#
#SBATCH --job-name=caribou_comparison
#SBATCH --partition=peerd
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
DATASET=""
LLM_BACKEND=""
MODE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)  DATASET="$2";      shift 2 ;;
        --llm)      LLM_BACKEND="$2";  shift 2 ;;
        --mode)     MODE="$2";         shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$DATASET" || -z "$LLM_BACKEND" || -z "$MODE" ]]; then
    echo "Usage: sbatch run_comparison.sh --dataset <id> --llm <chatgpt|deepseek|claude> --mode <full_system|single_agent|one_shot>"
    exit 1
fi

# ---------------------------------------------------------------------------
# Paths derived from dataset config
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CARIBOU_ROOT="$(git -C "${SLURM_SUBMIT_DIR:-$(pwd)}" rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$CARIBOU_ROOT" ]]; then
    CARIBOU_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || true)"
fi
if [[ -z "$CARIBOU_ROOT" ]]; then
    CARIBOU_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
fi
COMP_DIR="$CARIBOU_ROOT/benchmarking/celltyping_benchmarks"
DATASET_DIR="$COMP_DIR/datasets/$DATASET"
CONFIG_PATH="$DATASET_DIR/config.json"
PROMPT_PATH="$DATASET_DIR/prompt.txt"

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "ERROR: Dataset config not found: $CONFIG_PATH"
    echo "Available datasets:"; ls "$COMP_DIR/datasets/"; exit 1
fi
if [[ ! -f "$PROMPT_PATH" ]]; then
    echo "ERROR: Prompt not found: $PROMPT_PATH"; exit 1
fi

# Read values from config (requires python/jq; using python for portability)
DATASET_PATH=$(python3 -c "import json, pathlib; c=json.load(open('$CONFIG_PATH')); p=pathlib.Path(c['input_dataset_path']).expanduser(); print(p if p.is_absolute() else pathlib.Path('$CARIBOU_ROOT') / p)")
SLURM_MEM=$(python3 -c "import json; c=json.load(open('$CONFIG_PATH')); print(c.get('slurm_mem','64G'))")

BLUEPRINT_FULL="$CARIBOU_ROOT/caribou/src/caribou/agents/caribou_fully_connected_v2.json"
BLUEPRINT_SINGLE="$CARIBOU_ROOT/caribou/src/caribou/agents/caribou_single_agent.json"
RESULTS_DIR="$COMP_DIR/results/$DATASET"
SANDBOX_BACKEND="singularity"

mkdir -p "$RESULTS_DIR" logs

# ---------------------------------------------------------------------------
# Name the run
# ---------------------------------------------------------------------------
RUN_NAME="${LLM_BACKEND}_${MODE}_${SLURM_JOB_ID:-$$}"
RUN_DIR="$RESULTS_DIR/$RUN_NAME"
mkdir -p "$RUN_DIR"

# Re-name the SLURM job for legible queue output
if command -v scontrol &>/dev/null && [[ -n "${SLURM_JOB_ID:-}" ]]; then
    scontrol update JobId="$SLURM_JOB_ID" JobName="cmp_${DATASET}_${LLM_BACKEND}_${MODE}"
fi

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
START_TIME=$(date +%s)
echo "====================================================="
echo "  CARIBOU Comparison Run"
echo "  Dataset : $DATASET"
echo "  LLM     : $LLM_BACKEND"
echo "  Mode    : $MODE"
echo "  Output  : $RUN_DIR"
echo "  Started : $(date)"
echo "====================================================="

INITIAL_PROMPT="$(cat "$PROMPT_PATH")"
cd "$CARIBOU_ROOT"

if [[ "$MODE" == "full_system" ]]; then
    NUM_TURNS=30
    caribou run auto \
        --blueprint            "$BLUEPRINT_FULL" \
        --dataset              "$DATASET_PATH" \
        --sandbox              "$SANDBOX_BACKEND" \
        --llm                  "$LLM_BACKEND" \
        --turns                "$NUM_TURNS" \
        --prompt               "$INITIAL_PROMPT" \
        --driver-agent         "master_agent" \
        --output-dir           "$RUN_DIR" \
        --make-report \
        --agent-report-memory \
        2>&1 | tee "$RUN_DIR/run.log"

elif [[ "$MODE" == "single_agent" ]]; then
    NUM_TURNS=20
    caribou run auto \
        --blueprint      "$BLUEPRINT_SINGLE" \
        --dataset        "$DATASET_PATH" \
        --sandbox        "$SANDBOX_BACKEND" \
        --llm            "$LLM_BACKEND" \
        --turns          "$NUM_TURNS" \
        --prompt         "$INITIAL_PROMPT" \
        --driver-agent   "solo_agent" \
        --output-dir     "$RUN_DIR" \
        --compress-memory \
        2>&1 | tee "$RUN_DIR/run.log"

elif [[ "$MODE" == "full_system_no_mem" ]]; then
    NUM_TURNS=30
    caribou run auto \
        --blueprint      "$BLUEPRINT_FULL" \
        --dataset        "$DATASET_PATH" \
        --sandbox        "$SANDBOX_BACKEND" \
        --llm            "$LLM_BACKEND" \
        --turns          "$NUM_TURNS" \
        --prompt         "$INITIAL_PROMPT" \
        --driver-agent   "master_agent" \
        --output-dir     "$RUN_DIR" \
        --make-report \
        2>&1 | tee "$RUN_DIR/run.log"

elif [[ "$MODE" == "one_shot" ]]; then
    python "$CARIBOU_ROOT/benchmarking/task_benchmarks/src/one_shot_runner.py" \
        --dataset      "$DATASET_PATH" \
        --output-dir   "$RUN_DIR" \
        --llm          "$LLM_BACKEND" \
        --prompt       "$PROMPT_PATH" \
        --benchmark-id "comparison_${DATASET}" \
        2>&1 | tee "$RUN_DIR/run.log"

else
    echo "ERROR: Unknown mode '$MODE' (expected: full_system | full_system_no_mem | single_agent | one_shot)"
    exit 1
fi

# ---------------------------------------------------------------------------
# Record metadata
# ---------------------------------------------------------------------------
END_TIME=$(date +%s)
RUNTIME=$((END_TIME - START_TIME))
echo "Finished: $(date) | Runtime: ${RUNTIME}s"

python3 - <<PYEOF
import json, pathlib
meta = {
    "dataset":          "$DATASET",
    "llm_backend":      "$LLM_BACKEND",
    "mode":             "$MODE",
    "run_name":         "$RUN_NAME",
    "dataset_path":     "$DATASET_PATH",
    "prompt_path":      "$PROMPT_PATH",
    "blueprint_path":   "$( [[ "$MODE" == "single_agent" ]] && echo "$BLUEPRINT_SINGLE" || echo "$BLUEPRINT_FULL" )",
    "num_turns":        {"full_system": 30, "full_system_no_mem": 30, "single_agent": 20, "one_shot": 1}.get("$MODE", 0),
    "start_time":       $START_TIME,
    "end_time":         $END_TIME,
    "runtime_seconds":  $RUNTIME,
    "slurm_job_id":     "${SLURM_JOB_ID:-}",
}
pathlib.Path("$RUN_DIR/run_metadata.json").write_text(json.dumps(meta, indent=2))
PYEOF

echo "Run complete: $RUN_NAME"
