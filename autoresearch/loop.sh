#!/usr/bin/env bash
# Autoresearch loop — runs experiments until you kill it.
#
# Usage: ./autoresearch/loop.sh
#
# The loop:
#   1. Agent reads program.md + train.py
#   2. Agent edits train.py
#   3. Docker runs the experiment (sandboxed, 10 min timeout)
#   4. If metric improved → git commit
#   5. If not → git revert
#   6. Repeat
#
# Press Ctrl+C to stop between experiments.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
IMAGE_NAME="eclip-autoresearch"
LOG_DIR="$SCRIPT_DIR/logs"
BEST_METRIC_FILE="$SCRIPT_DIR/.best_metric"

mkdir -p "$LOG_DIR"

# Initialize best metric (higher = worse for mean_rank)
if [ ! -f "$BEST_METRIC_FILE" ]; then
    echo "999.0" > "$BEST_METRIC_FILE"
fi

EXPERIMENT=0

# Build Docker image once
echo "=== Building Docker image ==="
docker build -t "$IMAGE_NAME" -f "$SCRIPT_DIR/Dockerfile" "$PROJECT_DIR" 2>&1 | tail -5

echo "=== Autoresearch loop started ==="
echo "Press Ctrl+C between experiments to stop."
echo ""

while true; do
    EXPERIMENT=$((EXPERIMENT + 1))
    BEST=$(cat "$BEST_METRIC_FILE")
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="$LOG_DIR/exp_${EXPERIMENT}_${TIMESTAMP}.log"

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Experiment #$EXPERIMENT | Best so far: $BEST"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Save current train.py as backup
    cp "$SCRIPT_DIR/train.py" "$SCRIPT_DIR/.train.py.bak"

    # TODO: This is where the AI agent edits train.py
    # For now, just run the current version.
    # In practice, you'd call your LLM agent here to modify train.py
    # based on program.md and previous results.

    echo "Running experiment..."
    RESULT=$(docker run --rm \
        --gpus '"device=0"' \
        --memory=24g \
        --network=none \
        --stop-timeout=600 \
        -v "$PROJECT_DIR/data:/workspace/data:ro" \
        -v "$SCRIPT_DIR/train.py:/workspace/autoresearch/train.py:ro" \
        "$IMAGE_NAME" 2>&1) || true

    echo "$RESULT" > "$LOG_FILE"

    # Extract metric
    METRIC=$(echo "$RESULT" | grep ">>> val_metric" | tail -1 | awk '{print $NF}')
    MEMORY=$(echo "$RESULT" | grep ">>> peak_memory_gb" | tail -1 | awk '{print $NF}')

    if [ -z "$METRIC" ]; then
        echo "  FAILED — no metric found in output"
        echo "  See: $LOG_FILE"
        # Revert
        cp "$SCRIPT_DIR/.train.py.bak" "$SCRIPT_DIR/train.py"
        continue
    fi

    echo "  mean_rank: $METRIC | peak_mem: ${MEMORY:-?} GB"

    # Compare (lower is better)
    IMPROVED=$(uv run --project "$PROJECT_DIR" python -c "print('yes' if float('$METRIC') < float('$BEST') else 'no')")

    if [ "$IMPROVED" = "yes" ]; then
        echo "  ✓ IMPROVED ($BEST → $METRIC)"
        echo "$METRIC" > "$BEST_METRIC_FILE"
        cd "$SCRIPT_DIR"
        git add train.py
        git commit -m "experiment #$EXPERIMENT | mean_rank: $BEST → $METRIC" --quiet
        cd "$PROJECT_DIR"
    else
        echo "  ✗ No improvement ($METRIC >= $BEST), reverting"
        cp "$SCRIPT_DIR/.train.py.bak" "$SCRIPT_DIR/train.py"
    fi

    echo ""
done
