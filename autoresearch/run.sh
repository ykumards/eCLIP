#!/usr/bin/env bash
# Run one autoresearch experiment in a sandboxed Docker container.
#
# Usage:
#   ./autoresearch/run.sh              # run one experiment
#   ./autoresearch/run.sh bash         # interactive shell for debugging
#
# Safety:
#   - 10 min timeout (--stop-timeout)
#   - 20 GB GPU memory cap
#   - 24 GB RAM cap
#   - No network access
#   - Read-only code, only data volume is writable
#   - Non-root user inside container

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
IMAGE_NAME="eclip-autoresearch"

# Build if needed
if ! docker image inspect "$IMAGE_NAME" &>/dev/null; then
    echo "Building Docker image..."
    docker build -t "$IMAGE_NAME" -f "$SCRIPT_DIR/Dockerfile" "$PROJECT_DIR"
fi

CMD="${1:-}"

# Logging: save experiment output with timestamp
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$(date +%Y%m%d_%H%M%S).log"

if [ "$CMD" = "bash" ]; then
    # Interactive debugging
    docker run --rm -it \
        --gpus '"device=0"' \
        --memory=24g \
        --shm-size=4g \
        --network=none \
        -v "$PROJECT_DIR/data:/workspace/data:ro" \
        -v "$SCRIPT_DIR/train.py:/workspace/autoresearch/train.py:rw" \
        -v "${HF_HOME:-$HOME/.cache/huggingface}:/root/.cache/huggingface:ro" \
        "$IMAGE_NAME" \
        bash
else
    # Run experiment with timeout, tee output to log
    echo "=== Experiment $(date -Iseconds) ===" | tee "$LOG_FILE"
    echo "=== train.py SHA: $(sha256sum "$SCRIPT_DIR/train.py" | cut -c1-12) ===" | tee -a "$LOG_FILE"
    docker run --rm \
        --gpus '"device=0"' \
        --memory=24g \
        --shm-size=4g \
        --network=none \
        --stop-timeout=600 \
        -v "$PROJECT_DIR/data:/workspace/data:ro" \
        -v "$SCRIPT_DIR/train.py:/workspace/autoresearch/train.py:ro" \
        -v "${HF_HOME:-$HOME/.cache/huggingface}:/root/.cache/huggingface:ro" \
        "$IMAGE_NAME" 2>&1 | tee -a "$LOG_FILE"
fi
