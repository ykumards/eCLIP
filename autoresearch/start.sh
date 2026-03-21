#!/usr/bin/env bash
# Launch Claude Code in locked-down autoresearch mode.
#
# Usage:
#   ./autoresearch/start.sh              # default: opus, high effort
#   ./autoresearch/start.sh sonnet       # use sonnet
#   ./autoresearch/start.sh opus medium  # opus with medium effort
#   ./autoresearch/start.sh haiku        # cheapest, fastest
#
# Permissions:
#   - Can ONLY edit train.py and scratchpad.md
#   - Can ONLY run experiments via ./autoresearch/run.sh
#   - Can ONLY do git add/commit/checkout/log/diff/status
#   - Everything else is auto-denied (no prompt, just blocked)

set -euo pipefail
cd "$(dirname "$0")/.."

# Kill any running experiment containers on exit
trap 'docker kill $(docker ps -q --filter ancestor=eclip-autoresearch) 2>/dev/null || true' EXIT

MODEL="${1:-opus}"
EFFORT="${2:-high}"

claude \
  --model "$MODEL" \
  --permission-mode dontAsk \
  --allowedTools \
    "Edit(autoresearch/train.py)" \
    "Edit(autoresearch/scratchpad.md)" \
    "Read" \
    "Bash(./autoresearch/run.sh)" \
    "Bash(./autoresearch/run.sh *)" \
    "Bash(git add autoresearch/train.py autoresearch/scratchpad.md)" \
    "Bash(git commit *)" \
    "Bash(git checkout -- autoresearch/train.py)" \
    "Bash(git log *)" \
    "Bash(git diff *)" \
    "Bash(git status)" \
    "Bash(sleep *)" \
  --disallowedTools \
    "Bash(git push *)" \
    "Bash(rm *)" \
    "Bash(pip install *)" \
    "Bash(uv *)" \
    "Bash(curl *)" \
    "Bash(wget *)" \
    "Bash(python *)" \
    "Bash(docker *)" \
    "Bash(sudo *)" \
  -- \
  "Read autoresearch/program.md and autoresearch/train.py fully, then start the autoresearch experiment loop. Run experiments one at a time, following the priorities in program.md. After each experiment, update scratchpad.md with observations and plans. Wait 2 minutes (sleep 120) between experiments to let the GPU cool down."
