#!/usr/bin/env bash
# Launch Claude Code in locked-down autoresearch mode.
#
# Usage:
#   ./autoresearch/start.sh                    # fresh start, opus
#   ./autoresearch/start.sh resume             # resume from scratchpad
#   ./autoresearch/start.sh start sonnet       # fresh start, sonnet
#   ./autoresearch/start.sh resume opus medium # resume, opus, medium effort
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

MODEL="${2:-opus}"
EFFORT="${3:-high}"
MODE="${1:-start}"

COMMON_FLAGS=(
  --model "$MODEL"
  --permission-mode dontAsk
  --allowedTools
    "Edit(autoresearch/train.py)"
    "Edit(autoresearch/scratchpad.md)"
    "Read"
    "Bash(./autoresearch/run.sh)"
    "Bash(./autoresearch/run.sh *)"
    "Bash(git add autoresearch/train.py autoresearch/scratchpad.md)"
    "Bash(git commit *)"
    "Bash(git checkout -- autoresearch/train.py)"
    "Bash(git log *)"
    "Bash(git diff *)"
    "Bash(git status)"
    "Bash(sleep *)"
  --disallowedTools
    "Bash(git push *)"
    "Bash(rm *)"
    "Bash(pip install *)"
    "Bash(uv *)"
    "Bash(curl *)"
    "Bash(wget *)"
    "Bash(python *)"
    "Bash(docker *)"
    "Bash(sudo *)"
)

SCRATCHPAD_RULE="CRITICAL: After EVERY experiment (whether committed or reverted), you MUST edit autoresearch/scratchpad.md before moving on. Update: (1) Current Best if improved, (2) add the experiment to the Experiment Log table, (3) update Hypotheses with what you learned, (4) update Next Experiments with your plan. Do NOT just print results to chat — the scratchpad is your persistent memory across sessions. Wait 2 minutes (sleep 120) between experiments to let the GPU cool down."

START_PROMPT="Read autoresearch/program.md and autoresearch/train.py fully, then start the autoresearch experiment loop. Run experiments one at a time, following the priorities in program.md. $SCRATCHPAD_RULE"

RESUME_PROMPT="Read autoresearch/scratchpad.md, autoresearch/program.md, and autoresearch/train.py. Check git log for recent experiments. You are resuming an ongoing autoresearch loop — pick up where the last session left off. Do NOT repeat experiments already in the git log or scratchpad. Focus on what's in Next Experiments or try something new based on your observations. $SCRATCHPAD_RULE"

if [ "$MODE" = "resume" ]; then
  PROMPT="$RESUME_PROMPT"
else
  PROMPT="$START_PROMPT"
fi

claude "${COMMON_FLAGS[@]}" -- "$PROMPT"
