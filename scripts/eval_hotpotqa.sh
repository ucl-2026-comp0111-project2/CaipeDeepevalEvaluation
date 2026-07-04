#!/usr/bin/env sh
set -eu

# Runs HotpotQA evaluation with the repository default evaluation settings.
# Extra CLI options can be appended after the script name; later options override earlier ones.

SCRIPT_DIR=$(dirname "$0")
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
PYTHON_BIN=${PYTHON:-python3}

exec "$PYTHON_BIN" "$REPO_ROOT/src/deepeval_eval/hotpotqa_deepeval.py" eval --max-items 10 --top-k 5 --max-context-chars 12000 "$@"
