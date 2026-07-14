#!/usr/bin/env sh
set -eu

# Run HotpotQA agentic evaluation via CAIPE supervisor A2A endpoint.

SCRIPT_DIR=$(dirname "$0")
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
PYTHON_BIN=${PYTHON:-python3}

SUPERVISOR_URL=${CAIPE_SUPERVISOR_URL:-http://localhost:8000}

exec "$PYTHON_BIN" \
  "$REPO_ROOT/src/deepeval_eval/hotpotqa_deepeval.py" \
  eval \
  --agentic \
  --supervisor-url "$SUPERVISOR_URL" \
  --max-items 10 \
  --top-k 5 \
  --max-context-chars 12000 \
  "$@"
