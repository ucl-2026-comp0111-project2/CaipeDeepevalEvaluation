#!/usr/bin/env sh
set -eu

# Run EnterpriseRAG-Bench agentic evaluation via CAIPE supervisor A2A endpoint.
# Additional CLI arguments can be passed to override the defaults.

SCRIPT_DIR=$(dirname "$0")
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
PYTHON_BIN=${PYTHON:-python3}

exec "$PYTHON_BIN" \
  "$REPO_ROOT/src/deepeval_eval/enterprise_deepeval.py" \
  eval \
  --agentic \
  --supervisor-url "${CAIPE_SUPERVISOR_URL:-http://localhost:8000}" \
  --max-items 10 \
  --top-k 3 \
  --max-context-chars 6000 \
  "$@"