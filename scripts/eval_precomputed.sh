#!/usr/bin/env sh
set -eu

# Run DeepEval against benchmark ground-truth contexts/reference answers.
# Additional CLI arguments can be passed to override the defaults.

SCRIPT_DIR=$(dirname "$0")
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
PYTHON_BIN=${PYTHON:-python3}

exec "$PYTHON_BIN" \
  "$REPO_ROOT/src/deepeval_eval/precomputed_deepeval.py" \
  --benchmark hotpotqa \
  --max-items 20 \
  --answer-mode reference \
  "$@"
