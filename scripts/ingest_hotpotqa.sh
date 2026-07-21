#!/usr/bin/env sh
set -eu

# Run HotpotQA ingestion using the repository defaults.
# Additional CLI arguments can be passed to override the defaults.

SCRIPT_DIR=$(dirname "$0")
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
PYTHON_BIN=${PYTHON:-python3}

exec "$PYTHON_BIN" \
  "$REPO_ROOT/src/deepeval_eval/hotpotqa_deepeval.py" \
  ingest \
  --limit 100 \
  --questions-per-category 50 \
  --max-docs 1000 \
  --batch-size 50 \
  "$@"
