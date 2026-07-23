#!/usr/bin/env sh
set -eu

# Run EnterpriseRAG-Bench ingestion using the repository defaults.
# Additional CLI arguments can be passed to override the defaults.

SCRIPT_DIR=$(dirname "$0")
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
PYTHON_BIN=${PYTHON:-python3}

exec "$PYTHON_BIN" \
  "$REPO_ROOT/src/deepeval_eval/ingest.py" \
  --dataset-name enterprise \
  --sources confluence jira github hubspot fireflies linear google_drive gmail slack \
  --limit-per-source 1000 \
  --num-questions 100 \
  --questions-per-category 10 \
  --batch-size 50 \
  "$@"
