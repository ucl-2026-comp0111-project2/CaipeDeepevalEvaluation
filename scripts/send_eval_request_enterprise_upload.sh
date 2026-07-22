#!/bin/bash
set -e

# Ensure we are in the project root directory
cd "$(dirname "$0")/.."

# Load environment variables from .env file if it exists
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

API_URL="${API_URL:-http://localhost:8000}"
DEFAULT_ENTERPRISE_PATH="/Users/alexanghh/development/caipe_ragas/rag_eval/data/enterprise_rag_bench_questions.jsonl"
if [ -z "$QUESTIONS_PATH" ] || [ ! -f "$QUESTIONS_PATH" ] || [[ "$QUESTIONS_PATH" == *"hotpotqa"* ]]; then
  QUESTIONS_PATH="$DEFAULT_ENTERPRISE_PATH"
fi
DATASET_NAME="${DATASET_NAME:-enterprise}"
DATASOURCE_ID="${DATASOURCE_ID:-enterprise_rag_bench}"
ANSWER_MODE="${ANSWER_MODE:-reference}"
MAX_ITEMS="${MAX_ITEMS:-1}"
LIMIT_PER_CATEGORY="${LIMIT_PER_CATEGORY:-}"
TOP_K="${TOP_K:-5}"
MAX_CONTEXT_CHARS="${MAX_CONTEXT_CHARS:-16000}"
AGENTIC="${AGENTIC:-true}"
SAVE_TO_DB="${SAVE_TO_DB:-false}"
FORCE_RERUN="${FORCE_RERUN:-false}"

if [ ! -f "$QUESTIONS_PATH" ]; then
  echo "Error: Questions dataset file not found at '$QUESTIONS_PATH'"
  exit 1
fi

echo "Submitting evaluation job with uploaded dataset: ${QUESTIONS_PATH}"
echo "API Endpoint: ${API_URL}/eval/jobs/upload"

QUERY_PARAMS="dataset_name=${DATASET_NAME}&datasource_id=${DATASOURCE_ID}&answer_mode=${ANSWER_MODE}&max_items=${MAX_ITEMS}&top_k=${TOP_K}&max_context_chars=${MAX_CONTEXT_CHARS}&agentic=${AGENTIC}&save_to_db=${SAVE_TO_DB}&force_rerun=${FORCE_RERUN}"
if [ -n "$LIMIT_PER_CATEGORY" ]; then
  QUERY_PARAMS="${QUERY_PARAMS}&limit_per_category=${LIMIT_PER_CATEGORY}"
fi

# Upload dataset file and submit eval job
RESPONSE=$(curl -s -X POST "${API_URL}/eval/jobs/upload?${QUERY_PARAMS}" \
  -F "file=@${QUESTIONS_PATH}")

echo "Response from API:"
echo "$RESPONSE" | jq . 2>/dev/null || echo "$RESPONSE"

JOB_ID=$(echo "$RESPONSE" | jq -r '.job_id // empty')

if [ -z "$JOB_ID" ]; then
  echo "Failed to obtain job_id from API response."
  exit 1
fi

echo ""
echo "Evaluation job submitted successfully. Job ID: ${JOB_ID}"
echo "Polling job status at ${API_URL}/jobs/${JOB_ID}..."

# Poll job status until complete or failed
while true; do
  STATUS_RESP=$(curl -s "${API_URL}/jobs/${JOB_ID}")
  JOB_STATUS=$(echo "$STATUS_RESP" | jq -r '.status')
  echo "[$(date +'%H:%M:%S')] Job status: ${JOB_STATUS}"

  if [ "$JOB_STATUS" = "completed" ]; then
    echo ""
    echo "=========================================="
    echo "Job Completed Successfully!"
    echo "=========================================="
    echo "Fetching detailed job results..."
    RESULTS_DIR="results"
    mkdir -p "$RESULTS_DIR"
    CSV_FILE="${RESULTS_DIR}/job_${JOB_ID}_results.csv"
    JSON_FILE="${RESULTS_DIR}/job_${JOB_ID}_results.json"
    
    # Save CSV option for download
    curl -s "${API_URL}/jobs/${JOB_ID}/results?format=csv" > "$CSV_FILE"
    
    # Save and print JSON to console
    curl -s "${API_URL}/jobs/${JOB_ID}/results?format=json" > "$JSON_FILE"
    cat "$JSON_FILE" | jq . 2>/dev/null || cat "$JSON_FILE"
    
    echo ""
    echo "Results saved to:"
    echo "  - JSON (console standard): ${JSON_FILE}"
    echo "  - CSV (download option): ${CSV_FILE}"
    break
  elif [ "$JOB_STATUS" = "failed" ]; then
    echo ""
    echo "Job failed!"
    echo "$STATUS_RESP" | jq .
    exit 1
  fi

  sleep 3
done
