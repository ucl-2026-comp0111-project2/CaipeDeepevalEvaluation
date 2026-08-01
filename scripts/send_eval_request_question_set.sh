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
SET_ID="${SET_ID:-1}"
ANSWER_MODE="${ANSWER_MODE:-generate}"
MAX_ITEMS="${MAX_ITEMS:-}"
LIMIT_PER_CATEGORY="${LIMIT_PER_CATEGORY:-}"
TOP_K="${TOP_K:-3}"
MAX_CONTEXT_CHARS="${MAX_CONTEXT_CHARS:-12000}"
AGENTIC="${AGENTIC:-false}"
SAVE_TO_DB="${SAVE_TO_DB:-true}"
FORCE_RERUN="${FORCE_RERUN:-false}"

# Usage: ./scripts/send_eval_request_question_set.sh [set_id] [max_items] [top_k]
if [ -n "$1" ]; then
  SET_ID="$1"
fi
if [ -n "$2" ]; then
  MAX_ITEMS="$2"
fi
if [ -n "$3" ]; then
  TOP_K="$3"
fi

echo "=================================================================="
echo " Question Set Database Evaluation Job Submission"
echo "=================================================================="
echo "API Server:     ${API_URL}"
echo "Target Set ID:  ${SET_ID}"
echo "Answer Mode:    ${ANSWER_MODE}"
echo "Top K Docs:     ${TOP_K}"
echo "Save Results:   ${SAVE_TO_DB}"
echo "=================================================================="

# Retrieve OIDC token from Keycloak (or static DEEPEVAL_API_KEY / fallback if unconfigured)
if [ -z "$CAIPE_OIDC_TOKEN" ] && command -v kubectl >/dev/null 2>&1; then
  if kubectl get secret caipe-ui-secret -n caipe >/dev/null 2>&1; then
    CLIENT_ID=$(kubectl get secret caipe-ui-secret -n caipe -o jsonpath='{.data.OIDC_CLIENT_ID}' | base64 --decode 2>/dev/null || true)
    CLIENT_SECRET=$(kubectl get secret caipe-ui-secret -n caipe -o jsonpath='{.data.OIDC_CLIENT_SECRET}' | base64 --decode 2>/dev/null || true)

    if [ -n "$CLIENT_ID" ] && [ -n "$CLIENT_SECRET" ]; then
      KEYCLOAK_URL="${KEYCLOAK_URL:-https://keycloak.caipe.homelab/realms/caipe/protocol/openid-connect/token}"
      FETCHED_TOKEN=$(curl -sk -X POST "$KEYCLOAK_URL" \
        -d "client_id=${CLIENT_ID}" \
        -d "client_secret=${CLIENT_SECRET}" \
        -d "grant_type=client_credentials" | jq -r '.access_token // empty' 2>/dev/null || true)
      if [ -n "$FETCHED_TOKEN" ] && [ "$FETCHED_TOKEN" != "null" ]; then
        export CAIPE_OIDC_TOKEN="$FETCHED_TOKEN"
      fi
    fi

    if [ -z "$CAIPE_OIDC_TOKEN" ]; then
      FETCHED_KEY=$(kubectl get secret caipe-ui-secret -n caipe -o jsonpath='{.data.AGENTGATEWAY_TARGETS_TOKEN}' | base64 --decode 2>/dev/null || true)
      if [ -z "$FETCHED_KEY" ]; then
        FETCHED_KEY=$(kubectl get secret caipe-ui-secret -n caipe -o jsonpath='{.data.NEXTAUTH_SECRET}' | base64 --decode 2>/dev/null || true)
      fi
      if [ -n "$FETCHED_KEY" ]; then
        export DEEPEVAL_API_KEY="$FETCHED_KEY"
      fi
    fi
  fi
fi

AUTH_TOKEN="${CAIPE_OIDC_TOKEN:-${DEEPEVAL_API_KEY}}"
AUTH_HEADER=()
if [ -n "$AUTH_TOKEN" ]; then
  AUTH_HEADER=(-H "Authorization: Bearer ${AUTH_TOKEN}")
fi

# Submit evaluation job targeting PostgreSQL Question Set ID
PAYLOAD="{
  \"question_set_id\": ${SET_ID},
  \"answer_mode\": \"${ANSWER_MODE}\",
  \"top_k\": ${TOP_K},
  \"max_context_chars\": ${MAX_CONTEXT_CHARS},
  \"agentic\": ${AGENTIC},
  \"save_to_db\": ${SAVE_TO_DB},
  \"force_rerun\": ${FORCE_RERUN}
}"

if [ -n "$MAX_ITEMS" ]; then
  PAYLOAD=$(echo "$PAYLOAD" | jq ". + {\"max_items\": ${MAX_ITEMS}}")
fi
if [ -n "$LIMIT_PER_CATEGORY" ]; then
  PAYLOAD=$(echo "$PAYLOAD" | jq ". + {\"limit_per_category\": ${LIMIT_PER_CATEGORY}}")
fi

echo "Submitting evaluation job payload:"
echo "$PAYLOAD" | jq .

RESPONSE=$(curl -sS --fail-with-body "${AUTH_HEADER[@]}" -H "Content-Type: application/json" \
  -X POST "${API_URL}/eval/jobs/question-sets/${SET_ID}" \
  -d "$PAYLOAD")

echo ""
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

while true; do
  STATUS_RESP=$(curl -s "${AUTH_HEADER[@]}" "${API_URL}/jobs/${JOB_ID}")
  JOB_STATUS=$(echo "$STATUS_RESP" | jq -r '.status // empty')
  echo "Status: ${JOB_STATUS}"

  if [ "$JOB_STATUS" = "completed" ] || [ "$JOB_STATUS" = "failed" ]; then
    echo ""
    echo "Final Job Status: ${JOB_STATUS}"
    echo "$STATUS_RESP" | jq . 2>/dev/null || echo "$STATUS_RESP"
    break
  fi
  sleep 2
done
