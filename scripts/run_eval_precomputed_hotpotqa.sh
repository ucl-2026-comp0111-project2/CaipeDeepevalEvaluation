#!/bin/bash

# Ensure we are in the project root directory
cd "$(dirname "$0")/.."

# Load environment variables from .env file if it exists
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

# Override configuration variables for HOTPOTQA
export CAIPE_DATASOURCE_ID="hotpotqa_sample"
export QUESTIONS_PATH="${QUESTIONS_PATH:-../caipe_ragas/rag_eval/data/hotpotqa_full_questions.jsonl}"

# Fetch OIDC credentials from Kubernetes
CLIENT_ID=$(kubectl get secret caipe-ui-secret -n caipe -o jsonpath='{.data.OIDC_CLIENT_ID}' | base64 --decode)
CLIENT_SECRET=$(kubectl get secret caipe-ui-secret -n caipe -o jsonpath='{.data.OIDC_CLIENT_SECRET}' | base64 --decode)
export CAIPE_CLIENT_ID="${CLIENT_ID}"
export CAIPE_CLIENT_SECRET="${CLIENT_SECRET}"

# Fetch OIDC token from Keycloak
export CAIPE_AUTH_TOKEN=$(curl -sk -X POST "https://keycloak.caipe.homelab/realms/caipe/protocol/openid-connect/token" \
  -d "client_id=${CLIENT_ID}" \
  -d "client_secret=${CLIENT_SECRET}" \
  -d "grant_type=client_credentials" | jq -r '.access_token')

uv run python3 src/deepeval_eval/deepeval_evaluator.py \
  eval \
  --oracle-testing \
  --datasource-id "${CAIPE_DATASOURCE_ID}" \
  --questions-file "${QUESTIONS_PATH}" \
  --dataset-name hotpotqa \
  --limit-per-category 50 \
  --top-k 5 \
  "$@"