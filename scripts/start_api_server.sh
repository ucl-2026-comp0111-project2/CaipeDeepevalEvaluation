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

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

echo "Starting CAIPE DeepEval REST API server on ${HOST}:${PORT}..."

# Fetch OIDC credentials from Kubernetes if available
if command -v kubectl >/dev/null 2>&1 && kubectl get secret caipe-ui-secret -n caipe >/dev/null 2>&1; then
  CLIENT_ID=$(kubectl get secret caipe-ui-secret -n caipe -o jsonpath='{.data.OIDC_CLIENT_ID}' | base64 --decode 2>/dev/null || true)
  CLIENT_SECRET=$(kubectl get secret caipe-ui-secret -n caipe -o jsonpath='{.data.OIDC_CLIENT_SECRET}' | base64 --decode 2>/dev/null || true)

  if [ -n "$CLIENT_ID" ] && [ -n "$CLIENT_SECRET" ]; then
    export CAIPE_OIDC_TOKEN=$(curl -sk -X POST "https://keycloak.caipe.homelab/realms/caipe/protocol/openid-connect/token" \
      -d "client_id=${CLIENT_ID}" \
      -d "client_secret=${CLIENT_SECRET}" \
      -d "grant_type=client_credentials" | jq -r '.access_token' 2>/dev/null || true)
    echo "OIDC token fetched successfully."
  fi
fi

# Run the API server via uv
uv run python3 -c "from deepeval_eval.api import run_server; run_server(host='${HOST}', port=${PORT})"
