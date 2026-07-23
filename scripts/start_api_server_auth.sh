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

# Explicitly enforce authentication by turning off dev bypass flags
export ALLOW_UNAUTHENTICATED_ACCESS="${ALLOW_UNAUTHENTICATED_ACCESS:-false}"
export CAIPE_UNSAFE_RBAC_BYPASS="${CAIPE_UNSAFE_RBAC_BYPASS:-false}"

# Default OIDC Provider Configuration (for JWT Bearer token validation)
export OIDC_ISSUER="${OIDC_ISSUER:-https://caipe.homelab/realms/caipe}"
export OIDC_AUDIENCE="${OIDC_AUDIENCE:-caipe-ui}"
export OIDC_JWKS_URL="${OIDC_JWKS_URL:-https://keycloak.caipe.homelab/realms/caipe/protocol/openid-connect/certs}"

# Retrieve and set DEEPEVAL_API_KEY from Kubernetes secret if not already set
if [ -z "$DEEPEVAL_API_KEY" ] && command -v kubectl >/dev/null 2>&1; then
  if kubectl get secret caipe-ui-secret -n caipe >/dev/null 2>&1; then
    FETCHED_KEY=$(kubectl get secret caipe-ui-secret -n caipe -o jsonpath='{.data.AGENTGATEWAY_TARGETS_TOKEN}' | base64 --decode 2>/dev/null || true)
    if [ -z "$FETCHED_KEY" ]; then
      FETCHED_KEY=$(kubectl get secret caipe-ui-secret -n caipe -o jsonpath='{.data.NEXTAUTH_SECRET}' | base64 --decode 2>/dev/null || true)
    fi
    if [ -n "$FETCHED_KEY" ]; then
      export DEEPEVAL_API_KEY="$FETCHED_KEY"
      echo "DEEPEVAL_API_KEY retrieved successfully from Kubernetes secret 'caipe-ui-secret'."
    fi
  fi
fi

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

echo "Starting Auth-Protected CAIPE DeepEval REST API server on ${HOST}:${PORT}..."
echo "  - ALLOW_UNAUTHENTICATED_ACCESS: ${ALLOW_UNAUTHENTICATED_ACCESS}"
echo "  - CAIPE_UNSAFE_RBAC_BYPASS: ${CAIPE_UNSAFE_RBAC_BYPASS}"
echo "  - OIDC_ISSUER: ${OIDC_ISSUER}"
echo "  - OIDC_AUDIENCE: ${OIDC_AUDIENCE}"
if [ -n "$DEEPEVAL_API_KEY" ]; then
  echo "  - DEEPEVAL_API_KEY: [CONFIGURED - static API key validation enabled]"
else
  echo "  - DEEPEVAL_API_KEY: [NOT SET - provide DEEPEVAL_API_KEY in .env or environment]"
fi

# Run the protected API server via uv
uv run python3 -c "from deepeval_eval.api import run_server; run_server(host='${HOST}', port=${PORT})"
