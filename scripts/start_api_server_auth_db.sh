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

# KUBECONFIG fallback for Kubernetes secret retrieval
KUBECONFIG_ARG=""
if [ -n "$KUBECONFIG" ] && [ -f "$KUBECONFIG" ]; then
  KUBECONFIG_ARG="--kubeconfig=$KUBECONFIG"
fi

# Retrieve and set DEEPEVAL_API_KEY from Kubernetes secret if not already set
if [ -z "$DEEPEVAL_API_KEY" ] && command -v kubectl >/dev/null 2>&1; then
  if kubectl $KUBECONFIG_ARG get secret caipe-ui-secret -n caipe >/dev/null 2>&1; then
    FETCHED_KEY=$(kubectl $KUBECONFIG_ARG get secret caipe-ui-secret -n caipe -o jsonpath='{.data.AGENTGATEWAY_TARGETS_TOKEN}' | base64 --decode 2>/dev/null || true)
    if [ -z "$FETCHED_KEY" ]; then
      FETCHED_KEY=$(kubectl $KUBECONFIG_ARG get secret caipe-ui-secret -n caipe -o jsonpath='{.data.NEXTAUTH_SECRET}' | base64 --decode 2>/dev/null || true)
    fi
    if [ -n "$FETCHED_KEY" ]; then
      export DEEPEVAL_API_KEY="$FETCHED_KEY"
      echo "DEEPEVAL_API_KEY retrieved successfully from Kubernetes secret 'caipe-ui-secret'."
    fi
  fi
fi

# Retrieve PostgreSQL Admin Password from Kubernetes secret if not set
if [ -z "$POSTGRES_PASSWORD" ] && command -v kubectl >/dev/null 2>&1; then
  if kubectl $KUBECONFIG_ARG get secret caipe-postgres-credentials -n caipe >/dev/null 2>&1; then
    FETCHED_PG_PASS=$(kubectl $KUBECONFIG_ARG get secret caipe-postgres-credentials -n caipe -o jsonpath='{.data.POSTGRES_ADMIN_PASSWORD}' | base64 --decode 2>/dev/null || true)
    if [ -n "$FETCHED_PG_PASS" ]; then
      export POSTGRES_PASSWORD="$FETCHED_PG_PASS"
      echo "POSTGRES_PASSWORD retrieved successfully from Kubernetes secret 'caipe-postgres-credentials'."
    fi
  fi
fi

# PostgreSQL Database Configuration Defaults (NodePort 30543 on K3s node 192.168.8.132)
export POSTGRES_HOST="${POSTGRES_HOST:-192.168.8.132}"
export POSTGRES_PORT="${POSTGRES_PORT:-30543}"
export POSTGRES_DB="${POSTGRES_DB:-caipe_eval}"
export POSTGRES_USER="${POSTGRES_USER:-postgres}"

# Construct DATABASE_URL if not explicitly set
if [ -z "$DATABASE_URL" ] && [ -n "$POSTGRES_PASSWORD" ]; then
  export DATABASE_URL="postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_DB}"
fi

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

echo "Starting Auth-Protected & DB-Connected CAIPE DeepEval REST API server on ${HOST}:${PORT}..."
echo "  - ALLOW_UNAUTHENTICATED_ACCESS: ${ALLOW_UNAUTHENTICATED_ACCESS}"
echo "  - CAIPE_UNSAFE_RBAC_BYPASS: ${CAIPE_UNSAFE_RBAC_BYPASS}"
echo "  - OIDC_ISSUER: ${OIDC_ISSUER}"
echo "  - OIDC_AUDIENCE: ${OIDC_AUDIENCE}"
if [ -n "$DEEPEVAL_API_KEY" ]; then
  echo "  - DEEPEVAL_API_KEY: [CONFIGURED - static API key validation enabled]"
else
  echo "  - DEEPEVAL_API_KEY: [NOT SET - provide DEEPEVAL_API_KEY in .env or environment]"
fi
if [ -n "$DATABASE_URL" ]; then
  echo "  - DATABASE_URL: postgresql://${POSTGRES_USER}:****@${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_DB} [CONFIGURED]"
else
  echo "  - DATABASE_URL: [NOT SET - DB persistence disabled or credentials missing]"
fi

# Run the protected API server via uv
uv run python3 -c "from deepeval_eval.api import run_server; run_server(host='${HOST}', port=${PORT})"
