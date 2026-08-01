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
SET_ID="${SET_ID:-}"
PAGE="${PAGE:-1}"
LIMIT="${LIMIT:-50}"
CATEGORY="${CATEGORY:-}"
LEVEL="${LEVEL:-}"
QUERY="${QUERY:-}"

# Usage: ./scripts/list_question_sets.sh [set_id] [query] [page] [limit]
if [ -n "$1" ]; then
  SET_ID="$1"
fi
if [ -n "$2" ]; then
  QUERY="$2"
fi
if [ -n "$3" ]; then
  PAGE="$3"
fi
if [ -n "$4" ]; then
  LIMIT="$4"
fi

echo "=================================================================="
echo " Question Sets & Questions Listing Script"
echo "=================================================================="
echo "API Server: ${API_URL}"
if [ -n "$SET_ID" ]; then
  echo "Target:     Listing Questions in Set ID=${SET_ID}"
else
  echo "Target:     Listing Question Sets"
fi
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

if [ -z "$SET_ID" ]; then
  # List Question Sets
  REQ_URL="${API_URL}/api/v1/question-sets?page=${PAGE}&limit=${LIMIT}"
  if [ -n "$QUERY" ]; then
    REQ_URL="${REQ_URL}&query=${QUERY}"
  fi

  echo "Fetching question sets from: ${REQ_URL}"
  RESPONSE=$(curl -sS --fail-with-body "${AUTH_HEADER[@]}" -X GET "$REQ_URL")

  echo ""
  echo "Question Sets Result:"
  echo "$RESPONSE" | jq . 2>/dev/null || echo "$RESPONSE"

else
  # List Questions in a specific Question Set
  REQ_URL="${API_URL}/api/v1/question-sets/${SET_ID}/questions?page=${PAGE}&limit=${LIMIT}"
  if [ -n "$QUERY" ]; then
    REQ_URL="${REQ_URL}&query=${QUERY}"
  fi
  if [ -n "$CATEGORY" ]; then
    REQ_URL="${REQ_URL}&category=${CATEGORY}"
  fi
  if [ -n "$LEVEL" ]; then
    REQ_URL="${REQ_URL}&level=${LEVEL}"
  fi

  echo "Fetching questions in set ${SET_ID} from: ${REQ_URL}"
  RESPONSE=$(curl -sS --fail-with-body "${AUTH_HEADER[@]}" -X GET "$REQ_URL")

  echo ""
  echo "Questions Result (Set ID: ${SET_ID}):"
  echo "$RESPONSE" | jq . 2>/dev/null || echo "$RESPONSE"
fi
