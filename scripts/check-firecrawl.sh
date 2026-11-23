#!/usr/bin/env bash
set -euo pipefail

# Official Firecrawl Local health verification script
# Probes GET /health per https://github.com/firecrawl/firecrawl#firecrawl-local

API_URL="${FIRECRAWL_API_URL:-http://127.0.0.1:3002}"
API_URL="${API_URL%/}"

echo "Checking Firecrawl Local at ${API_URL}/health ..."

CURL_ARGS=(-fsS --max-time 5)
if [ -n "${FIRECRAWL_LOCAL_TOKEN:-}" ]; then
  CURL_ARGS+=(-H "Authorization: Bearer ${FIRECRAWL_LOCAL_TOKEN}")
fi

if BODY=$(curl "${CURL_ARGS[@]}" "${API_URL}/health" 2>/dev/null); then
  echo "OK: ${API_URL}/health -> ${BODY}"
  exit 0
fi

echo "ERROR: Firecrawl Local is not reachable at ${API_URL}/health" >&2
exit 1
