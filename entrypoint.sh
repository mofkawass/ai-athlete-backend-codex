#!/bin/sh
set -e

# Accept either base64 or raw JSON in GCP_SERVICE_ACCOUNT_JSON_BASE64
if [ -n "$GCP_SERVICE_ACCOUNT_JSON_BASE64" ]; then
  if echo "$GCP_SERVICE_ACCOUNT_JSON_BASE64" | head -c 1 | grep -q "{" ; then
    printf '%s' "$GCP_SERVICE_ACCOUNT_JSON_BASE64" > /app/gcp.json
  else
    CLEAN=$(printf '%s' "$GCP_SERVICE_ACCOUNT_JSON_BASE64" | tr -d '\r\n ')
    echo "$CLEAN" | base64 -d > /app/gcp.json
  fi
fi

# Start API
exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}
