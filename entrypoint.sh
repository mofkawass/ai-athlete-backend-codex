#!/usr/bin/env sh
set -euo pipefail

if [ -z "${GCP_SERVICE_ACCOUNT_JSON_BASE64:-}" ]; then
  echo "ERROR: GCP_SERVICE_ACCOUNT_JSON_BASE64 is empty"; exit 1
fi

VAL="${GCP_SERVICE_ACCOUNT_JSON_BASE64}"

# Write /app/gcp.json from base64 or raw JSON
case "$VAL" in
  \{*)  printf "%s" "$VAL" > /app/gcp.json ;;
  *)    printf "%s" "$VAL" | tr -d '\n\r' | base64 -d > /app/gcp.json ;;
esac

# Make ADC discoverable to Google libs
export GOOGLE_APPLICATION_CREDENTIALS=/app/gcp.json

# Start API
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8080}"

