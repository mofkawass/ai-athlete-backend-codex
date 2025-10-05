#!/usr/bin/env sh
# POSIX-compatible entrypoint for Railway / FastAPI
# Ensures stable runtime for GCP service key + uvicorn app

set -eu

echo ">>> Starting AI Athlete Backend (Codex Edition)..."

# --- Validate key ---
if [ -z "${GCP_SERVICE_ACCOUNT_JSON_BASE64:-}" ]; then
  echo "❌ ERROR: GCP_SERVICE_ACCOUNT_JSON_BASE64 is missing."
  exit 1
fi

# --- Decode or write raw JSON ---
VAL="${GCP_SERVICE_ACCOUNT_JSON_BASE64}"
if echo "$VAL" | grep -q "{"; then
  echo "Detected raw JSON key, writing /app/gcp.json"
  printf "%s" "$VAL" > /app/gcp.json
else
  echo "Detected base64 key, decoding to /app/gcp.json"
  printf "%s" "$VAL" | tr -d '\n\r' | base64 -d > /app/gcp.json
fi

# --- Export for Google SDKs ---
export GOOGLE_APPLICATION_CREDENTIALS=/app/gcp.json
echo "✅ GOOGLE_APPLICATION_CREDENTIALS set"

# --- Start FastAPI via Uvicorn ---
echo "🚀 Launching FastAPI on port ${PORT:-8080}"
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8080}"


