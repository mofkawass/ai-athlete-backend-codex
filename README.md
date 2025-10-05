# AI Athlete Backend (Codex path)

FastAPI service that generates signed GCS upload URLs, processes short videos with MediaPipe pose, overlays landmarks, and returns coaching tips.

## Env Vars
- `PORT=8080`
- `GCS_BUCKET=<your-bucket>`
- `GOOGLE_APPLICATION_CREDENTIALS=/app/gcp.json`
- `GCP_SERVICE_ACCOUNT_JSON_BASE64` = base64 of the service account JSON **or** the raw JSON (single line)
- `COACH_WEBHOOK_TOKEN` (optional)

## Deploy on Railway
1. Create a new repo and upload these files.
2. In Railway → Deploy from repo.
3. Variables: set the env vars above.
4. Open `/health` and `/test` on your Railway URL.
