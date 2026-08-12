# The AI Athlete V2 Architecture

## Goal
Build one mobile application for iOS and Android that records or selects a short sports video, uploads it securely, tracks the athlete, measures movement, and returns a concise AI coaching report.

## V2 user flow
1. Athlete opens the iOS/Android app.
2. Athlete records or selects a 5-10 second video.
3. Athlete chooses Auto Detect, Tennis, Soccer, or Running.
4. App asks what the athlete wants to improve.
5. App requests a signed upload URL from the Railway API.
6. App uploads the video directly to Google Cloud Storage.
7. Railway creates an analysis job.
8. Computer vision extracts pose landmarks, movement metrics, and representative frames.
9. OpenAI receives structured measurements plus selected frames and returns a structured coaching report.
10. Railway creates an annotated result video.
11. Original upload and temporary frames are deleted after processing.
12. Mobile app shows progress, the annotated video, strengths, top priorities, and drills.

## Technology
- Mobile: React Native with Expo, TypeScript, expo-camera, expo-video, expo-file-system.
- API: FastAPI on Railway.
- Video storage: Google Cloud Storage, temporary only.
- Computer vision: MediaPipe + OpenCV + ffmpeg.
- AI coaching: OpenAI Responses API with structured output.
- Source control: GitHub.

## Design principles
- OpenAI is the coaching/reasoning layer, not the raw pose measurement engine.
- CV metrics should be factual and reproducible.
- AI output is limited to the top three priorities so the athlete is not overwhelmed.
- Users can override sport detection.
- Never ship the OpenAI API key or GCP service-account key inside the mobile app.
- Videos are private and temporary by default.

## V2 phases
### Phase 1 - Mobile shell
- Camera recording.
- Gallery selection.
- Upload progress.
- Analysis progress.
- Result screen shell.

### Phase 2 - Backend V2
- Signed uploads.
- Persistent job model.
- Processing states.
- Automatic cleanup.
- Structured logs.

### Phase 3 - Tennis analysis
- Stroke identification.
- Hitting-arm identification.
- Preparation, acceleration/contact proxy, and follow-through phases.
- Knee flexion, stance width, shoulder rotation, elbow/wrist path, balance, and timing metrics.

### Phase 4 - OpenAI coaching
- Feed metrics and representative frames to OpenAI.
- Enforce a JSON schema for strengths, priorities, evidence, recommendations, and drills.
- Store model/version and confidence with each report.

### Phase 5 - Soccer and Running
Add sport-specific metric engines while reusing upload, job, UI, OpenAI, and storage infrastructure.

## Current safety strategy
The existing `main` branch remains the working prototype. V2 work happens on `v2-foundation` until it passes testing.
