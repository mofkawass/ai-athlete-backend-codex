# The AI Athlete V2 Architecture

## Goal
Build one mobile application for iOS and Android focused on tennis. The app records or selects a short tennis video, uploads it securely, tracks the player, measures movement, and returns a concise AI coaching report.

## V2 user flow
1. Player opens the iOS/Android app.
2. Player records or selects a 5-10 second tennis video.
3. App asks what the player wants to improve, starting with Forehand Swing, Footwork, or Preparation.
4. App requests a signed upload URL from the Railway API.
5. App uploads the video directly to Google Cloud Storage.
6. Railway creates an analysis job.
7. Computer vision extracts pose landmarks, tennis-specific movement metrics, and representative frames.
8. The tennis analysis engine identifies the hitting arm and estimates preparation, acceleration/contact proxy, and follow-through phases.
9. OpenAI receives structured tennis measurements plus selected frames and returns a structured coaching report.
10. Railway creates an annotated result video.
11. Original upload and temporary frames are deleted after processing.
12. Mobile app shows progress, the annotated video, strengths, top three priorities, and drills.

## Technology
- Mobile: React Native with Expo and TypeScript for iOS and Android.
- API: FastAPI on Railway.
- Video storage: Google Cloud Storage, temporary only.
- Computer vision: MediaPipe + OpenCV + ffmpeg.
- AI coaching: OpenAI Responses API with structured output.
- Source control: GitHub.

## Tennis MVP scope
The first production-quality analysis target is tennis forehand. We will expand only after the forehand analysis is useful and repeatable across real test videos.

Initial tennis measurements:
- hitting-arm identification
- knee flexion during preparation
- stance width
- shoulder rotation / unit turn
- elbow and wrist path
- wrist-speed proxy
- balance / body-centre movement
- preparation timing
- acceleration/contact proxy
- follow-through completion

Initial coaching focuses:
- Forehand Swing
- Footwork
- Preparation

## Design principles
- OpenAI is the tennis coaching/reasoning layer, not the raw pose measurement engine.
- CV metrics should be factual and reproducible.
- AI output is limited to the top three priorities so the player is not overwhelmed.
- Do not pretend a metric is known when the camera angle or landmark quality is insufficient.
- Never ship the OpenAI API key or GCP service-account key inside the mobile app.
- Videos are private and temporary by default.
- Source uploads are deleted immediately after successful processing, with bucket lifecycle rules as a backup.

## V2 phases
### Phase 1 - Mobile shell
- Camera recording.
- Gallery selection.
- Upload progress.
- Analysis progress.
- Result screen shell.
- Tennis focus selection.

### Phase 2 - Backend V2
- Signed uploads.
- Persistent job model.
- Processing states.
- Automatic source-video cleanup.
- Structured logs.

### Phase 3 - Tennis forehand analysis
- Hitting-arm identification.
- Stroke phase segmentation.
- Knee flexion, stance width, shoulder rotation, elbow/wrist path, balance, and timing metrics.
- Representative frame extraction.
- Confidence/quality checks for each metric.

### Phase 4 - OpenAI tennis coaching
- Feed tennis metrics and representative frames to OpenAI.
- Enforce a JSON schema for strengths, priorities, evidence, recommendations, and drills.
- Store model/version and confidence with each report.
- Cap feedback at three primary coaching priorities.

### Phase 5 - Tennis expansion
After forehand validation, add backhand, serve, return, and dedicated footwork analysis while reusing the same mobile, upload, job, storage, and coaching infrastructure.

## Current safety strategy
The existing `main` branch remains the working prototype. V2 work happens on `v2-foundation` until it passes testing.
