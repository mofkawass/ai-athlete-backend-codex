# app/sport_detect.py
import math
import cv2
import tempfile
from google.cloud import storage
import mediapipe as mp

mp_pose = mp.solutions.pose

def _angle(p1, p2, p3):
    a = math.dist((p1.x, p1.y), (p2.x, p2.y))
    b = math.dist((p2.x, p2.y), (p3.x, p3.y))
    c = math.dist((p1.x, p1.y), (p3.x, p3.y))
    denom = max(2 * a * b, 1e-6)
    cosv = max(min((a*a + b*b - c*c) / denom, 1.0), -1.0)
    return math.degrees(math.acos(cosv))

def detect_sport_from_gcs(storage_client: storage.Client, bucket_name: str, object_path: str) -> str:
    """Heuristic: identify tennis forehand-like motion."""
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(object_path)

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        blob.download_to_filename(tmp.name)
        video_path = tmp.name

    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)
    cap = cv2.VideoCapture(video_path)
    tennis_score = 0
    max_frames = 120  # ~4s @ 30fps

    frames = 0
    while cap.isOpened() and frames < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
            elbow    = lm[mp_pose.PoseLandmark.LEFT_ELBOW]
            wrist    = lm[mp_pose.PoseLandmark.LEFT_WRIST]

            arm_len = math.dist((elbow.x, elbow.y), (shoulder.x, shoulder.y))
            horiz   = abs(elbow.x - shoulder.x)
            ang     = _angle(shoulder, elbow, wrist)

            if arm_len > 0.20 and horiz > 0.15 and 60 < ang < 120:
                tennis_score += 1
        frames += 1

    cap.release()
    pose.close()

    # Tune threshold if needed
    return "tennis" if tennis_score >= max(6, int(frames * 0.2)) else "unknown"
