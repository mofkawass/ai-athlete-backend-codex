# app/pose_overlay.py
import os
import math
import cv2
import tempfile
from typing import Dict, Any, Optional
from google.cloud import storage
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

def _get_video_meta(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video")
    fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, w, h, n

def _estimate_simple_metrics(pose_landmarks_list):
    """Very lightweight summary: average left knee bend angle, count of frames with arm extension."""
    if not pose_landmarks_list:
        return {"avg_knee_angle_left": None, "arm_extension_frames": 0}

    pose = mp_pose.PoseLandmark
    angles = []
    arm_ext_frames = 0
    for lm in pose_landmarks_list:
        LHIP  = lm[pose.LEFT_HIP]
        LKNEE = lm[pose.LEFT_KNEE]
        LANK  = lm[pose.LEFT_ANKLE]
        LSH   = lm[pose.LEFT_SHOULDER]
        LELB  = lm[pose.LEFT_ELBOW]
        LWR   = lm[pose.LEFT_WRIST]

        # Knee angle (hip-knee-ankle)
        ang = _angle((LHIP.x, LHIP.y), (LKNEE.x, LKNEE.y), (LANK.x, LANK.y))
        if not math.isnan(ang):
            angles.append(ang)

        # Arm extension proxy (shoulder–elbow horizontal reach)
        horiz = abs(LELB.x - LSH.x)
        arm_ext_frames += 1 if horiz > 0.15 else 0

    avg_knee = sum(angles) / len(angles) if angles else None
    return {"avg_knee_angle_left": round(avg_knee, 1) if avg_knee else None,
            "arm_extension_frames": arm_ext_frames}

def _angle(a, b, c):
    """Angle at b in degrees (using 2D normalized coords)."""
    ax, ay = a; bx, by = b; cx, cy = c
    v1 = (ax - bx, ay - by)
    v2 = (cx - bx, cy - by)
    n1 = math.hypot(*v1) or 1e-6
    n2 = math.hypot(*v2) or 1e-6
    dot = (v1[0]*v2[0] + v1[1]*v2[1]) / (n1*n2)
    dot = max(min(dot, 1.0), -1.0)
    return math.degrees(math.acos(dot))

def _guess_sport_from_pose_series(pose_landmarks_list) -> str:
    """Tiny heuristic: tennis-like if many frames with sideways arm reach + ~90° elbow angle."""
    if not pose_landmarks_list:
        return "unknown"
    pose = mp_pose.PoseLandmark
    hits = 0
    total = 0
    for lm in pose_landmarks_list:
        total += 1
        LSH = lm[pose.LEFT_SHOULDER]
        LEL = lm[pose.LEFT_ELBOW]
        LWR = lm[pose.LEFT_WRIST]
        arm_len = math.hypot(LEL.x - LSH.x, LEL.y - LSH.y)
        horiz   = abs(LEL.x - LSH.x)
        ang     = _angle((LSH.x, LSH.y), (LEL.x, LEL.y), (LWR.x, LWR.y))
        if arm_len > 0.20 and horiz > 0.15 and 60 < ang < 120:
            hits += 1
    return "tennis" if hits >= max(6, int(0.2 * total)) else "unknown"

def process_video_and_overlay(
    input_path: str,
    output_blob_path: str,
    bucket: storage.Bucket,
    provided_sport: Optional[str] = None
) -> Dict[str, Any]:
    """
    Reads input video, draws pose overlay, uploads annotated MP4 to GCS (output_blob_path),
    returns a small analysis payload.
    """
    fps, w, h, _ = _get_video_meta(input_path)

    # Prepare writer (mp4)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_out:
        out_path = tmp_out.name

    cap = cv2.VideoCapture(input_path)
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    pose_results_series = []
    with mp_pose.Pose(static_image_mode=False, model_complexity=1) as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            # Draw overlay
            if res.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    res.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
                )
                pose_results_series.append(res.pose_landmarks.landmark)

            writer.write(frame)

    cap.release()
    writer.release()

    # Upload to GCS
    out_blob = bucket.blob(output_blob_path)
    out_blob.upload_from_filename(out_path, content_type="video/mp4")
    try:
        os.remove(out_path)
    except Exception:
        pass

    # Simple analytics
    sport = provided_sport or _guess_sport_from_pose_series(pose_results_series)
    metrics = _estimate_simple_metrics(pose_results_series)

    summary = f"Detected sport: {sport}. " \
              f"Avg left-knee angle: {metrics.get('avg_knee_angle_left')}°. " \
              f"Frames with arm extension: {metrics.get('arm_extension_frames')}."

    drills = []
    if sport == "tennis":
        drills = [
            "Shadow swings with high elbow finish (10 reps).",
            "Split-step before feed, focus on balance (3 sets of 10).",
            "Brush up on ball for topspin (mini-court, 5 minutes)."
        ]
    else:
        drills = ["Light mobility + controlled repetitions to build consistent form."]

    return {
        "sport": sport,
        "metrics": metrics,
        "summary": summary,
        "drills": drills
    }
