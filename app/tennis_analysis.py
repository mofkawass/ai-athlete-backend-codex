import math
from typing import Any, Dict, List, Tuple

import cv2
import mediapipe as mp
import numpy as np


def _angle(a, b, c) -> float:
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return 0.0
    cosine = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _xy(landmarks, idx) -> Tuple[float, float]:
    lm = landmarks[idx]
    return (float(lm.x), float(lm.y))


def analyze_tennis_forehand(video_path: str) -> Dict[str, Any]:
    """Extract conservative 2D pose metrics for an MVP tennis forehand analysis.

    These values are measurements/proxies, not claims about professional technique.
    The coaching layer should use them as evidence and communicate uncertainty.
    """
    pose_api = mp.solutions.pose
    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    sample_every = max(1, int(round(fps / 15.0)))

    frames: List[Dict[str, float]] = []
    wrist_history: List[Tuple[int, float, float]] = []

    with pose_api.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        frame_no = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_no % sample_every != 0:
                frame_no += 1
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)
            if not result.pose_landmarks:
                frame_no += 1
                continue

            lm = result.pose_landmarks.landmark
            LS, RS = pose_api.PoseLandmark.LEFT_SHOULDER.value, pose_api.PoseLandmark.RIGHT_SHOULDER.value
            LH, RH = pose_api.PoseLandmark.LEFT_HIP.value, pose_api.PoseLandmark.RIGHT_HIP.value
            LK, RK = pose_api.PoseLandmark.LEFT_KNEE.value, pose_api.PoseLandmark.RIGHT_KNEE.value
            LA, RA = pose_api.PoseLandmark.LEFT_ANKLE.value, pose_api.PoseLandmark.RIGHT_ANKLE.value
            LE, RE = pose_api.PoseLandmark.LEFT_ELBOW.value, pose_api.PoseLandmark.RIGHT_ELBOW.value
            LW, RW = pose_api.PoseLandmark.LEFT_WRIST.value, pose_api.PoseLandmark.RIGHT_WRIST.value

            left_knee = _angle(_xy(lm, LH), _xy(lm, LK), _xy(lm, LA))
            right_knee = _angle(_xy(lm, RH), _xy(lm, RK), _xy(lm, RA))
            left_elbow = _angle(_xy(lm, LS), _xy(lm, LE), _xy(lm, LW))
            right_elbow = _angle(_xy(lm, RS), _xy(lm, RE), _xy(lm, RW))

            shoulder_width = abs(lm[RS].x - lm[LS].x)
            ankle_width = abs(lm[RA].x - lm[LA].x)
            stance_ratio = ankle_width / shoulder_width if shoulder_width > 0.02 else 0.0

            shoulder_line_deg = math.degrees(math.atan2(lm[RS].y - lm[LS].y, lm[RS].x - lm[LS].x))
            hip_line_deg = math.degrees(math.atan2(lm[RH].y - lm[LH].y, lm[RH].x - lm[LH].x))
            torso_separation_proxy = abs(shoulder_line_deg - hip_line_deg)

            # Pick the wrist moving farther from its same-side shoulder as a simple hitting-arm proxy.
            left_reach = math.dist(_xy(lm, LS), _xy(lm, LW))
            right_reach = math.dist(_xy(lm, RS), _xy(lm, RW))
            wrist_idx = LW if left_reach >= right_reach else RW
            wrist_history.append((frame_no, float(lm[wrist_idx].x), float(lm[wrist_idx].y)))

            frames.append({
                "frame": float(frame_no),
                "left_knee_angle_deg": left_knee,
                "right_knee_angle_deg": right_knee,
                "left_elbow_angle_deg": left_elbow,
                "right_elbow_angle_deg": right_elbow,
                "stance_width_ratio": float(stance_ratio),
                "torso_separation_proxy_deg": float(torso_separation_proxy),
            })
            frame_no += 1

    cap.release()

    if not frames:
        return {
            "quality": {"pose_frames": 0, "usable": False, "reason": "No reliable pose landmarks found."},
            "metrics": {},
            "key_frames": [],
        }

    speeds: List[Tuple[int, float]] = []
    for (f1, x1, y1), (f2, x2, y2) in zip(wrist_history, wrist_history[1:]):
        dt = max((f2 - f1) / fps, 1e-6)
        speeds.append((f2, math.hypot(x2 - x1, y2 - y1) / dt))
    peak_frame = max(speeds, key=lambda item: item[1])[0] if speeds else int(frames[len(frames) // 2]["frame"])
    peak_speed = max((s for _, s in speeds), default=0.0)

    def vals(key: str) -> List[float]:
        return [float(f[key]) for f in frames]

    metrics = {
        "knee_flexion_min_deg": round(min(vals("left_knee_angle_deg") + vals("right_knee_angle_deg")), 1),
        "stance_width_ratio_median": round(float(np.median(vals("stance_width_ratio"))), 2),
        "torso_separation_proxy_max_deg": round(max(vals("torso_separation_proxy_deg")), 1),
        "left_elbow_angle_range_deg": round(max(vals("left_elbow_angle_deg")) - min(vals("left_elbow_angle_deg")), 1),
        "right_elbow_angle_range_deg": round(max(vals("right_elbow_angle_deg")) - min(vals("right_elbow_angle_deg")), 1),
        "wrist_speed_proxy_peak_per_s": round(float(peak_speed), 3),
    }

    key_frames = sorted(set([
        int(frames[0]["frame"]),
        max(0, int(peak_frame - fps * 0.4)),
        int(peak_frame),
        int(peak_frame + fps * 0.4),
        int(frames[-1]["frame"]),
    ]))

    coverage = len(frames) / max(1, math.ceil(max(total_frames, 1) / sample_every))
    return {
        "quality": {
            "pose_frames": len(frames),
            "coverage": round(min(1.0, coverage), 2),
            "usable": len(frames) >= 5,
        },
        "metrics": metrics,
        "key_frames": key_frames,
        "movement": "forehand",
        "sport": "tennis",
    }
