import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


VISIBILITY_MIN = 0.55


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


def _visible(landmarks, indexes: List[int], threshold: float = VISIBILITY_MIN) -> bool:
    return all(float(getattr(landmarks[idx], "visibility", 1.0)) >= threshold for idx in indexes)


def _line_angle_deg(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))


def _wrapped_angle_diff_deg(a: float, b: float) -> float:
    """Smallest absolute angle difference in the range 0..180 degrees."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _median(values: List[float]) -> Optional[float]:
    return float(np.median(values)) if values else None


def _percentile(values: List[float], percentile: float) -> Optional[float]:
    return float(np.percentile(values, percentile)) if values else None


def _rounded(value: Optional[float], digits: int = 1) -> Optional[float]:
    return round(float(value), digits) if value is not None and math.isfinite(value) else None


def _metric_quality(status: str, reason: str, confidence: str = "medium") -> Dict[str, str]:
    return {"status": status, "confidence": confidence, "reason": reason}


def analyze_tennis_forehand(video_path: str) -> Dict[str, Any]:
    """Extract conservative, phase-aware 2D pose metrics for a tennis forehand MVP.

    The output deliberately distinguishes measured values from low-confidence proxies.
    It does not measure the ball, racket, spin, impact, or true 3D torso rotation.
    """
    pose_api = mp.solutions.pose
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            "quality": {"pose_frames": 0, "usable": False, "reason": "Could not open video."},
            "metrics": {},
            "metric_quality": {},
            "key_frames": [],
        }

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    sample_every = max(1, int(round(fps / 15.0)))

    frames: List[Dict[str, Any]] = []

    with pose_api.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
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

            shoulder_pair_visible = _visible(lm, [LS, RS])
            hip_pair_visible = _visible(lm, [LH, RH])
            left_leg_visible = _visible(lm, [LH, LK, LA])
            right_leg_visible = _visible(lm, [RH, RK, RA])
            left_arm_visible = _visible(lm, [LS, LE, LW])
            right_arm_visible = _visible(lm, [RS, RE, RW])
            ankles_visible = _visible(lm, [LA, RA])

            shoulder_mid = None
            hip_mid = None
            torso_length = None
            shoulder_width = None
            projected_shoulder_ratio = None
            if shoulder_pair_visible:
                left_shoulder = _xy(lm, LS)
                right_shoulder = _xy(lm, RS)
                shoulder_mid = ((left_shoulder[0] + right_shoulder[0]) / 2.0, (left_shoulder[1] + right_shoulder[1]) / 2.0)
                shoulder_width = math.dist(left_shoulder, right_shoulder)
            if hip_pair_visible:
                left_hip = _xy(lm, LH)
                right_hip = _xy(lm, RH)
                hip_mid = ((left_hip[0] + right_hip[0]) / 2.0, (left_hip[1] + right_hip[1]) / 2.0)
            if shoulder_mid and hip_mid:
                torso_length = math.dist(shoulder_mid, hip_mid)
                if shoulder_width is not None and torso_length > 0.02:
                    projected_shoulder_ratio = shoulder_width / torso_length

            left_knee = _angle(_xy(lm, LH), _xy(lm, LK), _xy(lm, LA)) if left_leg_visible else None
            right_knee = _angle(_xy(lm, RH), _xy(lm, RK), _xy(lm, RA)) if right_leg_visible else None
            left_elbow = _angle(_xy(lm, LS), _xy(lm, LE), _xy(lm, LW)) if left_arm_visible else None
            right_elbow = _angle(_xy(lm, RS), _xy(lm, RE), _xy(lm, RW)) if right_arm_visible else None

            stance_ratio = None
            if ankles_visible and shoulder_pair_visible and shoulder_width and shoulder_width > 0.03:
                ankle_width = math.dist(_xy(lm, LA), _xy(lm, RA))
                candidate = ankle_width / shoulder_width
                if 0.15 <= candidate <= 4.0:
                    stance_ratio = float(candidate)

            shoulder_angle = None
            hip_angle = None
            line_separation = None
            if shoulder_pair_visible:
                shoulder_angle = _line_angle_deg(_xy(lm, LS), _xy(lm, RS))
            if hip_pair_visible:
                hip_angle = _line_angle_deg(_xy(lm, LH), _xy(lm, RH))
            if shoulder_angle is not None and hip_angle is not None:
                line_separation = _wrapped_angle_diff_deg(shoulder_angle, hip_angle)

            left_wrist = _xy(lm, LW) if left_arm_visible else None
            right_wrist = _xy(lm, RW) if right_arm_visible else None
            left_shoulder = _xy(lm, LS) if left_arm_visible else None
            right_shoulder = _xy(lm, RS) if right_arm_visible else None

            frames.append({
                "frame": int(frame_no),
                "left_knee_angle_deg": left_knee,
                "right_knee_angle_deg": right_knee,
                "left_elbow_angle_deg": left_elbow,
                "right_elbow_angle_deg": right_elbow,
                "stance_width_ratio": stance_ratio,
                "shoulder_line_angle_2d_deg": shoulder_angle,
                "shoulder_hip_line_separation_2d_deg": line_separation,
                "projected_shoulder_to_torso_ratio": projected_shoulder_ratio,
                "torso_length": torso_length,
                "left_wrist": left_wrist,
                "right_wrist": right_wrist,
                "left_shoulder": left_shoulder,
                "right_shoulder": right_shoulder,
            })
            frame_no += 1

    cap.release()

    expected_samples = max(1, math.ceil(max(total_frames, 1) / sample_every))
    coverage = len(frames) / expected_samples
    if len(frames) < 5 or coverage < 0.30:
        return {
            "quality": {
                "pose_frames": len(frames),
                "coverage": round(min(1.0, coverage), 2),
                "usable": False,
                "reason": "Not enough reliable full-body pose coverage.",
            },
            "metrics": {},
            "metric_quality": {},
            "key_frames": [],
        }

    projected_ratios = [f["projected_shoulder_to_torso_ratio"] for f in frames if f["projected_shoulder_to_torso_ratio"] is not None]
    projected_ratio_median = _median(projected_ratios)
    if projected_ratio_median is None:
        camera_view = "unknown"
    elif projected_ratio_median < 0.35:
        camera_view = "side"
    elif projected_ratio_median < 0.70:
        camera_view = "diagonal"
    else:
        camera_view = "front_or_back"

    def arm_path(side: str) -> Tuple[float, int]:
        points = []
        for frame in frames:
            wrist = frame[f"{side}_wrist"]
            torso_length = frame["torso_length"]
            if wrist is not None and torso_length is not None and torso_length > 0.03:
                points.append((frame["frame"], wrist, torso_length))
        path = 0.0
        segments = 0
        for (_, p1, t1), (_, p2, t2) in zip(points, points[1:]):
            scale = max((t1 + t2) / 2.0, 0.03)
            path += math.dist(p1, p2) / scale
            segments += 1
        return path, segments

    left_path, left_segments = arm_path("left")
    right_path, right_segments = arm_path("right")
    total_path = left_path + right_path
    selected_arm = "left" if left_path >= right_path else "right"
    arm_confidence = max(left_path, right_path) / total_path if total_path > 0 else 0.0
    hitting_arm = selected_arm if arm_confidence >= 0.58 else "uncertain"

    wrist_points = []
    for frame in frames:
        wrist = frame[f"{selected_arm}_wrist"]
        torso_length = frame["torso_length"]
        if wrist is not None and torso_length is not None and torso_length > 0.03:
            wrist_points.append((frame["frame"], wrist, torso_length))

    raw_speeds: List[Tuple[int, float]] = []
    for (f1, p1, t1), (f2, p2, t2) in zip(wrist_points, wrist_points[1:]):
        dt = max((f2 - f1) / fps, 1e-6)
        scale = max((t1 + t2) / 2.0, 0.03)
        raw_speeds.append((f2, math.dist(p1, p2) / scale / dt))

    smoothed_speeds: List[Tuple[int, float]] = []
    for index, (frame_no, _) in enumerate(raw_speeds):
        start = max(0, index - 1)
        end = min(len(raw_speeds), index + 2)
        smoothed_speeds.append((frame_no, float(np.median([speed for _, speed in raw_speeds[start:end]]))))

    if smoothed_speeds:
        peak_frame, peak_speed = max(smoothed_speeds, key=lambda item: item[1])
        p95_speed = _percentile([speed for _, speed in smoothed_speeds], 95)
    else:
        peak_frame = int(frames[len(frames) // 2]["frame"])
        peak_speed = None
        p95_speed = None

    prep_start = peak_frame - int(0.9 * fps)
    prep_end = peak_frame - int(0.15 * fps)
    contact_start = peak_frame - int(0.12 * fps)
    contact_end = peak_frame + int(0.12 * fps)
    follow_start = peak_frame + int(0.15 * fps)
    follow_end = peak_frame + int(0.9 * fps)

    prep_frames = [f for f in frames if prep_start <= f["frame"] <= prep_end]
    if len(prep_frames) < 3:
        prep_frames = [f for f in frames if f["frame"] < peak_frame]
    contact_frames = [f for f in frames if contact_start <= f["frame"] <= contact_end]
    follow_frames = [f for f in frames if follow_start <= f["frame"] <= follow_end]

    def phase_values(phase_frames: List[Dict[str, Any]], key: str) -> List[float]:
        return [float(f[key]) for f in phase_frames if f.get(key) is not None]

    prep_knees = phase_values(prep_frames, "left_knee_angle_deg") + phase_values(prep_frames, "right_knee_angle_deg")
    contact_knees = phase_values(contact_frames, "left_knee_angle_deg") + phase_values(contact_frames, "right_knee_angle_deg")
    prep_stance = phase_values(prep_frames, "stance_width_ratio")
    line_separations = phase_values(frames, "shoulder_hip_line_separation_2d_deg")
    shoulder_angles = phase_values(frames, "shoulder_line_angle_2d_deg")

    selected_elbow_key = f"{selected_arm}_elbow_angle_deg"
    prep_elbow = phase_values(prep_frames, selected_elbow_key)
    contact_elbow = phase_values(contact_frames, selected_elbow_key)
    follow_elbow = phase_values(follow_frames, selected_elbow_key)

    follow_wrist_height = []
    for frame in follow_frames:
        wrist = frame[f"{selected_arm}_wrist"]
        shoulder = frame[f"{selected_arm}_shoulder"]
        torso_length = frame["torso_length"]
        if wrist and shoulder and torso_length and torso_length > 0.03:
            follow_wrist_height.append((wrist[1] - shoulder[1]) / torso_length)

    if shoulder_angles:
        unwrapped = np.unwrap(np.radians(shoulder_angles))
        shoulder_rotation_range = float(np.degrees(np.max(unwrapped) - np.min(unwrapped)))
    else:
        shoulder_rotation_range = None

    stance_value = _median(prep_stance)
    stance_available = camera_view in {"diagonal", "front_or_back"} and len(prep_stance) >= 3 and stance_value is not None
    arm_metrics_available = arm_confidence >= 0.58 and max(left_segments, right_segments) >= 4

    metrics = {
        "knee_angle_preparation_median_deg": _rounded(_median(prep_knees), 1),
        "knee_angle_contact_proxy_median_deg": _rounded(_median(contact_knees), 1),
        "stance_width_ratio_preparation_median": _rounded(stance_value if stance_available else None, 2),
        "shoulder_hip_line_separation_2d_p90_deg": _rounded(_percentile(line_separations, 90), 1),
        "shoulder_line_rotation_range_2d_deg": _rounded(shoulder_rotation_range, 1),
        "hitting_arm_elbow_preparation_median_deg": _rounded(_median(prep_elbow) if arm_metrics_available else None, 1),
        "hitting_arm_elbow_contact_proxy_median_deg": _rounded(_median(contact_elbow) if arm_metrics_available else None, 1),
        "hitting_arm_elbow_follow_through_median_deg": _rounded(_median(follow_elbow) if arm_metrics_available else None, 1),
        "wrist_speed_proxy_p95_body_lengths_per_s": _rounded(p95_speed if arm_metrics_available else None, 2),
        "follow_through_wrist_height_relative_shoulder": _rounded(_median(follow_wrist_height) if arm_metrics_available else None, 2),
    }

    metric_quality = {
        "knee_angle_preparation_median_deg": _metric_quality(
            "valid" if len(prep_knees) >= 4 else "low_confidence",
            "Median of visible left/right knee angles during the preparation window.",
            "high" if len(prep_knees) >= 8 else "medium",
        ),
        "knee_angle_contact_proxy_median_deg": _metric_quality(
            "valid" if len(contact_knees) >= 2 else "low_confidence",
            "Median knee angle around peak 2D wrist-speed frame; this is a contact proxy, not ball contact.",
            "medium",
        ),
        "stance_width_ratio_preparation_median": _metric_quality(
            "valid" if stance_available else "unavailable",
            "Visible ankle distance divided by visible shoulder width during preparation. Suppressed for side views."
            if stance_available else f"Not reliable from the detected {camera_view} camera view or ankle visibility.",
            "medium" if stance_available else "low",
        ),
        "shoulder_hip_line_separation_2d_p90_deg": _metric_quality(
            "low_confidence" if line_separations else "unavailable",
            "Screen-plane shoulder/hip line separation only; it is not true 3D torso rotation.",
            "low",
        ),
        "shoulder_line_rotation_range_2d_deg": _metric_quality(
            "low_confidence" if shoulder_angles else "unavailable",
            "Range of the shoulder-line angle in the image plane; camera perspective strongly affects it.",
            "low",
        ),
        "hitting_arm_elbow_preparation_median_deg": _metric_quality(
            "valid" if arm_metrics_available and len(prep_elbow) >= 2 else "unavailable",
            f"Phase median for the inferred {selected_arm} arm. Arm inference confidence={arm_confidence:.2f}.",
            "medium" if arm_metrics_available else "low",
        ),
        "hitting_arm_elbow_contact_proxy_median_deg": _metric_quality(
            "valid" if arm_metrics_available and len(contact_elbow) >= 1 else "unavailable",
            "Elbow angle around peak 2D wrist-speed frame; technique style can legitimately vary.",
            "medium" if arm_metrics_available else "low",
        ),
        "hitting_arm_elbow_follow_through_median_deg": _metric_quality(
            "valid" if arm_metrics_available and len(follow_elbow) >= 2 else "unavailable",
            "Median elbow angle in the follow-through window.",
            "medium" if arm_metrics_available else "low",
        ),
        "wrist_speed_proxy_p95_body_lengths_per_s": _metric_quality(
            "valid" if arm_metrics_available and len(smoothed_speeds) >= 4 else "unavailable",
            "95th percentile 2D wrist speed normalized by torso length. It is descriptive and not a calibrated performance score.",
            "medium" if arm_metrics_available else "low",
        ),
        "follow_through_wrist_height_relative_shoulder": _metric_quality(
            "low_confidence" if arm_metrics_available and follow_wrist_height else "unavailable",
            "2D vertical wrist position relative to same-side shoulder during follow-through; negative means above shoulder.",
            "low",
        ),
    }

    warnings = []
    if camera_view == "side":
        warnings.append("Stance width is suppressed because a side view collapses lateral distance in 2D.")
    if hitting_arm == "uncertain":
        warnings.append("Hitting arm could not be inferred confidently; arm-specific coaching is suppressed.")
    warnings.append("No ball or racket tracking is used in this build; peak wrist speed is only a contact-phase proxy.")

    key_frames = sorted(set([
        int(frames[0]["frame"]),
        max(0, int(peak_frame - 0.6 * fps)),
        int(peak_frame),
        int(peak_frame + 0.6 * fps),
        int(frames[-1]["frame"]),
    ]))

    return {
        "quality": {
            "pose_frames": len(frames),
            "coverage": round(min(1.0, coverage), 2),
            "usable": True,
            "camera_view": camera_view,
            "hitting_arm": hitting_arm,
            "hitting_arm_candidate": selected_arm,
            "hitting_arm_confidence": round(arm_confidence, 2),
            "warnings": warnings,
        },
        "metrics": metrics,
        "metric_quality": metric_quality,
        "key_frames": key_frames,
        "movement": "forehand",
        "sport": "tennis",
        "phase_proxy": {
            "preparation_start_frame": max(0, prep_start),
            "preparation_end_frame": max(0, prep_end),
            "contact_proxy_frame": int(peak_frame),
            "follow_through_start_frame": int(follow_start),
            "follow_through_end_frame": int(follow_end),
        },
    }
