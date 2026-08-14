import json
import os
from typing import Any, Dict, List


def _valid_metrics(analysis: Dict[str, Any]) -> Dict[str, Any]:
    metrics = analysis.get("metrics", {})
    quality = analysis.get("metric_quality", {})
    valid = {}
    for key, value in metrics.items():
        status = quality.get(key, {}).get("status")
        if value is not None and status in {"valid", "low_confidence"}:
            valid[key] = value
    return valid


def _fallback(analysis: Dict[str, Any], focus: str) -> Dict[str, Any]:
    metrics = analysis.get("metrics", {})
    quality = analysis.get("metric_quality", {})
    priorities: List[Dict[str, str]] = []

    knee = metrics.get("knee_angle_preparation_median_deg")
    knee_status = quality.get("knee_angle_preparation_median_deg", {}).get("status")
    stance = metrics.get("stance_width_ratio_preparation_median")
    stance_status = quality.get("stance_width_ratio_preparation_median", {}).get("status")

    if knee is not None and knee_status == "valid" and knee > 165:
        priorities.append({
            "title": "Create a little more leg load",
            "evidence": f"Preparation knee-angle median was about {knee}°.",
            "recommendation": "Add a modest athletic knee bend during preparation while staying balanced.",
            "drill": "Pause-and-go shadow forehands: 2 x 10, holding the loaded position briefly before swinging.",
        })
    if stance is not None and stance_status == "valid" and stance < 0.75:
        priorities.append({
            "title": "Stabilize the preparation base",
            "evidence": f"Preparation stance-to-shoulder ratio was about {stance} from a camera view where this measure is usable.",
            "recommendation": "Experiment with a slightly wider, balanced base during preparation.",
            "drill": "Split-step to forehand stance: 3 x 8, checking balance before each shadow swing.",
        })

    defaults = [
        {
            "title": "Make the unit turn repeatable",
            "evidence": "The current build has reliable body pose coverage but does not yet track the ball or racket.",
            "recommendation": "Prepare early with a coordinated shoulder turn and arrive balanced before the forward swing.",
            "drill": "Unit-turn checkpoints: 2 x 10 slow shadow forehands with a brief preparation pause.",
        },
        {
            "title": "Build smooth acceleration",
            "evidence": "Wrist speed is currently a normalized 2D proxy, not racket-head speed.",
            "recommendation": "Let the swing accelerate progressively rather than forcing the arm from the start.",
            "drill": "Three-speed forehands at 50%, 70%, and 85% effort: 5 each.",
        },
        {
            "title": "Finish balanced and recover",
            "evidence": "The model can observe body pose through follow-through, while ball trajectory is not yet measured.",
            "recommendation": "Complete the follow-through under control and return toward a ready position.",
            "drill": "Hit-and-recover shadow sequence: 3 x 8.",
        },
    ]
    for item in defaults:
        if len(priorities) >= 3:
            break
        priorities.append(item)

    return {
        "source": "rules",
        "focus": focus,
        "strengths": ["Pose tracking produced usable movement data."],
        "priorities": priorities[:3],
    }


def generate_tennis_coaching(analysis: Dict[str, Any], focus: str = "swing") -> Dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return _fallback(analysis, focus)

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        schema = {
            "type": "object",
            "properties": {
                "strengths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 2,
                },
                "priorities": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 3,
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "evidence": {"type": "string"},
                            "recommendation": {"type": "string"},
                            "drill": {"type": "string"},
                        },
                        "required": ["title", "evidence", "recommendation", "drill"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["strengths", "priorities"],
            "additionalProperties": False,
        }

        coaching_payload = {
            "sport": "tennis",
            "movement": "forehand",
            "focus": focus,
            "quality": analysis.get("quality", {}),
            "metrics": _valid_metrics(analysis),
            "metric_quality": analysis.get("metric_quality", {}),
            "phase_proxy": analysis.get("phase_proxy", {}),
        }

        system_prompt = """You are the coaching reasoning layer for a tennis forehand MVP.
Use ONLY the supplied measurements as evidence. Never infer ball contact, racket angle, racket-head speed, spin, shot outcome, or true 3D rotation because those are not measured.
Treat metrics marked low_confidence as descriptive context only; do not turn them into strong corrective claims. Never use metrics marked unavailable.
Do not compare an uncalibrated speed proxy to an ideal value and do not prescribe a wrist snap. Modern forehand technique varies by grip, stance, hitting arm, and camera view.
Prefer phase-specific evidence. If the data does not support three genuine faults, return fewer than three priorities rather than inventing faults.
Strengths must also be supported by measured evidence. Keep recommendations concise, constructive, age-neutral, and tennis-specific."""

        response = client.responses.create(
            model=os.getenv("OPENAI_COACH_MODEL", "gpt-5-mini"),
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(coaching_payload)},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "tennis_coaching",
                    "strict": True,
                    "schema": schema,
                }
            },
        )
        parsed = json.loads(response.output_text)
        parsed["source"] = "openai"
        parsed["focus"] = focus
        return parsed
    except Exception as exc:
        result = _fallback(analysis, focus)
        result["ai_error"] = str(exc)
        return result
