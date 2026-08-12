import json
import os
from typing import Any, Dict


def _fallback(metrics: Dict[str, Any], focus: str) -> Dict[str, Any]:
    priorities = []
    knee = metrics.get("knee_flexion_min_deg")
    stance = metrics.get("stance_width_ratio_median")
    torso = metrics.get("torso_separation_proxy_max_deg")

    if knee is not None and knee > 155:
        priorities.append({"title": "Load the legs more", "evidence": f"Minimum knee angle was about {knee}°.", "recommendation": "Create a little more knee flexion before accelerating into the ball.", "drill": "Pause-and-go shadow forehands: 2 x 10."})
    if torso is not None and torso < 15:
        priorities.append({"title": "Create more upper-body coil", "evidence": f"Torso separation proxy peaked near {torso}°.", "recommendation": "Turn the shoulders earlier during preparation while staying balanced.", "drill": "Unit-turn checkpoints: 2 x 10 slow repetitions."})
    if stance is not None and stance < 0.8:
        priorities.append({"title": "Build a wider base", "evidence": f"Median stance-to-shoulder width ratio was {stance}.", "recommendation": "Use a slightly wider athletic base through preparation and loading.", "drill": "Split-step to forehand stance: 3 x 8."})

    defaults = [
        {"title": "Keep the preparation repeatable", "evidence": "Pose tracking was usable, but this MVP has limited camera-depth information.", "recommendation": "Use an early unit turn and arrive balanced before the forward swing.", "drill": "Shadow forehands with a preparation pause: 2 x 10."},
        {"title": "Accelerate smoothly", "evidence": "Wrist motion is currently measured only as a 2D speed proxy.", "recommendation": "Build racket-head speed progressively rather than forcing the arm early.", "drill": "Three-speed forehands at 50%, 70%, 85%: 5 each."},
        {"title": "Recover after the finish", "evidence": "The current model tracks body landmarks but not the ball trajectory.", "recommendation": "Finish balanced and recover toward a ready position immediately.", "drill": "Hit-and-recover shadow sequence: 3 x 8."},
    ]
    for item in defaults:
        if len(priorities) >= 3:
            break
        priorities.append(item)
    return {"source": "rules", "focus": focus, "strengths": ["Pose tracking produced usable movement data."], "priorities": priorities[:3]}


def generate_tennis_coaching(analysis: Dict[str, Any], focus: str = "swing") -> Dict[str, Any]:
    metrics = analysis.get("metrics", {})
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return _fallback(metrics, focus)

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        schema = {
            "type": "object",
            "properties": {
                "strengths": {"type": "array", "items": {"type": "string"}, "maxItems": 2},
                "priorities": {
                    "type": "array", "minItems": 3, "maxItems": 3,
                    "items": {"type": "object", "properties": {
                        "title": {"type": "string"}, "evidence": {"type": "string"},
                        "recommendation": {"type": "string"}, "drill": {"type": "string"}
                    }, "required": ["title", "evidence", "recommendation", "drill"], "additionalProperties": False}
                }
            }, "required": ["strengths", "priorities"], "additionalProperties": False
        }
        response = client.responses.create(
            model=os.getenv("OPENAI_COACH_MODEL", "gpt-5-mini"),
            input=[{"role": "system", "content": "You are the coaching reasoning layer for a tennis forehand MVP. Use ONLY the supplied 2D pose measurements as evidence. Do not invent ball, racket, contact, spin, or 3D measurements. Be concise, constructive, age-neutral, and prioritize exactly three actionable items."},
                   {"role": "user", "content": json.dumps({"sport": "tennis", "movement": "forehand", "focus": focus, "analysis": analysis})}],
            text={"format": {"type": "json_schema", "name": "tennis_coaching", "strict": True, "schema": schema}},
        )
        parsed = json.loads(response.output_text)
        parsed["source"] = "openai"
        parsed["focus"] = focus
        return parsed
    except Exception as exc:
        result = _fallback(metrics, focus)
        result["ai_error"] = str(exc)
        return result
