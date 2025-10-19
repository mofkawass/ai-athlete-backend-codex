# app/focus_rules.py
from typing import List, Dict

RECOMMENDATIONS_DB: Dict[str, Dict[str, List[str]]] = {
    "tennis": {
        "swing": [
            "Keep elbow high through contact to avoid power loss.",
            "Brush up on the ball for more topspin (wrist snap).",
            "Finish over the shoulder for consistent follow-through."
        ],
        "footwork": [
            "Add a split step before opponent contact for quicker reactions.",
            "Shorten recovery steps and stay centered after the shot.",
            "Transfer weight to the front foot at contact for balance."
        ],
        "preparation": [
            "Coil shoulders ~90° on takeback to load torque.",
            "Lower ready position to improve anticipation.",
            "Check grip (semi-western recommended for topspin)."
        ],
    },
    # Add more sports later (basketball/soccer/etc.)
}

def get_focus_recommendations(sport: str, focus: str, limit: int = 3):
    sport = (sport or "").lower().strip()
    focus = (focus or "").lower().strip()
    if sport not in RECOMMENDATIONS_DB:
        return [{"id": 0, "text": "General tip: keep movements controlled and repeatable."}]
    if focus not in RECOMMENDATIONS_DB[sport]:
        return [{"id": 0, "text": f"No specific tips for '{focus}'. Try another focus."}]
    tips = RECOMMENDATIONS_DB[sport][focus][:limit]
    return [{"id": i, "text": t} for i, t in enumerate(tips)]
