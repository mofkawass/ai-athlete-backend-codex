import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


SCHEMA_VERSION = "1.0"
ANALYZER_VERSION = "tennis-forehand-v2.1"
COACH_PROMPT_VERSION = "tennis-coach-v2.1"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def record_path(job_id: str) -> str:
    return f"records/{job_id}.json"


def save_record(bucket, record: Dict[str, Any]) -> None:
    payload = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    bucket.blob(record_path(record["job_id"])).upload_from_string(payload, content_type="application/json")


def load_record(bucket, job_id: str) -> Optional[Dict[str, Any]]:
    blob = bucket.blob(record_path(job_id))
    if not blob.exists():
        return None
    return json.loads(blob.download_as_text())


def create_record(
    job_id: str,
    focus: str,
    result: Dict[str, Any],
    openai_model: str,
    source_deleted: bool,
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "job_id": job_id,
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "sport": "tennis",
        "movement": "forehand",
        "focus": focus,
        "versions": {
            "analyzer": ANALYZER_VERSION,
            "coach_prompt": COACH_PROMPT_VERSION,
            "coach_model": openai_model,
        },
        "source_deleted": bool(source_deleted),
        "result": result,
        "review": {
            "status": "unreviewed",
            "human_review_requested": False,
            "user_feedback": [],
            "coach_reviews": [],
            "final_labels": [],
        },
    }


def append_user_feedback(
    record: Dict[str, Any],
    recommendation_index: int,
    rating: str,
    note: Optional[str] = None,
) -> Dict[str, Any]:
    item = {
        "created_at": utc_now_iso(),
        "recommendation_index": recommendation_index,
        "rating": rating,
    }
    if note:
        item["note"] = note
    record["review"]["user_feedback"].append(item)
    record["review"]["status"] = "user_feedback_received"
    record["updated_at"] = utc_now_iso()
    return record


def request_human_review(record: Dict[str, Any]) -> Dict[str, Any]:
    record["review"]["human_review_requested"] = True
    record["review"]["status"] = "human_review_requested"
    record["updated_at"] = utc_now_iso()
    return record


def append_coach_review(
    record: Dict[str, Any],
    coach_id: str,
    verdicts: List[Dict[str, Any]],
    overall_note: Optional[str] = None,
) -> Dict[str, Any]:
    item: Dict[str, Any] = {
        "created_at": utc_now_iso(),
        "coach_id": coach_id,
        "verdicts": verdicts,
    }
    if overall_note:
        item["overall_note"] = overall_note
    record["review"]["coach_reviews"].append(item)
    record["review"]["status"] = "coach_reviewed"
    record["updated_at"] = utc_now_iso()
    return record
