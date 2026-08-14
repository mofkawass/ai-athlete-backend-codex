from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
import logging
import os
import subprocess
import uuid
from datetime import timedelta

from fastapi import FastAPI, Body, Header, HTTPException, Query, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import storage
import cv2
import mediapipe as mp

from .tennis_analysis import analyze_tennis_forehand
from .ai_coach import generate_tennis_coaching
from .audit_store import (
    ANALYZER_VERSION,
    COACH_PROMPT_VERSION,
    SCHEMA_VERSION,
    append_coach_review,
    append_user_feedback,
    create_record,
    load_record,
    request_human_review,
    save_record,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai-athlete-v2")

API_VERSION = "0.3.0"
DEFAULT_COACH_MODEL = os.getenv("OPENAI_COACH_MODEL", "gpt-5-mini")

app = FastAPI(title="The AI Athlete Tennis API", version=API_VERSION, docs_url="/docs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- GCS / Storage ----------
BUCKET = os.environ.get("GCS_BUCKET", "").strip()
if not BUCKET:
    raise RuntimeError("GCS_BUCKET env var not set")

creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "/app/gcp.json")
storage_client = storage.Client.from_service_account_json(creds_path)
bucket = storage_client.bucket(BUCKET)

# During V2 testing, active job state remains in memory. Completed analysis/review
# records are persisted in GCS under records/<job_id>.json.
JOBS: Dict[str, Dict[str, Any]] = {}


def set_job_stage(job_id: str, stage: str) -> None:
    if job_id in JOBS:
        JOBS[job_id]["stage"] = stage


# ---------- Signed URL helpers ----------
def gcs_signed_put(object_name: str, content_type: str = "video/mp4", minutes: int = 15) -> Dict[str, str]:
    blob = bucket.blob(object_name)
    url = blob.generate_signed_url(
        version="v4",
        expiration=timedelta(minutes=minutes),
        method="PUT",
        content_type=content_type,
        headers={"Content-Type": content_type},
    )
    return {"url": url, "objectPath": object_name}


def gcs_signed_get(object_name: str, minutes: int = 60) -> str:
    return bucket.blob(object_name).generate_signed_url(
        version="v4",
        expiration=timedelta(minutes=minutes),
        method="GET",
    )


# ---------- Annotated video ----------
def draw_pose_overlay(in_path: str, out_path: str) -> Dict[str, Any]:
    pose_api = mp.solutions.pose
    drawing = mp.solutions.drawing_utils
    styles = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open uploaded video")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create annotated video")

    frame_count = 0
    with pose_api.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_result = pose.process(rgb)
            if pose_result.pose_landmarks:
                drawing.draw_landmarks(
                    frame,
                    pose_result.pose_landmarks,
                    pose_api.POSE_CONNECTIONS,
                    landmark_drawing_spec=styles.get_default_pose_landmarks_style(),
                )
            writer.write(frame)
            frame_count += 1

    cap.release()
    writer.release()
    return {"frames": frame_count, "width": width, "height": height, "fps": fps}


def transcode_to_web_mp4(in_path: str, out_path: str) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-i", in_path,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        out_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


# ---------- API ----------
@app.get("/health")
def health():
    return {
        "ok": True,
        "version": API_VERSION,
        "sport": "tennis",
        "movement": "forehand",
        "openai_enabled": bool(os.getenv("OPENAI_API_KEY", "").strip()),
        "record_schema_version": SCHEMA_VERSION,
        "analyzer_version": ANALYZER_VERSION,
        "coach_prompt_version": COACH_PROMPT_VERSION,
    }


@app.get("/signed-upload")
def signed_upload(name: str = Query(...), contentType: str = Query("video/mp4")):
    object_path = f"uploads/{uuid.uuid4()}-{name}"
    signed = gcs_signed_put(object_path, contentType)
    logger.info("Created signed upload for %s", object_path)
    return signed


def process_job(job_id: str, object_path: str, focus: Optional[str]):
    tmp_in = f"/tmp/{job_id}_input.mp4"
    tmp_raw = f"/tmp/{job_id}_overlay_raw.mp4"
    tmp_web = f"/tmp/{job_id}_overlay_web.mp4"
    result_gcs = f"results/{job_id}.mp4"
    source_deleted = False
    record_created = False
    chosen_focus = (focus or "swing").strip().lower()
    if chosen_focus not in {"swing", "preparation", "footwork"}:
        chosen_focus = "swing"

    try:
        logger.info("[%s] Starting tennis analysis for %s", job_id, object_path)

        set_job_stage(job_id, "downloading")
        bucket.blob(object_path).download_to_filename(tmp_in)

        set_job_stage(job_id, "measuring_tennis_forehand")
        tennis_analysis = analyze_tennis_forehand(tmp_in)
        if not tennis_analysis.get("quality", {}).get("usable", False):
            reason = tennis_analysis.get("quality", {}).get("reason", "Not enough reliable pose landmarks")
            raise RuntimeError(f"Video could not be analyzed reliably: {reason}")

        set_job_stage(job_id, "creating_overlay")
        video_meta = draw_pose_overlay(tmp_in, tmp_raw)
        transcode_to_web_mp4(tmp_raw, tmp_web)

        set_job_stage(job_id, "ai_coaching")
        coaching = generate_tennis_coaching(tennis_analysis, chosen_focus)

        set_job_stage(job_id, "uploading_result")
        bucket.blob(result_gcs).upload_from_filename(tmp_web, content_type="video/mp4")
        overlay_url = gcs_signed_get(result_gcs, minutes=240)

        priorities = coaching.get("priorities", [])[:3]
        recommendation_strings = [
            item.get("recommendation", "")
            for item in priorities
            if item.get("recommendation")
        ]
        drills = [item.get("drill", "") for item in priorities if item.get("drill")]

        result: Dict[str, Any] = {
            "sport": "tennis",
            "movement": "forehand",
            "focus": chosen_focus,
            "summary": "Tennis forehand analysis complete.",
            "metrics": {
                "frames": video_meta["frames"],
                "width": video_meta["width"],
                "height": video_meta["height"],
                "fps": video_meta["fps"],
            },
            "analysis": {
                "quality": tennis_analysis.get("quality", {}),
                "metrics": tennis_analysis.get("metrics", {}),
                "metric_quality": tennis_analysis.get("metric_quality", {}),
                "phase_proxy": tennis_analysis.get("phase_proxy", {}),
                "key_frames": tennis_analysis.get("key_frames", []),
                "recommendations": recommendation_strings,
            },
            "coaching": coaching,
            "drills": drills,
            "result_object_path": result_gcs,
            "overlay_url": overlay_url,
            "versions": {
                "api": API_VERSION,
                "record_schema": SCHEMA_VERSION,
                "analyzer": ANALYZER_VERSION,
                "coach_prompt": COACH_PROMPT_VERSION,
                "coach_model": DEFAULT_COACH_MODEL,
            },
        }

        # Persist the analysis before returning DONE. The source_deleted flag is
        # finalized in the finally block after the temporary upload is deleted.
        set_job_stage(job_id, "saving_record")
        record = create_record(
            job_id=job_id,
            focus=chosen_focus,
            result=result,
            openai_model=DEFAULT_COACH_MODEL,
            source_deleted=False,
        )
        save_record(bucket, record)
        record_created = True

        result["record_id"] = job_id
        JOBS[job_id]["status"] = "DONE"
        JOBS[job_id]["stage"] = "done"
        JOBS[job_id]["result"] = result
        logger.info("[%s] DONE and record persisted", job_id)

    except subprocess.CalledProcessError as exc:
        logger.exception("[%s] ffmpeg failed", job_id)
        JOBS[job_id]["status"] = "ERROR"
        JOBS[job_id]["stage"] = "error"
        JOBS[job_id]["result"] = {
            "error": "ffmpeg transcode failed",
            "stderr": exc.stderr.decode(errors="ignore")[-4000:] if exc.stderr else "",
        }
    except Exception as exc:
        logger.exception("[%s] Processing failed", job_id)
        JOBS[job_id]["status"] = "ERROR"
        JOBS[job_id]["stage"] = "error"
        JOBS[job_id]["result"] = {"error": str(exc)}
    finally:
        # Originals are temporary. Persistent learning data is the structured
        # analysis/review record, not the source video.
        try:
            source_blob = bucket.blob(object_path)
            if source_blob.exists():
                source_blob.delete()
                source_deleted = True
                logger.info("[%s] Deleted source object %s", job_id, object_path)
        except Exception:
            logger.exception("[%s] Could not delete source object %s", job_id, object_path)

        for path in (tmp_in, tmp_raw, tmp_web):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                logger.exception("[%s] Could not remove temp file %s", job_id, path)

        if job_id in JOBS:
            JOBS[job_id]["source_deleted"] = source_deleted
            if JOBS[job_id].get("result") is not None:
                JOBS[job_id]["result"]["source_deleted"] = source_deleted

        # Finalize the persisted record with actual source-deletion state.
        if record_created:
            try:
                persisted = load_record(bucket, job_id)
                if persisted:
                    persisted["source_deleted"] = source_deleted
                    persisted["result"]["source_deleted"] = source_deleted
                    save_record(bucket, persisted)
            except Exception:
                logger.exception("[%s] Could not finalize persisted record", job_id)


@app.post("/jobs")
def create_job(background_tasks: BackgroundTasks, payload: Dict[str, Any] = Body(...)):
    object_path = payload.get("objectPath")
    if not object_path:
        raise HTTPException(400, "objectPath required")
    if not str(object_path).startswith("uploads/"):
        raise HTTPException(400, "objectPath must point to the uploads/ prefix")

    requested_sport = str(payload.get("sport") or "tennis").lower()
    if requested_sport != "tennis":
        raise HTTPException(400, "V2 currently supports tennis only")

    focus = payload.get("focus") or "swing"
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        "status": "PROCESSING",
        "stage": "queued",
        "object_path": object_path,
        "sport": "tennis",
        "movement": "forehand",
        "focus": focus,
        "result": None,
        "source_deleted": False,
    }
    background_tasks.add_task(process_job, job_id, object_path, focus)
    return {"id": job_id, "status": "PROCESSING", "stage": "queued"}


@app.get("/status/{job_id}")
def status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        # Completed jobs can survive an application restart through the persistent record.
        persisted = load_record(bucket, job_id)
        if persisted:
            return {
                "id": job_id,
                "status": "DONE",
                "stage": "persisted",
                "source_deleted": persisted.get("source_deleted", False),
                "result": persisted.get("result"),
            }
        raise HTTPException(404, "not found")
    return {
        "id": job_id,
        "status": job["status"],
        "stage": job.get("stage"),
        "source_deleted": job.get("source_deleted", False),
        "result": job["result"],
    }


# ---------- Versioned analysis and review APIs ----------
class UserFeedbackRequest(BaseModel):
    recommendation_index: int = Field(ge=0, le=2)
    rating: str
    note: Optional[str] = Field(default=None, max_length=1000)


class CoachVerdict(BaseModel):
    recommendation_index: int = Field(ge=0, le=2)
    verdict: str
    corrected_recommendation: Optional[str] = Field(default=None, max_length=2000)
    note: Optional[str] = Field(default=None, max_length=2000)


class CoachReviewRequest(BaseModel):
    coach_id: str = Field(min_length=1, max_length=200)
    verdicts: List[CoachVerdict]
    overall_note: Optional[str] = Field(default=None, max_length=4000)


@app.get("/records/{job_id}")
def get_analysis_record(job_id: str):
    record = load_record(bucket, job_id)
    if not record:
        raise HTTPException(404, "analysis record not found")
    return record


@app.post("/records/{job_id}/feedback")
def submit_user_feedback(job_id: str, body: UserFeedbackRequest):
    rating = body.rating.strip().lower()
    if rating not in {"helpful", "not_helpful", "wrong"}:
        raise HTTPException(400, "rating must be helpful, not_helpful, or wrong")

    record = load_record(bucket, job_id)
    if not record:
        raise HTTPException(404, "analysis record not found")

    priorities = record.get("result", {}).get("coaching", {}).get("priorities", [])
    if body.recommendation_index >= len(priorities):
        raise HTTPException(400, "recommendation_index does not exist for this analysis")

    append_user_feedback(record, body.recommendation_index, rating, body.note)
    save_record(bucket, record)
    return {"ok": True, "review": record["review"]}


@app.post("/records/{job_id}/request-human-review")
def submit_human_review_request(job_id: str):
    record = load_record(bucket, job_id)
    if not record:
        raise HTTPException(404, "analysis record not found")
    request_human_review(record)
    save_record(bucket, record)
    return {"ok": True, "review": record["review"]}


@app.post("/records/{job_id}/coach-review")
def submit_coach_review(
    job_id: str,
    body: CoachReviewRequest,
    x_coach_token: str = Header(default=""),
):
    expected_token = os.getenv("COACH_REVIEW_TOKEN", "").strip()
    if not expected_token:
        raise HTTPException(503, "coach review submission is not enabled yet")
    if x_coach_token != expected_token:
        raise HTTPException(401, "unauthorized")

    record = load_record(bucket, job_id)
    if not record:
        raise HTTPException(404, "analysis record not found")

    allowed_verdicts = {"agree", "edit", "reject", "add_context"}
    verdicts = []
    for item in body.verdicts:
        verdict = item.verdict.strip().lower()
        if verdict not in allowed_verdicts:
            raise HTTPException(400, "coach verdict must be agree, edit, reject, or add_context")
        verdicts.append({
            "recommendation_index": item.recommendation_index,
            "verdict": verdict,
            "corrected_recommendation": item.corrected_recommendation,
            "note": item.note,
        })

    append_coach_review(record, body.coach_id, verdicts, body.overall_note)
    save_record(bucket, record)
    return {"ok": True, "review": record["review"]}


# ---------- Browser test harness ----------
@app.get("/test", response_class=HTMLResponse)
def test():
    return """<!doctype html><html><head><meta charset="utf-8">
<title>The AI Athlete - Tennis V2 Test</title>
<style>
body{font-family:system-ui;margin:24px;max-width:900px;background:#0b0f14;color:#eef2f7}
.card{background:#171d25;padding:18px;border-radius:16px;margin:14px 0}
button{padding:12px 18px;border-radius:10px;border:0;background:#2f6bff;color:white;font-weight:700;margin:4px}
button[disabled]{opacity:.55}.muted{color:#9ba7b6}pre{white-space:pre-wrap;overflow-wrap:anywhere}
video{max-width:100%;border-radius:14px;margin-top:12px}.priority{background:#f5f7fa;color:#111827;padding:12px;border-radius:12px;margin:10px 0}
.feedback button{background:#445064;font-size:12px;padding:7px 10px}
</style></head><body>
<h2>The AI Athlete - Tennis Forehand V2</h2>
<p class="muted">Upload a short forehand video. Tennis only for this test build.</p>
<div class="card">
<label>What do you want to improve? <select id="focus"><option value="swing">Swing</option><option value="preparation">Preparation</option><option value="footwork">Footwork</option></select></label><br><br>
<input type="file" id="file" accept="video/*"> <button id="go">Upload & Analyze</button>
<p id="status">Ready.</p></div>
<div id="report" class="card" style="display:none"></div>
<video id="video" controls style="display:none"></video>
<pre id="raw" class="card"></pre>
<script>
const base=location.origin,$=id=>document.getElementById(id);let currentJobId=null;
async function rate(index,rating){if(!currentJobId)return;const r=await fetch(`${base}/records/${currentJobId}/feedback`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({recommendation_index:index,rating})});if(r.ok){$('status').textContent=`Feedback saved: recommendation ${index+1} = ${rating}`;}else{$('status').textContent='Could not save feedback.';}}
async function requestCoach(){if(!currentJobId)return;const r=await fetch(`${base}/records/${currentJobId}/request-human-review`,{method:'POST'});$('status').textContent=r.ok?'Human review request recorded.':'Could not record human review request.';}
$('go').onclick=async()=>{const f=$('file').files[0];if(!f){$('status').textContent='Choose a video first.';return;} $('go').disabled=true;$('report').style.display='none';$('raw').textContent='';try{
$('status').textContent='1/4 Preparing secure upload...';const ct=f.type||'video/mp4';const sr=await fetch(`${base}/signed-upload?name=${Date.now()}.mp4&contentType=${encodeURIComponent(ct)}`);if(!sr.ok)throw new Error('Could not prepare upload');const {url,objectPath}=await sr.json();
$('status').textContent='2/4 Uploading video...';const put=await fetch(url,{method:'PUT',headers:{'Content-Type':ct},body:f});if(!put.ok)throw new Error('Upload failed '+put.status);
$('status').textContent='3/4 Starting tennis analysis...';const jr=await fetch(`${base}/jobs`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({objectPath,sport:'tennis',focus:$('focus').value})});if(!jr.ok)throw new Error(await jr.text());const {id}=await jr.json();currentJobId=id;
while(true){await new Promise(r=>setTimeout(r,1200));const data=await (await fetch(`${base}/status/${id}`)).json();$('status').textContent=`4/4 ${data.stage||'analyzing'}...`;if(data.status==='ERROR'){throw new Error(data.result?.error||'Analysis failed');}if(data.status==='DONE'){$('status').textContent='Done - versioned coaching report saved.';$('raw').textContent=JSON.stringify(data,null,2);const result=data.result;const priorities=result.coaching?.priorities||[];$('report').innerHTML='<h3>Top coaching priorities</h3>'+priorities.map((p,i)=>`<div class="priority"><b>${i+1}. ${p.title}</b><br><span>${p.evidence}</span><br><b>Do:</b> ${p.recommendation}<br><b>Drill:</b> ${p.drill}<div class="feedback"><button onclick="rate(${i},'helpful')">Helpful</button><button onclick="rate(${i},'not_helpful')">Not helpful</button><button onclick="rate(${i},'wrong')">Wrong</button></div></div>`).join('')+'<button onclick="requestCoach()">Request Human Coach Review</button>';$('report').style.display='block';if(result.overlay_url){$('video').src=result.overlay_url;$('video').style.display='block';}break;}}
}catch(e){$('status').textContent='Error: '+(e.message||e);}finally{$('go').disabled=false;}};
</script></body></html>"""


class TennisCoachRequest(BaseModel):
    metrics: Dict[str, Any]
    focus: str = "swing"


@app.post("/tennis/coach")
def tennis_coach(body: TennisCoachRequest):
    analysis = {
        "sport": "tennis",
        "movement": "forehand",
        "quality": {"usable": True},
        "metrics": body.metrics,
        "metric_quality": {key: {"status": "valid", "confidence": "test", "reason": "Manual coaching-layer test input."} for key in body.metrics},
        "key_frames": [],
    }
    return generate_tennis_coaching(analysis, body.focus)
