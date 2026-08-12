from pydantic import BaseModel
from typing import Dict, Any, Optional
import logging
import os
import subprocess
import uuid
from datetime import timedelta

from fastapi import FastAPI, Body, HTTPException, Query, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import storage
import cv2
import mediapipe as mp

from .tennis_analysis import analyze_tennis_forehand
from .ai_coach import generate_tennis_coaching


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai-athlete-v2")

app = FastAPI(title="The AI Athlete Tennis API", version="0.2.0", docs_url="/docs")

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

# V2 still uses an in-memory job store during the first test phase.
# Persistent jobs are the next infrastructure upgrade.
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
            result = pose.process(rgb)
            if result.pose_landmarks:
                drawing.draw_landmarks(
                    frame,
                    result.pose_landmarks,
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
        "version": "0.2.0",
        "sport": "tennis",
        "movement": "forehand",
        "openai_enabled": bool(os.getenv("OPENAI_API_KEY", "").strip()),
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
        chosen_focus = (focus or "swing").strip().lower()
        if chosen_focus not in {"swing", "preparation", "footwork"}:
            chosen_focus = "swing"
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
                "key_frames": tennis_analysis.get("key_frames", []),
                "recommendations": recommendation_strings,
            },
            "coaching": coaching,
            "drills": drills,
            "overlay_url": overlay_url,
        }

        JOBS[job_id]["status"] = "DONE"
        JOBS[job_id]["stage"] = "done"
        JOBS[job_id]["result"] = result
        logger.info("[%s] DONE", job_id)

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
        # Uploaded originals are temporary by design. Delete even when analysis fails
        # so an abandoned test does not accumulate storage charges.
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
        raise HTTPException(404, "not found")
    return {
        "id": job_id,
        "status": job["status"],
        "stage": job.get("stage"),
        "source_deleted": job.get("source_deleted", False),
        "result": job["result"],
    }


# ---------- Browser test harness ----------
@app.get("/test", response_class=HTMLResponse)
def test():
    return """<!doctype html><html><head><meta charset="utf-8">
<title>The AI Athlete - Tennis V2 Test</title>
<style>
body{font-family:system-ui;margin:24px;max-width:900px;background:#0b0f14;color:#eef2f7}
.card{background:#171d25;padding:18px;border-radius:16px;margin:14px 0}
button{padding:12px 18px;border-radius:10px;border:0;background:#2f6bff;color:white;font-weight:700}
button[disabled]{opacity:.55}.muted{color:#9ba7b6}pre{white-space:pre-wrap;overflow-wrap:anywhere}
video{max-width:100%;border-radius:14px;margin-top:12px}.priority{background:#f5f7fa;color:#111827;padding:12px;border-radius:12px;margin:10px 0}
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
const base=location.origin,$=id=>document.getElementById(id);
$('go').onclick=async()=>{const f=$('file').files[0];if(!f){$('status').textContent='Choose a video first.';return;} $('go').disabled=true;$('report').style.display='none';$('raw').textContent='';try{
$('status').textContent='1/4 Preparing secure upload...';const ct=f.type||'video/mp4';const sr=await fetch(`${base}/signed-upload?name=${Date.now()}.mp4&contentType=${encodeURIComponent(ct)}`);if(!sr.ok)throw new Error('Could not prepare upload');const {url,objectPath}=await sr.json();
$('status').textContent='2/4 Uploading video...';const put=await fetch(url,{method:'PUT',headers:{'Content-Type':ct},body:f});if(!put.ok)throw new Error('Upload failed '+put.status);
$('status').textContent='3/4 Starting tennis analysis...';const jr=await fetch(`${base}/jobs`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({objectPath,sport:'tennis',focus:$('focus').value})});if(!jr.ok)throw new Error(await jr.text());const {id}=await jr.json();
while(true){await new Promise(r=>setTimeout(r,1200));const data=await (await fetch(`${base}/status/${id}`)).json();$('status').textContent=`4/4 ${data.stage||'analyzing'}...`;if(data.status==='ERROR'){throw new Error(data.result?.error||'Analysis failed');}if(data.status==='DONE'){$('status').textContent='Done - coaching report ready.';$('raw').textContent=JSON.stringify(data,null,2);const result=data.result;const priorities=result.coaching?.priorities||[];$('report').innerHTML='<h3>Top coaching priorities</h3>'+priorities.map((p,i)=>`<div class="priority"><b>${i+1}. ${p.title}</b><br><span>${p.evidence}</span><br><b>Do:</b> ${p.recommendation}<br><b>Drill:</b> ${p.drill}</div>`).join('');$('report').style.display='block';if(result.overlay_url){$('video').src=result.overlay_url;$('video').style.display='block';}break;}}
}catch(e){$('status').textContent='Error: '+(e.message||e);}finally{$('go').disabled=false;}};
</script></body></html>"""


class TennisCoachRequest(BaseModel):
    metrics: Dict[str, Any]
    focus: str = "swing"


@app.post("/tennis/coach")
def tennis_coach(body: TennisCoachRequest):
    # Useful for testing the coaching layer independently from video upload.
    analysis = {
        "sport": "tennis",
        "movement": "forehand",
        "quality": {"usable": True},
        "metrics": body.metrics,
        "key_frames": [],
    }
    return generate_tennis_coaching(analysis, body.focus)
