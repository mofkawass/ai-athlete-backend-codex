from pydantic import BaseModel
from typing import Dict, Any, Optional
import os, uuid

from fastapi import FastAPI, Body, HTTPException, Query, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from google.cloud import storage

import cv2
import mediapipe as mp
from datetime import timedelta

# New: import our detection + focus tips helpers
from .sport_detect import detect_sport_from_gcs
from .focus_rules import get_focus_recommendations


app = FastAPI(title="AI Athlete API", version="0.1.0", docs_url="/docs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- GCS / Storage ----------
BUCKET = os.environ.get("GCS_BUCKET", "")
if not BUCKET:
    raise RuntimeError("GCS_BUCKET env var not set")

# Prefer ADC path set by entrypoint
creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "/app/gcp.json")
storage_client = storage.Client.from_service_account_json(creds_path)
bucket = storage_client.bucket(BUCKET)

# ---------- In-memory job store ----------
JOBS: Dict[str, Dict[str, Any]] = {}


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


# ---------- Very simple fallback sport guess (kept from your code) ----------
def simple_auto_sport(width: int, height: int, fps: float) -> str:
    ar = width / float(height or 1)
    if ar < 0.8:
        return "running"
    if ar > 1.6:
        return "soccer"
    return "tennis"


# ---------- Overlay / Pose drawing ----------
def draw_pose_overlay(in_path: str, out_path: str) -> Dict[str, Any]:
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)
    cap = cv2.VideoCapture(in_path)
    w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps= cap.get(cv2.CAP_PROP_FPS) or 24
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    frame_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if res.pose_landmarks:
            # simple dots overlay; can be upgraded to full skeleton
            for lm in res.pose_landmarks.landmark:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)
        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    pose.close()
    return {"frames": frame_count, "width": w, "height": h, "fps": fps}


# ---------- Generic coaching summary/drills by sport (kept) ----------
def coaching_tips(sport: str) -> Dict[str, Any]:
    if sport == "tennis":
        return {
            "summary": "Focus on stance & shoulder rotation.",
            "drills": ["Shadow swings x20", "Split-step timing 2x2min", "Serve toss consistency 10x"],
        }
    if sport == "soccer":
        return {
            "summary": "Improve stride rhythm and hip-knee alignment.",
            "drills": ["Cone dribbles 3x", "Wall passes 50x", "Sprint mechanics A-skips 2x20m"],
        }
    return {
        "summary": "Keep a tall posture and steady cadence.",
        "drills": ["Cadence 170–180bpm for 5min", "A/B skips 2x20m", "Ankling 2x20m"],
    }


# ---------- Health ----------
@app.get("/health")
def health():
    return {"ok": True}


# ---------- Signed upload ----------
@app.get("/signed-upload")
def signed_upload(name: str = Query(...), contentType: str = Query("video/mp4")):
    object_path = f"uploads/{name}"
    return {"url": gcs_signed_put(object_path, contentType)["url"], "objectPath": object_path}


# ---------- Background processing ----------
def process_job(job_id: str, object_path: str, provided_sport: Optional[str], provided_focus: Optional[str]):
    try:
        # Download input
        tmp_in = f"/tmp/{job_id}.mp4"
        bucket.blob(object_path).download_to_filename(tmp_in)

        # Overlay and write temp output
        tmp_out = f"/tmp/{job_id}_overlay.mp4"
        meta = draw_pose_overlay(tmp_in, tmp_out)

        # Upload result
        result_gcs = f"results/{job_id}.mp4"
        bucket.blob(result_gcs).upload_from_filename(tmp_out, content_type="video/mp4")

        # Determine sport (prefer provided → then heuristic → then detection model if you want)
        sport = provided_sport or simple_auto_sport(meta["width"], meta["height"], meta["fps"])

        # Summary+drills (existing)
        generic = coaching_tips(sport)

        # NEW: include focus tips if client provided a focus value
        focus = (provided_focus or "").strip() or None
        focus_tips = None
        if sport and focus:
            focus_tips = get_focus_recommendations(sport, focus, limit=3)

        result = {
            "sport": sport,
            "summary": generic["summary"],
            "metrics": {
                "frames": meta["frames"],
                "width": meta["width"],
                "height": meta["height"],
                "fps": meta["fps"],
            },
            "drills": generic["drills"],
            "overlay_url": gcs_signed_get(result_gcs, minutes=240),
        }

        # Attach focus info if present
        if focus:
            result["focus"] = focus
            result["focus_tips"] = focus_tips

        JOBS[job_id]["status"] = "DONE"
        JOBS[job_id]["result"] = result

    except Exception as e:
        JOBS[job_id]["status"] = "ERROR"
        JOBS[job_id]["result"] = {"error": str(e)}


# ---------- Create job ----------
@app.post("/jobs")
def create_job(background_tasks: BackgroundTasks, payload: Dict[str, Any] = Body(...)):
    object_path = payload.get("objectPath")
    if not object_path:
        raise HTTPException(400, "objectPath required")

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status": "PROCESSING", "object_path": object_path, "result": None}

    # Accept optional sport and focus from client
    provided_sport = payload.get("sport")
    provided_focus = payload.get("focus")  # NEW

    background_tasks.add_task(process_job, job_id, object_path, provided_sport, provided_focus)
    return {"id": job_id}


# ---------- Status ----------
@app.get("/status/{job_id}")
def status(job_id: str):
    j = JOBS.get(job_id)
    if not j:
        raise HTTPException(404, "not found")
    return {"status": j["status"], "result": j["result"]}


# ---------- Minimal browser test page ----------
@app.get("/test", response_class=HTMLResponse)
def test():
    return """<!doctype html><html><body style="font-family:sans-serif">
<h2>AI Athlete – Quick Test (Codex)</h2>
<label>Focus (optional): 
<select id="focus">
  <option value="">(none)</option>
  <option value="swing">swing</option>
  <option value="footwork">footwork</option>
  <option value="preparation">preparation</option>
</select>
</label>
<br><br>
<input type="file" id="file" accept="video/*">
<button onclick="go()">Upload & Analyze</button>
<pre id="log"></pre>
<video id="v" controls style="max-width:520px"></video>
<script>
const base = location.origin;
async function go(){
  const f = document.getElementById('file').files[0];
  const focus = document.getElementById('focus').value || null;
  if(!f){ alert('pick a file'); return; }
  const r1 = await fetch(`${base}/signed-upload?name=${Date.now()}.mp4&contentType=${f.type||'video/mp4'}`);
  const {url, objectPath} = await r1.json();
  await fetch(url, {method:'PUT', body:f, headers:{'Content-Type': f.type||'video/mp4'}});
  const payload = {objectPath};
  if (focus) payload.focus = focus;
  const r2 = await fetch(`${base}/jobs`, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)});
  const {id} = await r2.json();
  let s={status:'PROCESSING'};
  while(s.status==='PROCESSING'){
    await new Promise(t=>setTimeout(t,1500));
    s = await (await fetch(`${base}/status/`+id)).json();
  }
  document.getElementById('log').textContent = JSON.stringify(s,null,2);
  if(s.result && s.result.overlay_url){ document.getElementById('v').src = s.result.overlay_url; }
}
</script></body></html>"""


# ---------- SPORT DETECTION AND RECOMMENDATIONS ----------
class DetectSportReq(BaseModel):
    objectPath: str  # e.g., "uploads/clip123.mp4"

class RecommendReq(BaseModel):
    sport: str       # e.g., "tennis"
    focus: str       # e.g., "swing", "footwork", "preparation"

@app.post("/detect-sport")
def detect_sport(body: DetectSportReq):
    sport = detect_sport_from_gcs(storage_client, BUCKET, body.objectPath)
    return {"sport": sport}

@app.post("/recommendations")
def recommendations(body: RecommendReq):
    return {"recommendations": get_focus_recommendations(body.sport, body.focus, limit=3)}
