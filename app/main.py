from pydantic import BaseModel
from typing import Dict, Any, Optional
import os, uuid, subprocess, math
from datetime import timedelta

from fastapi import FastAPI, Body, HTTPException, Query, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from google.cloud import storage

import cv2
import mediapipe as mp

# Helpers for detection + focus tips
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


# ---------- Geometry helpers for analysis ----------
def _pt(lms, idx):
    lm = lms[idx]
    return (lm.x, lm.y)

def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def _angle(a, b, c):
    """Angle (deg) at point b formed by points a-b-c."""
    ab = _dist(a, b)
    cb = _dist(c, b)
    if ab == 0 or cb == 0:
        return None
    ac = _dist(a, c)
    cosv = (ab**2 + cb**2 - ac**2) / (2 * ab * cb)
    cosv = max(-1.0, min(1.0, cosv))
    return math.degrees(math.acos(cosv))

def _median(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    xs.sort()
    n = len(xs)
    m = n // 2
    return xs[m] if n % 2 else (xs[m - 1] + xs[m]) / 2.0


# ---------- Overlay / Pose drawing + metrics collection ----------
def draw_pose_overlay(in_path: str, out_path: str) -> Dict[str, Any]:
    """
    Renders simple landmark dots to out_path AND collects per-frame metrics:
      - knee angles (L/R)
      - elbow height drop (elbow_y - shoulder_y, choose larger arm drop)
      - stance width ratio (ankle distance / hip distance)
    Returns: frames, width, height, fps, metrics_calc{...}
    """
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)
    cap = cv2.VideoCapture(in_path)

    w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps= cap.get(cv2.CAP_PROP_FPS) or 24
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    knees_L, knees_R = [], []
    elbow_height = []
    stance_width = []

    P = mp_pose.PoseLandmark
    frame_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        if res.pose_landmarks:
            lms = res.pose_landmarks.landmark

            # draw simple dots
            for lm in lms:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

            # landmarks
            LS = _pt(lms, P.LEFT_SHOULDER.value)
            LE = _pt(lms, P.LEFT_ELBOW.value)
            RS = _pt(lms, P.RIGHT_SHOULDER.value)
            RE = _pt(lms, P.RIGHT_ELBOW.value)

            LH = _pt(lms, P.LEFT_HIP.value)
            LK = _pt(lms, P.LEFT_KNEE.value)
            LA = _pt(lms, P.LEFT_ANKLE.value)
            RH = _pt(lms, P.RIGHT_HIP.value)
            RK = _pt(lms, P.RIGHT_KNEE.value)
            RA = _pt(lms, P.RIGHT_ANKLE.value)

            # knee flexion angles
            knees_L.append(_angle(LH, LK, LA))
            knees_R.append(_angle(RH, RK, RA))

            # elbow drop (positive means elbow below shoulder)
            eh_R = (RE[1] - RS[1]) if (RS and RE) else None
            eh_L = (LE[1] - LS[1]) if (LS and LE) else None
            if eh_L is not None and eh_R is not None:
                elbow_height.append(max(eh_L, eh_R))
            else:
                elbow_height.append(eh_L if eh_L is not None else eh_R)

            # stance width normalized by hip width
            hip_w = _dist(LH, RH) if (LH and RH) else None
            ankle_w = _dist(LA, RA) if (LA and RA) else None
            if hip_w and hip_w > 0 and ankle_w is not None:
                stance_width.append(ankle_w / hip_w)

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    pose.close()

    metrics_summary = {
        "knee_angle_left_median": _median(knees_L),
        "knee_angle_right_median": _median(knees_R),
        "elbow_drop_median": _median(elbow_height),          # >0 => elbow below shoulder
        "stance_width_ratio_median": _median(stance_width),   # <0.7 => narrow base
    }

    return {
        "frames": frame_count,
        "width": w,
        "height": h,
        "fps": fps,
        "metrics_calc": metrics_summary,
    }


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


# ---------- Transcode to a browser-friendly MP4 ----------
def transcode_to_web_mp4(in_path: str, out_path: str) -> None:
    """
    Ensure the result plays & seeks in browsers:
    - H.264 video (yuv420p)
    - +faststart to move moov atom to the beginning
    - audio disabled (remove -an to keep)
    """
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

        # 1) Overlay render + metrics
        tmp_out_raw = f"/tmp/{job_id}_overlay_raw.mp4"
        meta = draw_pose_overlay(tmp_in, tmp_out_raw)

        # 2) Transcode to web-friendly MP4
        tmp_out_web = f"/tmp/{job_id}_overlay_web.mp4"
        transcode_to_web_mp4(tmp_out_raw, tmp_out_web)

        # 3) Upload result
        result_gcs = f"results/{job_id}.mp4"
        bucket.blob(result_gcs).upload_from_filename(tmp_out_web, content_type="video/mp4")

        # Determine sport (prefer provided → then heuristic)
        sport = provided_sport or simple_auto_sport(meta["width"], meta["height"], meta["fps"])

        # Summary+drills (existing)
        generic = coaching_tips(sport)

        # Focus tips (optional)
        focus = (provided_focus or "").strip() or None
        focus_tips = None
        if sport and focus:
            focus_tips = get_focus_recommendations(sport, focus, limit=3)

        # ---- derive simple recommendations from metrics_calc ----
        recs = []
        mc = meta.get("metrics_calc", {})

        # Knee flexion: if both legs too straight, encourage more bend
        kl = mc.get("knee_angle_left_median")
        kr = mc.get("knee_angle_right_median")
        if kl and kr and (kl > 170 and kr > 170):
            recs.append("Bend your knees ~10–20° more during preparation for better stability.")

        # Elbow drop: if elbow notably below shoulder
        ed = mc.get("elbow_drop_median")
        if ed is not None and ed > 0.10:
            recs.append("Keep your hitting elbow higher (closer to shoulder height) through the swing.")

        # Stance width: if narrow base
        sw = mc.get("stance_width_ratio_median")
        if sw is not None and sw < 0.70:
            recs.append("Adopt a wider base (increase ankle distance) to improve balance and power transfer.")

        result: Dict[str, Any] = {
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

        # Attach analysis if we have signals
        if mc:
            result["analysis"] = {
                "metrics": mc,
                "recommendations": recs[:3] if recs else []
            }

        # Attach focus info if present
        if focus:
            result["focus"] = focus
            result["focus_tips"] = focus_tips

        JOBS[job_id]["status"] = "DONE"
        JOBS[job_id]["result"] = result

    except subprocess.CalledProcessError as e:
        JOBS[job_id]["status"] = "ERROR"
        JOBS[job_id]["result"] = {
            "error": "ffmpeg transcode failed",
            "stderr": e.stderr.decode(errors="ignore") if e.stderr else "",
        }
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
    provided_focus = payload.get("focus")

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
    return """<!doctype html><html><head><meta charset="utf-8">
<title>AI Athlete – Test</title>
<style>
  body{font-family:system-ui;margin:24px;max-width:800px}
  .row{margin:8px 0}
  .status{padding:8px 12px;border-radius:8px;background:#f5f5f5;white-space:pre-wrap}
  button{padding:10px 16px;border-radius:8px;border:0;background:#2563eb;color:#fff;cursor:pointer}
  button[disabled]{opacity:.6;cursor:not-allowed}
  .ok{color:#16a34a}
  .err{color:#dc2626}
  .muted{color:#666}
</style>
</head>
<body>
  <h2>AI Athlete – Quick Test</h2>
  <div class="row">
    <label>Focus (optional): 
      <select id="focus">
        <option value="">(none)</option>
        <option value="swing">swing</option>
        <option value="footwork">footwork</option>
        <option value="preparation">preparation</option>
      </select>
    </label>
  </div>
  <div class="row">
    <input type="file" id="file" accept="video/*">
    <button id="go">Upload & Analyze</button>
  </div>
  <div class="row">
    <div id="status" class="status">Idle.</div>
  </div>
  <div class="row">
    <video id="v" controls style="max-width:100%;display:none"></video>
  </div>
  <div class="row">
    <pre id="out" class="status muted" style="background:#fafafa"></pre>
  </div>

<script>
const base = location.origin;
const $ = (id) => document.getElementById(id);
const log = (msg) => { $('status').textContent = msg; };
const append = (msg) => { $('status').textContent += "\\n" + msg; };

$('go').onclick = async () => {
  const btn = $('go');
  const file = $('file').files[0];
  const focus = $('focus').value || null;
  $('v').style.display = 'none';
  $('v').src = '';
  $('out').textContent = '';
  if (!file) { log("Please choose a short video (<=10s)."); return; }

  try {
    btn.disabled = true;
    log("1/4 Requesting signed upload URL…");

    const name = Date.now() + ".mp4";
    const ct = file.type || "video/mp4";
    const s = await fetch(`${base}/signed-upload?name=${encodeURIComponent(name)}&contentType=${encodeURIComponent(ct)}`);
    if (!s.ok) throw new Error("signed-upload failed: " + s.status);
    const { url, objectPath } = await s.json();
    append("✔ Signed URL received.");

    log("2/4 Uploading to GCS…");
    const put = await fetch(url, { method:'PUT', headers:{'Content-Type': ct}, body: file });
    if (!put.ok) {
      const t = await put.text().catch(()=>"(no body)");
      throw new Error("Upload failed: " + put.status + " " + t);
    }
    append("✔ Upload done.");

    log("3/4 Creating processing job…");
    const payload = { objectPath };
    if (focus) payload.focus = focus;
    const create = await fetch(`${base}/jobs`, {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify(payload)
    });
    if (!create.ok) throw new Error("jobs failed: " + create.status);
    const { id } = await create.json();
    append("✔ Job created: " + id);

    log("4/4 Processing video (polling)…");
    const started = Date.now();
    const timeoutMs = 120000; // 2 minutes safety timeout

    while (true) {
      await new Promise(r => setTimeout(r, 1200));
      const r = await fetch(`${base}/status/` + id);
      if (!r.ok) throw new Error("status failed: " + r.status);
      const data = await r.json();

      if (data.status === 'DONE') {
        append("✔ Processing complete.");
        $('out').textContent = JSON.stringify(data, null, 2);
        if (data.result && data.result.overlay_url) {
          $('v').src = data.result.overlay_url;
          $('v').style.display = 'block';
        }
        log("Done ✅");
        break;
      }
      if (data.status === 'ERROR') {
        $('out').textContent = JSON.stringify(data, null, 2);
        log("Error ❌ — see details below.");
        break;
      }
      if (Date.now() - started > timeoutMs) {
        log("Error ❌ Timed out waiting for processing (2 min).");
        break;
      }
      append("…still processing");
    }
  } catch (e) {
    log("Error ❌ " + (e?.message || e));
    console.error(e);
  } finally {
    $('go').disabled = false;
  }
};
</script>
</body></html>"""


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
