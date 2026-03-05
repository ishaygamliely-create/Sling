"""
api/server.py — Football Intelligence API
v2.1 schema | v0.1.0 API
"""

from __future__ import annotations
import base64, json, logging, sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    print("Install: pip install fastapi uvicorn python-multipart")
    sys.exit(1)

# Core pipeline — optional (unavailable in lightweight Vercel serverless deploy)
try:
    from core.pipeline import FootballIntelligencePipeline, SCHEMA_VERSION
    _CORE_AVAILABLE = True
except Exception:
    FootballIntelligencePipeline = None  # type: ignore
    SCHEMA_VERSION = "2.1.0"
    _CORE_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SLING_API_VERSION = "0.1.0"

app = FastAPI(
    title="Sling — Football Intelligence API",
    description="Tactical analysis engine for broadcast football clips.",
    version=SLING_API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://sling-seven.vercel.app",
        "https://*.vercel.app",
        "http://localhost:3000",
        "http://localhost:8000",
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline = None
_config: Dict = {
    'yolo_model': 'yolov8n.pt', 'detection_confidence': 0.35,
    'iou_threshold': 0.3, 'max_track_misses': 30,
    'min_hits': 1, 'formation_window_frames': 60,
}

def get_pipeline():
    global _pipeline
    if not _CORE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Pipeline not available in this deployment")
    if _pipeline is None:
        _pipeline = FootballIntelligencePipeline(_config)
    return _pipeline

class FrameRequest(BaseModel):
    image_b64: str

class ManualCalibRequest(BaseModel):
    pixel_points: List[List[float]]
    pitch_points:  List[List[float]]

class VideoAnalyzeRequest(BaseModel):
    video_url: str


@app.get("/", summary="Service root")
async def root():
    return {
        "service":         "Sling Football Intelligence API",
        "api_version":     SLING_API_VERSION,
        "schema_version":  SCHEMA_VERSION,
        "pipeline":        "available" if _CORE_AVAILABLE else "serverless mode",
        "links": {
            "health":  "/health",
            "analyze": "/analyze",
            "docs":    "/docs",
        },
    }


@app.get("/health", summary="Health check")
async def health():
    """Lightweight health check — works without pipeline/YOLO on Vercel."""
    info: Dict = {
        "status":          "ok",
        "api_version":     SLING_API_VERSION,
        "schema_version":  SCHEMA_VERSION,
        "pipeline":        "available" if _CORE_AVAILABLE else "not loaded (serverless mode)",
    }
    if _CORE_AVAILABLE and _pipeline is not None:
        info["calibration_confidence"] = round(_pipeline.calibrator.calibration_confidence, 3)
        info["is_calibrated"]          = _pipeline.calibrator.is_calibrated
        info["frame_count"]            = _pipeline.frame_count
        info["active_tracks"]          = len(_pipeline.tracker.tracks)
    return info


@app.post("/analyze", summary="Submit video for tactical analysis")
async def analyze(req: VideoAnalyzeRequest):
    """
    MVP endpoint — accepts a video_url and returns a stub analysis.
    Full async processing (queue + worker) will be wired in the next milestone.
    """
    import re
    if not re.match(r"^https?://", req.video_url):
        raise HTTPException(status_code=422, detail="video_url must be a valid http/https URL")

    base = {
        "api_version":    SLING_API_VERSION,
        "schema_version": SCHEMA_VERSION,
        "video_url":      req.video_url,
    }

    if not _CORE_AVAILABLE:
        return {
            **base,
            "status":   "accepted",
            "job_mode": "async",
            "analysis": {"formation": None, "pressing": None, "events": []},
            "notes":    "MVP placeholder — engine not loaded in serverless mode. "
                        "Full processing requires a worker deployment.",
        }

    # Core available: return metadata-only analysis (no heavy processing in request)
    return {
        **base,
        "status":   "accepted",
        "job_mode": "sync",
        "analysis": {
            "formation": None,
            "pressing":  None,
            "events":    [],
        },
        "notes":    "Engine available. Submit via /analyze/frame or /analyze/video "
                    "for full frame-by-frame analysis. Async job queue coming in v0.2.",
    }


@app.post("/analyze/frame")
async def analyze_frame(req: FrameRequest):
    try:
        img_bytes = base64.b64decode(req.image_b64)
        nparr     = np.frombuffer(img_bytes, np.uint8)
        frame     = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode image")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")
    p        = get_pipeline()
    analysis = p.process_frame(frame, p.frame_count)
    return json.loads(p.to_json(analysis))


@app.post("/analyze/video")
async def analyze_video(file: UploadFile = File(...)):
    import tempfile, os
    suffix = Path(file.filename or "video.mp4").suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    p = get_pipeline()
    async def event_stream():
        try:
            for analysis in p.process_video(tmp_path):
                data = p.to_json(analysis)
                yield f"data: {data}\n\n"
        finally:
            os.unlink(tmp_path)
    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket):
    await websocket.accept()
    p = get_pipeline()
    try:
        while True:
            data      = await websocket.receive_text()
            payload   = json.loads(data)
            img_bytes = base64.b64decode(payload.get("image_b64", ""))
            nparr     = np.frombuffer(img_bytes, np.uint8)
            frame     = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is not None:
                analysis = p.process_frame(frame, p.frame_count)
                await websocket.send_text(p.to_json(analysis))
    except Exception:
        pass
    finally:
        await websocket.close()


@app.get("/formation/history/{team_id}")
async def formation_history(team_id: int):
    if team_id not in (0, 1):
        raise HTTPException(status_code=400, detail="team_id must be 0 or 1")
    p = get_pipeline()
    return {"schema_version": SCHEMA_VERSION, "team": team_id,
            "snapshots": p.get_formation_timeline(team_id)}


@app.post("/calibrate/manual")
async def calibrate_manual(req: ManualCalibRequest):
    p  = get_pipeline()
    ok = p.calibrator.set_manual_points(pixel_pts=req.pixel_points, pitch_pts=req.pitch_points)
    return {"schema_version": SCHEMA_VERSION, "success": ok,
            "confidence": p.calibrator.calibration_confidence}


if __name__ == "__main__":
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
