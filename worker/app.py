"""
worker/app.py — Sling Worker FastAPI service (Fly.io).

Exposes:
  POST /jobs   — accepts a job, processes it in a background thread
  GET  /        — health / ready check

Auth:  all /jobs endpoints require  Authorization: Bearer <WORKER_AUTH_TOKEN>
"""

from __future__ import annotations
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Optional

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import redis_store
from processor import run_job_sync

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# One job at a time — prevents OOM on small Fly machines
_executor = ThreadPoolExecutor(max_workers=1)

app = FastAPI(
    title="Sling Worker",
    description="Heavy processing service — not for direct public access.",
    docs_url=None,   # hide docs on worker
    redoc_url=None,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["Authorization", "Content-Type"],
)


# ── Auth ───────────────────────────────────────────────────────────────────────

_WORKER_TOKEN = os.environ.get("WORKER_AUTH_TOKEN", "")


def _require_auth(authorization: Optional[str] = Header(default=None)) -> None:
    if not _WORKER_TOKEN:
        raise RuntimeError("WORKER_AUTH_TOKEN env var not set on worker")
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or token != _WORKER_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")


# ── Request models ─────────────────────────────────────────────────────────────

class JobRequest(BaseModel):
    job_id:    str
    video_url: str


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return {"service": "sling-worker", "status": "ready"}


@app.get("/health", include_in_schema=False)
async def health():
    """Unauthenticated health check — safe for load balancers and local curl."""
    dev_mode = os.environ.get("DEV_MODE") == "1"
    active   = len(redis_store._mem) if dev_mode else None
    return {
        "status":   "ok",
        "service":  "sling-worker",
        "dev_mode": dev_mode,
        "store":    "memory" if dev_mode else "redis",
        "active_jobs": active,   # count of keys in mem-store (DEV_MODE only)
    }


@app.get("/jobs/{job_id}", dependencies=[Depends(_require_auth)])
async def get_job(job_id: str):
    """
    Return current job state from the store (Redis in prod, mem-store in DEV_MODE).
    Used by the API gateway to proxy job reads in DEV_MODE so the Worker is
    always the single source of truth for job state.
    """
    job = redis_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return job


@app.post("/jobs", dependencies=[Depends(_require_auth)])
async def create_job(req: JobRequest, background_tasks: BackgroundTasks):
    """
    Accept a job and start processing in the background thread pool.
    Returns immediately so Vercel doesn't time out.
    """
    import re
    if not re.match(r"^https://", req.video_url):
        raise HTTPException(status_code=422, detail="video_url must be https://")

    # Ensure job exists in Redis (Vercel already wrote it as queued;
    # if called directly we create a minimal record)
    existing = redis_store.get_job(req.job_id)
    if existing is None:
        now = datetime.now(timezone.utc).isoformat()
        redis_store.save_job({
            "job_id":     req.job_id,
            "video_url":  req.video_url,
            "status":     "queued",
            "created_at": now,
            "updated_at": now,
            "progress":   {"frames_processed": 0, "frames_total_est": None},
            "result":     None,
            "error":      None,
        })

    # Dispatch to thread pool (non-blocking)
    loop = __import__("asyncio").get_event_loop()
    loop.run_in_executor(_executor, run_job_sync, req.job_id, req.video_url)

    logger.info(f"Job {req.job_id} dispatched to executor")
    return {"accepted": True, "job_id": req.job_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8080, log_level="info")
