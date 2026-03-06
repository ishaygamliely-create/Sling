"""
worker/redis_store.py — Upstash Redis job state management.

Key schema:  job:{job_id}  (TTL 24h, refreshed on every update)
Value:       JSON-encoded dict — see JOB_SCHEMA below.

Status transitions:
  queued  (written by Vercel /analyze)
  running (written by Worker on job start)
  done    (written by Worker on success)
  failed  (written by Worker on error)

DEV_MODE:
  Set env var DEV_MODE=1 to use an in-memory dict instead of Upstash.
  Upstash is never imported or required in DEV_MODE.
  Production behaviour is completely unchanged when DEV_MODE is unset.
"""

from __future__ import annotations
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

_DEV_MODE: bool = os.environ.get("DEV_MODE") == "1"

# In-memory store used only when DEV_MODE=1
_mem: Dict[str, str] = {}

_JOB_TTL = 86_400  # 24 hours (used for Upstash TTL; ignored in DEV_MODE)


def _redis():
    """Return an Upstash Redis client (only called in production)."""
    from upstash_redis import Redis  # imported lazily — not needed in DEV_MODE
    return Redis(
        url=os.environ["UPSTASH_REDIS_REST_URL"],
        token=os.environ["UPSTASH_REDIS_REST_TOKEN"],
    )


def _key(job_id: str) -> str:
    return f"job:{job_id}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Public API ────────────────────────────────────────────────────────────────

def save_job(job: Dict[str, Any]) -> None:
    """Write a full job dict (creates or overwrites)."""
    if _DEV_MODE:
        _mem[_key(job["job_id"])] = json.dumps(job)
        return
    r = _redis()
    r.set(_key(job["job_id"]), json.dumps(job), ex=_JOB_TTL)


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Return the job dict or None if not found."""
    if _DEV_MODE:
        raw = _mem.get(_key(job_id))
        return json.loads(raw) if raw is not None else None
    r = _redis()
    raw = r.get(_key(job_id))
    if raw is None:
        return None
    return json.loads(raw)


def update_job(job_id: str, **fields: Any) -> None:
    """
    Merge `fields` into the existing job dict and refresh TTL.
    Special handling: nested 'progress' dict is merged (not replaced).
    Raises KeyError if job not found.
    """
    if _DEV_MODE:
        raw = _mem.get(_key(job_id))
        if raw is None:
            raise KeyError(f"Job {job_id} not found in mem-store")
        job = json.loads(raw)
        if "progress" in fields and isinstance(fields["progress"], dict):
            job.setdefault("progress", {})
            job["progress"].update(fields.pop("progress"))
        job.update(fields)
        job["updated_at"] = _now()
        _mem[_key(job_id)] = json.dumps(job)
        return

    r = _redis()
    raw = r.get(_key(job_id))
    if raw is None:
        raise KeyError(f"Job {job_id} not found in Redis")
    job = json.loads(raw)
    if "progress" in fields and isinstance(fields["progress"], dict):
        job.setdefault("progress", {})
        job["progress"].update(fields.pop("progress"))
    job.update(fields)
    job["updated_at"] = _now()
    r.set(_key(job_id), json.dumps(job), ex=_JOB_TTL)
