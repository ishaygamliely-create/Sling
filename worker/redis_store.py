"""
worker/redis_store.py — Upstash Redis job state management.

Key schema:  job:{job_id}  (TTL 24h, refreshed on every update)
Value:       JSON-encoded dict — see JOB_SCHEMA below.

Status transitions:
  queued  (written by Vercel /analyze)
  running (written by Worker on job start)
  done    (written by Worker on success)
  failed  (written by Worker on error)
"""

from __future__ import annotations
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from upstash_redis import Redis

_JOB_TTL = 86_400  # 24 hours


def _redis() -> Redis:
    return Redis(
        url=os.environ["UPSTASH_REDIS_REST_URL"],
        token=os.environ["UPSTASH_REDIS_REST_TOKEN"],
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _key(job_id: str) -> str:
    return f"job:{job_id}"


def save_job(job: Dict[str, Any]) -> None:
    """Write a full job dict to Redis (creates or overwrites)."""
    r = _redis()
    r.set(_key(job["job_id"]), json.dumps(job), ex=_JOB_TTL)


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Return the job dict or None if not found."""
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
