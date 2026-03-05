"""
worker/processor.py — Video download + Sling engine + summary aggregation.

Entry point:  run_job_sync(job_id, video_url)
              Called inside a ThreadPoolExecutor (NOT the event loop thread).
"""

from __future__ import annotations
import logging
import os
import tempfile
from collections import Counter
from statistics import mean
from typing import Any, Dict, Optional

import httpx

import redis_store

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

MAX_BYTES: int = 80_000_000          # 80 MB hard limit
DOWNLOAD_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0)
ALLOWED_CONTENT_TYPES = {
    "video/mp4", "video/quicktime", "video/x-msvideo",
    "video/webm", "video/x-matroska", "application/octet-stream",
}
PROGRESS_INTERVAL = 50               # update Redis every N frames


# ── Video download ─────────────────────────────────────────────────────────────

def download_video(url: str, max_bytes: int = MAX_BYTES) -> str:
    """
    Stream-download `url` to a temp file.
    Returns the temp file path.  Caller is responsible for cleanup.
    """
    with httpx.Client(timeout=DOWNLOAD_TIMEOUT, follow_redirects=True) as client:
        with client.stream("GET", url) as resp:
            resp.raise_for_status()

            ct = resp.headers.get("content-type", "").split(";")[0].strip().lower()
            if ct and ct not in ALLOWED_CONTENT_TYPES:
                raise ValueError(f"Content-Type '{ct}' not allowed")

            cl = resp.headers.get("content-length")
            if cl and int(cl) > max_bytes:
                raise ValueError(
                    f"Remote file size {int(cl):,} bytes exceeds {max_bytes:,} byte limit"
                )

            suffix = ".mp4"
            fd, tmp_path = tempfile.mkstemp(suffix=suffix)
            downloaded = 0
            try:
                with os.fdopen(fd, "wb") as fh:
                    for chunk in resp.iter_bytes(chunk_size=65_536):
                        downloaded += len(chunk)
                        if downloaded > max_bytes:
                            raise ValueError(
                                f"Download exceeded {max_bytes:,} byte limit "
                                f"(received >{downloaded:,} bytes)"
                            )
                        fh.write(chunk)
            except Exception:
                os.unlink(tmp_path)
                raise

    logger.info(f"Downloaded {downloaded:,} bytes → {tmp_path}")
    return tmp_path


# ── Summary aggregation ────────────────────────────────────────────────────────

def aggregate_summary(pipeline) -> Dict[str, Any]:
    """
    Build MVP summary from formation timelines after process_video().
    Uses pipeline.get_formation_timeline(team) which returns settled snapshots.
    """
    summary: Dict[str, Any] = {
        "schema_version":             "2.1.0",
        "frames_processed":           pipeline.frame_count,
        "formation_home":             None,
        "formation_away":             None,
        "avg_pressing_height_home":   None,
        "avg_pressing_height_away":   None,
        "avg_defensive_line_x_home":  None,
        "avg_defensive_line_x_away":  None,
        "settled_home":               0,
        "settled_away":               0,
        "both_settled_ratio":         0.0,
    }

    settled_counts = []
    for team_id, label in [(0, "home"), (1, "away")]:
        snapshots = pipeline.get_formation_timeline(team_id)
        settled   = [s for s in snapshots if s.get("is_settled")]
        n_settled = len(settled)
        summary[f"settled_{label}"] = n_settled
        settled_counts.append(n_settled)

        if not settled:
            continue

        # Most common formation name
        formations = [s["closest_known"] for s in settled if s.get("closest_known")]
        if formations:
            summary[f"formation_{label}"] = Counter(formations).most_common(1)[0][0]

        # Average pressing height
        heights = [s["pressing_height"] for s in settled
                   if s.get("pressing_height") is not None]
        if heights:
            summary[f"avg_pressing_height_{label}"] = round(mean(heights), 2)

        # Average defensive line x
        def_lines = [s["defensive_line_x"] for s in settled
                     if s.get("defensive_line_x") is not None]
        if def_lines:
            summary[f"avg_defensive_line_x_{label}"] = round(mean(def_lines), 2)

    total_frames = pipeline.frame_count or 1
    summary["both_settled_ratio"] = round(sum(settled_counts) / (2 * total_frames), 3)
    return summary


# ── Main job runner ────────────────────────────────────────────────────────────

def run_job_sync(job_id: str, video_url: str) -> None:
    """
    Full pipeline: download → process_video → aggregate → store result.
    Designed to run in a ThreadPoolExecutor thread (not the asyncio event loop).
    """
    from core.pipeline import FootballIntelligencePipeline
    from config.settings import get_profile_config

    tmp_path: Optional[str] = None

    try:
        # ── 1. Mark running ───────────────────────────────────────────────────
        redis_store.update_job(job_id, status="running",
                               progress={"frames_processed": 0, "frames_total_est": None})
        logger.info(f"[{job_id}] status=running | url={video_url}")

        # ── 2. Download ───────────────────────────────────────────────────────
        tmp_path = download_video(video_url)

        # ── 3. Build pipeline (broadcast profile, then auto-fallback handled
        #      inside validate; here we just use broadcast defaults) ──────────
        cfg = get_profile_config("broadcast")
        cfg.update({
            "yolo_model":              "yolov8n.pt",
            "formation_window_frames": 60,
            "min_hits":                1,
            "max_track_misses":        30,
            "iou_threshold":           0.3,
        })
        pipeline = FootballIntelligencePipeline(cfg)

        # ── 4. Process video ──────────────────────────────────────────────────
        frames_processed = 0
        for _analysis in pipeline.process_video(tmp_path):
            frames_processed += 1
            if frames_processed % PROGRESS_INTERVAL == 0:
                redis_store.update_job(
                    job_id,
                    progress={"frames_processed": frames_processed, "frames_total_est": None},
                )

        logger.info(f"[{job_id}] processed {frames_processed} frames")

        # ── 5. Aggregate summary ──────────────────────────────────────────────
        result = aggregate_summary(pipeline)

        # ── 6. Mark done ──────────────────────────────────────────────────────
        redis_store.update_job(
            job_id,
            status="done",
            result=result,
            progress={"frames_processed": frames_processed,
                      "frames_total_est": frames_processed},
        )
        logger.info(f"[{job_id}] status=done")

    except Exception as exc:
        logger.exception(f"[{job_id}] FAILED: {exc}")
        redis_store.update_job(job_id, status="failed", error=str(exc))

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
            logger.info(f"[{job_id}] cleaned up {tmp_path}")
