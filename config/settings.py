# Football Intelligence System — Configuration

from __future__ import annotations
from typing import Optional

# ── Default config (broadcast-safe baseline) ─────────────────────────────────
DEFAULT_CONFIG = {
    # Detection
    "yolo_model":                 "yolov8n.pt",
    "detection_confidence":       0.35,      # broadcast default — DO NOT lower globally
    "ball_confidence":            0.3,
    "min_bbox_area_ratio":        0.0005,    # person bbox must be ≥ 0.05% of frame area
    "max_aspect_ratio":           4.0,       # height/width; > 4 = crowd sliver / artifact

    # Tracking
    "iou_threshold":              0.3,
    "max_track_misses":           30,
    "min_hits":                   2,
    "color_similarity_weight":    0.3,

    # Calibration
    "auto_calibrate":             True,
    "calibration_confidence_threshold": 0.5,

    # Formation
    "formation_window_frames":    150,
    "formation_line_bandwidth":   5.0,
    "spectral_dims":              6,
    "smoothing_alpha":            0.3,

    # Processing
    "target_fps":                 5,
    "buffer_frames":              900,
    "gpu_enabled":                True,

    # API
    "api_host":                   "0.0.0.0",
    "api_port":                   8000,
}

# ── Detection profiles ────────────────────────────────────────────────────────
#
# broadcast  — precision-first; for fixed-camera broadcast with clear players.
#              Strict conf threshold; tighter area/aspect guards.
# wild       — recall-first; for phone clips, YouTube, close-up or moving cam.
#              Lower conf threshold; relaxed area floor (still blocks slivers).
#
DETECTION_PROFILES: dict[str, dict] = {
    "broadcast": {
        "detection_confidence": 0.35,
        "min_bbox_area_ratio":  0.0002,   # 0.02% — ~415px on 1080p; blocks tiny noise
        "max_aspect_ratio":     4.0,
    },
    "wild": {
        "detection_confidence": 0.15,
        "min_bbox_area_ratio":  0.00005,  # 0.005% — ~104px on 1080p; near-disabled
        "max_aspect_ratio":     4.0,       # aspect ratio remains the main guard
    },
}

VALID_PROFILES = tuple(DETECTION_PROFILES.keys())   # ('broadcast', 'wild')


def get_profile_config(
    profile: str = "broadcast",
    override_conf: Optional[float] = None,
) -> dict:
    """
    Return a full pipeline config dict for the requested profile.

    Priority (highest to lowest):
      1. override_conf   — explicit --conf argument
      2. profile values  — broadcast / wild
      3. DEFAULT_CONFIG  — everything else

    Parameters
    ----------
    profile : str
        'broadcast' (default, 0.35) or 'wild' (0.15).
    override_conf : float | None
        When supplied, overrides detection_confidence regardless of profile.

    Returns
    -------
    dict
        Ready-to-pass config for FootballIntelligencePipeline.
    """
    if profile not in DETECTION_PROFILES:
        raise ValueError(
            f"Unknown profile '{profile}'. Valid options: {VALID_PROFILES}"
        )
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(DETECTION_PROFILES[profile])
    if override_conf is not None:
        cfg["detection_confidence"] = float(override_conf)
    return cfg
