# Football Intelligence System - Configuration

DEFAULT_CONFIG = {
    # Detection
    "yolo_model": "yolov8n.pt",               # yolov8n/s/m/l/x — trade speed/accuracy
    "detection_confidence": 0.4,
    "ball_confidence": 0.3,

    # Tracking
    "iou_threshold": 0.3,
    "max_track_misses": 30,                   # frames before track is killed
    "min_hits": 2,                            # min frames before track is output
    "color_similarity_weight": 0.3,           # vs IoU weight (0.7)

    # Calibration
    "auto_calibrate": True,
    "calibration_confidence_threshold": 0.5,  # below this → warn user

    # Formation
    "formation_window_frames": 150,           # ~6s at 25fps for temporal smoothing
    "formation_line_bandwidth": 5.0,          # meters, mean-shift bandwidth for line detection
    "spectral_dims": 6,                       # eigenvalue vector dimensions
    "smoothing_alpha": 0.3,                   # EMA factor

    # Processing
    "target_fps": 5,                          # frames to analyze per second
    "buffer_frames": 900,                     # frame history buffer (~3 min)
    "gpu_enabled": True,

    # API
    "api_host": "0.0.0.0",
    "api_port": 8000,
}
