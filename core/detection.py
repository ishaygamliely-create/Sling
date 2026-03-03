"""
Player Detection Module
Uses YOLOv8 for person detection + ball detection.
Falls back to HOG if YOLO unavailable.
Team separation via color clustering in HSV / LAB space.
"""

import numpy as np
import cv2
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RawDetection:
    bbox: Tuple[int, int, int, int]   # x1,y1,x2,y2
    confidence: float
    class_id: int                      # 0=person, 32=ball
    color_embedding: Optional[np.ndarray] = None


class TeamSeparator:
    """
    Separates players into two teams using:
    1. Extract jersey color from upper-body crop
    2. Project into LAB color space
    3. K-Means (k=2) on color embeddings
    4. Temporal voting for stability
    """

    def __init__(self):
        self.team_centroids: Optional[np.ndarray] = None
        self.assignment_history: Dict[int, List[int]] = {}
        self.n_stable_frames = 0

    def extract_jersey_color(self, frame: np.ndarray, bbox: Tuple) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        h = y2 - y1
        # Upper 40% of bounding box = jersey area
        crop = frame[y1:y1 + int(h * 0.4), x1:x2]
        if crop.size == 0:
            return np.zeros(6)

        # Convert to LAB for perceptual uniformity
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(float)

        # Remove near-white (referee) and very dark pixels
        mask = (lab[:, 0] > 30) & (lab[:, 0] < 220)
        lab = lab[mask]
        if len(lab) < 10:
            return np.zeros(6)

        # Return mean + std as 6-dim embedding
        return np.concatenate([lab.mean(0), lab.std(0)])

    def assign_teams(self, detections: List[RawDetection]) -> List[int]:
        """Returns team label (0 or 1) for each detection. -1 = referee."""
        embeddings = []
        valid_idx = []

        for i, det in enumerate(detections):
            if det.color_embedding is not None and not np.all(det.color_embedding == 0):
                embeddings.append(det.color_embedding[:3])   # use L,A,B mean
                valid_idx.append(i)

        if len(embeddings) < 4:
            return [0] * len(detections)

        embeddings = np.array(embeddings, dtype=np.float32)

        # K-Means k=2 (two teams)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1.0)
        _, labels, centers = cv2.kmeans(
            embeddings, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS
        )
        labels = labels.flatten()

        # Map cluster label → team (stable ordering)
        if self.team_centroids is not None:
            # Match to existing centroids to prevent label flipping
            d0 = np.linalg.norm(centers[0] - self.team_centroids[0])
            d1 = np.linalg.norm(centers[0] - self.team_centroids[1])
            if d1 < d0:
                labels = 1 - labels  # flip
        self.team_centroids = centers

        # Build final assignment
        team_assignments = [-1] * len(detections)
        for local_i, det_i in enumerate(valid_idx):
            team_assignments[det_i] = int(labels[local_i])

        return team_assignments

    def identify_referee(self, frame: np.ndarray, bbox: Tuple) -> bool:
        """Referees typically wear black or yellow distinct from both teams."""
        x1, y1, x2, y2 = bbox
        h = y2 - y1
        crop = frame[y1:y1 + int(h * 0.4), x1:x2]
        if crop.size == 0:
            return False
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # Black: low value; Yellow: hue 20-40, high saturation
        black_mask = hsv[:, :, 2] < 60
        yellow_mask = ((hsv[:, :, 0] > 20) & (hsv[:, :, 0] < 40) &
                       (hsv[:, :, 1] > 100))
        black_ratio = black_mask.mean()
        yellow_ratio = yellow_mask.mean()
        return (black_ratio > 0.4) or (yellow_ratio > 0.4)


class PlayerDetector:
    """
    Primary detection using YOLOv8 (ultralytics).
    Falls back gracefully to HOG pedestrian detector if YOLO unavailable.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.use_yolo = False
        self.team_separator = TeamSeparator()
        self._ball_pos: Optional[Tuple[float, float]] = None
        self._init_model()

    def _init_model(self):
        try:
            from ultralytics import YOLO
            model_path = self.config.get('yolo_model', 'yolov8n.pt')
            self.model = YOLO(model_path)
            self.use_yolo = True
            logger.info(f"YOLO model loaded: {model_path}")
        except ImportError:
            logger.warning("ultralytics not installed — falling back to HOG detector")
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame: np.ndarray) -> List[RawDetection]:
        if self.use_yolo:
            return self._detect_yolo(frame)
        return self._detect_hog(frame)

    def _detect_yolo(self, frame: np.ndarray) -> List[RawDetection]:
        conf_thresh      = self.config.get('detection_confidence', 0.4)
        min_area_ratio   = self.config.get('min_bbox_area_ratio', 0.0)
        max_aspect_ratio = self.config.get('max_aspect_ratio', 99.0)
        frame_area       = float(frame.shape[0] * frame.shape[1]) or 1.0

        results = self.model(frame, conf=conf_thresh, classes=[0, 32], verbose=False)

        detections = []
        self._ball_pos = None

        for r in results:
            for box in r.boxes:
                cls  = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if cls == 32:  # ball
                    self._ball_pos = ((x1 + x2) / 2, (y1 + y2) / 2)
                    continue

                if cls == 0:   # person
                    w = max(x2 - x1, 1)
                    h = max(y2 - y1, 1)

                    # Safety rail 1: minimum bounding-box area
                    if min_area_ratio > 0.0 and (w * h) / frame_area < min_area_ratio:
                        continue  # too small — likely a crowd pixel / far-field noise

                    # Safety rail 2: maximum height/width aspect ratio
                    if h / w > max_aspect_ratio:
                        continue  # extreme sliver — crowd edge / post artifact

                    emb = self.team_separator.extract_jersey_color(frame, (x1, y1, x2, y2))
                    detections.append(RawDetection(
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                        class_id=cls,
                        color_embedding=emb,
                    ))

        return detections


    def _detect_hog(self, frame: np.ndarray) -> List[RawDetection]:
        gray = cv2.resize(frame, (640, 360))
        scale = (frame.shape[1] / 640, frame.shape[0] / 360)
        boxes, weights = self.hog.detectMultiScale(gray, winStride=(8, 8), padding=(4, 4), scale=1.05)
        detections = []
        for (x, y, w, h), weight in zip(boxes, weights):
            x1 = int(x * scale[0])
            y1 = int(y * scale[1])
            x2 = int((x + w) * scale[0])
            y2 = int((y + h) * scale[1])
            emb = self.team_separator.extract_jersey_color(frame, (x1, y1, x2, y2))
            detections.append(RawDetection(
                bbox=(x1, y1, x2, y2),
                confidence=float(weight),
                class_id=0,
                color_embedding=emb
            ))
        return detections

    def get_ball_pos(self) -> Optional[Tuple[float, float]]:
        return self._ball_pos

    def assign_teams(self, detections: List[RawDetection], frame: np.ndarray) -> List[int]:
        return self.team_separator.assign_teams(detections)
