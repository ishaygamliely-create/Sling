"""
Multi-Object Tracking Module
Uses ByteTrack-style algorithm:
  - Hungarian assignment on IoU + appearance similarity
  - Kalman Filter for motion prediction
  - Re-ID buffer for occlusion recovery (up to N frames)
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# KALMAN FILTER (constant velocity model)
# ─────────────────────────────────────────────

class KalmanTracker:
    """4-state Kalman: [cx, cy, vx, vy]"""

    def __init__(self, cx: float, cy: float):
        self.x = np.array([cx, cy, 0., 0.], dtype=float)
        # State transition
        self.F = np.eye(4)
        self.F[0, 2] = 1.0
        self.F[1, 3] = 1.0
        # Measurement matrix (observe cx, cy)
        self.H = np.zeros((2, 4))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        # Covariance
        self.P = np.eye(4) * 50.0
        self.Q = np.eye(4) * 0.1       # process noise
        self.R = np.eye(2) * 5.0       # measurement noise

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2]

    def update(self, z: np.ndarray):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    @property
    def position(self) -> Tuple[float, float]:
        return float(self.x[0]), float(self.x[1])

    @property
    def velocity(self) -> Tuple[float, float]:
        return float(self.x[2]), float(self.x[3])


# ─────────────────────────────────────────────
# TRACKED OBJECT
# ─────────────────────────────────────────────

@dataclass
class TrackedObject:
    track_id: int
    team: int
    kalman: KalmanTracker
    bbox: Tuple[int, int, int, int]
    confidence: float
    color_embedding: Optional[np.ndarray] = None
    age: int = 0
    hits: int = 1
    misses: int = 0
    pixel_pos: Tuple[float, float] = (0., 0.)
    pitch_pos: Tuple[float, float] = (0., 0.)

    def predict(self):
        pos = self.kalman.predict()
        self.pixel_pos = (float(pos[0]), float(pos[1]))
        self.age += 1

    def update(self, bbox, conf, embedding):
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        self.kalman.update(np.array([cx, cy]))
        self.pixel_pos = self.kalman.position
        self.bbox = bbox
        self.confidence = conf
        self.color_embedding = embedding
        self.hits += 1
        self.misses = 0


# ─────────────────────────────────────────────
# MULTI-OBJECT TRACKER
# ─────────────────────────────────────────────

class MultiObjectTracker:
    """
    ByteTrack-inspired tracker:
    - High-confidence detections assigned first
    - Low-confidence detections used for recovery
    - Re-ID via appearance embedding similarity
    """

    def __init__(self, config: Dict):
        self.config = config
        self.tracks: Dict[int, TrackedObject] = {}
        self.next_id = 1
        self.max_misses = config.get('max_track_misses', 30)
        self.iou_threshold = config.get('iou_threshold', 0.3)
        self.min_hits = config.get('min_hits', 2)

    def update(self, detections, frame) -> List:
        """
        detections: List[RawDetection]
        Returns List[Player-like objects with tracking IDs]
        """
        from core.detection import RawDetection
        from core.pipeline import Player

        # 1. Team assignment
        teams = self._assign_teams(detections, frame)

        # 2. Predict existing tracks
        for track in self.tracks.values():
            track.predict()

        if not detections:
            self._prune_tracks()
            return self._get_active_players()

        # 3. Build cost matrix: IoU + color similarity
        track_ids = list(self.tracks.keys())
        det_bboxes = [d.bbox for d in detections]
        track_bboxes = [self.tracks[tid].bbox for tid in track_ids]

        if track_ids and det_bboxes:
            iou_matrix = self._compute_iou_matrix(track_bboxes, det_bboxes)
            color_matrix = self._compute_color_matrix(
                [self.tracks[tid].color_embedding for tid in track_ids],
                [d.color_embedding for d in detections]
            )
            cost_matrix = 1.0 - (0.7 * iou_matrix + 0.3 * color_matrix)

            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            matched_tracks = set()
            matched_dets = set()

            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < (1.0 - self.iou_threshold):
                    tid = track_ids[r]
                    self.tracks[tid].update(
                        detections[c].bbox,
                        detections[c].confidence,
                        detections[c].color_embedding
                    )
                    matched_tracks.add(r)
                    matched_dets.add(c)

            # 4. Unmatched detections → new tracks
            for c, det in enumerate(detections):
                if c not in matched_dets:
                    cx = (det.bbox[0] + det.bbox[2]) / 2
                    cy = (det.bbox[1] + det.bbox[3]) / 2
                    new_track = TrackedObject(
                        track_id=self.next_id,
                        team=teams[c],
                        kalman=KalmanTracker(cx, cy),
                        bbox=det.bbox,
                        confidence=det.confidence,
                        color_embedding=det.color_embedding,
                        pixel_pos=(cx, cy),
                    )
                    self.tracks[self.next_id] = new_track
                    self.next_id += 1

            # 5. Unmatched tracks → increment miss counter
            for r, tid in enumerate(track_ids):
                if r not in matched_tracks:
                    self.tracks[tid].misses += 1
        else:
            # No existing tracks → create new ones
            for c, det in enumerate(detections):
                cx = (det.bbox[0] + det.bbox[2]) / 2
                cy = (det.bbox[1] + det.bbox[3]) / 2
                t = TrackedObject(
                    track_id=self.next_id,
                    team=teams[c],
                    kalman=KalmanTracker(cx, cy),
                    bbox=det.bbox,
                    confidence=det.confidence,
                    color_embedding=det.color_embedding,
                    pixel_pos=(cx, cy),
                )
                self.tracks[self.next_id] = t
                self.next_id += 1

        self._prune_tracks()
        return self._get_active_players()

    def _prune_tracks(self):
        dead = [tid for tid, t in self.tracks.items() if t.misses > self.max_misses]
        for tid in dead:
            del self.tracks[tid]

    def _get_active_players(self):
        from core.pipeline import Player
        players = []
        for track in self.tracks.values():
            if track.hits >= self.min_hits:
                players.append(Player(
                    id=track.track_id,
                    team=track.team,
                    pixel_pos=track.pixel_pos,
                    pitch_pos=track.pitch_pos,
                    bbox=track.bbox,
                    confidence=track.confidence,
                    velocity=track.kalman.velocity,
                    color_embedding=track.color_embedding,
                ))
        return players

    def _assign_teams(self, detections, frame):
        from core.detection import TeamSeparator
        sep = TeamSeparator()
        return sep.assign_teams(detections)

    def _compute_iou_matrix(self, boxes_a, boxes_b) -> np.ndarray:
        M, N = len(boxes_a), len(boxes_b)
        iou = np.zeros((M, N))
        for i, a in enumerate(boxes_a):
            for j, b in enumerate(boxes_b):
                iou[i, j] = self._iou(a, b)
        return iou

    def _iou(self, a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
        return inter / union if union > 0 else 0.0

    def _compute_color_matrix(self, emb_a, emb_b) -> np.ndarray:
        M, N = len(emb_a), len(emb_b)
        mat = np.zeros((M, N))
        for i, ea in enumerate(emb_a):
            for j, eb in enumerate(emb_b):
                if ea is not None and eb is not None:
                    na = np.linalg.norm(ea)
                    nb = np.linalg.norm(eb)
                    if na > 0 and nb > 0:
                        mat[i, j] = np.dot(ea, eb) / (na * nb)
        return np.clip(mat, 0, 1)
