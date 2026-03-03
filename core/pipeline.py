"""
core/pipeline.py
================
Orchestrator + shared data structures for the Football Intelligence System.

v2.1.0 changes:
  - SCHEMA_VERSION = "2.1.0" defined at module level.
  - model_versions.formation built from actual SPECTRAL_K constant (not hardcoded).
  - to_json() emits schema_version + model_versions as first two keys on every frame.
    These are wire-only — NOT fields on FrameAnalysis or any dataclass.
  - FormationSnapshot gains direction_known (bool) and graph_health (dict).
  - All to_dict() methods cast every leaf to plain Python primitives (no numpy leakage).
"""

from __future__ import annotations

import json
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "2.1.0"


@dataclass
class Player:
    id:               int
    team:             int
    pixel_pos:        Tuple[float, float]
    pitch_pos:        Tuple[float, float]
    bbox:             Tuple[int, int, int, int]
    confidence:       float
    velocity:         Tuple[float, float]      = (0.0, 0.0)
    color_embedding:  Optional[np.ndarray]     = None

    def to_dict(self) -> Dict:
        return {
            'id':         int(self.id),
            'team':       int(self.team),
            'pixel_pos':  [round(float(v), 1) for v in self.pixel_pos],
            'pitch_pos':  [round(float(v), 2) for v in self.pitch_pos],
            'bbox':       [int(v) for v in self.bbox],
            'confidence': round(float(self.confidence), 3),
            'velocity':   [round(float(v), 3) for v in self.velocity],
        }


@dataclass
class FormationSnapshot:
    timestamp:        float
    team:             int
    player_positions: List[Tuple[float, float]]
    structural_graph: Dict
    line_structure:   List[List[int]]
    centroid:         Tuple[float, float]
    width:            float
    depth:            float
    pressing_height:  float
    compactness:      float
    overload_zones:   Dict[str, int]
    formation_vector: np.ndarray
    closest_known:    Optional[str]
    known_confidence: float
    is_settled:          bool                  = True
    stability_score:     float                 = 1.0
    attacking_direction: int                   = 1
    direction_known:     bool                  = True
    defensive_line_x:    float                 = 0.0
    attacking_line_x:    float                 = 0.0
    distance_histogram:  Optional[np.ndarray]  = None
    graph_health:        Optional[Dict]        = None

    def to_dict(self) -> Dict:
        fv   = self.formation_vector
        hist = self.distance_histogram
        return {
            'timestamp':           round(float(self.timestamp), 2),
            'team':                int(self.team),
            'closest_known':       self.closest_known,
            'known_confidence':    round(float(self.known_confidence), 3),
            'attacking_direction': int(self.attacking_direction),
            'direction_known':     bool(self.direction_known),
            'is_settled':          bool(self.is_settled),
            'stability_score':     round(float(self.stability_score), 3),
            'pressing_height':     round(float(self.pressing_height), 2),
            'width':               round(float(self.width), 2),
            'depth':               round(float(self.depth), 2),
            'compactness':         round(float(self.compactness), 2),
            'defensive_line_x':    round(float(self.defensive_line_x), 2),
            'attacking_line_x':    round(float(self.attacking_line_x), 2),
            'centroid':            [round(float(v), 2) for v in self.centroid],
            'formation_vector':    [round(float(v), 4) for v in fv] if fv is not None else [],
            'distance_histogram':  [round(float(v), 4) for v in hist] if hist is not None else [],
            'overload_zones':      {str(k): int(v) for k, v in self.overload_zones.items()},
            'line_structure':      [[int(i) for i in line] for line in self.line_structure],
            'structural_graph': {
                'nodes': int(self.structural_graph.get('nodes', 0)),
                'edges': [
                    {'a': int(e['a']), 'b': int(e['b']),
                     'distance': round(float(e['distance']), 2),
                     'weight':   round(float(e['weight']), 4)}
                    for e in self.structural_graph.get('edges', [])
                ],
            },
            'player_positions': [
                [round(float(p[0]), 2), round(float(p[1]), 2)]
                for p in self.player_positions
            ],
            'graph_health': self.graph_health,
        }


@dataclass
class TacticalCounter:
    title:              str
    confidence:         float
    target_zone:        str
    mechanism:          str
    reasoning:          str
    supporting_metrics: Dict
    risk_tradeoffs:     Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        def _clean(v):
            if isinstance(v, (np.floating, np.integer)):
                return round(float(v), 4)
            if isinstance(v, float):
                return round(v, 4)
            return v
        return {
            'title':              str(self.title),
            'confidence':         round(float(self.confidence), 3),
            'target_zone':        str(self.target_zone),
            'mechanism':          str(self.mechanism),
            'reasoning':          str(self.reasoning),
            'supporting_metrics': {str(k): _clean(v) for k, v in self.supporting_metrics.items()},
            'risk_tradeoffs':     {str(k): str(v) for k, v in self.risk_tradeoffs.items()},
        }


@dataclass
class FrameAnalysis:
    """
    In-memory frame result.
    schema_version and model_versions are NOT stored here —
    they are injected by to_json() into every wire response.
    """
    frame_id:               int
    timestamp:              float
    processing_ms:          float
    calibration_confidence: float
    players:                List[Player]
    ball_pos:               Optional[Tuple[float, float]]
    home_formation:         Optional[FormationSnapshot]
    away_formation:         Optional[FormationSnapshot]
    home_counters:          List[TacticalCounter]
    away_counters:          List[TacticalCounter]

    @property
    def processing_time_ms(self) -> float:
        """Alias used by demo.py (kept in sync with processing_ms)."""
        return self.processing_ms


class FootballIntelligencePipeline:

    def __init__(self, config: Dict):
        self.config      = config
        self.frame_count = 0
        self._init_components()
        from core.formation import SPECTRAL_K
        self.model_versions: Dict[str, str] = {
            'detector':  config.get('yolo_model', 'hog-fallback'),
            'tracker':   'bytetrack-kalman-v1',
            'formation': f'knn-laplacian-v2-spectral{SPECTRAL_K}',
            'counter':   'metric-grounded-v2-11rules',
            'schema':    SCHEMA_VERSION,
        }
        logger.info("Pipeline initialised (schema %s)", SCHEMA_VERSION)

    def _init_components(self):
        from core.detection   import PlayerDetector
        from core.tracking    import MultiObjectTracker
        from core.calibration import PitchCalibrator
        from core.formation   import DynamicFormationEngine
        from core.counter     import CounterTacticEngine
        self.detector   = PlayerDetector(self.config)
        self.tracker    = MultiObjectTracker(self.config)
        self.calibrator = PitchCalibrator(self.config)
        self.formation  = DynamicFormationEngine(self.config)
        self.counter    = CounterTacticEngine(self.config)

    def process_frame(self, frame: np.ndarray, frame_id: int) -> FrameAnalysis:
        t0 = time.perf_counter()
        detections = self.detector.detect(frame)
        ball_pixel = self.detector.get_ball_pos()
        players    = self.tracker.update(detections, frame)
        self.calibrator.calibrate(frame)
        conf = self.calibrator.calibration_confidence
        for p in players:
            p.pitch_pos = self.calibrator.to_pitch_coords(p.pixel_pos)
        ball_pitch = self.calibrator.to_pitch_coords(ball_pixel) if ball_pixel else None
        snap_home = self.formation.analyze(players, team=0, frame_id=frame_id, calib_confidence=conf)
        snap_away = self.formation.analyze(players, team=1, frame_id=frame_id, calib_confidence=conf)
        home_counters = away_counters = []
        if snap_home and snap_away and snap_home.is_settled and snap_away.is_settled:
            home_counters = self.counter.generate(snap_away, snap_home)
            away_counters = self.counter.generate(snap_home, snap_away)
        ms = (time.perf_counter() - t0) * 1000
        self.frame_count += 1
        return FrameAnalysis(
            frame_id=frame_id, timestamp=round(frame_id/25.0, 3),
            processing_ms=round(ms, 1), calibration_confidence=round(conf, 3),
            players=players, ball_pos=ball_pitch,
            home_formation=snap_home, away_formation=snap_away,
            home_counters=home_counters, away_counters=away_counters,
        )

    def process_positions(
        self,
        home_positions: List[Tuple[float, float]],
        away_positions: List[Tuple[float, float]],
        home_velocities: Optional[List[Tuple[float, float]]] = None,
        away_velocities: Optional[List[Tuple[float, float]]] = None,
        calib_confidence: float = 1.0,
        frame_id: Optional[int] = None,
    ) -> 'FrameAnalysis':
        """Positions-mode entry point: takes pre-computed pitch coords directly."""
        import time as _time
        t0 = _time.perf_counter()
        if frame_id is None:
            frame_id = self.frame_count
        home_velocities = home_velocities or [(0.0, 0.0)] * len(home_positions)
        away_velocities = away_velocities or [(0.0, 0.0)] * len(away_positions)
        players: List[Player] = []
        for i, (pos, vel) in enumerate(zip(home_positions, home_velocities)):
            players.append(Player(
                id=i, team=0, pixel_pos=(0.0, 0.0), pitch_pos=pos,
                bbox=(0, 0, 1, 1), confidence=1.0, velocity=vel,
            ))
        for i, (pos, vel) in enumerate(zip(away_positions, away_velocities)):
            players.append(Player(
                id=100+i, team=1, pixel_pos=(0.0, 0.0), pitch_pos=pos,
                bbox=(0, 0, 1, 1), confidence=1.0, velocity=vel,
            ))
        snap_home = self.formation.analyze(players, team=0, frame_id=frame_id,
                                           calib_confidence=calib_confidence)
        snap_away = self.formation.analyze(players, team=1, frame_id=frame_id,
                                           calib_confidence=calib_confidence)
        home_counters = away_counters = []
        if snap_home and snap_away and snap_home.is_settled and snap_away.is_settled:
            home_counters = self.counter.generate(snap_away, snap_home)
            away_counters = self.counter.generate(snap_home, snap_away)
        ms = (_time.perf_counter() - t0) * 1000
        self.frame_count += 1
        return FrameAnalysis(
            frame_id=frame_id, timestamp=round(frame_id / 25.0, 3),
            processing_ms=round(ms, 1), calibration_confidence=float(calib_confidence),
            players=players, ball_pos=None,
            home_formation=snap_home, away_formation=snap_away,
            home_counters=home_counters, away_counters=away_counters,
        )

    def process_video(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        frame_id = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                yield self.process_frame(frame, frame_id)
                frame_id += 1
        finally:
            cap.release()

    def get_formation_timeline(self, team: int) -> List[Dict]:
        return self.formation.get_timeline(team)

    def to_json(self, analysis: FrameAnalysis) -> str:
        d: Dict[str, Any] = {
            'schema_version': SCHEMA_VERSION,
            'model_versions': dict(self.model_versions),
            'frame_id':               int(analysis.frame_id),
            'timestamp':              float(analysis.timestamp),
            'processing_ms':          float(analysis.processing_ms),
            'calibration_confidence': float(analysis.calibration_confidence),
            'players': [p.to_dict() for p in analysis.players],
            'ball': ({'pitch_pos': [round(float(v), 2) for v in analysis.ball_pos]}
                     if analysis.ball_pos else None),
            'home_formation': (analysis.home_formation.to_dict() if analysis.home_formation else None),
            'away_formation': (analysis.away_formation.to_dict() if analysis.away_formation else None),
            'home_counters': [c.to_dict() for c in analysis.home_counters],
            'away_counters': [c.to_dict() for c in analysis.away_counters],
        }
        return json.dumps(d)
