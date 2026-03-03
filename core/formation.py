"""
Formation Recognition Engine v2.1
===================================

v2.1 changes:
  - FormationGraph.build() gains adaptive sigma + connectivity check.
    If graph is disconnected or any node degree < MIN_DEGREE, sigma is
    multiplied by SIGMA_STEP until fixed or SIGMA_MAX_MULT reached, then
    radius cap is dropped (pure kNN). Diagnostics in graph['graph_health'].
  - distance_histogram uses bounding-box diagonal normalisation (zoom invariant).
  - DirectionNormaliser: GK-anchor method. Bug fixed: `continue` not `return 0`
    in middle-region case so both extremes are always checked.
  - LineDetector: purely coordinate-based x-band grouping, MIN_LINE_SIZE=2,
    singletons never returned. Works in both positions mode and video mode.
"""

from __future__ import annotations

import math
import logging
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

PITCH_W = 105.0
PITCH_H = 68.0

SPECTRAL_K      = 8
KNN_K           = 4
RADIUS_CAP      = 25.0
HIST_BINS       = 12
BLEND_SPECTRAL  = 0.6
BLEND_HIST      = 0.4

MIN_DEGREE      = 1
SIGMA_STEP      = 1.5
SIGMA_MAX_MULT  = 4.0

MIN_OUTFIELD            = 7
SETTLE_VELOCITY_THR     = 2.5
SETTLE_PLAYER_STABILITY = 0.7
SETTLE_MIN_CALIB_CONF   = 0.4

GK_GOAL_THR_LO  = 20.0
GK_GOAL_THR_HI  = 85.0
GK_ISO_DIST_M   = 8.0
VOTE_WINDOW     = 75
HYSTERESIS_FRAC = 0.70
VELOCITY_WINDOW = 60

KNOWN_FORMATIONS: Dict[str, List[Tuple[float, float]]] = {
    '4-4-2':   [(0.15,0.18),(0.15,0.38),(0.15,0.62),(0.15,0.82),
                (0.50,0.15),(0.50,0.38),(0.50,0.62),(0.50,0.85),
                (0.78,0.32),(0.78,0.68)],
    '4-3-3':   [(0.15,0.18),(0.15,0.38),(0.15,0.62),(0.15,0.82),
                (0.45,0.20),(0.45,0.50),(0.45,0.80),
                (0.78,0.12),(0.78,0.50),(0.78,0.88)],
    '4-2-3-1': [(0.15,0.18),(0.15,0.38),(0.15,0.62),(0.15,0.82),
                (0.36,0.35),(0.36,0.65),
                (0.55,0.15),(0.55,0.50),(0.55,0.85),
                (0.80,0.50)],
    '3-5-2':   [(0.15,0.25),(0.15,0.50),(0.15,0.75),
                (0.45,0.10),(0.45,0.32),(0.45,0.50),(0.45,0.68),(0.45,0.90),
                (0.78,0.35),(0.78,0.65)],
    '5-3-2':   [(0.13,0.10),(0.13,0.30),(0.13,0.50),(0.13,0.70),(0.13,0.90),
                (0.45,0.25),(0.45,0.50),(0.45,0.75),
                (0.78,0.35),(0.78,0.65)],
    '3-4-3':   [(0.15,0.25),(0.15,0.50),(0.15,0.75),
                (0.42,0.10),(0.42,0.38),(0.42,0.62),(0.42,0.90),
                (0.78,0.18),(0.78,0.50),(0.78,0.82)],
    '4-1-4-1': [(0.15,0.18),(0.15,0.38),(0.15,0.62),(0.15,0.82),
                (0.32,0.50),
                (0.55,0.12),(0.55,0.35),(0.55,0.65),(0.55,0.88),
                (0.82,0.50)],
    '4-4-1-1': [(0.15,0.18),(0.15,0.38),(0.15,0.62),(0.15,0.82),
                (0.50,0.15),(0.50,0.38),(0.50,0.62),(0.50,0.85),
                (0.68,0.50),(0.82,0.50)],
}


class FormationGraph:

    def build(self, positions: List[Tuple[float, float]]) -> Dict:
        n = len(positions)
        if n < 3:
            return {
                'nodes': n, 'edges': [], 'adjacency': {i: {} for i in range(n)},
                'graph_health': self._make_health(RADIUS_CAP/3.0, 1.0, 0, n<=1, False, max(n,1)),
            }

        pts      = np.array(positions, dtype=float)
        diff     = pts[:, None, :] - pts[None, :, :]
        dist_mat = np.sqrt((diff**2).sum(-1))
        np.fill_diagonal(dist_mat, np.inf)

        base_sigma = RADIUS_CAP / 3.0
        sigma      = base_sigma
        fallback   = False
        adjacency: Dict = {}
        edge_set:  Dict = {}

        for _attempt in range(6):
            adjacency, edge_set = self._build_with_sigma(n, dist_mat, sigma, use_radius=not fallback)
            degree    = [len(adjacency[i]) for i in range(n)]
            connected, n_comp = self._check_connectivity(adjacency, n)
            if min(degree) >= MIN_DEGREE and connected:
                break
            mult = sigma / base_sigma
            if mult >= SIGMA_MAX_MULT:
                fallback  = True
                adjacency, edge_set = self._build_with_sigma(n, dist_mat, sigma, use_radius=False)
                degree    = [len(adjacency[i]) for i in range(n)]
                connected, n_comp = self._check_connectivity(adjacency, n)
                break
            sigma *= SIGMA_STEP

        mult   = sigma / base_sigma
        health = self._make_health(sigma, mult, min(degree), connected, fallback, n_comp)
        if mult > 1.0 or fallback:
            logger.debug("Graph adapted: sigma×%.2f fallback=%s min_deg=%d connected=%s",
                         mult, fallback, min(degree), connected)
        return {'nodes': n, 'edges': list(edge_set.values()), 'adjacency': adjacency, 'graph_health': health}

    def spectral_signature(self, graph: Dict) -> np.ndarray:
        n = graph['nodes']
        if n < 2:
            return np.full(SPECTRAL_K, 2.0)
        adj = graph['adjacency']
        A   = np.zeros((n, n))
        for i, nbrs in adj.items():
            for j, w in nbrs.items():
                A[i, j] = w
        degree     = A.sum(axis=1)
        d_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0)
        L_sym      = np.eye(n) - np.diag(d_inv_sqrt) @ A @ np.diag(d_inv_sqrt)
        eigs        = np.sort(np.clip(np.linalg.eigvalsh(L_sym), 0.0, 2.0))
        non_trivial = eigs[eigs > 1e-6]
        if len(non_trivial) >= SPECTRAL_K:
            return non_trivial[:SPECTRAL_K]
        return np.pad(non_trivial, (0, SPECTRAL_K - len(non_trivial)), constant_values=2.0)

    def distance_histogram(self, positions: List[Tuple[float, float]]) -> np.ndarray:
        MIN_DIAG_M = 15.0
        pts = np.array(positions, dtype=float)
        if len(pts) < 2:
            return np.zeros(HIST_BINS)
        bb_w  = float(pts[:,0].max() - pts[:,0].min())
        bb_h  = float(pts[:,1].max() - pts[:,1].min())
        scale = max(math.hypot(bb_w, bb_h), MIN_DIAG_M)
        diff  = pts[:, None, :] - pts[None, :, :]
        dists = np.sqrt((diff**2).sum(-1)) / scale
        upper = dists[np.triu_indices(len(pts), k=1)]
        hist, _ = np.histogram(upper, bins=HIST_BINS, range=(0.0, 1.5))
        hist = hist.astype(float)
        s    = hist.sum()
        return hist / s if s > 0 else hist

    @staticmethod
    def _build_with_sigma(n, dist_mat, sigma, use_radius):
        adjacency: Dict[int, Dict[int, float]] = {i: {} for i in range(n)}
        edge_set:  Dict = {}
        for i in range(n):
            row = dist_mat[i]
            if use_radius:
                effective_radius = sigma * 3.0   # grows with sigma so loop actually expands connectivity
                within = np.where(row <= effective_radius)[0]
                if len(within) == 0:
                    within = np.array([int(np.argmin(row))])
            else:
                within = np.array([j for j in range(n) if j != i])
            near = within[np.argsort(row[within])][:KNN_K]
            for j in near:
                d = float(dist_mat[i, j])
                w = float(np.exp(-(d**2) / (2 * sigma**2)))
                adjacency[i][int(j)] = w
                adjacency[int(j)][i] = w
                key = (min(i, int(j)), max(i, int(j)))
                edge_set[key] = {'a': key[0], 'b': key[1], 'distance': round(d,2), 'weight': round(w,4)}
        return adjacency, edge_set

    @staticmethod
    def _check_connectivity(adjacency, n):
        if n == 0:
            return True, 0
        visited    = set()
        components = 0
        for start in range(n):
            if start in visited:
                continue
            components += 1
            queue = [start]
            while queue:
                node = queue.pop()
                if node in visited:
                    continue
                visited.add(node)
                queue.extend(nb for nb in adjacency.get(node, {}) if nb not in visited)
        return components == 1, components

    @staticmethod
    def _make_health(sigma, mult, min_deg, connected, fallback, n_comp):
        return {
            'sigma_used':        round(float(sigma), 3),
            'sigma_multiplier':  round(float(mult),  2),
            'min_degree':        int(min_deg),
            'is_connected':      bool(connected),
            'fallback_knn_only': bool(fallback),
            'n_components':      int(n_comp),
        }


class DirectionNormaliser:
    """
    GK-anchor direction estimator.

    v2.1 bugfix: loop uses `continue` (not `return 0`) when a candidate is
    isolated but in the middle-region dead zone, so both leftmost AND
    rightmost players are always checked before giving up.
    """

    def __init__(self):
        self._dir:           Dict[int, int]   = {}
        self._votes:         Dict[int, deque] = {0: deque(maxlen=VOTE_WINDOW),
                                                   1: deque(maxlen=VOTE_WINDOW)}
        self._known:         Dict[int, bool]  = {0: False, 1: False}
        self._centroid_hist: Dict[int, deque] = {0: deque(maxlen=VELOCITY_WINDOW),
                                                   1: deque(maxlen=VELOCITY_WINDOW)}

    def update_and_normalise(self, positions, team):
        if not positions:
            return positions, self._known.get(team, False)
        pts = np.array(positions)
        self._centroid_hist[team].append(float(pts[:, 0].mean()))
        vote = self._gk_vote(pts)
        if vote == 0:
            vote = self._velocity_vote(team)
        if vote != 0:
            self._votes[team].append(vote)
        direction = self._committed_direction(team)
        if direction == 0:
            return [(float(p[0]), float(p[1])) for p in positions], False
        if direction == 1:
            return [(float(p[0]), float(p[1])) for p in positions], True
        return [(PITCH_W - float(p[0]), float(p[1])) for p in positions], True

    def get_direction(self, team):
        return self._dir.get(team, 0)

    def direction_known(self, team):
        return self._known.get(team, False)

    def _gk_vote(self, pts):
        if len(pts) < 3:
            return 0
        xs = pts[:, 0]
        for idx in [int(np.argmin(xs)), int(np.argmax(xs))]:
            gk_x     = xs[idx]
            other_xs = np.delete(xs, idx)
            gap      = float(np.min(np.abs(other_xs - gk_x)))
            if gap >= GK_ISO_DIST_M:
                if gk_x <= GK_GOAL_THR_LO: return +1
                if gk_x >= GK_GOAL_THR_HI: return -1
                continue   # isolated but middle region — check other extreme
        lx, rx = float(xs.min()), float(xs.max())
        if lx <= GK_GOAL_THR_LO: return +1
        if rx >= GK_GOAL_THR_HI: return -1
        return 0

    def _velocity_vote(self, team):
        hist = list(self._centroid_hist[team])
        if len(hist) < VELOCITY_WINDOW // 2:
            return 0
        q     = max(1, len(hist) // 4)
        early = float(np.mean(hist[:q]))
        late  = float(np.mean(hist[-q:]))
        net   = late - early
        if abs(net) < 3.0:
            return 0
        return +1 if net > 0 else -1

    def _committed_direction(self, team):
        votes = list(self._votes[team])
        n     = len(votes)
        if n == 0:
            return 0
        pos_frac = sum(1 for v in votes if v == +1) / n
        neg_frac = 1.0 - pos_frac
        if pos_frac >= HYSTERESIS_FRAC:
            self._dir[team] = +1; self._known[team] = True
        elif neg_frac >= HYSTERESIS_FRAC:
            self._dir[team] = -1; self._known[team] = True
        return self._dir.get(team, 0)


class LineDetector:
    """
    Groups players into lines by x-band clustering (coordinate-based, no image needed).
    Groups with < MIN_LINE_SIZE players are discarded. Singletons are never returned.
    """
    LINE_DIST_THR = 3.0
    MIN_LINE_SIZE = 2

    def detect_lines(self, positions):
        if len(positions) < self.MIN_LINE_SIZE:
            return []
        pts  = np.array(positions)
        xs   = pts[:, 0]
        used: set = set()
        lines = []
        for i in np.argsort(xs):
            if i in used:
                continue
            group = [int(i)]
            for j in range(len(positions)):
                if j != i and j not in used and abs(xs[i] - xs[j]) <= self.LINE_DIST_THR:
                    group.append(int(j))
            if len(group) >= self.MIN_LINE_SIZE:
                for idx in group:
                    used.add(idx)
                lines.append(sorted(group))
        return lines


class TacticalMetrics:

    def compute(self, positions, team):
        if not positions:
            return {}
        pts = np.array(positions, dtype=float)
        x, y = pts[:, 0], pts[:, 1]
        diff        = pts[:, None, :] - pts[None, :, :]
        dists       = np.sqrt((diff**2).sum(-1))
        upper       = dists[np.triu_indices(len(pts), k=1)]
        compactness = float(upper.mean()) if len(upper) > 0 else 0.0
        x_bins = np.array([0, PITCH_W/3, 2*PITCH_W/3, PITCH_W])
        y_bins = np.array([0, PITCH_H/3, 2*PITCH_H/3, PITCH_H])
        xn = ['defensive','middle','attacking']
        yn = ['left','centre','right']
        zones: Dict[str, int] = {}
        for xi in range(3):
            for yi in range(3):
                m = ((pts[:,0] >= x_bins[xi]) & (pts[:,0] < x_bins[xi+1]) &
                     (pts[:,1] >= y_bins[yi]) & (pts[:,1] < y_bins[yi+1]))
                if m.sum():
                    zones[f"{xn[xi]}_{yn[yi]}"] = int(m.sum())
        return {
            'centroid':         (float(x.mean()), float(y.mean())),
            'width':            float(y.max() - y.min()),
            'depth':            float(x.max() - x.min()),
            'compactness':      compactness,
            'pressing_height':  float(x.mean()),
            'defensive_line_x': float(x.min()),
            'attacking_line_x': float(x.max()),
            'vertical_spread':  float(y.std()),
            'overload_zones':   zones,
        }


class FormationMatcher:

    def __init__(self):
        self._g         = FormationGraph()
        self._templates: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._precompute()

    def _precompute(self):
        for name, pos in KNOWN_FORMATIONS.items():
            pts_m = [(x*PITCH_W, y*PITCH_H) for x,y in pos]
            graph = self._g.build(pts_m)
            spec  = self._g.spectral_signature(graph)
            hist  = self._g.distance_histogram(pts_m)
            self._templates[name] = (spec, hist)

    def match(self, spec, hist):
        best_name = None
        best_sim  = -1.0
        for name, (ts, th) in self._templates.items():
            s = BLEND_SPECTRAL * self._cos(spec, ts) + BLEND_HIST * self._cos(hist, th)
            if s > best_sim:
                best_sim  = s
                best_name = name
        return best_name, float(np.clip(best_sim, 0.0, 1.0))

    @staticmethod
    def _cos(a, b):
        n  = min(len(a), len(b))
        a, b = a[:n], b[:n]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            return 0.0
        return float(np.clip(np.dot(a, b) / (na * nb), 0.0, 1.0))


class SettledStateDetector:

    def __init__(self, window=30):
        self._vel: deque = deque(maxlen=window)
        self._cnt: deque = deque(maxlen=window)

    def update(self, velocities, player_count, calib_conf):
        mean_spd = float(np.mean([np.hypot(vx, vy) for vx, vy in velocities])) if velocities else 0.0
        self._vel.append(mean_spd)
        self._cnt.append(player_count)
        avg_spd     = float(np.mean(self._vel))
        spd_score   = float(np.clip(1.0 - avg_spd / SETTLE_VELOCITY_THR, 0, 1))
        cnt_ok      = sum(1 for c in self._cnt if c >= MIN_OUTFIELD) / max(len(self._cnt), 1)
        calib_score = float(np.clip(calib_conf, 0, 1))
        stability   = 0.4 * spd_score + 0.35 * cnt_ok + 0.25 * calib_score
        is_settled  = (avg_spd < SETTLE_VELOCITY_THR and
                       cnt_ok >= SETTLE_PLAYER_STABILITY and
                       calib_conf >= SETTLE_MIN_CALIB_CONF)
        return is_settled, float(stability)


class DynamicFormationEngine:

    SMOOTHING_ALPHA = 0.25

    def __init__(self, config):
        self.config         = config
        window              = config.get('formation_window_frames', 150)
        self.graph_builder  = FormationGraph()
        self.line_detector  = LineDetector()
        self.metrics_engine = TacticalMetrics()
        self.matcher        = FormationMatcher()
        self.direction      = DirectionNormaliser()
        self.settle_detector = {0: SettledStateDetector(30), 1: SettledStateDetector(30)}
        self._smooth_spec   = {0: None, 1: None}
        self._smooth_hist   = {0: None, 1: None}
        self._frame_history = {0: deque(maxlen=window), 1: deque(maxlen=window)}
        self._timeline      = {0: [], 1: []}

    def analyze(self, players, team, frame_id, calib_confidence=1.0):
        from core.pipeline import FormationSnapshot
        outfield      = [p for p in players if p.team == team]
        positions_raw = [p.pitch_pos for p in outfield]
        velocities    = [getattr(p, 'velocity', (0.0, 0.0)) for p in outfield]
        is_settled, stability = self.settle_detector[team].update(velocities, len(positions_raw), calib_confidence)
        if len(positions_raw) < MIN_OUTFIELD:
            return None
        positions, dir_known = self.direction.update_and_normalise(positions_raw, team)
        self._frame_history[team].append(positions)
        graph       = self.graph_builder.build(positions)
        raw_spec    = self.graph_builder.spectral_signature(graph)
        raw_hist    = self.graph_builder.distance_histogram(positions)
        smooth_spec = self._ema(self._smooth_spec[team], raw_spec)
        smooth_hist = self._ema(self._smooth_hist[team], raw_hist)
        self._smooth_spec[team] = smooth_spec
        self._smooth_hist[team] = smooth_hist
        lines   = self.line_detector.detect_lines(positions)
        m       = self.metrics_engine.compute(positions, team)
        closest, conf = self.matcher.match(smooth_spec, smooth_hist)
        ts = round(frame_id / 25.0, 2)
        if is_settled:
            last = self._timeline[team][-1] if self._timeline[team] else None
            if last is None or abs(ts - last['timestamp']) > 2.0:
                self._timeline[team].append({
                    'timestamp': ts, 'closest_known': closest,
                    'known_confidence': round(conf, 3), 'stability_score': round(stability, 3),
                    'pressing_height': round(m.get('pressing_height', 0), 2),
                    'attacking_direction': int(self.direction.get_direction(team)),
                    'direction_known': bool(dir_known),
                })
                if len(self._timeline[team]) > 100:
                    self._timeline[team] = self._timeline[team][-100:]
        return FormationSnapshot(
            timestamp=ts, team=team, player_positions=positions,
            structural_graph={'nodes': graph['nodes'], 'edges': graph['edges']},
            line_structure=lines,
            centroid=m.get('centroid', (0.0, 0.0)),
            width=m.get('width', 0.0), depth=m.get('depth', 0.0),
            pressing_height=m.get('pressing_height', 0.0),
            compactness=m.get('compactness', 0.0),
            overload_zones=m.get('overload_zones', {}),
            formation_vector=smooth_spec, closest_known=closest, known_confidence=conf,
            is_settled=is_settled, stability_score=stability,
            attacking_direction=self.direction.get_direction(team),
            direction_known=dir_known,
            defensive_line_x=m.get('defensive_line_x', 0.0),
            attacking_line_x=m.get('attacking_line_x', 0.0),
            distance_histogram=smooth_hist,
            graph_health=graph.get('graph_health'),
        )

    def get_timeline(self, team):
        return self._timeline.get(team, [])

    def _ema(self, prev, curr):
        if prev is None or len(prev) != len(curr):
            return curr.copy()
        return self.SMOOTHING_ALPHA * curr + (1 - self.SMOOTHING_ALPHA) * prev
