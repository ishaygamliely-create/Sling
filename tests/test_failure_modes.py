"""
tests/test_failure_modes.py — 24 regression tests
Run: python -m unittest tests/test_failure_modes.py -v
     pytest tests/test_failure_modes.py -v
"""

from __future__ import annotations
import json, math, os, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np


def make_players(positions, team=0, base_id=0, velocity=(0.3, 0.1)):
    class FP:
        def __init__(self, id_, t, x, y, v):
            self.id=id_; self.team=t; self.pitch_pos=(x,y); self.velocity=v
            self.pixel_pos=(0.0,0.0); self.bbox=(0,0,1,1); self.confidence=1.0
        def to_dict(self):
            return {
                'id':int(self.id),'team':int(self.team),
                'pixel_pos':[0.0,0.0],
                'pitch_pos':[round(float(self.pitch_pos[0]),2),round(float(self.pitch_pos[1]),2)],
                'bbox':[0,0,1,1],'confidence':1.0,
                'velocity':[round(float(v),3) for v in self.velocity],
            }
    return [FP(base_id+i, team, x, y, velocity) for i,(x,y) in enumerate(positions)]

BASE_433 = [(8,34),(18,12),(18,27),(18,41),(18,56),(45,20),(45,34),(45,50),(72,12),(72,34),(72,56)]

def jittered(rng, base=BASE_433, sigma=0.5):
    return [(x+rng.normal(0,sigma), y+rng.normal(0,sigma)) for x,y in base]


# ── 1. Occlusion dropout ─────────────────────────────────────────────────────
class TestOcclusionDropout(unittest.TestCase):

    def test_stability_survives_30pct_dropout(self):
        from core.formation import DynamicFormationEngine, SPECTRAL_K
        rng    = np.random.default_rng(42)
        engine = DynamicFormationEngine({'formation_window_frames': 30})
        scores = []
        none_count = 0
        for fi in range(50):
            players = make_players(jittered(rng), team=0)
            visible = [p for p,m in zip(players, rng.random(len(players))>0.30) if m]
            snap    = engine.analyze(visible, team=0, frame_id=fi, calib_confidence=0.8)
            if snap is None: none_count += 1; continue
            scores.append(snap.stability_score)
            self.assertEqual(len(snap.formation_vector), SPECTRAL_K)
        self.assertGreater(len(scores), 50*0.4)
        min_s = min(scores)
        self.assertGreater(min_s, 0.20, f"stability_score collapsed to {min_s:.3f}")
        print(f"\n  [PASS] Occlusion: min_stability={min_s:.3f}, snapshots={len(scores)}/50, none={none_count}")


# ── 2. Wrong homography freeze ────────────────────────────────────────────────
class TestWrongHomographyFreeze(unittest.TestCase):

    def _good_frame(self, w=1280, h=720):
        import cv2
        frame = np.zeros((h,w,3), dtype=np.uint8); frame[:] = (34,85,34)
        cv2.rectangle(frame,(64,58),(w-64,h-58),(255,255,255),3)
        cv2.line(frame,(w//2,58),(w//2,h-58),(255,255,255),2)
        return frame

    def _bad_frame(self, w=1280, h=720):
        frame = np.zeros((1280,720,3), dtype=np.uint8); frame[:] = (20,20,20)
        return frame

    def test_freeze_on_bad_frame(self):
        from core.calibration import PitchCalibrator
        cal = PitchCalibrator({})
        cal.calibrate(self._good_frame())
        initial_conf = cal.calibration_confidence
        self.assertTrue(cal.is_calibrated)
        cal.calibrate(self._bad_frame())
        pt = cal.to_pitch_coords((640.0, 360.0))
        self.assertFalse(any(math.isnan(v) for v in pt))
        self.assertFalse(any(math.isinf(v) for v in pt))
        x, y = pt
        self.assertGreaterEqual(x, 0.0); self.assertLessEqual(x, 105.0)
        self.assertGreaterEqual(y, 0.0); self.assertLessEqual(y, 68.0)
        ok = cal.set_manual_points(
            pixel_pts=[[64,58],[1216,58],[0,662],[1280,662]],
            pitch_pts=[[0,0],[105,0],[0,68],[105,68]])
        self.assertTrue(ok)
        self.assertAlmostEqual(cal.calibration_confidence, 1.0)
        cal.calibrate(self._bad_frame())
        self.assertAlmostEqual(cal.calibration_confidence, 1.0,
            msg="Manual lock should survive bad frame")
        print(f"\n  [PASS] Freeze: initial={initial_conf:.3f}, locked conf={cal.calibration_confidence:.3f}")


# ── 3. Broadcast zoom stability ───────────────────────────────────────────────
class TestBroadcastZoomStability(unittest.TestCase):

    SIM_THRESHOLD  = 0.85
    HIST_THRESHOLD = 0.90

    def _cos(self, a, b):
        n=min(len(a),len(b)); a,b=a[:n],b[:n]
        na,nb=np.linalg.norm(a),np.linalg.norm(b)
        if na<1e-9 or nb<1e-9: return 0.0
        return float(np.dot(a,b)/(na*nb))

    def test_spectral_zoom_invariance(self):
        from core.formation import FormationGraph
        fg   = FormationGraph()
        base = np.array([(18,12),(18,27),(18,41),(18,56),(45,20),(45,34),
                         (45,52),(72,12),(72,34),(72,56)], dtype=float)
        centre = np.array([52.5, 34.0])
        results = {}
        for scale in [0.7, 1.0, 1.3]:
            scaled = centre + (base-centre)*scale
            scaled[:,0] = np.clip(scaled[:,0],0,105); scaled[:,1] = np.clip(scaled[:,1],0,68)
            pos = list(map(tuple, scaled))
            g   = fg.build(pos)
            results[scale] = (fg.spectral_signature(g), fg.distance_histogram(pos))
        ref_spec, ref_hist = results[1.0]
        for scale in [0.7, 1.3]:
            ts, th = results[scale]
            ss = self._cos(ref_spec, ts); hs = self._cos(ref_hist, th)
            print(f"\n  [INFO] zoom={scale:.1f}×: spec_sim={ss:.3f} hist_sim={hs:.3f}")
            self.assertGreater(ss, self.SIM_THRESHOLD)
            self.assertGreater(hs, self.HIST_THRESHOLD)
        print("\n  [PASS] Zoom invariance passed for [0.7×, 1.3×]")

    def test_spectral_descriptor_fixed_length(self):
        from core.formation import FormationGraph, SPECTRAL_K
        fg = FormationGraph()
        for n in [7, 9, 10, 11]:
            rng = np.random.default_rng(n)
            pts = [(rng.uniform(10,95), rng.uniform(5,63)) for _ in range(n)]
            sig = fg.spectral_signature(fg.build(pts))
            self.assertEqual(len(sig), SPECTRAL_K)
            self.assertTrue(all(0.0 <= v <= 2.001 for v in sig))
        print(f"\n  [PASS] Spectral descriptor fixed at SPECTRAL_K={SPECTRAL_K}")


# ── 4. Direction normaliser ───────────────────────────────────────────────────
from core.formation import VOTE_WINDOW

class TestDirectionNormaliser(unittest.TestCase):

    def _run(self, engine, frames, team=0):
        snaps = []
        for fi, pos in enumerate(frames):
            snap = engine.analyze(make_players(pos, team=team), team=team, frame_id=fi, calib_confidence=0.8)
            snaps.append(snap)
        return snaps

    def test_gk_left_attacks_right(self):
        from core.formation import DynamicFormationEngine
        engine = DynamicFormationEngine({'formation_window_frames': 30})
        rng    = np.random.default_rng(1)
        def frame():
            pts  = [(5, 34)]
            pts += [(18+rng.normal(0,.5), 12+i*12+rng.normal(0,.5)) for i in range(4)]
            pts += [(45+rng.normal(0,.5), 20+i*15+rng.normal(0,.5)) for i in range(3)]
            pts += [(72+rng.normal(0,.5), 12+i*22+rng.normal(0,.5)) for i in range(3)]
            return pts
        snaps   = self._run(engine, [frame() for _ in range(VOTE_WINDOW+10)])
        settled = [s for s in snaps if s and s.direction_known]
        self.assertGreater(len(settled), VOTE_WINDOW*0.5)
        self.assertEqual(settled[-1].attacking_direction, 1)
        print("\n  [PASS] GK at x=5 → attacking_direction=+1")

    def test_gk_right_attacks_left(self):
        from core.formation import DynamicFormationEngine
        engine = DynamicFormationEngine({'formation_window_frames': 30})
        rng    = np.random.default_rng(2)
        def frame():
            pts  = [(100, 34)]
            pts += [(87+rng.normal(0,.5), 12+i*12+rng.normal(0,.5)) for i in range(4)]
            pts += [(60+rng.normal(0,.5), 20+i*15+rng.normal(0,.5)) for i in range(3)]
            pts += [(33+rng.normal(0,.5), 12+i*22+rng.normal(0,.5)) for i in range(3)]
            return pts
        snaps   = self._run(engine, [frame() for _ in range(VOTE_WINDOW+10)], team=0)
        settled = [s for s in snaps if s and s.direction_known]
        self.assertGreater(len(settled), VOTE_WINDOW*0.5)
        self.assertEqual(settled[-1].attacking_direction, -1)
        print("\n  [PASS] GK at x=100 → attacking_direction=-1")

    def test_midfield_crossings_no_flip(self):
        from core.formation import DynamicFormationEngine
        engine = DynamicFormationEngine({'formation_window_frames': 30})
        rng    = np.random.default_rng(3)
        def settled_frame():
            return [(5,34)]+[(18,12+i*12) for i in range(4)]+[(45,20+i*15) for i in range(3)]+[(72,12+i*22) for i in range(3)]
        self._run(engine, [settled_frame()]*(VOTE_WINDOW+5))
        def crossing_frame(fwd_x):
            return [(5,34)]+[(18,12+i*12) for i in range(4)]+[(fwd_x,20+i*15) for i in range(3)]+[(fwd_x+20,12+i*22) for i in range(3)]
        osc = [crossing_frame(65 if i%10<5 else 40) for i in range(40)]
        for s in [s for s in self._run(engine, osc) if s and s.direction_known]:
            self.assertEqual(s.attacking_direction, 1)
        print("\n  [PASS] Direction stable (+1) through midfield crossings")


# ── 5. GK vote bug regression ─────────────────────────────────────────────────
class TestGKVoteBugRegression(unittest.TestCase):

    AWAY_PTS = np.array([(97,34),(87,10),(87,27),(87,41),(87,58),
                          (67,24),(67,46),(55,10),(55,34),(55,58),(42,34)])
    LO=20.0; HI=85.0; ISO=8.0

    def _broken(self, pts):
        xs = pts[:,0]
        for idx in [int(np.argmin(xs)), int(np.argmax(xs))]:
            gk_x=xs[idx]; other_xs=np.delete(xs,idx)
            gap=float(np.min(np.abs(other_xs-gk_x)))
            if gap>=self.ISO:
                if gk_x<=self.LO: return +1
                if gk_x>=self.HI: return -1
                return 0   # BUG
        lx,rx=float(xs.min()),float(xs.max())
        if lx<=self.LO: return +1
        if rx>=self.HI: return -1
        return 0

    def _fixed(self, pts):
        xs = pts[:,0]
        for idx in [int(np.argmin(xs)), int(np.argmax(xs))]:
            gk_x=xs[idx]; other_xs=np.delete(xs,idx)
            gap=float(np.min(np.abs(other_xs-gk_x)))
            if gap>=self.ISO:
                if gk_x<=self.LO: return +1
                if gk_x>=self.HI: return -1
                continue   # FIX
        lx,rx=float(xs.min()),float(xs.max())
        if lx<=self.LO: return +1
        if rx>=self.HI: return -1
        return 0

    def test_broken_returns_zero(self):
        self.assertEqual(self._broken(self.AWAY_PTS), 0)

    def test_fixed_returns_minus_one(self):
        self.assertEqual(self._fixed(self.AWAY_PTS), -1)

    def test_broken_never_commits(self):
        rng   = np.random.default_rng(99)
        votes = [self._broken(self.AWAY_PTS+rng.normal(0,.3,self.AWAY_PTS.shape)) for _ in range(100)]
        self.assertTrue(all(v==0 for v in votes))

    def test_fixed_always_votes_minus_one(self):
        rng   = np.random.default_rng(99)
        votes = [self._fixed(self.AWAY_PTS+rng.normal(0,.3,self.AWAY_PTS.shape)) for _ in range(100)]
        self.assertTrue(all(v==-1 for v in votes))

    def test_live_engine_commits_both_teams(self):
        from core.formation import DynamicFormationEngine, VOTE_WINDOW
        HOME = [(8,34),(18,10),(18,27),(18,41),(18,58),(45,18),(45,34),(45,52),(72,10),(72,34),(72,58)]
        engine = DynamicFormationEngine({'formation_window_frames': 30})
        rng    = np.random.default_rng(7)
        home_ok = away_ok = False
        fi = 0
        for fi in range(VOTE_WINDOW+30):
            n  = rng.normal(0,.4,(11,2))
            sh = engine.analyze(make_players([(HOME[i][0]+n[i,0],HOME[i][1]+n[i,1]) for i in range(11)],team=0),team=0,frame_id=fi,calib_confidence=0.9)
            sa = engine.analyze(make_players([(self.AWAY_PTS[i,0]+n[i,0],self.AWAY_PTS[i,1]+n[i,1]) for i in range(11)],team=1),team=1,frame_id=fi,calib_confidence=0.9)
            if sh and sh.direction_known: home_ok=True
            if sa and sa.direction_known: away_ok=True
            if home_ok and away_ok: break
        self.assertTrue(home_ok, "HOME direction never committed")
        self.assertTrue(away_ok, "AWAY direction never committed (GK vote bug?)")
        print(f"\n  [PASS] Both teams committed direction within {fi+1} frames")

    def test_why_bug_occurs_documentation(self):
        xs        = self.AWAY_PTS[:,0]
        left_idx  = int(np.argmin(xs)); right_idx = int(np.argmax(xs))
        left_gap  = float(np.min(np.abs(np.delete(xs,left_idx)-xs[left_idx])))
        right_gap = float(np.min(np.abs(np.delete(xs,right_idx)-xs[right_idx])))
        self.assertGreaterEqual(left_gap,  self.ISO)
        self.assertGreater(xs[left_idx],   self.LO)
        self.assertLess(xs[left_idx],      self.HI)
        self.assertGreaterEqual(right_gap, self.ISO)
        self.assertGreaterEqual(xs[right_idx], self.HI)
        print(f"\n  [PASS] Bug root cause confirmed: leftmost=striker x={xs[left_idx]:.0f} (middle), rightmost=GK x={xs[right_idx]:.0f} (>HI)")


# ── 6. Schema version ─────────────────────────────────────────────────────────
class TestSchemaVersion(unittest.TestCase):

    def _make_frames(self, n=15):
        from core.pipeline import FootballIntelligencePipeline, SCHEMA_VERSION, FrameAnalysis
        from core.formation import DynamicFormationEngine
        from core.counter   import CounterTacticEngine

        config = {'yolo_model': 'hog-fallback', 'formation_window_frames': 30}
        pipeline = FootballIntelligencePipeline.__new__(FootballIntelligencePipeline)
        pipeline.config      = config
        pipeline.frame_count = 0
        from core.formation import SPECTRAL_K
        pipeline.model_versions = {
            'detector': 'hog-fallback', 'tracker': 'bytetrack-kalman-v1',
            'formation': f'knn-laplacian-v2-spectral{SPECTRAL_K}',
            'counter': 'metric-grounded-v2-11rules', 'schema': SCHEMA_VERSION,
        }
        pipeline.formation = DynamicFormationEngine(config)
        pipeline.counter   = CounterTacticEngine(config)

        HOME = [(8,34),(18,10),(18,27),(18,41),(18,58),(45,18),(45,34),(45,52),(72,10),(72,34),(72,58)]
        AWAY = [(97,34),(87,10),(87,27),(87,41),(87,58),(67,24),(67,46),(55,10),(55,34),(55,58),(42,34)]
        rng  = np.random.default_rng(42)
        frames = []
        for fi in range(n):
            nn = rng.normal(0,.3,(11,2))
            hp = make_players([(HOME[i][0]+nn[i,0],HOME[i][1]+nn[i,1]) for i in range(11)],team=0)
            ap = make_players([(AWAY[i][0]+nn[i,0],AWAY[i][1]+nn[i,1]) for i in range(11)],team=1)
            sh = pipeline.formation.analyze(hp,0,fi,0.9)
            sa = pipeline.formation.analyze(ap,1,fi,0.9)
            hc = ac = []
            if sh and sa and sh.is_settled and sa.is_settled:
                hc = pipeline.counter.generate(sa,sh)
                ac = pipeline.counter.generate(sh,sa)
            fa = FrameAnalysis(frame_id=fi,timestamp=round(fi/25.,3),processing_ms=1.0,
                               calibration_confidence=0.9,players=hp+ap,ball_pos=None,
                               home_formation=sh,away_formation=sa,home_counters=hc,away_counters=ac)
            frames.append(pipeline.to_json(fa))
            pipeline.frame_count += 1
        return frames, SCHEMA_VERSION

    def test_schema_version_on_every_frame(self):
        frames, expected = self._make_frames()
        for i, raw in enumerate(frames):
            d = json.loads(raw)
            self.assertIn('schema_version', d, f"frame {i}: missing schema_version")
            self.assertEqual(d['schema_version'], expected)
        print(f"\n  [PASS] schema_version='{expected}' on all 15 frames")

    def test_model_versions_on_every_frame(self):
        frames, _ = self._make_frames()
        required  = {'detector','tracker','formation','counter','schema'}
        for i, raw in enumerate(frames):
            d  = json.loads(raw)
            self.assertIn('model_versions', d)
            for k in required:
                self.assertIn(k, d['model_versions'])
                self.assertIsInstance(d['model_versions'][k], str)
        print(f"\n  [PASS] model_versions with keys {required} on all 15 frames")

    def test_no_numpy_types_in_output(self):
        frames, _ = self._make_frames()
        for i, raw in enumerate(frames):
            try: json.loads(raw)
            except json.JSONDecodeError as e: self.fail(f"frame {i}: {e}")
        print("\n  [PASS] All frames JSON-clean (no numpy type leakage)")

    def test_direction_known_and_stability_present(self):
        frames, _ = self._make_frames()
        for i, raw in enumerate(frames):
            d = json.loads(raw)
            for key in ('home_formation','away_formation'):
                fmn = d.get(key)
                if fmn is None: continue
                self.assertIn('direction_known', fmn)
                self.assertIn('stability_score', fmn)
                self.assertIsInstance(fmn['direction_known'], bool)
                ss = fmn['stability_score']
                self.assertGreaterEqual(ss, 0.0); self.assertLessEqual(ss, 1.0)
        print("\n  [PASS] direction_known + stability_score present and valid")


# ── 7. Graph health ───────────────────────────────────────────────────────────
class TestGraphHealth(unittest.TestCase):

    REQUIRED = {'sigma_used','sigma_multiplier','min_degree','is_connected','fallback_knn_only','n_components'}

    def test_graph_health_in_build_output(self):
        from core.formation import FormationGraph
        fg  = FormationGraph()
        pts = [(18,10),(18,27),(18,41),(18,58),(45,20),(45,34),(45,52),(72,10),(72,34),(72,56),(8,34)]
        g   = fg.build(pts)
        self.assertIn('graph_health', g)
        gh = g['graph_health']
        for k in self.REQUIRED: self.assertIn(k, gh)
        self.assertIsInstance(gh['sigma_used'],        float)
        self.assertIsInstance(gh['sigma_multiplier'],  float)
        self.assertIsInstance(gh['min_degree'],        int)
        self.assertIsInstance(gh['is_connected'],      bool)
        self.assertIsInstance(gh['fallback_knn_only'], bool)
        self.assertIsInstance(gh['n_components'],      int)
        print(f"\n  [PASS] graph_health keys: {sorted(gh.keys())}")

    def test_graph_health_in_snapshot_to_dict(self):
        from core.formation import DynamicFormationEngine
        engine = DynamicFormationEngine({'formation_window_frames': 30})
        rng    = np.random.default_rng(5)
        BASE   = [(8,34),(18,10),(18,27),(18,41),(18,58),(45,18),(45,34),(45,52),(72,10),(72,34),(72,58)]
        for fi in range(5):
            n  = rng.normal(0,.3,(11,2))
            pl = make_players([(BASE[i][0]+n[i,0],BASE[i][1]+n[i,1]) for i in range(11)],team=0)
            snap = engine.analyze(pl,0,fi,0.9)
            if snap is None: continue
            d = snap.to_dict()
            self.assertIn('graph_health', d)
            if d['graph_health'] is not None:
                for k in self.REQUIRED: self.assertIn(k, d['graph_health'])
                json.dumps(d['graph_health'])
        print("\n  [PASS] graph_health present in FormationSnapshot.to_dict()")

    def test_nominal_graph_connected_sigma_one(self):
        from core.formation import FormationGraph
        fg  = FormationGraph()
        pts = [(18,10),(18,27),(18,41),(18,58),(45,20),(45,34),(45,52),(72,10),(72,34),(72,56),(8,34)]
        g   = fg.build(pts)
        gh  = g['graph_health']
        self.assertTrue(gh['is_connected'])
        self.assertEqual(gh['n_components'], 1)
        self.assertFalse(gh['fallback_knn_only'])
        print(f"\n  [PASS] Nominal graph: sigma×{gh['sigma_multiplier']:.2f}, connected={gh['is_connected']}")

    def test_spread_formation_adapts_sigma(self):
        from core.formation import FormationGraph
        fg  = FormationGraph()
        pts = [(0,0),(0,34),(0,68),(35,0),(35,34),(35,68),(70,0),(70,34),(70,68),(105,0),(105,34)]
        g   = fg.build(pts)
        gh  = g['graph_health']
        self.assertTrue(gh['is_connected'])
        self.assertEqual(gh['n_components'], 1)
        print(f"\n  [PASS] Spread: sigma×{gh['sigma_multiplier']:.2f} fallback={gh['fallback_knn_only']} connected={gh['is_connected']}")


class TestDetectionProfiles(unittest.TestCase):
    """Regression tests for detection profile system (config/settings.py)."""

    def test_broadcast_profile_confidence(self):
        """broadcast profile must set detection_confidence = 0.35."""
        from config.settings import get_profile_config
        cfg = get_profile_config('broadcast')
        self.assertEqual(cfg['detection_confidence'], 0.35)
        print('\n  [PASS] broadcast profile: detection_confidence=0.35')

    def test_wild_profile_confidence(self):
        """wild profile must set detection_confidence = 0.15."""
        from config.settings import get_profile_config
        cfg = get_profile_config('wild')
        self.assertEqual(cfg['detection_confidence'], 0.15)
        print('\n  [PASS] wild profile: detection_confidence=0.15')

    def test_explicit_conf_overrides_profile(self):
        """--conf override must take priority over any profile setting."""
        from config.settings import get_profile_config
        cfg = get_profile_config('broadcast', override_conf=0.20)
        self.assertEqual(cfg['detection_confidence'], 0.20)
        cfg2 = get_profile_config('wild', override_conf=0.50)
        self.assertEqual(cfg2['detection_confidence'], 0.50)
        print('\n  [PASS] explicit override takes priority over profile')


if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in [TestOcclusionDropout, TestWrongHomographyFreeze, TestBroadcastZoomStability,
                TestDirectionNormaliser, TestGKVoteBugRegression, TestSchemaVersion,
                TestGraphHealth, TestDetectionProfiles]:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
