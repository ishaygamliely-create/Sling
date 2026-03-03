#!/usr/bin/env python3
"""
validate_broadcast.py — Real-clip validation for Football Intelligence System v2.1

Usage:
    python validate_broadcast.py --video data/real/clip.mov
    python validate_broadcast.py --video clip.mov --profile wild --out reports/wild.json
    python validate_broadcast.py --video clip.mov --conf 0.20 --max-frames 300

Priority order for detection_confidence:
    --conf (explicit) > --profile > default (broadcast = 0.35)

Auto-fallback (broadcast only, no --conf override):
    Runs broadcast first. If avg players/frame < 14 OR no settled snapshots by
    frame 200, automatically re-runs with wild profile and logs
    "AUTO-FALLBACK TRIGGERED".

Exit codes:
    0 = PASS   (all hard checks satisfied)
    1 = FAIL   (one or more hard checks failed)
    2 = setup error (video not found, import error, etc.)
"""

from __future__ import annotations
import argparse, json, math, platform, statistics, sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))

# Auto-fallback thresholds
_FALLBACK_CHECK_FRAME   = 200    # check at this frame
_FALLBACK_MIN_PLAYERS   = 14     # avg players/frame below this → fallback
_FALLBACK_MIN_SETTLED   = 1      # total settled snapshots below this → fallback


# ── Environment info ──────────────────────────────────────────────────────────

def _env_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        'python': platform.python_version(),
        'os':     platform.system(),
    }
    for pkg, key in [('cv2', 'opencv'), ('numpy', 'numpy')]:
        try:
            import importlib
            m = importlib.import_module(pkg)
            info[key] = m.__version__
        except ImportError:
            info[key] = None
    try:
        import ultralytics
        info['ultralytics'] = ultralytics.__version__
        info['ultralytics_present'] = True
    except ImportError:
        info['ultralytics'] = None
        info['ultralytics_present'] = False
    return info


# ── Core validation logic ─────────────────────────────────────────────────────

def validate(
    video_path:     str,
    out_path:       str,
    yolo_model:     str,
    max_frames:     int,
    profile:        str  = 'broadcast',
    override_conf:  Optional[float] = None,
    _is_fallback:   bool = False,
) -> bool:
    """
    Run validation on a broadcast clip.

    Parameters
    ----------
    video_path    : Path to video file.
    out_path      : Path for JSON report output.
    yolo_model    : YOLO weights filename.
    max_frames    : Maximum number of frames to process.
    profile       : 'broadcast' (0.35) or 'wild' (0.15).
    override_conf : Explicit confidence override (highest priority).
    _is_fallback  : Internal flag — prevents recursive fallback loop.
    """
    from core.pipeline import FootballIntelligencePipeline, SCHEMA_VERSION
    from config.settings import get_profile_config, VALID_PROFILES

    if profile not in VALID_PROFILES:
        print(f'Error: unknown profile "{profile}". Valid: {VALID_PROFILES}',
              file=sys.stderr)
        sys.exit(2)

    cfg     = get_profile_config(profile, override_conf)
    eff_conf = cfg['detection_confidence']

    # Auto-fallback only on first broadcast run without an explicit --conf
    auto_fallback_enabled = (
        profile == 'broadcast'
        and override_conf is None
        and not _is_fallback
    )

    pipeline_cfg = dict(cfg)
    pipeline_cfg['yolo_model'] = yolo_model

    pipeline = FootballIntelligencePipeline(pipeline_cfg)

    # ── Per-frame accumulators ────────────────────────────────────────────────
    calib_confs:    List[float] = []
    player_counts:  List[int]   = []
    proc_ms:        List[float] = []
    home_snaps:     List[Dict]  = []
    away_snaps:     List[Dict]  = []
    both_settled_frames:         int = 0
    direction_known_home_frames: int = 0
    direction_known_away_frames: int = 0

    # Camera cut detection (calib drop > 0.15 heuristic)
    CUT_DROP_THRESHOLD = 0.15
    prev_calib: Optional[float] = None
    cut_frames: List[int] = []

    schema_failures: List[str] = []
    nan_failures:    List[str] = []
    graph_failures:  List[str] = []

    first_settled_home: Optional[int] = None
    first_settled_away: Optional[int] = None

    # Auto-fallback flag
    fallback_triggered = False
    fallback_reason    = ''

    SEP = '═' * 70
    print(f'\n{SEP}')
    print('  FOOTBALL INTELLIGENCE — BROADCAST VALIDATION')
    print(SEP)
    print(f'  Video       : {video_path}')
    print(f'  YOLO model  : {yolo_model}')
    print(f'  Profile     : {profile}{"  [auto-fallback]" if _is_fallback else ""}')
    print(f'  Conf thresh : {eff_conf}')
    print(f'  Max frames  : {max_frames}')
    print(f'  Schema      : {SCHEMA_VERSION}')
    print(f'{SEP}\n')

    for analysis in pipeline.process_video(video_path):
        if analysis.frame_id >= max_frames:
            break

        raw = pipeline.to_json(analysis)
        d   = json.loads(raw)
        fi  = d['frame_id']

        # ── Schema checks ─────────────────────────────────────────────────
        if 'schema_version' not in d:
            schema_failures.append(f'frame {fi}: missing schema_version')
        elif d['schema_version'] != SCHEMA_VERSION:
            schema_failures.append(
                f'frame {fi}: schema={d["schema_version"]} expected={SCHEMA_VERSION}')

        if 'model_versions' not in d:
            schema_failures.append(f'frame {fi}: missing model_versions')
        else:
            for k in ('detector', 'tracker', 'formation', 'counter', 'schema'):
                if k not in d['model_versions']:
                    schema_failures.append(
                        f'frame {fi}: model_versions missing \'{k}\'')

        # ── Frame-level metrics ───────────────────────────────────────────
        cur_calib = d['calibration_confidence']
        calib_confs.append(cur_calib)
        player_counts.append(len(d['players']))
        proc_ms.append(d['processing_ms'])

        # Camera cut detection
        if prev_calib is not None and (prev_calib - cur_calib) > CUT_DROP_THRESHOLD:
            cut_frames.append(fi)
        prev_calib = cur_calib

        # ── NaN/Inf in pitch coords ───────────────────────────────────────
        for p in d['players']:
            pp = p.get('pitch_pos', [0, 0])
            if any(math.isnan(v) or math.isinf(v) for v in pp):
                nan_failures.append(
                    f'frame {fi} player {p["id"]}: bad pitch_pos {pp}')

        # ── Formation snapshots ───────────────────────────────────────────
        h_settled = a_settled = False
        h_dir_known = a_dir_known = False

        for label, key, store, is_home in [
            ('HOME', 'home_formation', home_snaps, True),
            ('AWAY', 'away_formation', away_snaps, False),
        ]:
            fmn = d.get(key)
            if fmn is None:
                continue
            store.append(fmn)

            gh = fmn.get('graph_health')
            if gh is not None:
                try:
                    json.dumps(gh)
                except Exception as e:
                    graph_failures.append(f'frame {fi} {key}: {e}')

            if fmn.get('is_settled'):
                if is_home:
                    h_settled = True
                    if first_settled_home is None:
                        first_settled_home = fi
                else:
                    a_settled = True
                    if first_settled_away is None:
                        first_settled_away = fi

            if fmn.get('direction_known'):
                if is_home:
                    h_dir_known = True
                else:
                    a_dir_known = True

        if h_settled and a_settled:
            both_settled_frames += 1
        if h_dir_known:
            direction_known_home_frames += 1
        if a_dir_known:
            direction_known_away_frames += 1

        # ── Auto-fallback check at frame 200 ─────────────────────────────
        if auto_fallback_enabled and fi == _FALLBACK_CHECK_FRAME:
            avg_p   = statistics.mean(player_counts) if player_counts else 0
            settled = sum(1 for s in home_snaps if s.get('is_settled')) + \
                      sum(1 for s in away_snaps if s.get('is_settled'))
            reasons = []
            if avg_p < _FALLBACK_MIN_PLAYERS:
                reasons.append(
                    f'avg_players={avg_p:.1f} < {_FALLBACK_MIN_PLAYERS}')
            if settled < _FALLBACK_MIN_SETTLED:
                reasons.append(
                    f'settled_snapshots={settled} < {_FALLBACK_MIN_SETTLED}')
            if reasons:
                fallback_triggered = True
                fallback_reason    = '; '.join(reasons)
                print(f'\n  ⚠  AUTO-FALLBACK TRIGGERED: broadcast → wild')
                print(f'     Reason : {fallback_reason}')
                print(f'     Restarting with wild profile (conf={
                      get_profile_config("wild")["detection_confidence"]}) …\n')
                return validate(
                    video_path, out_path, yolo_model, max_frames,
                    profile='wild', override_conf=None, _is_fallback=True,
                )

        # Progress indicator every 50 frames
        if fi % 50 == 0:
            avg_c = statistics.mean(calib_confs) if calib_confs else 0
            print(f'  frame {fi:>4}  calib_avg={avg_c:.2f}  '
                  f'players={player_counts[-1]:>2}  '
                  f'settled_both={both_settled_frames}')

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    total = len(calib_confs)
    if total == 0:
        print('ERROR: no frames processed (bad video path or codec?)',
              file=sys.stderr)
        sys.exit(2)

    settled_home = sum(1 for s in home_snaps if s.get('is_settled'))
    settled_away = sum(1 for s in away_snaps if s.get('is_settled'))
    avg_conf     = round(statistics.mean(calib_confs), 3)

    settled_ratio     = round(both_settled_frames / total, 3)
    dir_known_ratio_h = round(direction_known_home_frames / total, 3)
    dir_known_ratio_a = round(direction_known_away_frames / total, 3)

    # ── PASS / FAIL checks ────────────────────────────────────────────────────
    checks = [
        {
            'name':    'schema_version on every frame',
            'pass':    len(schema_failures) == 0,
            'details': f'{len(schema_failures)} failures' if schema_failures
                       else 'all frames OK',
        },
        {
            'name':    'model_versions with all required keys',
            'pass':    not any('model_versions' in f for f in schema_failures),
            'details': 'all frames OK' if not any(
                'model_versions' in f for f in schema_failures
            ) else 'see schema_failures',
        },
        {
            'name':    'no NaN/Inf in pitch coordinates',
            'pass':    len(nan_failures) == 0,
            'details': f'{len(nan_failures)} bad positions' if nan_failures
                       else 'all clean',
        },
        {
            'name':    'at least 1 settled snapshot for HOME',
            'pass':    settled_home >= 1,
            'details': f'{settled_home} settled frames',
        },
        {
            'name':    'at least 1 settled snapshot for AWAY',
            'pass':    settled_away >= 1,
            'details': f'{settled_away} settled frames',
        },
        {
            'name':    'graph_health blocks JSON-serialisable',
            'pass':    len(graph_failures) == 0,
            'details': f'{len(graph_failures)} failures' if graph_failures
                       else 'all serialisable',
        },
        {
            'name':    'avg calibration_confidence >= 0.30',
            'pass':    avg_conf >= 0.30,
            'details': f'avg={avg_conf:.3f}',
        },
    ]
    hard_pass = all(c['pass'] for c in checks)

    # ── Build report dict ─────────────────────────────────────────────────────
    report: Dict[str, Any] = {
        'schema_version': SCHEMA_VERSION,
        'result':         'PASS' if hard_pass else 'FAIL',
        'profile_used':   'wild' if _is_fallback else profile,
        'conf_used':      eff_conf,
        'auto_fallback':  _is_fallback,
        'video':          str(video_path),
        'env':            _env_info(),
        'summary': {
            'frames_processed':   total,
            'both_settled_ratio': settled_ratio,
            'direction_known_ratio': {
                'home': dir_known_ratio_h,
                'away': dir_known_ratio_a,
            },
            'calibration_confidence': {
                'avg': avg_conf,
                'min': round(min(calib_confs), 3),
                'max': round(max(calib_confs), 3),
            },
            'players_per_frame': {
                'avg': round(statistics.mean(player_counts), 1),
                'min': min(player_counts),
                'max': max(player_counts),
            },
            'processing_ms': {
                'avg': round(statistics.mean(proc_ms), 1),
                'min': round(min(proc_ms), 1),
                'max': round(max(proc_ms), 1),
            },
            'formation_snapshots': {
                'home_total':   len(home_snaps),
                'home_settled': settled_home,
                'home_settled_pct':
                    round(settled_home / max(len(home_snaps), 1) * 100, 1),
                'away_total':   len(away_snaps),
                'away_settled': settled_away,
                'away_settled_pct':
                    round(settled_away / max(len(away_snaps), 1) * 100, 1),
                'first_settled_home_frame': first_settled_home,
                'first_settled_away_frame': first_settled_away,
            },
        },
        'checks':    checks,
        'cut_detection': {
            'heuristic':     f'calib drop > {CUT_DROP_THRESHOLD} between frames',
            'cuts_detected': len(cut_frames),
            'cut_pct':       round(len(cut_frames) / max(total - 1, 1) * 100, 1),
            'cut_at_frames': cut_frames[:50],
        },
        'schema_failures': schema_failures[:20],
        'nan_failures':    nan_failures[:20],
        'graph_failures':  graph_failures[:20],
    }

    # ── Terminal output ───────────────────────────────────────────────────────
    cc = report['summary']['calibration_confidence']
    pc = report['summary']['players_per_frame']
    pm = report['summary']['processing_ms']
    fs = report['summary']['formation_snapshots']

    print(f'\n{"─"*70}')
    print('  METRICS')
    print(f'{"─"*70}')
    print(f'  Profile used           : {report["profile_used"]}'
          f'{"  [auto-fallback]" if _is_fallback else ""}')
    print(f'  Conf threshold         : {eff_conf}')
    print(f'  Frames processed       : {total}')
    print(f'  Camera cuts detected   : {len(cut_frames)} '
          f'({len(cut_frames)/max(total-1,1)*100:.1f}% frames — '
          f'calib drop > {CUT_DROP_THRESHOLD})')
    print(f'  Calibration confidence : avg={cc["avg"]:.3f}  '
          f'min={cc["min"]:.3f}  max={cc["max"]:.3f}')
    print(f'  Players / frame        : avg={pc["avg"]:.1f}  '
          f'min={pc["min"]}  max={pc["max"]}')
    print(f'  Processing ms / frame  : avg={pm["avg"]:.1f}  '
          f'min={pm["min"]:.1f}  max={pm["max"]:.1f}')
    print(f'  Both-settled ratio     : {settled_ratio:.1%}')
    print(f'  Direction known (HOME) : {dir_known_ratio_h:.1%}')
    print(f'  Direction known (AWAY) : {dir_known_ratio_a:.1%}')
    print(f'  HOME snapshots         : {fs["home_total"]} total  '
          f'{fs["home_settled"]} settled ({fs["home_settled_pct"]}%)')
    print(f'  AWAY snapshots         : {fs["away_total"]} total  '
          f'{fs["away_settled"]} settled ({fs["away_settled_pct"]}%)')

    print(f'\n{"─"*70}')
    print('  CHECKS')
    print(f'{"─"*70}')
    for c in checks:
        icon = '✓' if c['pass'] else '✗'
        print(f'  [{icon}] {c["name"]}')
        print(f'       → {c["details"]}')

    result_line = 'PASS ✅' if hard_pass else 'FAIL ❌'
    print(f'\n{SEP}')
    print(f'  RESULT: {result_line}')
    if _is_fallback:
        print(f'  (auto-fallback from broadcast → wild was applied)')
    print(SEP)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as fp:
        json.dump(report, fp, indent=2, default=str)
    print(f'  Report saved → {out_path}\n')

    return hard_pass


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Football Intelligence — Real-clip validation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument('--video',      required=True,
                    help='Path to broadcast video file')
    ap.add_argument('--yolo',       default='yolov8n.pt',
                    help='YOLO weights (default: yolov8n.pt)')
    ap.add_argument('--max-frames', type=int, default=300,
                    help='Maximum frames to analyse (default: 300)')
    ap.add_argument('--profile',    default='broadcast',
                    choices=['broadcast', 'wild'],
                    help='Detection profile (default: broadcast). '
                         'broadcast=conf 0.35 | wild=conf 0.15')
    ap.add_argument('--conf',       type=float, default=None,
                    help='Override detection_confidence (highest priority)')
    ap.add_argument('--out',        default='reports/clip_report.json',
                    help='Output JSON path (default: reports/clip_report.json)')
    args = ap.parse_args()

    if not Path(args.video).exists():
        print(f'Error: video not found: {args.video}', file=sys.stderr)
        sys.exit(2)

    ok = validate(
        video_path=args.video,
        out_path=args.out,
        yolo_model=args.yolo,
        max_frames=args.max_frames,
        profile=args.profile,
        override_conf=args.conf,
    )
    sys.exit(0 if ok else 1)
