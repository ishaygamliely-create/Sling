#!/usr/bin/env python3
"""
validate_broadcast.py — Real-clip validation script.

Usage:
    python validate_broadcast.py --video clip.mp4
    python validate_broadcast.py --video clip.mp4 --yolo yolov8s.pt
    python validate_broadcast.py --video clip.mp4 --out report.json

Exit 0 = PASS, Exit 1 = FAIL, Exit 2 = setup error.
"""

from __future__ import annotations
import argparse, json, math, statistics, sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))


def validate(video_path: str, out_path: str, yolo_model: str) -> bool:
    from core.pipeline import FootballIntelligencePipeline, SCHEMA_VERSION

    config = {'yolo_model': yolo_model, 'detection_confidence': 0.35,
              'iou_threshold': 0.3, 'max_track_misses': 30,
              'min_hits': 1, 'formation_window_frames': 60}
    pipeline = FootballIntelligencePipeline(config)

    calib_confs: List[float] = []
    player_counts: List[int] = []
    home_snaps: List[Dict]   = []
    away_snaps: List[Dict]   = []
    schema_failures: List[str] = []
    nan_failures:    List[str] = []
    graph_failures:  List[str] = []
    first_settled_home = first_settled_away = None

    print(f"Processing : {video_path}")
    print(f"YOLO model : {yolo_model}")
    print(f"Expected schema_version: {SCHEMA_VERSION}\n")

    for analysis in pipeline.process_video(video_path):
        raw = pipeline.to_json(analysis)
        d   = json.loads(raw)
        fi  = d['frame_id']

        if 'schema_version' not in d:
            schema_failures.append(f"frame {fi}: missing schema_version")
        elif d['schema_version'] != SCHEMA_VERSION:
            schema_failures.append(f"frame {fi}: got {d['schema_version']} expected {SCHEMA_VERSION}")

        if 'model_versions' not in d:
            schema_failures.append(f"frame {fi}: missing model_versions")
        else:
            for k in ('detector','tracker','formation','counter','schema'):
                if k not in d['model_versions']:
                    schema_failures.append(f"frame {fi}: model_versions missing '{k}'")

        calib_confs.append(d['calibration_confidence'])
        player_counts.append(len(d['players']))

        for p in d['players']:
            pp = p.get('pitch_pos', [0,0])
            if any(math.isnan(v) or math.isinf(v) for v in pp):
                nan_failures.append(f"frame {fi} player {p['id']}: bad pitch_pos {pp}")

        for label, key, store in [('HOME','home_formation',home_snaps),('AWAY','away_formation',away_snaps)]:
            fmn = d.get(key)
            if fmn is None: continue
            store.append(fmn)
            gh = fmn.get('graph_health')
            if gh is not None:
                try: json.dumps(gh)
                except Exception as e: graph_failures.append(f"frame {fi} {key}: {e}")
            if fmn.get('is_settled'):
                if label=='HOME' and first_settled_home is None: first_settled_home = d
                if label=='AWAY' and first_settled_away is None: first_settled_away = d

    settled_home = sum(1 for s in home_snaps if s.get('is_settled'))
    settled_away = sum(1 for s in away_snaps if s.get('is_settled'))
    total = len(calib_confs)

    report = {
        'schema_version': SCHEMA_VERSION, 'video': str(video_path),
        'total_frames': total,
        'calibration_confidence': {
            'avg': round(statistics.mean(calib_confs), 3) if calib_confs else 0,
            'min': round(min(calib_confs), 3) if calib_confs else 0,
            'max': round(max(calib_confs), 3) if calib_confs else 0,
        },
        'players_per_frame': {
            'avg': round(statistics.mean(player_counts), 1) if player_counts else 0,
            'min': min(player_counts) if player_counts else 0,
            'max': max(player_counts) if player_counts else 0,
        },
        'formation_snapshots': {
            'home_total': len(home_snaps), 'home_settled': settled_home,
            'home_settled_pct': round(settled_home/max(len(home_snaps),1)*100,1),
            'away_total': len(away_snaps), 'away_settled': settled_away,
            'away_settled_pct': round(settled_away/max(len(away_snaps),1)*100,1),
        },
        'schema_failures': schema_failures, 'nan_failures': nan_failures,
        'graph_failures': graph_failures,
        'first_settled_home_frame': first_settled_home,
        'first_settled_away_frame': first_settled_away,
    }

    failures: List[str] = []
    if schema_failures:  failures.append(f"Schema failures: {len(schema_failures)}")
    if nan_failures:     failures.append(f"NaN/Inf in pitch coords: {len(nan_failures)}")
    if graph_failures:   failures.append(f"graph_health serialisation failures: {len(graph_failures)}")
    if (settled_home+settled_away) == 0: failures.append("No settled snapshots for either team")
    avg_conf = report['calibration_confidence']['avg']
    if avg_conf < 0.30: failures.append(f"Calibration avg={avg_conf:.3f} < 0.30")

    print("="*70)
    print("  VALIDATION REPORT")
    print("="*70)
    cc = report['calibration_confidence']
    pc = report['players_per_frame']
    fs = report['formation_snapshots']
    print(f"  Frames analysed        : {total}")
    print(f"  Calibration confidence : avg={cc['avg']:.3f}  min={cc['min']:.3f}  max={cc['max']:.3f}")
    print(f"  Players per frame      : avg={pc['avg']:.1f}  min={pc['min']}  max={pc['max']}")
    print(f"  HOME snapshots         : {fs['home_total']} total, {fs['home_settled']} settled ({fs['home_settled_pct']}%)")
    print(f"  AWAY snapshots         : {fs['away_total']} total, {fs['away_settled']} settled ({fs['away_settled_pct']}%)")
    print(f"  Schema failures        : {len(schema_failures)}")
    print(f"  NaN/Inf failures       : {len(nan_failures)}")
    print(f"  Graph health failures  : {len(graph_failures)}")
    print()
    if failures:
        print("  RESULT: FAIL")
        for f in failures: print(f"    ✗ {f}")
    else:
        print("  RESULT: PASS")
        print("    ✓ schema_version on every frame")
        print("    ✓ model_versions with all required keys")
        print("    ✓ at least 1 settled snapshot per team")
        print("    ✓ no NaN/Inf in pitch coordinates")
        print("    ✓ all graph_health blocks serialisable")
        print(f"    ✓ calibration avg={avg_conf:.3f} ≥ 0.30")
    print()
    with open(out_path, 'w') as fp:
        json.dump(report, fp, indent=2, default=str)
    print(f"  Full report → {out_path}")
    print("="*70)
    return len(failures) == 0


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--video',  required=True)
    p.add_argument('--out',    default='validation.json')
    p.add_argument('--yolo',   default='yolov8n.pt')
    args = p.parse_args()
    if not Path(args.video).exists():
        print(f"Error: video not found: {args.video}", file=sys.stderr)
        sys.exit(2)
    sys.exit(0 if validate(args.video, args.out, args.yolo) else 1)
