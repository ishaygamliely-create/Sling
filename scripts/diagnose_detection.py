#!/usr/bin/env python3
"""
scripts/diagnose_detection.py
Breaks down the detection pipeline for specific frames to isolate
where player count loss occurs: YOLO → conf filter → team assign → tracker.

Usage:
    python scripts/diagnose_detection.py --video data/real/clip.mov --frames 395,400,405
    python scripts/diagnose_detection.py --video data/real/clip.mov --lowest 5
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def diagnose_frame(video_path: str, frame_num: int,
                   yolo_model: str = 'yolov8n.pt',
                   conf_thresh: float = 0.35) -> dict:
    """
    Returns detection counts at every stage of the pipeline for one frame.

    Stages:
      A  raw_yolo_all    — all YOLO detections at conf=0.0 (no filter)
      B  after_conf      — after applying conf >= conf_thresh
      C  team_home       — after K-Means team assignment, assigned to team 0
         team_away       — assigned to team 1
         referee         — assigned -1 (referee heuristic)
         no_embedding    — persons with zero/null color embedding (very small crop)

    Note: the 'players_per_frame' in the validation report is FURTHER reduced
    by the tracker (must be tracked) and calibrator (must project onto pitch),
    so it can be lower than C.
    """
    import cv2 as cv
    import numpy as np
    from ultralytics import YOLO
    from core.detection import TeamSeparator, RawDetection

    cap = cv.VideoCapture(video_path)
    cap.set(cv.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return {'frame': frame_num, 'error': 'could not read frame'}

    model     = YOLO(yolo_model)
    separator = TeamSeparator()

    # ── A: Raw YOLO at conf=0.0 ───────────────────────────────────────────────
    results_all = model(frame, conf=0.0, classes=[0], verbose=False)
    raw_yolo_all = sum(len(r.boxes) for r in results_all)

    # ── B: After conf threshold ───────────────────────────────────────────────
    results_filtered = model(frame, conf=conf_thresh, classes=[0], verbose=False)
    after_conf_detections: list[RawDetection] = []
    for r in results_filtered:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            emb = separator.extract_jersey_color(frame, (x1, y1, x2, y2))
            after_conf_detections.append(RawDetection(
                bbox=(x1, y1, x2, y2),
                confidence=float(box.conf[0]),
                class_id=0,
                color_embedding=emb,
            ))

    after_conf_count = len(after_conf_detections)
    no_embedding     = sum(1 for d in after_conf_detections
                           if d.color_embedding is None or np.all(d.color_embedding == 0))

    # ── C: After team assignment ──────────────────────────────────────────────
    if after_conf_detections:
        team_labels = separator.assign_teams(after_conf_detections)
    else:
        team_labels = []

    team_home = sum(1 for t in team_labels if t == 0)
    team_away = sum(1 for t in team_labels if t == 1)
    referee   = sum(1 for t in team_labels if t == -1)

    return {
        'frame':           frame_num,
        'A_raw_yolo_all':  raw_yolo_all,
        'B_after_conf':    after_conf_count,
        'B_no_embedding':  no_embedding,
        'C_team_home':     team_home,
        'C_team_away':     team_away,
        'C_referee':       referee,
        'note': ('players_per_frame in report is further reduced by tracker + '
                 'calibrator, so may be lower than C_team_home + C_team_away'),
    }


def _print_result(r: dict):
    if 'error' in r:
        print(f'  Frame {r["frame"]:>4}: ERROR — {r["error"]}')
        return
    print(f'\n  ── Frame {r["frame"]} ──')
    print(f'    A) YOLO raw (conf=0.00)       : {r["A_raw_yolo_all"]:>3}  (all person detections)')
    print(f'    B) After conf ≥ threshold     : {r["B_after_conf"]:>3}  '
          f'({r["B_no_embedding"]} with no valid color embedding)')
    print(f'    C) After team assignment      :')
    print(f'       Team HOME (0)              : {r["C_team_home"]:>3}')
    print(f'       Team AWAY (1)              : {r["C_team_away"]:>3}')
    print(f'       Referee  (-1)              : {r["C_referee"]:>3}')
    total_c = r["C_team_home"] + r["C_team_away"] + r["C_referee"]
    print(f'       Total C                    : {total_c:>3}')
    print(f'    D) players_per_frame (report) : ??? (tracker+calib further filter)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Detection pipeline diagnostics per frame')
    ap.add_argument('--video',   required=True)
    ap.add_argument('--yolo',    default='yolov8n.pt')
    ap.add_argument('--conf',    type=float, default=0.35)
    ap.add_argument('--frames',  type=str, default=None,
                    help='Comma-separated frame numbers, e.g. 395,400,405')
    ap.add_argument('--report',  type=str, default=None,
                    help='Optional path to clip_report.json to auto-pick lowest-player frames')
    ap.add_argument('--lowest',  type=int, default=5,
                    help='How many lowest-player frames to analyse from report (default 5)')
    args = ap.parse_args()

    target_frames: list[int] = []

    # From explicit --frames list
    if args.frames:
        target_frames = [int(f.strip()) for f in args.frames.split(',')]

    # From report JSON — pick frames with lowest player counts
    if args.report and Path(args.report).exists():
        with open(args.report) as f:
            rep = json.load(f)
        # report doesn't store per-frame list; remind user to specify --frames
        print('[INFO] --report loaded but per-frame breakdown not stored in report.')
        print('[INFO] Use --frames 395,400,405 to specify frames manually.')

    if not target_frames:
        target_frames = [395, 400, 405]  # default: around the known low frame

    print('\n' + '═' * 60)
    print('  DETECTION PIPELINE DIAGNOSTICS')
    print('═' * 60)
    print(f'  Video      : {args.video}')
    print(f'  YOLO model : {args.yolo}')
    print(f'  Conf thresh: {args.conf}')
    print(f'  Frames     : {target_frames}')
    print('═' * 60)

    results = []
    for fn in target_frames:
        r = diagnose_frame(args.video, fn, yolo_model=args.yolo, conf_thresh=args.conf)
        _print_result(r)
        results.append(r)

    print('\n' + '═' * 60)
    print('  LEGEND')
    print('  A = YOLO total persons at conf=0.0 (before any filter)')
    print('  B = After confidence threshold (this is what the pipeline uses)')
    print('  C = After K-Means team separation')
    print('  D = After tracker + calibrator (= players_per_frame in report)')
    print('  If A >> B: YOLO sees many but confidence filter removes them')
    print('  If B >> C: team separator is the bottleneck')
    print('  If C >> D: tracker/calibrator is the bottleneck')
    print('═' * 60 + '\n')
