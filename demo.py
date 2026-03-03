#!/usr/bin/env python3
"""
demo.py — Football Intelligence System demo (three modes)

MODES
─────
  positions  (default — no YOLO/camera required)
    Feeds deterministic pitch-coordinate positions directly into the pipeline.
    Proves:  positions → FormationSnapshot → counters JSON
    Requires only: numpy, opencv-python-headless, scipy

  video
    Runs full pipeline on a broadcast video file (requires ultralytics for YOLO).
    Falls back to HOG if YOLO unavailable (works on real broadcast; not circles).

  generate
    Writes a synthetic 30-second match video to data/samples/synthetic_match.mp4
    (useful to verify the video generator; no analysis is performed).

USAGE
─────
  # Full end-to-end proof with zero external deps beyond numpy/cv2/scipy:
  python demo.py

  # Explicit positions mode, 120 analysis frames, save JSON:
  python demo.py --mode positions --frames 120 --output demo_positions.json

  # Run on a real broadcast clip:
  python demo.py --mode video --video path/to/match.mp4

  # Generate synthetic video only:
  python demo.py --mode generate --frames 750
"""

import sys, json, time, argparse, logging, math
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s — %(message)s',
)
logger = logging.getLogger('demo')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DETERMINISTIC POSITION SEQUENCES
# ══════════════════════════════════════════════════════════════════════════════
#
# All positions in pitch metres (0–105 x, 0–68 y).
# x=0 = own goal, x=105 = opponent goal.
# GK is index 0 in each list and sits near goal line → feeds GK-anchor.

HOME_BASE = [
    # GK
    (6.0,  34.0),
    # 4 defenders
    (18.0, 11.0), (18.0, 26.5), (18.0, 41.5), (18.0, 57.0),
    # 3 central mids
    (40.0, 20.0), (42.0, 34.0), (40.0, 48.0),
    # 3 forwards (4-3-3)
    (65.0, 14.0), (68.0, 34.0), (65.0, 54.0),
]

AWAY_BASE = [
    # GK
    (99.0, 34.0),
    # 4 defenders
    (87.0, 11.0), (87.0, 26.5), (87.0, 41.5), (87.0, 57.0),
    # 2 defensive mids (4-2-3-1)
    (73.0, 27.0), (73.0, 41.0),
    # 3 attacking mids
    (58.0, 16.0), (55.0, 34.0), (58.0, 52.0),
    # CF
    (42.0, 34.0),
]


def _perturb(base, frame_idx, sigma_x=2.5, sigma_y=1.8):
    """Deterministic per-frame position perturbation using sin/cos series."""
    out = []
    for i, (x, y) in enumerate(base):
        dx = sigma_x * math.sin(frame_idx * 0.11 + i * 0.9)
        dy = sigma_y * math.cos(frame_idx * 0.08 + i * 1.3)
        out.append((
            float(np.clip(x + dx, 0.5, 104.5)),
            float(np.clip(y + dy, 0.5, 67.5)),
        ))
    return out


def _velocities(base, frame_idx, sigma_x=2.5, sigma_y=1.8):
    """Analytical derivative of _perturb — displacement per frame (m/frame).
    The settle detector threshold (2.5) is in m/frame units, matching the
    velocity=(0.3, 0.1) used in unit tests.
    """
    out = []
    for i in range(len(base)):
        vx = sigma_x * 0.11 * math.cos(frame_idx * 0.11 + i * 0.9)
        vy = sigma_y * 0.08 * -math.sin(frame_idx * 0.08 + i * 1.3)
        out.append((float(vx), float(vy)))
    return out


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — POSITIONS MODE
# ══════════════════════════════════════════════════════════════════════════════

def run_positions_mode(n_frames: int, output_json: str) -> int:
    """
    Feed deterministic positions directly into formation + counter pipeline.
    Returns number of frames that produced at least one counter suggestion.
    Exits with code 1 if no counters were generated (integration failure).
    """
    from core.pipeline import FootballIntelligencePipeline

    config = {
        'formation_window_frames': 60,
        'buffer_frames':           150,
        'max_track_misses':        30,
        'min_hits':                1,
        'target_fps':              5,
        'iou_threshold':           0.3,
    }

    pipeline = FootballIntelligencePipeline(config)

    results       = []
    counters_seen = 0
    SEP = '═' * 72
    DIV = '─' * 72

    print(f'\n{SEP}')
    print('  FOOTBALL INTELLIGENCE SYSTEM — POSITIONS MODE')
    print('  (formation + counters, no YOLO / camera required)')
    print(SEP)
    print(f'  Scenario : Home 4-3-3  vs  Away 4-2-3-1')
    print(f'  Frames   : {n_frames}  (~{n_frames/25:.0f}s @ 25fps equiv)')
    print(f'{SEP}\n')

    for fi in range(n_frames):
        home_pos = _perturb(HOME_BASE, fi)
        away_pos = _perturb(AWAY_BASE, fi)
        home_vel = _velocities(HOME_BASE, fi)
        away_vel = _velocities(AWAY_BASE, fi)

        analysis = pipeline.process_positions(
            home_pos, away_pos, home_vel, away_vel, calib_confidence=1.0,
        )

        jd = json.loads(pipeline.to_json(analysis))
        results.append(jd)

        hf = jd.get('home_formation')
        af = jd.get('away_formation')
        hc = jd.get('home_counters', [])
        ac = jd.get('away_counters', [])

        if hc or ac:
            counters_seen += 1

        # Print every 15th frame (and first 3)
        if fi % 15 == 0 or fi < 3:
            _print_frame_summary(fi + 1, analysis.processing_time_ms, hf, af, hc, ac)

    # Formation timeline
    print(f'{DIV}')
    print('  FORMATION TIMELINE')
    print(f'{DIV}')
    for team, label in [(0, 'HOME'), (1, 'AWAY')]:
        tl = pipeline.get_formation_timeline(team)
        print(f'\n  {label} ({len(tl)} settled snapshots):')
        for e in tl[-8:]:
            print(f'    t={e["timestamp"]:>6.2f}s  '
                  f'{e.get("closest_known","?"):>8}  '
                  f'conf={e.get("known_confidence",0):.2f}  '
                  f'stab={e.get("stability_score",0):.2f}  '
                  f'press={e.get("pressing_height",0):.1f}m')

    # Final counter report — find last frame that had suggestions
    final_hc, final_ac = [], []
    for r in reversed(results):
        if r.get('home_counters') or r.get('away_counters'):
            final_hc = r.get('home_counters', [])
            final_ac = r.get('away_counters', [])
            break

    print(f'\n{DIV}')
    print('  COUNTER SUGGESTIONS (last settled frame)')
    print(f'{DIV}')
    _print_counters('HOME exploits AWAY', final_hc)
    _print_counters('AWAY exploits HOME', final_ac)

    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)

    print(f'\n{SEP}')
    print(f'  SUMMARY')
    print(f'  Frames analysed      : {n_frames}')
    print(f'  Frames with counters : {counters_seen}')
    print(f'  Full JSON            : {output_json}')
    print(f'{SEP}\n')

    return counters_seen


def _print_frame_summary(frame_num, ms, hf, af, hc, ac):
    def _fmt_snap(snap, label):
        if not snap:
            return f'    {label}: no snapshot'
        s      = '✓' if snap.get('is_settled') else '~'
        form   = snap.get('closest_known', '?')
        conf   = snap.get('known_confidence', 0.0)
        stab   = snap.get('stability_score',  0.0)
        ph     = snap.get('pressing_height',  0.0)
        dl     = snap.get('defensive_line_x', 0.0)
        d      = snap.get('attacking_direction', 0)
        dk     = snap.get('direction_known', False)
        d_str  = '+1→' if d == 1 else '-1←' if d == -1 else '0?'
        dk_str = '✓' if dk else '?'
        return (f'    {label} {s} {form:>8}  conf={conf:.2f}  stab={stab:.2f}  '
                f'press={ph:.1f}m  def={dl:.1f}m  dir={d_str}[{dk_str}]')

    print(f'  Frame {frame_num:>4} | {ms:.1f}ms')
    print(_fmt_snap(hf, 'HOME'))
    print(_fmt_snap(af, 'AWAY'))
    if hc:
        top = hc[0]
        print(f'    ↳ COUNTER [{top["confidence"]:.2f}] {top["title"]}')
        sm = top.get('supporting_metrics', {})
        print(f'      {json.dumps(sm, separators=(",",":"))}')
    print()


def _print_counters(label: str, counters: list):
    if not counters:
        print(f'\n  {label}: no suggestions (teams not yet settled)')
        return
    print(f'\n  {label}:')
    for i, c in enumerate(counters[:3], 1):
        rt = c.get('risk_tradeoffs', {})
        sm = c.get('supporting_metrics', {})
        print(f'  {i}. [{c["confidence"]:.2f}] {c["title"]}')
        print(f'     mechanism : {c["mechanism"]}')
        print(f'     metrics   : {json.dumps(sm, separators=(",",":"))}')
        print(f'     reward    : {rt.get("reward","—")}')
        print(f'     risk      : {rt.get("risk","—")}')
        print(f'     condition : {rt.get("condition","—")}')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — SYNTHETIC VIDEO GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def _draw_pitch(frame, H_inv):
    lines = [
        [(0,0),(105,0)],[(105,0),(105,68)],[(105,68),(0,68)],[(0,68),(0,0)],
        [(52.5,0),(52.5,68)],
        [(0,13.84),(16.5,13.84)],[(16.5,13.84),(16.5,54.16)],[(16.5,54.16),(0,54.16)],
        [(105,13.84),(88.5,13.84)],[(88.5,13.84),(88.5,54.16)],[(88.5,54.16),(105,54.16)],
        [(0,24.84),(5.5,24.84)],[(5.5,24.84),(5.5,43.16)],[(5.5,43.16),(0,43.16)],
        [(105,24.84),(99.5,24.84)],[(99.5,24.84),(99.5,43.16)],[(99.5,43.16),(105,43.16)],
    ]
    def to_px(px, py):
        p = cv2.perspectiveTransform(np.array([[[px, py]]], dtype=np.float32), H_inv)
        return int(p[0,0,0]), int(p[0,0,1])
    for ln in lines:
        cv2.line(frame, to_px(*ln[0]), to_px(*ln[1]), (255,255,255), 2, cv2.LINE_AA)
    cp  = to_px(52.5, 34.0)
    rp  = to_px(52.5 + 9.15, 34.0)
    rad = int(math.hypot(rp[0]-cp[0], rp[1]-cp[1]))
    cv2.circle(frame, cp, rad, (255,255,255), 2)
    return frame


def generate_synthetic_video(output_path: str, n_frames: int = 750, fps: int = 25):
    W, H   = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    src    = np.float32([[64,72],[1216,72],[0,686],[1280,686]])
    dst    = np.float32([[0,0],[105,0],[0,68],[105,68]])
    H_mat, _ = cv2.findHomography(src, dst)
    H_inv    = np.linalg.inv(H_mat)

    def to_pixel(px, py):
        p = cv2.perspectiveTransform(np.array([[[px, py]]], dtype=np.float32), H_inv)
        return int(p[0,0,0]), int(p[0,0,1])

    logger.info(f"Generating {n_frames}-frame synthetic video → {output_path}")
    for fi in range(n_frames):
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        for s in range(10):
            x1 = int(W * s / 10); x2 = int(W * (s+1) / 10)
            frame[:, x1:x2] = (34,85,34) if s % 2 == 0 else (28,72,28)
        frame = _draw_pitch(frame, H_inv)
        for i, (x, y) in enumerate(_perturb(HOME_BASE, fi)):
            px, py = to_pixel(x, y)
            cv2.circle(frame, (px,py), 9, (220,80,80), -1)
            cv2.circle(frame, (px,py), 9, (255,255,255), 1)
            cv2.putText(frame, str(i+1), (px-4,py+4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
        for i, (x, y) in enumerate(_perturb(AWAY_BASE, fi)):
            px, py = to_pixel(x, y)
            cv2.circle(frame, (px,py), 9, (80,80,220), -1)
            cv2.circle(frame, (px,py), 9, (255,255,255), 1)
            cv2.putText(frame, str(i+1), (px-4,py+4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
        bx = 52.5 + 18.0 * math.sin(fi / fps * 0.9)
        by = 34.0 + 12.0 * math.cos(fi / fps * 1.1)
        bpx, bpy = to_pixel(bx, by)
        cv2.circle(frame, (bpx,bpy), 5, (255,255,255), -1)
        cv2.circle(frame, (bpx,bpy), 5, (0,128,255), 2)
        cv2.putText(frame, f"Frame {fi:04d}  t={fi/fps:.2f}s  SYNTHETIC",
                    (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        writer.write(frame)
    writer.release()
    logger.info(f"Saved: {output_path}")
    return H_mat


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — VIDEO MODE
# ══════════════════════════════════════════════════════════════════════════════

def run_video_mode(video_path: str, output_json: str, max_frames: int):
    from core.pipeline import FootballIntelligencePipeline

    config = {
        'yolo_model':              'yolov8n.pt',
        'detection_confidence':    0.35,
        'iou_threshold':           0.3,
        'max_track_misses':        30,
        'min_hits':                1,
        'target_fps':              5,
        'formation_window_frames': 60,
        'buffer_frames':           300,
    }
    pipeline = FootballIntelligencePipeline(config)
    results  = []

    print('\n' + '═'*72)
    print('  FOOTBALL INTELLIGENCE SYSTEM — VIDEO MODE')
    print('═'*72 + '\n')

    for n, analysis in enumerate(pipeline.process_video(video_path), 1):
        if n > max_frames:
            break
        jd = json.loads(pipeline.to_json(analysis))
        results.append(jd)
        _print_frame_summary(
            jd['frame_id'], jd['processing_ms'],
            jd.get('home_formation'), jd.get('away_formation'),
            jd.get('home_counters',[]), jd.get('away_counters',[]),
        )

    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'  Full output → {output_json}')
    print('═'*72 + '\n')


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Football Intelligence System Demo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--mode', choices=['positions','video','generate'],
                        default='positions',
                        help='positions (default) | video | generate')
    parser.add_argument('--video',  type=str, default=None)
    parser.add_argument('--frames', type=int, default=150,
                        help='Frames for positions or generate mode [default 150]')
    parser.add_argument('--max-analysis-frames', type=int, default=120,
                        help='Max frames to analyse in video mode [default 120]')
    parser.add_argument('--output', type=str, default='demo_output.json')
    args = parser.parse_args()

    if args.mode == 'positions':
        count = run_positions_mode(args.frames, args.output)
        sys.exit(0 if count > 0 else 1)

    elif args.mode == 'generate':
        Path('data/samples').mkdir(parents=True, exist_ok=True)
        generate_synthetic_video('data/samples/synthetic_match.mp4',
                                 n_frames=args.frames)

    elif args.mode == 'video':
        video_path = args.video
        if video_path is None:
            Path('data/samples').mkdir(parents=True, exist_ok=True)
            video_path = 'data/samples/synthetic_match.mp4'
            if not Path(video_path).exists():
                generate_synthetic_video(video_path, n_frames=750)
        run_video_mode(video_path, args.output, args.max_analysis_frames)
