"""
Pitch Calibration Module v2 — Broadcast-noise hardened

Key changes vs v1:
  1. Confidence-gated homography updates
     - The homography is FROZEN when calibration confidence < FREEZE_THRESHOLD.
     - A new calibration attempt only REPLACES the stored H if the new
       confidence is HIGHER than the current one. This prevents degraded
       camera-cut frames from overwriting a good calibration.

  2. Confidence score is multi-component:
       - RANSAC inlier ratio (0-1)
       - Number of detected pitch lines (saturates at N_LINES_GOOD)
       - Orthogonality of detected lines (lines should be ≈ 90° on pitch)
     Final confidence = weighted product. Exposed as `calibration_confidence`.

  3. Camera-cut / zoom detection
     - Sudden large changes in the dominant line angles → camera cut.
     - On detected cut: mark as `cut_detected`, reset confidence to 0.
       Next frame will trigger a fresh calibration attempt.

  4. Broadcast-aware default homography
     - The fallback now uses typical broadcast camera geometry per shot type
       (wide, medium, close). Shot type is estimated from green area fraction.

  5. Temporal homography smoothing
     - When a new calibration succeeds with good confidence, the homography
       is linearly blended with the previous one (α=0.3) to avoid jumps.
     - This handles gradual zoom-in/out without abrupt coordinate jumps.

  6. Manual override always wins
     - set_manual_points() sets confidence=1.0 and disables auto-recalibration
       until next camera cut is detected.

How zoom/cuts are handled:
  - Zoom changes the pixel scale but not the pitch geometry → homography
    becomes wrong. We detect zoom via pitch-line length change (>40% jump).
  - On zoom change: recalibrate. If lines insufficient: keep old H frozen
    and flag low confidence so formation engine suppresses output.
  - On camera cut (new view): full reset + recalibrate.
"""

import numpy as np
import cv2
from typing import Optional, Tuple, List, Dict
import logging

logger = logging.getLogger(__name__)

PITCH_W = 105.0
PITCH_H = 68.0

PITCH_LANDMARKS = {
    'tl_corner': (0.0, 0.0),
    'tr_corner': (PITCH_W, 0.0),
    'bl_corner': (0.0, PITCH_H),
    'br_corner': (PITCH_W, PITCH_H),
    'tl_penalty': (16.5, 13.84),
    'tr_penalty': (88.5, 13.84),
    'bl_penalty': (16.5, 54.16),
    'br_penalty': (88.5, 54.16),
    'center': (52.5, 34.0),
    'tl_6yd': (5.5, 24.84),
    'tr_6yd': (99.5, 24.84),
    'bl_6yd': (5.5, 43.16),
    'br_6yd': (99.5, 43.16),
}

FREEZE_THRESHOLD = 0.35         # below this → freeze H, flag low confidence
N_LINES_GOOD = 12               # enough lines for full calibration
BLEND_ALPHA = 0.30              # homography temporal blend weight
ZOOM_JUMP_THRESHOLD = 0.40      # >40% change in mean line length → zoom change
CUT_ANGLE_THRESHOLD = 25.0      # degrees — dominant line angle jump → cut


class PitchCalibrator:

    def __init__(self, config: Dict):
        self.config = config
        self.H: Optional[np.ndarray] = None
        self.H_inv: Optional[np.ndarray] = None
        self.is_calibrated = False
        self.calibration_confidence = 0.0
        self.cut_detected = False
        self._manual_locked = False
        self._prev_line_length: Optional[float] = None
        self._prev_dominant_angle: Optional[float] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def calibrate(self, frame: np.ndarray) -> bool:
        """
        Attempt (re)calibration from frame.
        New H only accepted if new confidence > current confidence.
        """
        if self._manual_locked and not self.cut_detected:
            return True

        try:
            field_mask = self._field_mask(frame)
            line_img = self._extract_lines(frame, field_mask)
            lines = self._hough_lines(line_img)

            # Cut / zoom detection
            self._detect_cut_zoom(lines)
            if self.cut_detected:
                self.calibration_confidence = 0.0
                self._manual_locked = False
                logger.info("Camera cut detected — resetting calibration")

            if lines is None or len(lines) < 4:
                if not self.is_calibrated:
                    self._default_homography(frame)
                return self.is_calibrated

            intersections = self._intersections(lines)
            src_pts, dst_pts = self._match_landmarks(intersections, frame.shape)

            if len(src_pts) < 4:
                if not self.is_calibrated:
                    self._default_homography(frame)
                return self.is_calibrated

            H_new, mask = cv2.findHomography(
                np.array(src_pts, dtype=np.float32),
                np.array(dst_pts, dtype=np.float32),
                cv2.RANSAC, 5.0
            )

            if H_new is None:
                if not self.is_calibrated:
                    self._default_homography(frame)
                return self.is_calibrated

            # Compute multi-component confidence
            inlier_ratio = float(mask.sum()) / len(mask) if mask is not None else 0.0
            line_score = min(1.0, len(lines) / N_LINES_GOOD)
            ortho_score = self._orthogonality_score(lines)
            new_conf = 0.5 * inlier_ratio + 0.3 * line_score + 0.2 * ortho_score

            logger.debug(f"Calibration: inlier={inlier_ratio:.2f} lines={line_score:.2f} "
                         f"ortho={ortho_score:.2f} → conf={new_conf:.2f}")

            if new_conf < FREEZE_THRESHOLD:
                # Not good enough to update
                if not self.is_calibrated:
                    self._default_homography(frame)
                    self.calibration_confidence = new_conf
                logger.warning(f"Calibration confidence {new_conf:.2f} below freeze threshold "
                               f"{FREEZE_THRESHOLD} — keeping previous H")
                return self.is_calibrated

            if new_conf <= self.calibration_confidence and self.is_calibrated:
                # Don't downgrade
                return True

            # Temporal blend if we have a previous H
            if self.H is not None:
                H_blended = BLEND_ALPHA * H_new + (1 - BLEND_ALPHA) * self.H
                # Re-normalise so H[2,2] = 1
                H_blended /= H_blended[2, 2]
                self.H = H_blended
            else:
                self.H = H_new

            self.H_inv = np.linalg.inv(self.H)
            self.is_calibrated = True
            self.calibration_confidence = new_conf
            self.cut_detected = False
            logger.info(f"Pitch calibrated — confidence: {new_conf:.2%}")
            return True

        except Exception as e:
            logger.error(f"Calibration error: {e}")
            if not self.is_calibrated:
                self._default_homography(frame)
            return self.is_calibrated

    def set_manual_points(self, pixel_pts: List[Tuple], pitch_pts: List[Tuple]) -> bool:
        """Manual calibration. Sets confidence=1.0, locks out auto-recal."""
        H, _ = cv2.findHomography(
            np.float32(pixel_pts), np.float32(pitch_pts), cv2.RANSAC, 5.0
        )
        if H is None:
            return False
        self.H = H
        self.H_inv = np.linalg.inv(H)
        self.is_calibrated = True
        self.calibration_confidence = 1.0
        self._manual_locked = True
        self.cut_detected = False
        logger.info("Manual calibration applied — confidence: 1.0")
        return True

    def to_pitch_coords(self, pixel_pos: Tuple[float, float]) -> Tuple[float, float]:
        if self.H is None:
            return pixel_pos
        pt = np.array([[[pixel_pos[0], pixel_pos[1]]]], dtype=np.float32)
        t = cv2.perspectiveTransform(pt, self.H)
        x = float(np.clip(t[0, 0, 0], 0, PITCH_W))
        y = float(np.clip(t[0, 0, 1], 0, PITCH_H))
        return (x, y)

    def to_pixel_coords(self, pitch_pos: Tuple[float, float]) -> Tuple[float, float]:
        if self.H_inv is None:
            return pitch_pos
        pt = np.array([[[pitch_pos[0], pitch_pos[1]]]], dtype=np.float32)
        t = cv2.perspectiveTransform(pt, self.H_inv)
        return (float(t[0, 0, 0]), float(t[0, 0, 1]))

    # ── Private ───────────────────────────────────────────────────────────────

    def _field_mask(self, frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv,
                           np.array([30, 35, 35]),
                           np.array([90, 255, 255]))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    def _extract_lines(self, frame: np.ndarray, field_mask: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # CLAHE to handle lighting variation / stadium shadows
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        # Adaptive threshold — handles different turf brightness
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, -15
        )
        return cv2.bitwise_and(thresh, thresh, mask=field_mask)

    def _hough_lines(self, line_img: np.ndarray):
        edges = cv2.Canny(line_img, 30, 100)
        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi/180,
            threshold=40, minLineLength=40, maxLineGap=25
        )
        return lines

    def _detect_cut_zoom(self, lines):
        """Update cut/zoom flags based on line statistics."""
        if lines is None:
            return

        lengths = []
        angles = []
        for line in lines[:, 0]:
            x1, y1, x2, y2 = line
            l = float(np.hypot(x2 - x1, y2 - y1))
            a = float(np.degrees(np.arctan2(y2 - y1, x2 - x1))) % 180
            lengths.append(l)
            angles.append(a)

        mean_len = float(np.mean(lengths))
        # Dominant angle (most common within 5° bins)
        hist, edges = np.histogram(angles, bins=36, range=(0, 180))
        dom_angle = float(edges[np.argmax(hist)])

        if self._prev_line_length is not None:
            zoom_change = abs(mean_len - self._prev_line_length) / (self._prev_line_length + 1e-6)
            angle_change = min(abs(dom_angle - self._prev_dominant_angle),
                               180 - abs(dom_angle - self._prev_dominant_angle))
            if angle_change > CUT_ANGLE_THRESHOLD:
                self.cut_detected = True
            elif zoom_change > ZOOM_JUMP_THRESHOLD:
                # Zoom — recalibrate but don't reset
                logger.info(f"Zoom change {zoom_change:.1%} — scheduling recalibration")
                self.calibration_confidence = max(0.0, self.calibration_confidence - 0.3)

        self._prev_line_length = mean_len
        self._prev_dominant_angle = dom_angle

    def _orthogonality_score(self, lines) -> float:
        """Score how well lines form ≈90° pairs (expected for pitch markings)."""
        if lines is None or len(lines) < 4:
            return 0.0
        angles = []
        for line in lines[:, 0]:
            x1, y1, x2, y2 = line
            angles.append(float(np.degrees(np.arctan2(y2-y1, x2-x1))) % 180)
        angles = np.array(angles)
        # Cluster into two groups
        h, e = np.histogram(angles, bins=18, range=(0, 180))
        top2 = np.argsort(h)[-2:]
        if len(top2) < 2:
            return 0.0
        a1 = float(e[top2[0]])
        a2 = float(e[top2[1]])
        diff = abs(a1 - a2) % 180
        perp = min(diff, 180 - diff)   # close to 90 is good
        return float(1.0 - abs(perp - 90.0) / 90.0)

    def _intersections(self, lines) -> List[Tuple[float, float]]:
        pts = []
        ls = lines[:, 0]
        for i in range(len(ls)):
            for j in range(i+1, len(ls)):
                p = self._line_intersect(ls[i], ls[j])
                if p:
                    pts.append(p)
        return pts

    @staticmethod
    def _line_intersect(l1, l2) -> Optional[Tuple[float, float]]:
        x1,y1,x2,y2 = l1
        x3,y3,x4,y4 = l2
        d = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if abs(d) < 1e-6:
            return None
        t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / d
        return (x1 + t*(x2-x1), y1 + t*(y2-y1))

    def _match_landmarks(self, pts, shape) -> Tuple[List, List]:
        h, w = shape[:2]
        if not pts:
            return [], []
        arr = np.array(pts)
        valid = arr[(arr[:,0]>=0)&(arr[:,0]<w)&(arr[:,1]>=0)&(arr[:,1]<h)]
        if len(valid) < 4:
            return [], []

        src = [
            valid[np.argmin( valid[:,0] + valid[:,1])].tolist(),
            valid[np.argmin(-valid[:,0] + valid[:,1])].tolist(),
            valid[np.argmin( valid[:,0] - valid[:,1])].tolist(),
            valid[np.argmin(-valid[:,0] - valid[:,1])].tolist(),
        ]
        dst = [list(PITCH_LANDMARKS['tl_corner']), list(PITCH_LANDMARKS['tr_corner']),
               list(PITCH_LANDMARKS['bl_corner']), list(PITCH_LANDMARKS['br_corner'])]
        return src, dst

    def _default_homography(self, frame: np.ndarray) -> bool:
        """
        Broadcast-aware fallback. Estimates shot type from green area fraction:
          - >70% green → wide shot (full pitch visible)
          - 40-70%     → medium shot (half-pitch)
          - <40%       → close-up (limited view)
        """
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(hsv, np.array([30,35,35]), np.array([90,255,255]))
        green_ratio = green.mean() / 255.0

        if green_ratio > 0.70:
            # Wide: full pitch in frame
            src = np.float32([[w*0.04,h*0.08],[w*0.96,h*0.08],
                               [w*0.00,h*0.96],[w*1.00,h*0.96]])
            dst = np.float32([[0,0],[PITCH_W,0],[0,PITCH_H],[PITCH_W,PITCH_H]])
            conf = 0.35
        elif green_ratio > 0.40:
            # Medium: ~half pitch
            src = np.float32([[w*0.02,h*0.05],[w*0.98,h*0.05],
                               [w*0.00,h*0.98],[w*1.00,h*0.98]])
            dst = np.float32([[0,0],[PITCH_W/2,0],[0,PITCH_H],[PITCH_W/2,PITCH_H]])
            conf = 0.25
        else:
            # Close-up: small region
            src = np.float32([[w*0.0,h*0.0],[w*1.0,h*0.0],
                               [w*0.0,h*1.0],[w*1.0,h*1.0]])
            dst = np.float32([[20,14],[55,14],[20,54],[55,54]])
            conf = 0.15

        H, _ = cv2.findHomography(src, dst)
        if H is not None:
            self.H = H
            self.H_inv = np.linalg.inv(H)
            self.is_calibrated = True
            self.calibration_confidence = conf
            logger.warning(f"Default homography (shot_type conf={conf:.2f}, "
                           f"green_ratio={green_ratio:.2f})")
        return self.is_calibrated
