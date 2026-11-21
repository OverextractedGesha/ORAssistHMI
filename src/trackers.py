"""trackers.py - MediaPipe Tasks API Trackers for OR-SterileAssist.

Provides high-level trackers:
- FaceTracker: Detects surgeon/operator face (BlazeFace short-range) for sterile zone distance/presence check.
- HandTracker: Real-time hand landmark tracker (21 points) in VIDEO mode.
- draw_hand: Enhanced rendering of hand skeleton with distinct surgical fingertip highlights.
"""
import os
import sys
import time
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = python.BaseOptions

FACE_MODEL = "blaze_face_short_range.tflite"
HAND_MODEL = "hand_landmarker.task"

MODEL_URLS = {
    FACE_MODEL: "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
    HAND_MODEL: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
}


def model_path(model_name: str) -> str:
    """Locates the MediaPipe model file in local, parent, or PyInstaller frozen directories, or downloads if missing."""
    if getattr(sys, "frozen", False):
        exe_dir = os.path.dirname(sys.executable)
        meipass = getattr(sys, "_MEIPASS", exe_dir)
        candidates = [
            os.path.join(meipass, "models", model_name),
            os.path.join(meipass, model_name),
            os.path.join(exe_dir, "models", model_name),
            os.path.join(exe_dir, model_name),
        ]
        dest_base = exe_dir
    else:
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        dest_base = cur_dir
        candidates = [
            os.path.join(cur_dir, "models", model_name),
            os.path.join(cur_dir, "..", "models", model_name),
            os.path.join(cur_dir, "..", "..", "models", model_name),
            os.path.join(r"C:\Users\LENOVO\Documents\5th Semester\HMI\models", model_name),
            os.path.join(r"C:\vision_app\models", model_name),
        ]
    for p in candidates:
        if os.path.isfile(p):
            return os.path.abspath(p)

    dest_dir = os.path.join(dest_base, "models")
    os.makedirs(dest_dir, exist_ok=True)
    dest_file = os.path.join(dest_dir, model_name)
    if not os.path.isfile(dest_file):
        url = MODEL_URLS.get(model_name)
        if url:
            print(f"[trackers] Downloading model {model_name}...")
            urllib.request.urlretrieve(url, dest_file)
            print(f"[trackers] Download complete: {model_name}")
    return os.path.abspath(dest_file)


def to_mp_image(frame_bgr):
    """Converts OpenCV BGR image to MediaPipe Image (RGB)."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


class FaceTracker:
    """Surgeon/Operator face detector using BlazeFace."""

    def __init__(self):
        options = vision.FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=model_path(FACE_MODEL)),
            min_detection_confidence=0.5,
        )
        self.detector = vision.FaceDetector.create_from_options(options)

    def find(self, frame_bgr):
        """Detects faces and returns list of bounding boxes [(x, y, w, h), ...] sorted by size (largest first)."""
        result = self.detector.detect(to_mp_image(frame_bgr))
        boxes = []
        for d in result.detections:
            b = d.bounding_box
            x = max(0, int(b.origin_x))
            y = max(0, int(b.origin_y))
            w = int(b.width)
            h = int(b.height)
            boxes.append((x, y, w, h))
        return sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)


class HandTracker:
    """Real-time 21-point hand landmark tracker in continuous VIDEO mode with EMA anti-jitter smoothing."""

    def __init__(self, num_hands=1, alpha=0.5):
        self.num_hands = num_hands
        self.alpha = alpha
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path(HAND_MODEL)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self._start_time = time.monotonic()
        self._last_ms = 0
        self._prev_landmarks = None
        self._missing_frames = 0

    def find(self, frame_bgr):
        """Detects primary hand landmarks with EMA smoothing. Returns 21 [(x, y), ...] pixel coordinates or None."""
        h, w = frame_bgr.shape[:2]
        now_ms = int((time.monotonic() - self._start_time) * 1000)
        if now_ms <= self._last_ms:
            now_ms = self._last_ms + 1
        self._last_ms = now_ms

        result = self.landmarker.detect_for_video(to_mp_image(frame_bgr), now_ms)
        if result.hand_landmarks and len(result.hand_landmarks) > 0:
            raw_pts = [(float(p.x * w), float(p.y * h)) for p in result.hand_landmarks[0]]
            if self._prev_landmarks is not None and self._missing_frames <= 3:
                # Exponential Moving Average (EMA) smoothing: alpha * current + (1 - alpha) * previous
                smoothed = []
                for (cx, cy), (px, py) in zip(raw_pts, self._prev_landmarks):
                    sx = self.alpha * cx + (1.0 - self.alpha) * px
                    sy = self.alpha * cy + (1.0 - self.alpha) * py
                    smoothed.append((sx, sy))
                self._prev_landmarks = smoothed
            else:
                self._prev_landmarks = raw_pts
            self._missing_frames = 0
            return [(int(round(x)), int(round(y))) for x, y in self._prev_landmarks]
        else:
            self._missing_frames += 1
            if self._missing_frames > 3:
                self._prev_landmarks = None
            return None


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17),                               # Palm base
]

FINGERTIP_INDICES = [4, 8, 12, 16, 20]


def draw_hand(frame, pts):
    """Renders hand skeleton with bones (emerald green), joints (red), and vivid fingertip highlights."""
    if not pts:
        return

    # Draw bones
    for p1_idx, p2_idx in HAND_CONNECTIONS:
        if p1_idx < len(pts) and p2_idx < len(pts):
            cv2.line(frame, pts[p1_idx], pts[p2_idx], (0, 230, 115), 2)

    # Draw joints
    for idx, p in enumerate(pts):
        if idx not in FINGERTIP_INDICES:
            cv2.circle(frame, p, 4, (0, 0, 230), -1)

    # Highlight fingertips:
    # 4: Thumb (Yellow)
    # 8: Index (Cyan pointer reticle)
    # 12, 16, 20: Bright Orange
    for tip_idx in FINGERTIP_INDICES:
        if tip_idx < len(pts):
            pt = pts[tip_idx]
            if tip_idx == 8:  # Surgical Index Pointer
                cv2.circle(frame, pt, 8, (255, 255, 0), -1)  # Cyan filled
                cv2.circle(frame, pt, 11, (0, 255, 255), 2)  # Yellow ring
            elif tip_idx == 4:  # Thumb
                cv2.circle(frame, pt, 7, (0, 215, 255), -1)  # Gold/Yellow
                cv2.circle(frame, pt, 9, (255, 255, 255), 1)
            else:
                cv2.circle(frame, pt, 6, (0, 140, 255), -1)  # Orange
                cv2.circle(frame, pt, 8, (255, 255, 255), 1)
