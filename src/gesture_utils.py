"""gesture_utils.py - Landmark calculations, finger counting, and stable gesture debouncing.

Used for surgical gesture extraction, Midas touch mitigation, and virtual laser pointer tracking.
"""
import math

# MediaPipe Hand Landmark Indices (21 points)
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4

INDEX_FINGER_MCP = 5
INDEX_FINGER_PIP = 6
INDEX_FINGER_DIP = 7
INDEX_FINGER_TIP = 8

MIDDLE_FINGER_MCP = 9
MIDDLE_FINGER_PIP = 10
MIDDLE_FINGER_DIP = 11
MIDDLE_FINGER_TIP = 12

RING_FINGER_MCP = 13
RING_FINGER_PIP = 14
RING_FINGER_DIP = 15
RING_FINGER_TIP = 16

PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

# Finger tip and PIP joint pairs for the 4 fingers
FINGERS = [
    (INDEX_FINGER_TIP, INDEX_FINGER_PIP),
    (MIDDLE_FINGER_TIP, MIDDLE_FINGER_PIP),
    (RING_FINGER_TIP, RING_FINGER_PIP),
    (PINKY_TIP, PINKY_PIP),
]


def dist(p1, p2):
    """Euclidean distance between two 2D points (x, y)."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def count_fingers(lm):
    """Counts how many fingers are extended straight (0 to 5).

    - Index, Middle, Ring, Pinky: extended if distance from tip to wrist
      is greater than distance from middle joint (PIP) to wrist.
    - Thumb: extended if distance from thumb tip to pinky MCP base is greater
      than distance from thumb IP joint to pinky MCP base.
    """
    if not lm or len(lm) < 21:
        return 0
    wrist = lm[WRIST]
    n = 0
    for tip, pip in FINGERS:
        d_tip = dist(lm[tip], wrist)
        d_pip = dist(lm[pip], wrist)
        if d_tip > d_pip:
            n += 1
    base = lm[PINKY_MCP]
    if dist(lm[THUMB_TIP], base) > dist(lm[THUMB_IP], base):
        n += 1
    return n


def is_pinch(lm, thresh=40):
    """Detects pinch gesture: True if thumb tip (4) and index tip (8) distance < thresh pixels."""
    if not lm or len(lm) < 21:
        return False
    return dist(lm[THUMB_TIP], lm[INDEX_FINGER_TIP]) < thresh


def pinch_ratio(lm):
    """Normalized pinch aperture ratio relative to hand size (wrist to middle finger MCP)."""
    if not lm or len(lm) < 21:
        return 1.0
    scale = max(dist(lm[WRIST], lm[MIDDLE_FINGER_MCP]), 1.0)
    return dist(lm[THUMB_TIP], lm[INDEX_FINGER_TIP]) / scale


def is_index_pointer(lm) -> bool:
    """Returns True when the index finger is extended, middle/ring/pinky are bent, and not pinching.
    
    Enables single-finger surgical laser pointing.
    """
    if not lm or len(lm) < 21:
        return False

    wrist = lm[WRIST]
    # Index finger extended
    index_extended = dist(lm[INDEX_FINGER_TIP], wrist) > dist(lm[INDEX_FINGER_PIP], wrist)
    if not index_extended:
        return False

    # Middle, Ring, Pinky bent
    other_fingers = [
        (MIDDLE_FINGER_TIP, MIDDLE_FINGER_PIP),
        (RING_FINGER_TIP, RING_FINGER_PIP),
        (PINKY_TIP, PINKY_PIP),
    ]
    for tip, pip in other_fingers:
        if dist(lm[tip], wrist) >= dist(lm[pip], wrist):
            return False

    # Not pinching
    if pinch_ratio(lm) <= 0.35:
        return False

    return True


def angle(a, b, c):
    """Calculates angle at joint b between ba and bc in degrees (0 - 180)."""
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    mag_ba = math.hypot(*ba)
    mag_bc = math.hypot(*bc)
    if mag_ba * mag_bc == 0:
        return 180.0
    dot = ba[0] * bc[0] + ba[1] * bc[1]
    cos_val = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cos_val))


class Hysteresis:
    """Hysteresis filter for discrete state transitions with dual thresholds."""

    def __init__(self, on_below, off_above):
        self.on_below = on_below
        self.off_above = off_above
        self.active = False

    def update(self, val):
        if not self.active and val <= self.on_below:
            self.active = True
            return True
        elif self.active and val >= self.off_above:
            self.active = False
        return False


class StableGesture:
    """Gesture debouncer requiring a pose to be held for N frames before emitting an event.
    
    Mitigates Midas touch effects.
    """

    def __init__(self, frames=15):
        self.needed = frames
        self.last = None
        self.count = 0

    def update(self, gesture):
        if gesture == self.last:
            self.count += 1
        else:
            self.last, self.count = gesture, 1
        if gesture is not None and self.count == self.needed:
            return gesture
        return None

    @property
    def progress(self):
        """Gesture hold progress from 0.0 to 1.0 for UI progress bars."""
        if self.last is None or self.count == 0:
            return 0.0
        return min(self.count / self.needed, 1.0)

    def reset(self):
        """Resets the debouncer."""
        self.last = None
        self.count = 0
