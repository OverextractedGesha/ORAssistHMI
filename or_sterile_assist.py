"""or_sterile_assist.py - Touchless Intra-Operative CT & Vital Monitor.

Project ETS Human-Machine Interface (HMI) Biomedical Application.
Features:
- Touchless CT navigation via OpenCV & MediaPipe Hand Tracking (pinch-to-scroll, 1-finger surgical laser pointer with dwell marking).
- Face presence & distance monitoring via BlazeFace (sterile zone auto-lock).
- Multi-window CT viewing (Brain, Bone, Subdural) on realistic 3D Head CT phantom.
- Intra-operative vital signs monitoring with SpO2 pitch sonification (IEC 60601-1-8 standard).
- IEC 60601-1-8 standard medical alarms with 2-minute alarm silence countdown.
- Bilingual speech recognition (Indonesian online Google ASR, English offline Vosk ASR, clinical dictation).
- Half-duplex audio protection preventing self-triggering during TTS speech output.
- Live ambient noise calibration with audio dBFS meter.
- Full redundant keyboard, mouse, and tactile button controls.
- Intra-operative surgical logging with .txt report export.
"""
import os
import sys

# Safe stdout/stderr redirection for pythonw.exe (GUI-only launcher without console)
if sys.stdout is None or sys.stderr is None:
    _log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "or_sterile_assist.log")
    try:
        _log_file = open(_log_path, "a", encoding="utf-8", buffering=1)
        if sys.stdout is None:
            sys.stdout = _log_file
        if sys.stderr is None:
            sys.stderr = _log_file
    except Exception:
        pass
import time
import datetime
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import cv2
from PIL import Image, ImageTk
import sounddevice as sd

# Local module imports
from audio_utils import (
    EARCONS,
    ALARMS,
    SAMPLE_RATE,
    sequence,
    pulse_ox_tone,
    spo2_to_pitch,
)
from trackers import FaceTracker, HandTracker, draw_hand
from gesture_utils import (
    count_fingers,
    is_pinch,
    pinch_ratio,
    is_index_pointer,
    dist,
    StableGesture,
    Hysteresis,
)
from ct_phantom import (
    N_SLICES,
    SIZE as CT_SIZE,
    WINDOWS,
    make_phantom,
    to_display_rgb,
    get_hu_at,
)
from speech_engine import SpeechEngine

# Clinical Dark Palette
BG_DARK = "#12151b"
BG_PANEL = "#1a1e27"
BG_CARD = "#222733"
BG_INPUT = "#161922"
FG_PRIMARY = "#eceff4"
FG_MUTED = "#8892b0"
ACCENT_CYAN = "#00e5ff"
ACCENT_TEAL = "#26a69a"
ACCENT_YELLOW = "#ffd54f"
ACCENT_GREEN = "#4caf50"
ACCENT_RED = "#ff5252"
ACCENT_ORANGE = "#ff9800"


class ORSterileAssistApp:
    """Main Touchless Intra-Operative Biomedical Application."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("OR-SterileAssist — Touchless Intra-Operative CT & Vital Monitor (ETS HMI)")
        self.root.geometry("1340x840")
        self.root.minsize(1280, 780)
        self.root.configure(bg=BG_DARK)

        # State Variables
        self.is_running = True
        self.current_slice = 38  # Default near target lesion
        self.current_window = "Brain"
        self.bookmarks = set([38])
        self.markers = {}  # {slice_idx: [(u, v, label, hu_val), ...]}
        self.cursor_uv = (0.5, 0.5)
        self.dwell_progress = 0.0

        # UI State Cache & Anti-Jitter flags
        self._ui_cache = {}
        self._updating_slider = False
        self._last_render_state = None

        # Laser pointer dwell state
        self.last_laser_uv = None
        self.laser_dwell_count = 0
        self.LASER_DWELL_TARGET = 42  # ~1.4 seconds at 30 fps
        self.last_laser_time = 0.0

        # Pinch Clutching Hysteresis (on_below=0.26, off_above=0.45)
        self.pinch_grab = Hysteresis(on_below=0.26, off_above=0.45)
        self.pinch_anchor_y = 0.0
        self.pinch_anchor_slice = self.current_slice
        self.last_pinch_time = 0.0

        # Face Presence Debouncer (15 frames)
        self.missing_face_frames = 0
        self._last_face_status = "NO FACE"
        self._last_face_locked = False

        # Gesture Debouncer (15 frames = ~0.5s at 30 fps)
        self.gesture_debouncer = StableGesture(frames=15)
        self.gesture_locked = False
        self.face_autolock = tk.BooleanVar(value=True)

        # Patient Vitals
        self.vitals = {
            "hr": 74,
            "spo2": 98,
            "bp_sys": 118,
            "bp_dia": 76,
            "temp": 36.8,
            "alarm_severity": "none",  # "none", "low", "medium", "high"
        }
        self.alarm_silenced_until = 0.0  # monotonic timestamp

        # Pulse Ox Sonification
        self.pulse_ox_active = tk.BooleanVar(value=False)
        self.pulse_ox_job = None

        # Pre-seed marker on slice 38
        self.markers[38] = [(0.66, 0.39, "Hyperdense Lesion", 78.0)]

        # 2. Initialize Models & Engines
        print("[OR-SterileAssist] Generating 3D Head CT Phantom...")
        self.ct_volume = make_phantom()
        print("[OR-SterileAssist] CT Phantom ready.")

        # Audio & Speech Engine
        self.speech = SpeechEngine()

        # Computer Vision Trackers & Camera
        self.hand_tracker = None
        self.face_tracker = None
        self._init_trackers()

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            # Try default backend if CAP_DSHOW fails
            self.cap = cv2.VideoCapture(0)
        self.cam_available = self.cap.isOpened()
        if self.cam_available:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # 3. Build User Interface
        self._setup_styles()
        self._build_ui()
        self._bind_shortcuts()

        # 4. Start Event Loops
        self.last_cam_time = time.monotonic()
        self.root.after(30, self._camera_loop)
        self.root.after(50, self._speech_poll_loop)
        self.root.after(1000, self._vitals_monitor_loop)

        # Initial sound confirmation
        self.speech.play_sound(sequence(EARCONS["berhasil"]))

    def _set_label_cached(self, key: str, widget, **kwargs):
        """Updates widget config only if kwargs differ from cached values, preventing Tkinter layout recalculation."""
        prev = self._ui_cache.get(key)
        if prev != kwargs:
            self._ui_cache[key] = kwargs
            widget.config(**kwargs)

    def _init_trackers(self):
        """Initializes MediaPipe trackers safely."""
        try:
            self.hand_tracker = HandTracker()
            self.face_tracker = FaceTracker()
            print("[OR-SterileAssist] MediaPipe trackers initialized.")
        except Exception as e:
            print(f"[OR-SterileAssist] Tracker initialization warning: {e}")

    def _setup_styles(self):
        """Configures ttk widget styles for a sleek dark surgical appearance."""
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TProgressbar", thickness=12, troughcolor=BG_CARD, background=ACCENT_CYAN)
        style.configure("Green.Horizontal.TProgressbar", thickness=10, troughcolor=BG_CARD, background=ACCENT_GREEN)
        style.configure("Dwell.Horizontal.TProgressbar", thickness=8, troughcolor=BG_CARD, background=ACCENT_YELLOW)

    def _build_ui(self):
        """Constructs the responsive 3-column layout."""
        # Main container
        self.main_frame = tk.Frame(self.root, bg=BG_DARK)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Column Grid Configuration: 3 Columns
        self.main_frame.columnconfigure(0, weight=4, minsize=420)  # Left: CT Viewer
        self.main_frame.columnconfigure(1, weight=3, minsize=360)  # Mid: Camera & Sensor
        self.main_frame.columnconfigure(2, weight=5, minsize=460)  # Right: Vitals & Speech Log
        self.main_frame.rowconfigure(0, weight=1)

        self._build_left_ct_panel()
        self._build_middle_sensor_panel()
        self._build_right_vitals_speech_panel()

    # =========================================================================
    # LEFT COLUMN: Touchless CT Viewer & Slice Navigation
    # =========================================================================
    def _build_left_ct_panel(self):
        panel = tk.Frame(self.main_frame, bg=BG_PANEL, relief=tk.RIDGE, bd=1, width=440)
        panel.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        panel.grid_propagate(False)
        panel.pack_propagate(False)

        # Header Title
        hdr = tk.Frame(panel, bg=BG_PANEL)
        hdr.pack(fill=tk.X, padx=8, pady=(8, 4))
        tk.Label(hdr, text="NEURO CT SCANNER", font=("Segoe UI", 12, "bold"), fg=ACCENT_CYAN, bg=BG_PANEL).pack(side=tk.LEFT)
        self.window_badge = tk.Label(hdr, text="Brain (L:40 W:80)", font=("Segoe UI", 9, "bold"), fg=BG_DARK, bg=ACCENT_TEAL, padx=6, pady=2)
        self.window_badge.pack(side=tk.RIGHT)

        # CT Canvas / Label
        ct_border = tk.Frame(panel, bg="#0a0c10", padx=2, pady=2)
        ct_border.pack(padx=8, pady=4)
        self.ct_label = tk.Label(ct_border, bg="#000000", width=CT_SIZE, height=CT_SIZE)
        self.ct_label.pack()

        # Mouse Interaction Bindings on CT Label
        self.ct_label.bind("<Motion>", self._on_ct_mouse_motion)
        self.ct_label.bind("<Button-1>", self._on_ct_mouse_click)
        self.ct_label.bind("<MouseWheel>", self._on_ct_mouse_wheel)

        # Live CT Readout Bar
        self.ct_status_lbl = tk.Label(
            panel,
            text="Slice: 38/60 | Window: Brain | Cursor HU: +78 HU | Target: ROI-1",
            font=("Consolas", 9),
            fg=FG_PRIMARY,
            bg=BG_CARD,
            pady=4,
            width=54,
            anchor="w",
        )
        self.ct_status_lbl.pack(fill=tk.X, padx=8, pady=4)

        # Slice Slider (Scrubber)
        slider_frame = tk.Frame(panel, bg=BG_PANEL)
        slider_frame.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(slider_frame, text="Slice Scrub:", font=("Segoe UI", 9), fg=FG_MUTED, bg=BG_PANEL).pack(side=tk.LEFT, padx=4)
        self.slice_slider = tk.Scale(
            slider_frame,
            from_=0,
            to=N_SLICES - 1,
            orient=tk.HORIZONTAL,
            bg=BG_PANEL,
            fg=FG_PRIMARY,
            troughcolor=BG_INPUT,
            highlightthickness=0,
            command=self._on_slider_change,
        )
        self.slice_slider.set(self.current_slice)
        self.slice_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        # Redundant Tactile Control Buttons
        btn_grid = tk.Frame(panel, bg=BG_PANEL)
        btn_grid.pack(fill=tk.X, padx=8, pady=6)
        btn_grid.columnconfigure((0, 1, 2), weight=1)

        tk.Button(btn_grid, text="◀ Prev (P)", font=("Segoe UI", 9, "bold"), bg=BG_CARD, fg=FG_PRIMARY, activebackground=ACCENT_CYAN, command=self.prev_slice).grid(row=0, column=0, padx=2, pady=2, sticky="ew")
        tk.Button(btn_grid, text="Next ▶ (N)", font=("Segoe UI", 9, "bold"), bg=BG_CARD, fg=FG_PRIMARY, activebackground=ACCENT_CYAN, command=self.next_slice).grid(row=0, column=1, padx=2, pady=2, sticky="ew")
        tk.Button(btn_grid, text="Window (W)", font=("Segoe UI", 9, "bold"), bg=BG_CARD, fg=ACCENT_TEAL, command=self.cycle_window).grid(row=0, column=2, padx=2, pady=2, sticky="ew")

        tk.Button(btn_grid, text="★ Bookmark (B)", font=("Segoe UI", 9), bg=BG_CARD, fg=ACCENT_YELLOW, command=self.bookmark_current_slice).grid(row=1, column=0, padx=2, pady=2, sticky="ew")
        tk.Button(btn_grid, text="⊕ Mark Lesion (M)", font=("Segoe UI", 9), bg=BG_CARD, fg=ACCENT_CYAN, command=self.mark_current_lesion).grid(row=1, column=1, padx=2, pady=2, sticky="ew")
        tk.Button(btn_grid, text="⊗ Clear Mark (C)", font=("Segoe UI", 9), bg=BG_CARD, fg=FG_MUTED, command=self.clear_markers_on_slice).grid(row=1, column=2, padx=2, pady=2, sticky="ew")

        # Quick Jump & Bookmarks Bar
        jump_frame = tk.Frame(panel, bg=BG_PANEL)
        jump_frame.pack(fill=tk.X, padx=8, pady=(2, 6))
        tk.Button(jump_frame, text="⚡ Jump to Lesion (Slice 38)", font=("Segoe UI", 9, "bold"), bg="#1c3d44", fg=ACCENT_CYAN, command=lambda: self.set_slice(38)).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        self.bookmark_list_lbl = tk.Label(jump_frame, text="Bookmarks: [38]", font=("Segoe UI", 8), fg=ACCENT_YELLOW, bg=BG_PANEL)
        self.bookmark_list_lbl.pack(side=tk.RIGHT, padx=4)

        # Initial CT render
        self._render_ct_display()

    # =========================================================================
    # MIDDLE COLUMN: Computer Vision, Gestures & Visual Feedback
    # =========================================================================
    def _build_middle_sensor_panel(self):
        panel = tk.Frame(self.main_frame, bg=BG_PANEL, relief=tk.RIDGE, bd=1, width=380)
        panel.grid(row=0, column=1, sticky="nsew", padx=4, pady=4)
        panel.grid_propagate(False)
        panel.pack_propagate(False)

        # Header Title & Lock Toggle
        hdr = tk.Frame(panel, bg=BG_PANEL)
        hdr.pack(fill=tk.X, padx=8, pady=(8, 4))
        tk.Label(hdr, text="STERILE VISION SENSOR", font=("Segoe UI", 12, "bold"), fg=ACCENT_CYAN, bg=BG_PANEL).pack(side=tk.LEFT)
        self.lock_badge = tk.Label(hdr, text="[ACTIVE]", font=("Segoe UI", 9, "bold"), fg=BG_DARK, bg=ACCENT_GREEN, padx=6, pady=2)
        self.lock_badge.pack(side=tk.RIGHT)

        # Camera Display (320 x 240)
        cam_border = tk.Frame(panel, bg="#0a0c10", padx=2, pady=2)
        cam_border.pack(padx=8, pady=4)
        self.cam_label = tk.Label(cam_border, bg="#000000", width=320, height=240)
        self.cam_label.pack()

        # Face & Distance Status Banner
        self.face_status_lbl = tk.Label(
            panel,
            text="Surgeon: [DETECTING...]",
            font=("Segoe UI", 9, "bold"),
            fg=FG_PRIMARY,
            bg=BG_CARD,
            pady=4,
            width=36,
            anchor="w",
        )
        self.face_status_lbl.pack(fill=tk.X, padx=8, pady=4)

        # Current Gesture Status Card (Fixed height 96 with pack_propagate False)
        gest_card = tk.Frame(panel, bg=BG_CARD, padx=8, pady=6, height=96)
        gest_card.pack(fill=tk.X, padx=8, pady=4)
        gest_card.pack_propagate(False)

        tk.Label(gest_card, text="ACTIVE GESTURE MODE", font=("Segoe UI", 8, "bold"), fg=FG_MUTED, bg=BG_CARD).pack(anchor="w")
        self.gesture_mode_lbl = tk.Label(
            gest_card,
            text="[IDLE] NO HAND DETECTED",
            font=("Segoe UI", 10, "bold"),
            fg=ACCENT_YELLOW,
            bg=BG_CARD,
            width=34,
            anchor="w",
        )
        self.gesture_mode_lbl.pack(anchor="w", pady=2)

        # Hold / Dwell Progress Bar (Midas Touch Mitigation)
        dwell_hdr = tk.Frame(gest_card, bg=BG_CARD)
        dwell_hdr.pack(fill=tk.X, pady=(4, 2))
        tk.Label(dwell_hdr, text="Gesture Hold / Laser Dwell Lock:", font=("Segoe UI", 8), fg=FG_MUTED, bg=BG_CARD).pack(side=tk.LEFT)
        self.dwell_pct_lbl = tk.Label(dwell_hdr, text="0%", font=("Segoe UI", 8, "bold"), fg=ACCENT_CYAN, bg=BG_CARD)
        self.dwell_pct_lbl.pack(side=tk.RIGHT)

        self.dwell_bar = ttk.Progressbar(gest_card, style="Dwell.Horizontal.TProgressbar", length=280, mode="determinate")
        self.dwell_bar.pack(fill=tk.X, pady=2)

        # Gesture Guide Cheatsheet (Course Standard Gesture Map)
        guide_box = tk.LabelFrame(panel, text="Surgical Gesture Mappings", font=("Segoe UI", 8, "bold"), fg=FG_MUTED, bg=BG_PANEL)
        guide_box.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        guides = [
            ("[Pinch] Thumb+Index", "Scroll slices Up / Down"),
            ("[1 Finger] Laser", "Point at CT & dwell 1.4s to mark"),
            ("[2 Fingers] Hold 0.5s", "Cycle window: Brain/Bone/Subdural"),
            ("[3 Fingers] Hold 0.5s", "Bookmark current slice"),
            ("[4 Fingers] Hold 0.5s", "Voice summary of patient vitals"),
            ("[5 Fingers] Hold 0.5s", "Toggle sterile gesture lock"),
        ]
        for gest, action in guides:
            row = tk.Frame(guide_box, bg=BG_PANEL)
            row.pack(fill=tk.X, padx=4, pady=1)
            tk.Label(row, text=gest, font=("Segoe UI", 8, "bold"), fg=ACCENT_CYAN, bg=BG_PANEL, width=22, anchor="w").pack(side=tk.LEFT)
            tk.Label(row, text=action, font=("Segoe UI", 8), fg=FG_PRIMARY, bg=BG_PANEL, anchor="w").pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Sterile Controls Row
        ctrl_row = tk.Frame(panel, bg=BG_PANEL)
        ctrl_row.pack(fill=tk.X, padx=8, pady=(4, 8))
        self.lock_btn = tk.Button(
            ctrl_row,
            text="Pause Gesture Lock (Space)",
            font=("Segoe UI", 9, "bold"),
            bg=BG_CARD,
            fg=ACCENT_YELLOW,
            command=self.toggle_gesture_lock,
        )
        self.lock_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        tk.Checkbutton(
            ctrl_row,
            text="Face Auto-Lock",
            variable=self.face_autolock,
            font=("Segoe UI", 8),
            fg=FG_PRIMARY,
            bg=BG_PANEL,
            selectcolor=BG_CARD,
            activebackground=BG_PANEL,
        ).pack(side=tk.RIGHT, padx=4)

    # =========================================================================
    # RIGHT COLUMN: Vitals, IEC 60601-1-8 Alarms & Voice Dictation
    # =========================================================================
    def _build_right_vitals_speech_panel(self):
        panel = tk.Frame(self.main_frame, bg=BG_PANEL, relief=tk.RIDGE, bd=1)
        panel.grid(row=0, column=2, sticky="nsew", padx=4, pady=4)
        panel.rowconfigure(0, weight=4)  # Vitals & Alarms
        panel.rowconfigure(1, weight=5)  # Speech & Dictation
        panel.columnconfigure(0, weight=1)
        panel.grid_propagate(False)
        panel.pack_propagate(False)

        # TOP HALF: Intra-Operative Vitals & Alarms
        vitals_panel = tk.Frame(panel, bg=BG_PANEL, height=330)
        vitals_panel.grid(row=0, column=0, sticky="nsew", padx=8, pady=(8, 4))
        vitals_panel.grid_propagate(False)
        vitals_panel.pack_propagate(False)

        v_hdr = tk.Frame(vitals_panel, bg=BG_PANEL)
        v_hdr.pack(fill=tk.X)
        tk.Label(v_hdr, text="INTRA-OPERATIVE VITALS & IEC 60601-1-8 ALARMS", font=("Segoe UI", 11, "bold"), fg=ACCENT_CYAN, bg=BG_PANEL).pack(side=tk.LEFT)
        self.alarm_badge = tk.Label(v_hdr, text="[NORMAL]", font=("Segoe UI", 9, "bold"), fg=BG_DARK, bg=ACCENT_GREEN, padx=6, pady=2)
        self.alarm_badge.pack(side=tk.RIGHT)

        # 4 Vital Cards Grid
        cards_grid = tk.Frame(vitals_panel, bg=BG_PANEL)
        cards_grid.pack(fill=tk.X, pady=6)
        cards_grid.columnconfigure((0, 1, 2, 3), weight=1)

        # Card 1: Heart Rate
        c1 = tk.Frame(cards_grid, bg=BG_CARD, padx=6, pady=6)
        c1.grid(row=0, column=0, padx=3, sticky="nsew")
        tk.Label(c1, text="HEART RATE", font=("Segoe UI", 7, "bold"), fg=FG_MUTED, bg=BG_CARD).pack()
        self.hr_lbl = tk.Label(c1, text="74", font=("Segoe UI", 16, "bold"), fg=ACCENT_GREEN, bg=BG_CARD)
        self.hr_lbl.pack()
        tk.Label(c1, text="bpm", font=("Segoe UI", 8), fg=FG_MUTED, bg=BG_CARD).pack()

        # Card 2: SpO2
        c2 = tk.Frame(cards_grid, bg=BG_CARD, padx=6, pady=6)
        c2.grid(row=0, column=1, padx=3, sticky="nsew")
        tk.Label(c2, text="SpO2 SAT", font=("Segoe UI", 7, "bold"), fg=FG_MUTED, bg=BG_CARD).pack()
        self.spo2_lbl = tk.Label(c2, text="98 %", font=("Segoe UI", 16, "bold"), fg=ACCENT_CYAN, bg=BG_CARD)
        self.spo2_lbl.pack()
        self.spo2_pitch_lbl = tk.Label(c2, text="Pitch: 784 Hz", font=("Segoe UI", 8), fg=FG_MUTED, bg=BG_CARD)
        self.spo2_pitch_lbl.pack()

        # Card 3: Blood Pressure
        c3 = tk.Frame(cards_grid, bg=BG_CARD, padx=6, pady=6)
        c3.grid(row=0, column=2, padx=3, sticky="nsew")
        tk.Label(c3, text="NIBP (BP)", font=("Segoe UI", 7, "bold"), fg=FG_MUTED, bg=BG_CARD).pack()
        self.bp_lbl = tk.Label(c3, text="118/76", font=("Segoe UI", 16, "bold"), fg=FG_PRIMARY, bg=BG_CARD)
        self.bp_lbl.pack()
        tk.Label(c3, text="mmHg", font=("Segoe UI", 8), fg=FG_MUTED, bg=BG_CARD).pack()

        # Card 4: Temperature
        c4 = tk.Frame(cards_grid, bg=BG_CARD, padx=6, pady=6)
        c4.grid(row=0, column=3, padx=3, sticky="nsew")
        tk.Label(c4, text="TEMP", font=("Segoe UI", 7, "bold"), fg=FG_MUTED, bg=BG_CARD).pack()
        self.temp_lbl = tk.Label(c4, text="36.8", font=("Segoe UI", 16, "bold"), fg=FG_PRIMARY, bg=BG_CARD)
        self.temp_lbl.pack()
        tk.Label(c4, text="°C", font=("Segoe UI", 8), fg=FG_MUTED, bg=BG_CARD).pack()

        # Clinical Scenario Simulations (Demonstration Buttons)
        scen_frame = tk.Frame(vitals_panel, bg=BG_PANEL)
        scen_frame.pack(fill=tk.X, pady=4)
        tk.Label(scen_frame, text="Demo Scenarios:", font=("Segoe UI", 8, "bold"), fg=FG_MUTED, bg=BG_PANEL).pack(side=tk.LEFT, padx=4)

        tk.Button(scen_frame, text="Normal (98%)", font=("Segoe UI", 8), bg=BG_CARD, fg=ACCENT_GREEN, command=self._set_normal_scenario).pack(side=tk.LEFT, padx=2)
        tk.Button(scen_frame, text="Mild Hypoxia (91%)", font=("Segoe UI", 8), bg=BG_CARD, fg=ACCENT_YELLOW, command=self._set_mild_hypoxia_scenario).pack(side=tk.LEFT, padx=2)
        tk.Button(scen_frame, text="Critical Desat (84%)", font=("Segoe UI", 8), bg=BG_CARD, fg=ACCENT_RED, command=self._set_critical_desat_scenario).pack(side=tk.LEFT, padx=2)

        # Pulse Ox Sonification & Alarm Silence Row
        audio_ctrl = tk.Frame(vitals_panel, bg=BG_CARD, padx=8, pady=4)
        audio_ctrl.pack(fill=tk.X, pady=4)

        tk.Checkbutton(
            audio_ctrl,
            text="SpO2 Sonification Beep (Pitch = SpO2)",
            variable=self.pulse_ox_active,
            font=("Segoe UI", 8, "bold"),
            fg=ACCENT_CYAN,
            bg=BG_CARD,
            selectcolor=BG_INPUT,
            activebackground=BG_CARD,
            command=self._on_pulse_ox_toggle,
        ).pack(side=tk.LEFT)

        self.silence_btn = tk.Button(
            audio_ctrl,
            text="[MUTE] Silence Alarm 2 Min (A)",
            font=("Segoe UI", 8, "bold"),
            bg="#3b1b1b",
            fg=ACCENT_YELLOW,
            command=self.silence_alarm,
        )
        self.silence_btn.pack(side=tk.RIGHT, padx=4)

        # Test IEC Alarms Row
        test_row = tk.Frame(vitals_panel, bg=BG_PANEL)
        test_row.pack(fill=tk.X, pady=2)
        tk.Label(test_row, text="Test IEC Alarms:", font=("Segoe UI", 8), fg=FG_MUTED, bg=BG_PANEL).pack(side=tk.LEFT, padx=4)
        tk.Button(test_row, text="Low Priority", font=("Segoe UI", 8), bg=BG_CARD, fg=FG_PRIMARY, command=lambda: self.speech.play_sound(sequence(ALARMS["low"]), half_duplex=True)).pack(side=tk.LEFT, padx=2)
        tk.Button(test_row, text="Med Priority", font=("Segoe UI", 8), bg=BG_CARD, fg=ACCENT_YELLOW, command=lambda: self.speech.play_sound(sequence(ALARMS["medium"]), half_duplex=True)).pack(side=tk.LEFT, padx=2)
        tk.Button(test_row, text="High Priority", font=("Segoe UI", 8), bg=BG_CARD, fg=ACCENT_RED, command=lambda: self.speech.play_sound(sequence(ALARMS["high"]), half_duplex=True)).pack(side=tk.LEFT, padx=2)

        # BOTTOM HALF: Bilingual Voice Control, Meter & Intra-Operative Notes
        speech_panel = tk.Frame(panel, bg=BG_PANEL)
        speech_panel.grid(row=1, column=0, sticky="nsew", padx=8, pady=(4, 8))
        speech_panel.grid_propagate(False)
        speech_panel.pack_propagate(False)

        s_hdr = tk.Frame(speech_panel, bg=BG_PANEL)
        s_hdr.pack(fill=tk.X)
        tk.Label(s_hdr, text="BILINGUAL VOICE CONTROL & CLINICAL DICTATION", font=("Segoe UI", 11, "bold"), fg=ACCENT_CYAN, bg=BG_PANEL).pack(side=tk.LEFT)
        self.hd_badge = tk.Label(s_hdr, text="[LISTENING]", font=("Segoe UI", 9, "bold"), fg=BG_DARK, bg=ACCENT_GREEN, padx=6, pady=2)
        self.hd_badge.pack(side=tk.RIGHT)
        self.hd_badge.pack(side=tk.RIGHT)

        # Voice Mode Selector Radio Buttons
        mode_frame = tk.Frame(speech_panel, bg=BG_CARD, padx=6, pady=4)
        mode_frame.pack(fill=tk.X, pady=4)

        self.voice_mode_var = tk.StringVar(value="ID_ONLINE")
        tk.Label(mode_frame, text="ASR Mode:", font=("Segoe UI", 8, "bold"), fg=FG_MUTED, bg=BG_CARD).pack(side=tk.LEFT, padx=4)

        modes = [
            ("ID Online (Google)", "ID_ONLINE"),
            ("EN Offline (Vosk)", "EN_OFFLINE"),
            ("Dictation (ID)", "DICTATION_ID"),
        ]
        for text, m in modes:
            tk.Radiobutton(
                mode_frame,
                text=text,
                variable=self.voice_mode_var,
                value=m,
                font=("Segoe UI", 8),
                fg=FG_PRIMARY,
                bg=BG_CARD,
                selectcolor=BG_INPUT,
                activebackground=BG_CARD,
                command=self._on_voice_mode_change,
            ).pack(side=tk.LEFT, padx=4)

        # Mic Level Meter & Calibration Bar
        meter_row = tk.Frame(speech_panel, bg=BG_PANEL)
        meter_row.pack(fill=tk.X, pady=2)

        self.mic_toggle_btn = tk.Button(meter_row, text="Mic ON", font=("Segoe UI", 8, "bold"), bg=BG_CARD, fg=ACCENT_GREEN, command=self._on_toggle_mic)
        self.mic_toggle_btn.pack(side=tk.LEFT, padx=2)

        tk.Button(meter_row, text="Calibrate (2s)", font=("Segoe UI", 8), bg=BG_CARD, fg=FG_PRIMARY, command=self.speech.start_calibration).pack(side=tk.LEFT, padx=2)

        self.dbfs_lbl = tk.Label(meter_row, text="-60.0 dBFS", font=("Consolas", 8), fg=FG_MUTED, bg=BG_PANEL, width=10)
        self.dbfs_lbl.pack(side=tk.LEFT, padx=2)

        self.meter_bar = ttk.Progressbar(meter_row, style="Green.Horizontal.TProgressbar", length=140, mode="determinate")
        self.meter_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        self.thresh_lbl = tk.Label(meter_row, text="Thresh: -44 dB", font=("Consolas", 8), fg=ACCENT_YELLOW, bg=BG_PANEL)
        self.thresh_lbl.pack(side=tk.RIGHT, padx=2)

        # Voice Feedback Label
        self.heard_lbl = tk.Label(
            speech_panel,
            text='Heard: Ready for voice commands (e.g. "iris berikutnya", "bookmark", "tanda vital")',
            font=("Segoe UI", 8, "italic"),
            fg=ACCENT_CYAN,
            bg=BG_INPUT,
            anchor="w",
            padx=6,
            pady=3,
        )
        self.heard_lbl.pack(fill=tk.X, pady=2)

        # Intra-Operative Notes Box
        notes_hdr = tk.Frame(speech_panel, bg=BG_PANEL)
        notes_hdr.pack(fill=tk.X, pady=(4, 2))
        tk.Label(notes_hdr, text="Intra-Operative Clinical Log & Dictation:", font=("Segoe UI", 8, "bold"), fg=FG_MUTED, bg=BG_PANEL).pack(side=tk.LEFT)

        tk.Button(notes_hdr, text="Export (.txt)", font=("Segoe UI", 8), bg=BG_CARD, fg=ACCENT_TEAL, command=self.export_report).pack(side=tk.RIGHT, padx=2)
        tk.Button(notes_hdr, text="Speak Log Summary", font=("Segoe UI", 8), bg=BG_CARD, fg=ACCENT_CYAN, command=self.speak_notes_summary).pack(side=tk.RIGHT, padx=2)

        self.notes_text = tk.Text(
            speech_panel,
            height=7,
            bg=BG_INPUT,
            fg=FG_PRIMARY,
            insertbackground=ACCENT_CYAN,
            font=("Consolas", 8),
            wrap=tk.WORD,
            bd=0,
            padx=6,
            pady=4,
        )
        self.notes_text.pack(fill=tk.BOTH, expand=True)

        # Prepopulate Patient Header in Notes
        init_notes = (
            "=== INTRA-OPERATIVE SURGICAL LOG ===\n"
            f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            "Patient ID: OR-702 (Adult Neuro / Cranial CT)\n"
            "Procedure: Right Parietotemporal Lesion Resection\n"
            "Surgeon: Lead Neurosurgeon\n"
            "----------------------------------------\n"
            "[LOG] System initialized. Default slice: 38/60 (Brain Window).\n"
            "[LOG] Initial lesion marked on Slice 38: (x=0.66, y=0.39, +78 HU).\n"
        )
        self.notes_text.insert(tk.END, init_notes)
        self.notes_text.see(tk.END)

    # =========================================================================
    # KEYBOARD & REDUNDANT SHORTCUT BINDINGS
    # =========================================================================
    def _is_text_focused(self, event=None) -> bool:
        """Checks if keyboard focus is currently in a text editing widget."""
        focused = self.root.focus_get()
        candidates = [focused]
        if event is not None and getattr(event, "widget", None) is not None:
            candidates.append(event.widget)
        for w in candidates:
            if w is not None and isinstance(w, (tk.Text, tk.Entry, ttk.Entry)):
                return True
        return False

    def _guarded_action(self, action):
        """Creates an event handler that ignores single-key shortcuts when typing in a text widget."""
        def handler(event):
            if self._is_text_focused(event):
                return
            action()
        return handler

    def _bind_shortcuts(self):
        """Binds redundant keyboard controls for surgical efficiency."""
        self.root.bind("<Left>", self._guarded_action(self.prev_slice))
        self.root.bind("<Down>", self._guarded_action(self.prev_slice))
        self.root.bind("<p>", self._guarded_action(self.prev_slice))
        self.root.bind("<P>", self._guarded_action(self.prev_slice))

        self.root.bind("<Right>", self._guarded_action(self.next_slice))
        self.root.bind("<Up>", self._guarded_action(self.next_slice))
        self.root.bind("<n>", self._guarded_action(self.next_slice))
        self.root.bind("<N>", self._guarded_action(self.next_slice))

        self.root.bind("<w>", self._guarded_action(self.cycle_window))
        self.root.bind("<W>", self._guarded_action(self.cycle_window))

        self.root.bind("<b>", self._guarded_action(self.bookmark_current_slice))
        self.root.bind("<B>", self._guarded_action(self.bookmark_current_slice))

        self.root.bind("<m>", self._guarded_action(self.mark_current_lesion))
        self.root.bind("<M>", self._guarded_action(self.mark_current_lesion))

        self.root.bind("<c>", self._guarded_action(self.clear_markers_on_slice))
        self.root.bind("<C>", self._guarded_action(self.clear_markers_on_slice))

        self.root.bind("<a>", self._guarded_action(self.silence_alarm))
        self.root.bind("<A>", self._guarded_action(self.silence_alarm))

        self.root.bind("<v>", self._guarded_action(self.speak_vitals_summary))
        self.root.bind("<V>", self._guarded_action(self.speak_vitals_summary))

        self.root.bind("<space>", self._guarded_action(self.toggle_gesture_lock))

        # Return focus from notes editor back to root on Escape
        self.notes_text.bind("<Escape>", lambda e: self.root.focus_set())

    # =========================================================================
    # CT NAVIGATION & RENDERING LOGIC
    # =========================================================================
    def set_slice(self, slice_num: int):
        """Changes the active CT slice safely with slider feedback guard."""
        new_slice = int(np.clip(slice_num, 0, N_SLICES - 1))
        if new_slice != self.current_slice:
            self.current_slice = new_slice
            self._updating_slider = True
            try:
                self.slice_slider.set(self.current_slice)
            finally:
                self._updating_slider = False
            self._render_ct_display()

    def prev_slice(self):
        """Navigates to previous slice."""
        if self.current_slice > 0:
            self.set_slice(self.current_slice - 1)

    def next_slice(self):
        """Navigates to next slice."""
        if self.current_slice < N_SLICES - 1:
            self.set_slice(self.current_slice + 1)

    def _on_slider_change(self, val):
        """Slider scrubbing event handler guarded against feedback loops."""
        if getattr(self, "_updating_slider", False):
            return
        val_int = int(float(val))
        if val_int != self.current_slice:
            self.current_slice = val_int
            self._render_ct_display()

    def cycle_window(self):
        """Cycles through Brain, Bone, Subdural windows."""
        names = list(WINDOWS.keys())
        idx = (names.index(self.current_window) + 1) % len(names)
        self.current_window = names[idx]
        level, width = WINDOWS[self.current_window]
        self.window_badge.config(text=f"{self.current_window} (L:{level} W:{width})")
        self.speech.play_sound(sequence(EARCONS["ok"]))
        self._render_ct_display(force=True)
        self.log_message(f"Window changed: {self.current_window} (L:{level} W:{width})")

    def bookmark_current_slice(self):
        """Toggles bookmark on the current slice."""
        if self.current_slice in self.bookmarks:
            self.bookmarks.remove(self.current_slice)
            action = "removed"
            self.speech.play_sound(sequence(EARCONS["error"]))
        else:
            self.bookmarks.add(self.current_slice)
            action = "added"
            self.speech.play_sound(sequence(EARCONS["bookmark"]))
            if self.voice_mode_var.get() == "ID_ONLINE":
                self.speech.say(f"Iris {self.current_slice} disimpan.", lang="id")
            else:
                self.speech.say(f"Slice {self.current_slice} bookmarked.", lang="en")

        bm_str = ", ".join(str(s) for s in sorted(self.bookmarks)) if self.bookmarks else "None"
        self.bookmark_list_lbl.config(text=f"Bookmarks: [{bm_str}]")
        self.log_message(f"Bookmark {action}: Slice {self.current_slice}")
        self._render_ct_display(force=True)

    def mark_current_lesion(self, uv=None):
        """Places a surgical target marker at the specified or current cursor position."""
        target_uv = uv or self.cursor_uv
        u, v = target_uv
        hu = get_hu_at(self.ct_volume, self.current_slice, u, v)

        if self.current_slice not in self.markers:
            self.markers[self.current_slice] = []

        marker_num = len(self.markers[self.current_slice]) + 1
        label = f"Target-{marker_num}"
        self.markers[self.current_slice].append((u, v, label, hu))

        self.speech.play_sound(sequence(EARCONS["marker"]))
        if self.voice_mode_var.get() == "ID_ONLINE":
            self.speech.say(f"Target ditandai {hu:+.0f} Hounsfield.", lang="id")
        else:
            self.speech.say(f"Target marked at {hu:+.0f} Hounsfield.", lang="en")

        self.log_message(f"ROI Marker '{label}' placed on Slice {self.current_slice} at (u={u:.2f}, v={v:.2f}): {hu:+.1f} HU")
        self._render_ct_display(force=True)

    def clear_markers_on_slice(self):
        """Clears all markers on current slice."""
        if self.current_slice in self.markers and self.markers[self.current_slice]:
            count = len(self.markers[self.current_slice])
            self.markers[self.current_slice].clear()
            self.speech.play_sound(sequence(EARCONS["error"]))
            self.log_message(f"Cleared {count} markers on Slice {self.current_slice}")
            self._render_ct_display(force=True)

    def _on_ct_mouse_motion(self, event):
        """Updates cursor crosshair and live HU readout during mouse motion."""
        u = float(event.x) / CT_SIZE
        v = float(event.y) / CT_SIZE
        self.cursor_uv = (np.clip(u, 0.0, 1.0), np.clip(v, 0.0, 1.0))
        self._render_ct_display()

    def _on_ct_mouse_click(self, event):
        """Places marker on left-click on the CT image and returns focus to root."""
        self.root.focus_set()
        u = float(event.x) / CT_SIZE
        v = float(event.y) / CT_SIZE
        self.mark_current_lesion((np.clip(u, 0.0, 1.0), np.clip(v, 0.0, 1.0)))

    def _on_ct_mouse_wheel(self, event):
        """Scrolls slices on mouse wheel over CT."""
        if event.delta > 0:
            self.next_slice()
        elif event.delta < 0:
            self.prev_slice()

    def _render_ct_display(self, force: bool = False):
        """Renders the current 2D CT slice with annotations and displays on ct_label (cached)."""
        render_state = (
            self.current_slice,
            self.current_window,
            round(self.cursor_uv[0], 3),
            round(self.cursor_uv[1], 3),
            round(self.dwell_progress, 2),
            len(self.markers.get(self.current_slice, [])),
            self.current_slice in self.bookmarks,
        )
        if not force and self._last_render_state == render_state:
            return
        self._last_render_state = render_state

        level, width = WINDOWS[self.current_window]
        slice_hu = self.ct_volume[self.current_slice]
        slice_markers = self.markers.get(self.current_slice, [])
        is_bookmarked = self.current_slice in self.bookmarks

        rgb = to_display_rgb(
            slice_hu=slice_hu,
            level=level,
            width=width,
            cursor_uv=self.cursor_uv,
            markers=slice_markers,
            is_bookmarked=is_bookmarked,
            dwell_progress=self.dwell_progress,
        )

        # Convert to PIL and PhotoImage
        img = Image.fromarray(rgb)
        photo = ImageTk.PhotoImage(img)
        self.ct_label.configure(image=photo)
        self.ct_label.image = photo  # CRITICAL HCI RULE: Keep reference to prevent GC!

        # Update Readout Bar (cached to prevent redraw jitter)
        u, v = self.cursor_uv
        hu_val = get_hu_at(self.ct_volume, self.current_slice, u, v)
        text_status = f"Slice: {self.current_slice + 1:02d}/{N_SLICES} | Win: {self.current_window:<8} | Cursor: ({u:.2f}, {v:.2f}) -> {hu_val:+.1f} HU"
        self._set_label_cached("ct_status", self.ct_status_lbl, text=text_status)

    # =========================================================================
    # MIDDLE COLUMN: Camera & Gesture Tracking Loop
    # =========================================================================
    def toggle_gesture_lock(self):
        """Toggles sterile gesture lock."""
        self.gesture_locked = not self.gesture_locked
        if self.gesture_locked:
            self._set_label_cached("lock_badge", self.lock_badge, text="[PAUSED]", bg=ACCENT_YELLOW, fg=BG_DARK)
            self.lock_btn.config(text="Resume Gesture Lock (Space)", fg=ACCENT_GREEN)
            self._set_label_cached("gesture_mode", self.gesture_mode_lbl, text="[LOCKED] GESTURES PAUSED", fg=ACCENT_YELLOW)
            self.speech.say("Pelacakan gestur dijeda.", lang="id")
        else:
            self._set_label_cached("lock_badge", self.lock_badge, text="[ACTIVE]", bg=ACCENT_GREEN, fg=BG_DARK)
            self.lock_btn.config(text="Pause Gesture Lock (Space)", fg=ACCENT_YELLOW)
            self._set_label_cached("gesture_mode", self.gesture_mode_lbl, text="[IDLE] GESTURES ACTIVE", fg=ACCENT_CYAN)
            self.speech.say("Pelacakan gestur aktif.", lang="id")
        self.log_message(f"Sterile Gesture Lock: {'PAUSED' if self.gesture_locked else 'ACTIVE'}")

    def _camera_loop(self):
        """Main OpenCV camera & MediaPipe tracking loop executed every 30ms via root.after."""
        try:
            if self.cam_available and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    # Mirror frame horizontally for intuitive interaction
                    frame = cv2.flip(frame, 1)
                else:
                    frame = self._generate_simulated_frame()
            else:
                frame = self._generate_simulated_frame()

            # Process Face Tracker (Surgeon distance & sterile zone presence with debouncing)
            face_status = self._last_face_status
            face_locked = self._last_face_locked
            if self.face_tracker is not None:
                boxes = self.face_tracker.find(frame)
                if boxes:
                    self.missing_face_frames = 0
                    fx, fy, fw, fh = boxes[0]
                    cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 200), 2)
                    frame_w = frame.shape[1]
                    ratio = fw / float(frame_w)
                    if ratio > 0.55:
                        face_status = "Surgeon: [TOO CLOSE] RETRACT"
                        face_locked = False
                        cv2.putText(frame, "TOO CLOSE", (fx, fy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
                    elif ratio < 0.14:
                        face_status = "Surgeon: [AWAY / BG]"
                        face_locked = bool(self.face_autolock.get())
                        cv2.putText(frame, "AWAY", (fx, fy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)
                    else:
                        face_status = "Surgeon: [PRESENT] OPTIMAL"
                        face_locked = False
                        cv2.putText(frame, "PRESENT", (fx, fy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                else:
                    self.missing_face_frames += 1
                    if self.missing_face_frames > 15:
                        face_status = "Surgeon: [AWAY / NO FACE]"
                        face_locked = bool(self.face_autolock.get())

            self._last_face_status = face_status
            self._last_face_locked = face_locked

            face_fg = ACCENT_RED if face_locked else (ACCENT_GREEN if "OPTIMAL" in face_status else ACCENT_YELLOW)
            face_text = face_status + (" [AUTO-LOCKED]" if face_locked else "")
            self._set_label_cached("face_status", self.face_status_lbl, text=face_text, fg=face_fg)

            # Process Hand Tracker
            hand_pts = None
            if self.hand_tracker is not None and not self.gesture_locked and not face_locked:
                hand_pts = self.hand_tracker.find(frame)

            detected_gesture = None
            gesture_text = "[IDLE] NO HAND DETECTED"
            dwell_pct = 0.0

            if hand_pts:
                draw_hand(frame, hand_pts)

                # Check Half-Duplex rule: Ignore gesture commands while speaking
                if self.speech.is_speaking():
                    gesture_text = "[HALF-DUPLEX] MUTED (SPEAKING)"
                    self.gesture_debouncer.reset()
                    self.pinch_grab.active = False
                else:
                    # 1. Pinch to Scroll Slices with Hysteresis FIRST (on_below=0.26, off_above=0.45)
                    # Evaluating pinch first guarantees active clutching is never hijacked by 1-finger laser pointer
                    pr = pinch_ratio(hand_pts)
                    pinch_y = (hand_pts[4][1] + hand_pts[8][1]) / 2.0
                    newly_grabbed = self.pinch_grab.update(pr)

                    if newly_grabbed:
                        self.pinch_anchor_y = pinch_y
                        self.pinch_anchor_slice = self.current_slice
                        self.last_pinch_time = time.monotonic()
                        self.gesture_debouncer.reset()

                    if self.pinch_grab.active:
                        self.last_pinch_time = time.monotonic()
                        self.gesture_debouncer.reset()
                        gesture_text = "[PINCH-CLUTCH] SCROLLING"
                        self.laser_dwell_count = 0
                        self.dwell_progress = 0.0

                        # Proportional slice scrolling (moving hand up increases slice)
                        dy = self.pinch_anchor_y - pinch_y
                        target_slice = int(round(self.pinch_anchor_slice + dy / 14.0))
                        if target_slice != self.current_slice:
                            self.set_slice(target_slice)

                    # 2. Surgical Laser Pointer Check (1 Extended Index Finger, outside pinch & cooldown)
                    elif is_index_pointer(hand_pts) and (time.monotonic() - self.last_pinch_time >= 0.35):
                        self.last_laser_time = time.monotonic()
                        self.gesture_debouncer.reset()
                        gesture_text = "[POINTER] SURGICAL LASER"

                        ix, iy = hand_pts[8]
                        u_norm = np.clip((ix / frame.shape[1] - 0.12) / 0.76, 0.0, 1.0)
                        v_norm = np.clip((iy / frame.shape[0] - 0.12) / 0.76, 0.0, 1.0)
                        self.cursor_uv = (u_norm, v_norm)

                        if self.last_laser_uv is not None and dist(self.cursor_uv, self.last_laser_uv) < 0.045:
                            self.laser_dwell_count += 1
                        else:
                            self.laser_dwell_count = 1
                            self.last_laser_uv = self.cursor_uv

                        dwell_pct = min(1.0, self.laser_dwell_count / float(self.LASER_DWELL_TARGET))
                        self.dwell_progress = dwell_pct

                        if dwell_pct >= 1.0:
                            self.mark_current_lesion(self.cursor_uv)
                            self.laser_dwell_count = 0
                            self.dwell_progress = 0.0
                            self.last_laser_time = time.monotonic()

                        self._render_ct_display()

                    else:
                        # 3. Post-Pinch & Post-Laser Cooldown guard (0.45 s)
                        now_mono = time.monotonic()
                        is_cooldown = (now_mono - self.last_pinch_time < 0.45) or (now_mono - self.last_laser_time < 0.45)
                        if is_cooldown:
                            gesture_text = "[COOLDOWN] STABILIZING..."
                            self.gesture_debouncer.reset()
                            self.laser_dwell_count = 0
                            self.dwell_progress = 0.0
                        else:
                            self.laser_dwell_count = 0
                            self.dwell_progress = 0.0
                            n_fingers = count_fingers(hand_pts)
                            if n_fingers in (2, 3, 4, 5):
                                detected_gesture = n_fingers
                                names = {
                                    2: "[HOLD] 2 FINGERS (WINDOW)",
                                    3: "[HOLD] 3 FINGERS (BOOKMARK)",
                                    4: "[HOLD] 4 FINGERS (VITALS)",
                                    5: "[HOLD] 5 FINGERS (LOCK)",
                                }
                                gesture_text = names.get(n_fingers, "")

            else:
                self.pinch_grab.active = False
                self.laser_dwell_count = 0
                self.dwell_progress = 0.0

            # Update StableGesture Debouncer
            triggered = self.gesture_debouncer.update(detected_gesture)
            if detected_gesture is not None and dwell_pct == 0.0:
                dwell_pct = self.gesture_debouncer.progress

            # Execute Stable Gesture Event
            if triggered == 2:
                self.cycle_window()
            elif triggered == 3:
                self.bookmark_current_slice()
            elif triggered == 4:
                self.speak_vitals_summary()
            elif triggered == 5:
                self.toggle_gesture_lock()

            # Update UI Progress Bar & Gesture Label (Cached)
            dwell_int = int(dwell_pct * 100)
            if self._ui_cache.get("dwell_val") != dwell_int:
                self._ui_cache["dwell_val"] = dwell_int
                self.dwell_bar["value"] = dwell_int
                self._set_label_cached("dwell_pct", self.dwell_pct_lbl, text=f"{dwell_int}%")
            self._set_label_cached("gesture_mode", self.gesture_mode_lbl, text=gesture_text)

            # Render camera preview to 320x240
            preview = cv2.resize(frame, (320, 240))
            rgb_cam = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
            cam_photo = ImageTk.PhotoImage(Image.fromarray(rgb_cam))
            self.cam_label.configure(image=cam_photo)
            self.cam_label.image = cam_photo  # CRITICAL: Keep reference!

        except Exception as e:
            print(f"[OR-SterileAssist] Camera loop exception: {e}")

        # Schedule next frame in 30ms (never use while True)
        if self.is_running and self.root.winfo_exists():
            self.root.after(30, self._camera_loop)

    def _generate_simulated_frame(self):
        """Generates a synthetic camera preview frame if physical webcam is busy or absent."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:] = (20, 24, 32)
        cv2.putText(img, "SIMULATED SURGICAL SENSOR", (160, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 229, 255), 2)
        cv2.putText(img, "(No physical camera found - mouse/keys active)", (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        # Moving indicator dot
        t = time.monotonic()
        cx = int(320 + 140 * np.sin(t * 1.5))
        cy = int(340 + 40 * np.cos(t * 2.0))
        cv2.circle(img, (cx, cy), 14, (0, 255, 128), -1)
        cv2.putText(img, "Simulated Hand Target", (cx + 18, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 128), 1)
        return img

    # =========================================================================
    # SPEECH ENGINE POLLING & VOICE COMMAND DISPATCH
    # =========================================================================
    def _speech_poll_loop(self):
        """Polls events from the SpeechEngine queue and updates GUI smoothly."""
        try:
            while not self.speech.events_queue.empty():
                event = self.speech.events_queue.get_nowait()
                etype = event.get("type")

                if etype == "meter":
                    # Update dBFS level meter
                    level = event.get("dbfs", -60.0)
                    norm_val = int(np.clip((level + 60.0) / 60.0, 0.0, 1.0) * 100.0)
                    if self._ui_cache.get("meter_val") != norm_val:
                        self._ui_cache["meter_val"] = norm_val
                        self.meter_bar["value"] = norm_val
                    self._set_label_cached("dbfs_lbl", self.dbfs_lbl, text=f"{level:.1f} dBFS")

                    # Half-Duplex Status Badge
                    if event.get("speaking"):
                        self._set_label_cached("hd_badge", self.hd_badge, text="[SPEAKING]", bg=ACCENT_YELLOW, fg=BG_DARK)
                    elif not event.get("mic_enabled"):
                        self._set_label_cached("hd_badge", self.hd_badge, text="[MIC OFF]", bg=ACCENT_RED, fg=BG_DARK)
                    else:
                        self._set_label_cached("hd_badge", self.hd_badge, text="[LISTENING]", bg=ACCENT_GREEN, fg=BG_DARK)

                elif etype == "calibration_done":
                    thresh = event.get("threshold_db", -44.0)
                    self.thresh_lbl.config(text=f"Thresh: {thresh:.1f} dB")
                    self.heard_lbl.config(text=event.get("msg", "Calibration done."))
                    self.speech.play_sound(sequence(EARCONS["berhasil"]))

                elif etype == "command":
                    cmd_id = event.get("cmd")
                    matched_text = event.get("text", "")
                    conf = event.get("conf", 0.9)
                    reply = event.get("reply", "")

                    self.heard_lbl.config(text=f'Heard: "{matched_text}" (Conf: {conf:.2f}) -> {cmd_id}')
                    self.log_message(f'Voice Command executed: "{matched_text}" -> {cmd_id}')
                    self._execute_voice_command(cmd_id, reply)

                elif etype == "dictation":
                    text = event.get("text", "")
                    self.heard_lbl.config(text=f'Dictated: "{text}"')
                    self.log_dictation(text)
                    self.speech.play_sound(sequence(EARCONS["ok"]))

                elif etype == "unrecognized":
                    text = event.get("text", "")
                    self.heard_lbl.config(text=f'Unrecognized: "{text}"')

                elif etype == "status":
                    self.heard_lbl.config(text=event.get("msg", ""))

        except Exception as e:
            print(f"[OR-SterileAssist] Speech poll error: {e}")

        # Poll again in 50ms (never block mainloop)
        if self.is_running and self.root.winfo_exists():
            self.root.after(50, self._speech_poll_loop)

    def _execute_voice_command(self, cmd_id: str, spoken_reply: str):
        """Dispatches verified speech commands."""
        lang = "id" if self.voice_mode_var.get() == "ID_ONLINE" else "en"

        if cmd_id == "NEXT_SLICE":
            self.next_slice()
            self.speech.say(spoken_reply or "Iris berikutnya.", lang=lang)

        elif cmd_id == "PREV_SLICE":
            self.prev_slice()
            self.speech.say(spoken_reply or "Iris sebelumnya.", lang=lang)

        elif cmd_id == "WINDOW_BRAIN":
            self.current_window = "Brain"
            level, width = WINDOWS["Brain"]
            self.window_badge.config(text=f"Brain (L:{level} W:{width})")
            self._render_ct_display()
            self.speech.say(spoken_reply or "Jendela otak.", lang=lang)

        elif cmd_id == "WINDOW_BONE":
            self.current_window = "Bone"
            level, width = WINDOWS["Bone"]
            self.window_badge.config(text=f"Bone (L:{level} W:{width})")
            self._render_ct_display()
            self.speech.say(spoken_reply or "Jendela tulang.", lang=lang)

        elif cmd_id == "WINDOW_SUBDURAL":
            self.current_window = "Subdural"
            level, width = WINDOWS["Subdural"]
            self.window_badge.config(text=f"Subdural (L:{level} W:{width})")
            self._render_ct_display()
            self.speech.say(spoken_reply or "Jendela subdural.", lang=lang)

        elif cmd_id == "BOOKMARK_SLICE":
            self.bookmark_current_slice()

        elif cmd_id == "MARK_LESION":
            self.mark_current_lesion()

        elif cmd_id == "CLEAR_MARKERS":
            self.clear_markers_on_slice()
            self.speech.say(spoken_reply or "Tanda dihapus.", lang=lang)

        elif cmd_id == "PATIENT_VITALS":
            self.speak_vitals_summary()

        elif cmd_id == "HEART_RATE":
            hr_txt = f"Detak jantung pasien {self.vitals['hr']} kali per menit." if lang == "id" else f"Patient heart rate is {self.vitals['hr']} beats per minute."
            self.speech.say(hr_txt, lang=lang)

        elif cmd_id == "OXYGEN":
            spo2_txt = f"Saturasi oksigen {self.vitals['spo2']} persen." if lang == "id" else f"Oxygen saturation is {self.vitals['spo2']} percent."
            self.speech.say(spo2_txt, lang=lang)

        elif cmd_id == "BLOOD_PRESSURE":
            bp_txt = f"Tekanan darah {self.vitals['bp_sys']} per {self.vitals['bp_dia']} milimeter merkuri." if lang == "id" else f"Blood pressure is {self.vitals['bp_sys']} over {self.vitals['bp_dia']} millimeters of mercury."
            self.speech.say(bp_txt, lang=lang)

        elif cmd_id == "SILENCE_ALARM":
            self.silence_alarm()
            silence_txt = "Alarm diheningkan dua menit." if lang == "id" else "Alarm silenced for two minutes."
            self.speech.say(silence_txt, lang=lang)

        elif cmd_id == "PAUSE_CAMERA":
            if not self.gesture_locked:
                self.toggle_gesture_lock()

        elif cmd_id == "RESUME_CAMERA":
            if self.gesture_locked:
                self.toggle_gesture_lock()

        elif cmd_id == "HELP":
            self.speech.say(spoken_reply, lang=lang)

    def _on_voice_mode_change(self):
        """Handles switching speech recognition mode."""
        mode = self.voice_mode_var.get()
        self.speech.set_mode(mode)
        self.log_message(f"ASR Mode changed to: {mode}")

    def _on_toggle_mic(self):
        """Toggles microphone listening."""
        is_on = self.speech.toggle_mic()
        self.mic_toggle_btn.config(
            text="Mic ON" if is_on else "Mic OFF",
            fg=ACCENT_GREEN if is_on else ACCENT_RED,
        )

    # =========================================================================
    # VITALS, ALARMS & PULSE OX SONIFICATION
    # =========================================================================
    def _set_normal_scenario(self):
        """Sets normal baseline vitals."""
        self.vitals.update({
            "hr": 74,
            "spo2": 98,
            "bp_sys": 118,
            "bp_dia": 76,
            "temp": 36.8,
            "alarm_severity": "none",
        })
        self._update_vitals_ui()
        self.log_message("Clinical scenario: Normal Baseline (SpO2 98%)")

    def _set_mild_hypoxia_scenario(self):
        """Sets mild hypoxia with IEC Medium alarm."""
        self.vitals.update({
            "hr": 94,
            "spo2": 91,
            "bp_sys": 136,
            "bp_dia": 88,
            "temp": 37.1,
            "alarm_severity": "medium",
        })
        self._update_vitals_ui()
        self.log_message("Clinical scenario: Mild Hypoxia (SpO2 91% - IEC Medium Alarm)")

    def _set_critical_desat_scenario(self):
        """Sets critical desaturation with IEC High alarm."""
        self.vitals.update({
            "hr": 122,
            "spo2": 84,
            "bp_sys": 158,
            "bp_dia": 102,
            "temp": 37.5,
            "alarm_severity": "high",
        })
        self._update_vitals_ui()
        self.log_message("Clinical scenario: CRITICAL DESATURATION (SpO2 84% - IEC High Alarm)")

    def _update_vitals_ui(self):
        """Refreshes vital card values and alarm status."""
        self.hr_lbl.config(text=str(self.vitals["hr"]))
        self.spo2_lbl.config(text=f"{self.vitals['spo2']} %")
        pitch = spo2_to_pitch(self.vitals["spo2"])
        self.spo2_pitch_lbl.config(text=f"Pitch: {pitch:.0f} Hz")
        self.bp_lbl.config(text=f"{self.vitals['bp_sys']}/{self.vitals['bp_dia']}")
        self.temp_lbl.config(text=f"{self.vitals['temp']:.1f}")

        sev = self.vitals["alarm_severity"]
        if sev == "none":
            self._set_label_cached("alarm_badge", self.alarm_badge, text="[NORMAL]", bg=ACCENT_GREEN, fg=BG_DARK)
        elif sev == "low":
            self._set_label_cached("alarm_badge", self.alarm_badge, text="[LOW ALARM]", bg=ACCENT_YELLOW, fg=BG_DARK)
        elif sev == "medium":
            self._set_label_cached("alarm_badge", self.alarm_badge, text="[MED ALARM]", bg=ACCENT_ORANGE, fg=BG_DARK)
        elif sev == "high":
            self._set_label_cached("alarm_badge", self.alarm_badge, text="[CRITICAL ALARM]", bg=ACCENT_RED, fg=BG_DARK)

    def silence_alarm(self):
        """Silences alarms for 2 minutes (IEC 60601-1-8 standard)."""
        self.alarm_silenced_until = time.monotonic() + 120.0  # 120 seconds
        self.speech.play_sound(sequence(EARCONS["ok"]))
        self.log_message("IEC 60601-1-8 Alarm silenced for 120 seconds.")

    def _vitals_monitor_loop(self):
        """Periodic 1-second vital signs monitor & alarm trigger loop."""
        now = time.monotonic()
        silenced = now < self.alarm_silenced_until

        if silenced:
            remaining = int(self.alarm_silenced_until - now)
            self._set_label_cached("silence_btn", self.silence_btn, text=f"[MUTE] Silenced ({remaining}s)", fg=ACCENT_YELLOW)
        else:
            self._set_label_cached("silence_btn", self.silence_btn, text="[MUTE] Silence Alarm 2 Min (A)", fg=ACCENT_YELLOW)

        # Trigger alarm audio bursts if active and not silenced and not speaking
        sev = self.vitals["alarm_severity"]
        if sev in ("medium", "high") and not silenced and not self.speech.is_speaking():
            pattern = ALARMS.get(sev)
            if pattern is not None:
                self.speech.play_sound(sequence(pattern, volume=0.22), half_duplex=True)

        # Schedule next check in 1 second
        if self.is_running and self.root.winfo_exists():
            self.root.after(1000, self._vitals_monitor_loop)

    def _on_pulse_ox_toggle(self):
        """Handles Pulse Oximeter Sonification toggle."""
        if self.pulse_ox_active.get():
            self.log_message("SpO2 Pulse Sonification ENABLED.")
            self._schedule_pulse_ox_beep()
        else:
            self.log_message("SpO2 Pulse Sonification DISABLED.")
            if self.pulse_ox_job is not None:
                self.root.after_cancel(self.pulse_ox_job)
                self.pulse_ox_job = None

    def _schedule_pulse_ox_beep(self):
        """Plays a rhythmic pulse oximeter chirp at the interval of the current heart rate."""
        if not self.pulse_ox_active.get():
            return

        hr = max(30, self.vitals["hr"])
        interval_ms = int(60000.0 / hr)

        # Play short chirp if not currently speaking TTS or alarm
        if not self.speech.is_speaking():
            chirp = pulse_ox_tone(self.vitals["spo2"], duration=0.065, volume=0.20)
            try:
                sd.play(chirp, SAMPLE_RATE)
            except Exception:
                pass

        # Schedule next pulse
        self.pulse_ox_job = self.root.after(interval_ms, self._schedule_pulse_ox_beep)

    def speak_vitals_summary(self):
        """Synthesizes and speaks a complete patient vitals summary."""
        lang = "id" if self.voice_mode_var.get() == "ID_ONLINE" else "en"
        if lang == "id":
            text = (
                f"Laporan tanda vital: Detak jantung {self.vitals['hr']} kali per menit, "
                f"saturasi oksigen {self.vitals['spo2']} persen, "
                f"tekanan darah {self.vitals['bp_sys']} per {self.vitals['bp_dia']}, "
                f"suhu {self.vitals['temp']} derajat Celsius. Status {self.vitals['alarm_severity']}."
            )
        else:
            text = (
                f"Patient vitals summary: Heart rate {self.vitals['hr']} beats per minute, "
                f"oxygen saturation {self.vitals['spo2']} percent, "
                f"blood pressure {self.vitals['bp_sys']} over {self.vitals['bp_dia']}, "
                f"temperature {self.vitals['temp']} degrees Celsius."
            )
        self.speech.say(text, sound=sequence(EARCONS["attention"]), lang=lang)
        self.log_message("Spoke Patient Vitals Summary.")

    # =========================================================================
    # SURGICAL LOGGING & REPORT EXPORT
    # =========================================================================
    def log_message(self, msg: str):
        """Appends a timestamped log entry to the notes text widget."""
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        entry = f"[{ts}] {msg}\n"
        self.notes_text.insert(tk.END, entry)
        self.notes_text.see(tk.END)

    def log_dictation(self, text: str):
        """Appends transcribed surgeon dictation to notes."""
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        entry = f"[{ts} DICTATION] \"{text}\"\n"
        self.notes_text.insert(tk.END, entry)
        self.notes_text.see(tk.END)

    def speak_notes_summary(self):
        """Speaks a concise summary of bookmarks and marked targets."""
        lang = "id" if self.voice_mode_var.get() == "ID_ONLINE" else "en"
        n_bm = len(self.bookmarks)
        total_markers = sum(len(m) for m in self.markers.values())
        if lang == "id":
            summary = f"Ringkasan operasi: {n_bm} iris disimpan dalam bookmark, dan {total_markers} target lesi telah ditandai."
        else:
            summary = f"Surgical summary: {n_bm} slices bookmarked, and {total_markers} lesion targets marked."
        self.speech.say(summary, lang=lang)

    def export_report(self):
        """Exports the intra-operative surgical notes to a formatted text file."""
        content = self.notes_text.get("1.0", tk.END).strip()
        default_filename = f"OR_Surgical_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            initialfile=default_filename,
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if file_path:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                    f.write("\n\n=== END OF SURGICAL REPORT ===\n")
                messagebox.showinfo("Export Success", f"Surgical report saved to:\n{file_path}")
                self.speech.play_sound(sequence(EARCONS["berhasil"]))
                self.log_message(f"Report exported: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Export Failed", f"Could not save file: {e}")

    def on_closing(self):
        """Gracefully releases audio and camera resources upon window close."""
        print("[OR-SterileAssist] Shutting down application...")
        self.is_running = False
        if self.pulse_ox_job is not None:
            try:
                self.root.after_cancel(self.pulse_ox_job)
            except Exception:
                pass
            self.pulse_ox_job = None
        if hasattr(self, "speech") and self.speech:
            self.speech.stop()
        if hasattr(self, "cap") and self.cap and self.cap.isOpened():
            try:
                self.cap.release()
            except Exception:
                pass
        try:
            self.root.destroy()
        except Exception:
            pass


def main():
    """Application entry point."""
    root = tk.Tk()
    app = ORSterileAssistApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
