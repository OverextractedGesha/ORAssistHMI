"""or_sterile_assist_v2.py - Medical-Grade PySide6 Intra-Operative Workstation.

Project ETS Human-Machine Interface (HMI) Biomedical Workstation v2.0.
Implements the 'OR Obsidian' dark-room surgical UI with:
- Real Philips Cranial CT (.npz / .nii / .nii.gz) & Synthetic Phantom switching
- PACS-grade vector DICOM viewport with calibrated millimeter scale bar & slice rail
- Touchless MediaPipe hand/face tracking (pinch-clutch scrolling, 1-finger dwell laser,
  2-5 finger debounced commands, 15-frame missing-face sterile auto-lock)
- 60 FPS sweeping ECG II & PLETH SpO2 waveforms with IEC 60601-1-8 visual/audio alarms
- SpO2 pulse oximeter pitch sonification synchronized to heart rate
- Bilingual half-duplex ASR/TTS voice commands & clinical dictation with .txt export
"""
from __future__ import annotations

import datetime
import os
import queue
import sys
import time

# Safe sys.path resolution whether launched from repository root or src/
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# Safe stdout/stderr redirection for pythonw.exe or frozen PyInstaller GUI builds
if sys.stdout is None or sys.stderr is None:
    if getattr(sys, "frozen", False):
        _base_dir = os.path.dirname(sys.executable)
        _log_path = os.path.join(_base_dir, "or_sterile_assist_v2.log")
    else:
        _log_dir = os.path.abspath(os.path.join(_SRC_DIR, "..", "logs"))
        os.makedirs(_log_dir, exist_ok=True)
        _log_path = os.path.join(_log_dir, "or_sterile_assist_v2.log")
    try:
        _log_file = open(_log_path, "a", encoding="utf-8", buffering=1)
        if sys.stdout is None:
            sys.stdout = _log_file
        if sys.stderr is None:
            sys.stderr = _log_file
    except Exception:
        pass

import cv2
import numpy as np
import sounddevice as sd
from PySide6.QtCore import QEvent, QObject, Qt, QTimer
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from audio_utils import ALARMS, EARCONS, SAMPLE_RATE, pulse_ox_tone, sequence
from ct_phantom import (
    N_SLICES,
    SIZE as CT_SIZE,
    WINDOWS,
    _parse_lesion_span,
    _parse_lesion_uv,
    get_hu_at,
    load_nifti_ct,
    load_real_ct,
    make_phantom,
    window_slice_grayscale,
)
from gesture_utils import (
    Hysteresis,
    StableGesture,
    count_fingers,
    dist,
    is_index_pointer,
    pinch_ratio,
)
from speech_engine import SpeechEngine
from trackers import FaceTracker, HandTracker, draw_hand
from ui_v2 import (
    APP_QSS,
    COLOR_AMBER,
    COLOR_CYAN,
    COLOR_DANGER,
    COLOR_NOMINAL,
    CTPanel,
    SpeechConsolePanel,
    TEXT_MUTED,
    TEXT_PRIMARY,
    VisionHudPanel,
    VitalsMonitorPanel,
    repolish,
    tabular_font,
    ui_font,
)


class ORSterileAssistV2Window(QMainWindow):
    """Medical-grade PySide6 workstation window for OR-SterileAssist v2."""

    LASER_DWELL_TARGET = 42  # ~1.4 seconds at 30 FPS

    def __init__(self, headless: bool = False, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.headless = bool(headless)
        self.is_running = True

        self.setWindowTitle("OR-SterileAssist v2 — Medical Workstation (ETS HMI)")
        self.setMinimumSize(1280, 800)
        self.resize(1380, 860)
        self.setStyleSheet(
            APP_QSS
            + "\nQFrame#Panel QPushButton { padding: 4px 5px; }\n"
            + "QFrame#Panel QCheckBox { font-size: 8pt; spacing: 4px; }\n"
        )
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Core CT & Annotation State
        self.current_window = "Brain"
        self.cursor_uv: tuple[float, float] = (0.5, 0.5)
        self.dwell_progress = 0.0
        self.bookmarks: set[int] = set()
        self.markers: dict[int, list[tuple[float, float, str, float]]] = {}
        self.lesion_span: tuple[int, int] | None = None
        self._last_render_key: tuple | None = None

        # Laser Pointer Dwell State
        self.last_laser_uv: tuple[float, float] | None = None
        self.laser_dwell_count = 0
        self.last_laser_time = 0.0

        # Pinch Clutching Hysteresis (on_below=0.26, off_above=0.45)
        self.pinch_grab = Hysteresis(on_below=0.26, off_above=0.45)
        self.pinch_anchor_y = 0.0
        self.pinch_anchor_slice = 0
        self.last_pinch_time = 0.0

        # Face Presence Debouncer (15 frames)
        self.missing_face_frames = 0
        self._last_face_status = "[CHECKING]"
        self._last_face_locked = False

        # Gesture Debouncer (15 frames = ~0.5s at 30 FPS)
        self.gesture_debouncer = StableGesture(frames=15)
        self.gesture_locked = False
        self.face_autolock_enabled = True

        # Speech / Audio State
        self.voice_mode = "ID_ONLINE"

        # Patient Vitals & IEC Alarm State
        self.vitals: dict = {
            "hr": 74,
            "spo2": 98,
            "bp_sys": 118,
            "bp_dia": 76,
            "temp": 36.8,
            "alarm_severity": "none",
        }
        self.alarm_silenced_until = 0.0
        self.pulse_ox_active = False
        self._last_wave_time = time.monotonic()

        # Load Default Clinical CT Volume
        print("[OR-SterileAssist v2] Loading default cranial CT volume...")
        self.ct_volume, self.ct_meta = load_real_ct()

        # Initialize Subsystems
        self.speech = SpeechEngine()
        self.hand_tracker: HandTracker | None = None
        self.face_tracker: FaceTracker | None = None
        self._init_trackers()

        self.cap = None
        self.cam_available = False
        if not self.headless:
            self._init_camera()

        # Build UI & Connect Signals
        self._build_ui()
        self._connect_signals()

        # Apply loaded CT volume to UI & initialize surgical log
        self._apply_loaded_volume(self.ct_volume, self.ct_meta, log_change=False)
        self._init_surgical_log()
        self._set_scenario("normal", log_change=False)

        # Install application-level key filter for guarded surgical shortcuts
        self._key_filter_installed = False
        qapp = QApplication.instance()
        if qapp is not None:
            qapp.installEventFilter(self)
            self._key_filter_installed = True

        # Start Non-Blocking Qt Timers
        self.cam_timer = QTimer(self)
        self.cam_timer.timeout.connect(self._camera_loop)
        self.cam_timer.start(33)

        self.wave_timer = QTimer(self)
        self.wave_timer.timeout.connect(self._waveform_loop)
        self.wave_timer.start(16)

        self.speech_timer = QTimer(self)
        self.speech_timer.timeout.connect(self._speech_poll_loop)
        self.speech_timer.start(50)

        self.vitals_timer = QTimer(self)
        self.vitals_timer.timeout.connect(self._vitals_monitor_loop)
        self.vitals_timer.start(1000)

        self.pulse_ox_timer = QTimer(self)
        self.pulse_ox_timer.setSingleShot(True)
        self.pulse_ox_timer.timeout.connect(self._on_pulse_ox_tick)

        if not self.headless:
            self._play_sound(sequence(EARCONS["berhasil"]))

    # -------------------------------------------------------------------------
    # Audio / TTS Helpers (Safe in Headless Mode)
    # -------------------------------------------------------------------------
    def _play_sound(self, sound_array: np.ndarray, half_duplex: bool = False) -> None:
        if not self.headless and self.speech is not None:
            self.speech.play_sound(sound_array, half_duplex=half_duplex)

    def _say(self, text: str, sound=None, lang: str = "id", rate: int = 160) -> None:
        if not self.headless and self.speech is not None:
            self.speech.say(text, sound=sound, lang=lang, rate=rate)

    # -------------------------------------------------------------------------
    # Initialization Helpers
    # -------------------------------------------------------------------------
    def _init_trackers(self) -> None:
        try:
            self.hand_tracker = HandTracker(alpha=0.5)
            self.face_tracker = FaceTracker()
            print("[OR-SterileAssist v2] MediaPipe trackers initialized.")
        except Exception as e:
            print(f"[OR-SterileAssist v2] Tracker initialization warning: {e}")

    def _init_camera(self) -> None:
        try:
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if cap is not None and not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(0)
            self.cap = cap
            self.cam_available = bool(cap is not None and cap.isOpened())
            if self.cam_available:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        except Exception as e:
            print(f"[OR-SterileAssist v2] Camera initialization warning: {e}")
            self.cap = None
            self.cam_available = False

    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)

        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(8, 6, 8, 8)
        root_layout.setSpacing(6)

        # Slim Top Surgical Header Bar
        header = QFrame()
        header.setObjectName("Panel")
        header.setFixedHeight(38)
        hdr_lay = QHBoxLayout(header)
        hdr_lay.setContentsMargins(12, 4, 12, 4)
        hdr_lay.setSpacing(10)

        self.header_title_lbl = QLabel("OR-STERILEASSIST v2 | CRANIAL NAVIGATION WORKSTATION")
        self.header_title_lbl.setFont(ui_font(10, bold=True))
        self.header_title_lbl.setStyleSheet(f"color: {COLOR_CYAN}; letter-spacing: 0.8px;")
        hdr_lay.addWidget(self.header_title_lbl)

        self.proc_badge = QLabel("OR-702")
        self.proc_badge.setFont(tabular_font(8, bold=True))
        self.proc_badge.setStyleSheet(
            f"color: {TEXT_PRIMARY}; background-color: rgba(0, 229, 255, 0.10); "
            f"border: 1px solid {COLOR_CYAN}; border-radius: 3px; padding: 2px 7px;"
        )
        hdr_lay.addWidget(self.proc_badge)

        self.header_source_badge = QLabel("REAL CRANIAL CT")
        self.header_source_badge.setFont(tabular_font(8, bold=True))
        hdr_lay.addWidget(self.header_source_badge)

        hdr_lay.addStretch(1)

        self.clock_lbl = QLabel(datetime.datetime.now().strftime("%H:%M:%S"))
        self.clock_lbl.setFont(tabular_font(9, bold=True))
        self.clock_lbl.setStyleSheet(f"color: {TEXT_PRIMARY};")
        hdr_lay.addWidget(self.clock_lbl)

        self.version_badge = QLabel("v2.0")
        self.version_badge.setFont(tabular_font(8, bold=True))
        self.version_badge.setStyleSheet(
            f"color: {COLOR_NOMINAL}; background-color: rgba(0, 230, 118, 0.12); "
            f"border: 1px solid {COLOR_NOMINAL}; border-radius: 3px; padding: 2px 7px;"
        )
        hdr_lay.addWidget(self.version_badge)

        root_layout.addWidget(header)

        # 3-Column Clinical Workspace (4 : 3 : 5)
        body_row = QHBoxLayout()
        body_row.setSpacing(8)

        self.ct_panel = CTPanel()
        self.vision_panel = VisionHudPanel()

        right_col = QVBoxLayout()
        right_col.setSpacing(8)
        self.vitals_panel = VitalsMonitorPanel()
        self.speech_panel = SpeechConsolePanel()
        self.speech_panel.hd_badge.setMinimumWidth(108)
        self.speech_panel.hd_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.notes_edit = self.speech_panel.log_edit

        right_col.addWidget(self.vitals_panel, stretch=4)
        right_col.addWidget(self.speech_panel, stretch=5)

        body_row.addWidget(self.ct_panel, stretch=4)
        body_row.addWidget(self.vision_panel, stretch=3)
        body_row.addLayout(right_col, stretch=5)

        root_layout.addLayout(body_row, stretch=1)

    def _connect_signals(self) -> None:
        # Left Column: CT Navigation
        self.ct_panel.load_nifti_requested.connect(self.prompt_load_nifti)
        self.ct_panel.switch_source_requested.connect(self.switch_ct_source)
        self.ct_panel.window_selected.connect(self.set_window)
        self.ct_panel.prev_slice_requested.connect(self.prev_slice)
        self.ct_panel.next_slice_requested.connect(self.next_slice)
        self.ct_panel.bookmark_toggled.connect(self.bookmark_current_slice)
        self.ct_panel.mark_roi_requested.connect(lambda: self.mark_current_lesion())
        self.ct_panel.clear_roi_requested.connect(self.clear_markers_on_slice)
        self.ct_panel.default_slice_requested.connect(self.jump_to_default_slice)
        self.ct_panel.slice_changed.connect(self.set_slice)

        self.ct_panel.viewport.slice_wheeled.connect(self._on_viewport_wheel)
        self.ct_panel.viewport.cursor_moved.connect(self._on_viewport_cursor_moved)
        self.ct_panel.viewport.clicked_uv.connect(self._on_viewport_clicked)

        # Middle Column: Vision & Sterile Lock
        self.vision_panel.toggle_lock_requested.connect(self.toggle_gesture_lock)
        self.vision_panel.face_autolock_toggled.connect(self._on_face_autolock_toggled)

        # Right Top: Vitals & IEC Alarms
        self.vitals_panel.silence_requested.connect(self.silence_alarm)
        self.vitals_panel.scenario_requested.connect(self._set_scenario)
        self.vitals_panel.sonification_toggled.connect(self._on_pulse_ox_toggle)
        self.vitals_panel.alarm_test_requested.connect(self._on_alarm_test)
        self.vitals_panel.speak_vitals_requested.connect(self.speak_vitals_summary)

        # Right Bottom: Speech & Surgical Log
        self.speech_panel.asr_mode_changed.connect(self._on_voice_mode_change)
        self.speech_panel.mic_toggled.connect(self._on_toggle_mic)
        self.speech_panel.calibrate_requested.connect(self.speech.start_calibration)
        self.speech_panel.speak_log_requested.connect(self.speak_notes_summary)
        self.speech_panel.export_log_requested.connect(self.export_report)

    def _init_surgical_log(self) -> None:
        patient_id = self.ct_meta.get("patient_id", "OR-702")
        series_desc = self.ct_meta.get("series_desc", "CRANIAL CT")
        init_notes = (
            "=== INTRA-OPERATIVE SURGICAL LOG ===\n"
            f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Patient ID: {patient_id}\n"
            f"Series: {series_desc}\n"
            "Procedure: Cranial Navigation & Lesion Resection\n"
            "Surgeon: Lead Neurosurgeon\n"
            "----------------------------------------\n"
            f"[LOG] System initialized (v2.0). Default slice: {self.current_slice}/{self.n_slices} ({self.current_window} Window)."
        )
        default_slice = int(
            np.clip(int(self.ct_meta.get("default_slice", self.n_slices // 2)), 0, self.n_slices - 1)
        )
        default_markers = self.markers.get(default_slice, [])
        if default_markers:
            u, v, label, hu = default_markers[0]
            span_txt = (
                f" | Span: slices {self.lesion_span[0]}-{self.lesion_span[1]}"
                if self.lesion_span is not None
                else ""
            )
            init_notes += (
                f"\n[LOG] Pre-marked lesion '{label}' on Slice {default_slice} "
                f"at (u={u:.2f}, v={v:.2f}): {hu:+.1f} HU{span_txt}."
            )
        self.speech_panel.set_log_text(init_notes)

    # -------------------------------------------------------------------------
    # CT Volume Loading & Source Switching
    # -------------------------------------------------------------------------
    def _update_source_badges(self) -> None:
        is_real = bool(self.ct_meta.get("is_real", False))
        if is_real:
            self.ct_panel.set_source_badge(True, "REAL CRANIAL CT")
            self.header_source_badge.setText("REAL CRANIAL CT")
            self.header_source_badge.setStyleSheet(
                f"color: {COLOR_NOMINAL}; background-color: rgba(0, 230, 118, 0.12); "
                f"border: 1px solid {COLOR_NOMINAL}; border-radius: 3px; padding: 2px 7px;"
            )
        else:
            self.ct_panel.set_source_badge(False, "SYNTHETIC PHANTOM")
            self.header_source_badge.setText("SYNTHETIC PHANTOM")
            self.header_source_badge.setStyleSheet(
                f"color: {COLOR_AMBER}; background-color: rgba(255, 179, 0, 0.12); "
                f"border: 1px solid {COLOR_AMBER}; border-radius: 3px; padding: 2px 7px;"
            )

    def _apply_loaded_volume(self, volume: np.ndarray, meta: dict, log_change: bool = True) -> None:
        self.ct_volume = volume
        self.ct_meta = dict(meta)
        self.n_slices = max(1, int(volume.shape[0]))
        default_slice = int(np.clip(int(meta.get("default_slice", self.n_slices // 2)), 0, self.n_slices - 1))
        self.current_slice = default_slice
        self.pinch_anchor_slice = default_slice
        self.pinch_grab.active = False
        self.laser_dwell_count = 0
        self.dwell_progress = 0.0
        self.last_laser_uv = None

        if not meta.get("is_real", False):
            # Synthetic Phantom with calibrated right parietotemporal lesion on slice 38
            self.bookmarks = {default_slice}
            self.markers = {default_slice: [(0.66, 0.39, "Hyperdense Lesion", 78.0)]}
            self.lesion_span = (34, 43)
        else:
            self.bookmarks = {default_slice}
            self.lesion_span = _parse_lesion_span(meta.get("lesion_span"), self.n_slices)
            lesion_uv = _parse_lesion_uv(meta.get("lesion_marker_uv"))
            if lesion_uv is not None:
                u, v = lesion_uv
                label = str(meta.get("lesion_label") or "Focal Lesion")
                hu_val = get_hu_at(volume, default_slice, u, v)
                self.markers = {default_slice: [(u, v, label, hu_val)]}
            else:
                self.markers = {}

        self._update_source_badges()
        self.ct_panel.set_active_window(self.current_window)
        self._render_ct_display(force=True)

        if log_change:
            src_label = "REAL CRANIAL CT" if meta.get("is_real", False) else "SYNTHETIC PHANTOM"
            self.log_message(
                f"CT Source switched -> {src_label} ({meta.get('patient_id', 'Unknown')}, "
                f"{self.n_slices} slices, default {default_slice})"
            )

    def switch_ct_source(self) -> None:
        """Toggles between bundled Real Philips Cranial CT and Synthetic Phantom."""
        if self.ct_meta.get("is_real", False):
            vol = make_phantom()
            meta = {
                "is_real": False,
                "patient_id": "OR-702 (Synthetic Phantom)",
                "series_desc": "CRANIAL PHANTOM",
                "source": "Synthetic",
                "spacing_mm": [2.5, 0.65, 0.65],
                "default_slice": 38,
                "n_slices": N_SLICES,
                "height": CT_SIZE,
                "width": CT_SIZE,
            }
        else:
            vol, meta = load_real_ct()

        self._apply_loaded_volume(vol, meta, log_change=True)
        self._play_sound(sequence(EARCONS["ok"]))

    def load_nifti_file(self, nii_path: str) -> None:
        """Loads a NIfTI volume from disk into the PACS viewport."""
        vol, meta = load_nifti_ct(nii_path)
        self._apply_loaded_volume(vol, meta, log_change=True)
        self._play_sound(sequence(EARCONS["berhasil"]))

    def prompt_load_nifti(self) -> None:
        """Opens a file dialog to load a .nii or .nii.gz cranial CT volume."""
        if self.headless:
            return
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Cranial CT NIfTI Volume",
            "",
            "NIfTI Volumes (*.nii *.nii.gz);;All Files (*.*)",
        )
        if not file_path:
            return
        try:
            self.load_nifti_file(file_path)
        except Exception as exc:
            QMessageBox.warning(self, "NIfTI Load Failed", f"Could not load NIfTI volume:\n{exc}")
            self._play_sound(sequence(EARCONS["error"]))

    # -------------------------------------------------------------------------
    # CT Navigation & PACS Rendering
    # -------------------------------------------------------------------------
    def set_slice(self, slice_num: int) -> None:
        new_slice = int(np.clip(slice_num, 0, self.n_slices - 1))
        if new_slice != self.current_slice:
            self.current_slice = new_slice
            self._render_ct_display()

    def prev_slice(self) -> None:
        if self.current_slice > 0:
            self.set_slice(self.current_slice - 1)

    def next_slice(self) -> None:
        if self.current_slice < self.n_slices - 1:
            self.set_slice(self.current_slice + 1)

    def jump_to_default_slice(self) -> None:
        default_idx = int(np.clip(int(self.ct_meta.get("default_slice", self.n_slices // 2)), 0, self.n_slices - 1))
        self.set_slice(default_idx)
        self.log_message(f"Jumped to default slice {default_idx}.")

    def set_window(self, window_name: str) -> None:
        if window_name not in WINDOWS:
            return
        if window_name != self.current_window:
            self.current_window = window_name
            self.ct_panel.set_active_window(window_name)
            level, width = WINDOWS[window_name]
            self._play_sound(sequence(EARCONS["ok"]))
            self._render_ct_display(force=True)
            self.log_message(f"Window changed: {window_name} (L:{level} W:{width})")

    def cycle_window(self) -> None:
        names = list(WINDOWS.keys())
        idx = (names.index(self.current_window) + 1) % len(names)
        self.set_window(names[idx])

    def bookmark_current_slice(self) -> None:
        if self.current_slice in self.bookmarks:
            self.bookmarks.remove(self.current_slice)
            action = "removed"
            self._play_sound(sequence(EARCONS["error"]))
        else:
            self.bookmarks.add(self.current_slice)
            action = "added"
            self._play_sound(sequence(EARCONS["bookmark"]))
            if self.voice_mode == "ID_ONLINE":
                self._say(f"Iris {self.current_slice} disimpan.", lang="id")
            else:
                self._say(f"Slice {self.current_slice} bookmarked.", lang="en")

        self.log_message(f"Bookmark {action}: Slice {self.current_slice}")
        self._render_ct_display(force=True)

    def mark_current_lesion(self, uv: tuple[float, float] | None = None) -> None:
        target_uv = uv if uv is not None else self.cursor_uv
        u = float(np.clip(target_uv[0], 0.0, 1.0))
        v = float(np.clip(target_uv[1], 0.0, 1.0))
        hu = get_hu_at(self.ct_volume, self.current_slice, u, v)

        if self.current_slice not in self.markers:
            self.markers[self.current_slice] = []

        marker_num = len(self.markers[self.current_slice]) + 1
        label = f"Target-{marker_num}"
        self.markers[self.current_slice].append((u, v, label, hu))

        self._play_sound(sequence(EARCONS["marker"]))
        if self.voice_mode == "ID_ONLINE":
            self._say(f"Target ditandai {hu:+.0f} Hounsfield.", lang="id")
        else:
            self._say(f"Target marked at {hu:+.0f} Hounsfield.", lang="en")

        self.log_message(
            f"ROI Marker '{label}' placed on Slice {self.current_slice} at (u={u:.2f}, v={v:.2f}): {hu:+.1f} HU"
        )
        self._render_ct_display(force=True)

    def clear_markers_on_slice(self) -> None:
        if self.current_slice in self.markers and self.markers[self.current_slice]:
            count = len(self.markers[self.current_slice])
            self.markers[self.current_slice].clear()
            self._play_sound(sequence(EARCONS["error"]))
            self.log_message(f"Cleared {count} markers on Slice {self.current_slice}")
            self._render_ct_display(force=True)

    def _on_viewport_wheel(self, step: int) -> None:
        if step > 0:
            self.next_slice()
        elif step < 0:
            self.prev_slice()

    def _on_viewport_cursor_moved(self, u: float, v: float) -> None:
        self.cursor_uv = (float(np.clip(u, 0.0, 1.0)), float(np.clip(v, 0.0, 1.0)))
        self._render_ct_display()

    def _on_viewport_clicked(self, u: float, v: float) -> None:
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        self.cursor_uv = (float(np.clip(u, 0.0, 1.0)), float(np.clip(v, 0.0, 1.0)))
        self.mark_current_lesion(self.cursor_uv)

    def _render_ct_display(self, force: bool = False) -> None:
        render_key = (
            id(self.ct_volume),
            self.current_slice,
            self.current_window,
            round(self.cursor_uv[0], 3),
            round(self.cursor_uv[1], 3),
            round(self.dwell_progress, 2),
            len(self.markers.get(self.current_slice, [])),
            self.current_slice in self.bookmarks,
        )
        if not force and self._last_render_key == render_key:
            return
        self._last_render_key = render_key

        wl, ww = WINDOWS[self.current_window]
        slice_hu = self.ct_volume[self.current_slice]
        gray = window_slice_grayscale(slice_hu, wl, ww)

        u, v = self.cursor_uv
        cursor_hu = get_hu_at(self.ct_volume, self.current_slice, u, v)
        slice_markers = self.markers.get(self.current_slice, [])
        is_bm = self.current_slice in self.bookmarks

        self.ct_panel.viewport.update_view(
            slice_uint8=gray,
            metadata=self.ct_meta,
            slice_idx=self.current_slice,
            total_slices=self.n_slices,
            window_name=self.current_window,
            wl=wl,
            ww=ww,
            cursor_uv=self.cursor_uv,
            cursor_hu=cursor_hu,
            markers=slice_markers,
            is_bookmarked=is_bm,
            dwell_progress=self.dwell_progress,
        )
        self.ct_panel.slice_rail.set_state(
            current_slice=self.current_slice,
            total_slices=self.n_slices,
            bookmarks=self.bookmarks,
            lesion_span=self.lesion_span,
        )
        self.ct_panel.set_bookmarks_summary(self.bookmarks, self.current_slice)

    # -------------------------------------------------------------------------
    # Touchless Vision & Gesture Loop (33ms / ~30 FPS)
    # -------------------------------------------------------------------------
    def toggle_gesture_lock(self) -> None:
        self.gesture_locked = not self.gesture_locked
        if self.gesture_locked:
            self.vision_panel.set_lock_state(True, "manual")
            self.vision_panel.set_gesture_status("[LOCKED] GESTURES PAUSED", COLOR_AMBER, 0.0)
            self._say("Pelacakan gestur dijeda.", lang="id")
        else:
            self.vision_panel.set_lock_state(False, "")
            self.vision_panel.set_gesture_status("[IDLE] GESTURES ACTIVE", COLOR_CYAN, 0.0)
            self._say("Pelacakan gestur aktif.", lang="id")
        self.log_message(f"Sterile Gesture Lock: {'PAUSED' if self.gesture_locked else 'ACTIVE'}")

    def _on_face_autolock_toggled(self, enabled: bool) -> None:
        self.face_autolock_enabled = bool(enabled)
        self.log_message(f"Face Auto-Lock: {'ENABLED' if self.face_autolock_enabled else 'DISABLED'}")

    def _generate_simulated_frame(self) -> np.ndarray:
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:] = (16, 22, 30)
        cv2.putText(img, "SIMULATED SURGICAL SENSOR", (155, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 229, 255), 2)
        cv2.putText(
            img,
            "(No physical camera - mouse/keys active)",
            (130, 238),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (160, 175, 190),
            1,
        )
        t = time.monotonic()
        cx = int(320 + 130 * np.sin(t * 1.5))
        cy = int(340 + 36 * np.cos(t * 2.0))
        cv2.circle(img, (cx, cy), 12, (0, 230, 118), -1)
        cv2.putText(img, "Simulated Hand Target", (cx + 16, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 230, 118), 1)
        return img

    def _camera_loop(self) -> None:
        if not self.is_running:
            return
        try:
            if self.cam_available and self.cap is not None and self.cap.isOpened():
                ret, frame = self.cap.read()
                frame = cv2.flip(frame, 1) if ret else self._generate_simulated_frame()
            else:
                frame = self._generate_simulated_frame()

            # 1. Face Tracker & 15-frame Missing-Face Auto-Lock
            face_status = self._last_face_status
            face_locked = self._last_face_locked
            if self.face_tracker is not None and not self.headless:
                boxes = self.face_tracker.find(frame)
                if boxes:
                    self.missing_face_frames = 0
                    fx, fy, fw, fh = boxes[0]
                    cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 200), 2)
                    ratio = fw / float(max(1, frame.shape[1]))
                    if ratio > 0.55:
                        face_status = "[TOO CLOSE] RETRACT"
                        face_locked = False
                        cv2.putText(frame, "TOO CLOSE", (fx, max(14, fy - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
                    elif ratio < 0.14:
                        face_status = "[AWAY / BG]"
                        face_locked = bool(self.face_autolock_enabled)
                        cv2.putText(frame, "AWAY", (fx, max(14, fy - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)
                    else:
                        face_status = "[PRESENT] OPTIMAL"
                        face_locked = False
                        cv2.putText(frame, "PRESENT", (fx, max(14, fy - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                else:
                    self.missing_face_frames += 1
                    if self.missing_face_frames > 15:
                        face_status = "[AWAY / NO FACE]"
                        face_locked = bool(self.face_autolock_enabled)
            elif self.headless:
                face_status = "[PRESENT] STERILE HEADLESS"
                face_locked = False

            self._last_face_status = face_status
            self._last_face_locked = face_locked

            face_banner = face_status + (" [AUTO-LOCKED]" if face_locked else "")
            self.vision_panel.set_face_status(face_banner, face_locked)

            if self.gesture_locked:
                self.vision_panel.set_lock_state(True, "manual")
            elif face_locked:
                self.vision_panel.set_lock_state(True, "face auto-lock")
            else:
                self.vision_panel.set_lock_state(False, "")

            # 2. Hand Tracker & Gesture Processing
            hand_pts = None
            if self.hand_tracker is not None and not self.headless and not self.gesture_locked and not face_locked:
                hand_pts = self.hand_tracker.find(frame)

            detected_gesture: int | None = None
            gesture_text = "[LOCKED] GESTURES PAUSED" if (self.gesture_locked or face_locked) else "[IDLE] NO HAND DETECTED"
            gesture_color = COLOR_AMBER if self.gesture_locked else COLOR_DANGER if face_locked else COLOR_CYAN
            dwell_norm = 0.0

            if hand_pts:
                draw_hand(frame, hand_pts)

                if self.speech.is_speaking():
                    gesture_text = "[HALF-DUPLEX] MUTED (SPEAKING)"
                    gesture_color = COLOR_AMBER
                    self.gesture_debouncer.reset()
                    self.pinch_grab.active = False
                else:
                    # A. Pinch Clutching FIRST
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
                        gesture_color = COLOR_CYAN
                        self.laser_dwell_count = 0
                        self.dwell_progress = 0.0

                        dy = self.pinch_anchor_y - pinch_y
                        target_slice = int(round(self.pinch_anchor_slice + dy / 14.0))
                        if target_slice != self.current_slice:
                            self.set_slice(target_slice)

                    # B. 1-Finger Surgical Laser Pointer + 42-frame (~1.4s) Dwell
                    elif is_index_pointer(hand_pts) and (time.monotonic() - self.last_pinch_time >= 0.35):
                        self.last_laser_time = time.monotonic()
                        self.gesture_debouncer.reset()
                        gesture_text = "[POINTER] SURGICAL LASER"
                        gesture_color = COLOR_NOMINAL

                        ix, iy = hand_pts[8]
                        u_norm = float(np.clip((ix / frame.shape[1] - 0.12) / 0.76, 0.0, 1.0))
                        v_norm = float(np.clip((iy / frame.shape[0] - 0.12) / 0.76, 0.0, 1.0))
                        self.cursor_uv = (u_norm, v_norm)

                        if self.last_laser_uv is not None and dist(self.cursor_uv, self.last_laser_uv) < 0.045:
                            self.laser_dwell_count += 1
                        else:
                            self.laser_dwell_count = 1
                            self.last_laser_uv = self.cursor_uv

                        dwell_norm = min(1.0, self.laser_dwell_count / float(self.LASER_DWELL_TARGET))
                        self.dwell_progress = dwell_norm

                        if dwell_norm >= 1.0:
                            self.mark_current_lesion(self.cursor_uv)
                            self.laser_dwell_count = 0
                            self.dwell_progress = 0.0
                            self.last_laser_time = time.monotonic()

                        self._render_ct_display()

                    else:
                        # C. Post-Pinch & Post-Laser Cooldown Guard (0.45s)
                        now_mono = time.monotonic()
                        is_cooldown = (now_mono - self.last_pinch_time < 0.45) or (now_mono - self.last_laser_time < 0.45)
                        if is_cooldown:
                            gesture_text = "[COOLDOWN] STABILIZING..."
                            gesture_color = COLOR_AMBER
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
                                gesture_text = names[n_fingers]
                                gesture_color = COLOR_AMBER
            else:
                self.pinch_grab.active = False
                self.laser_dwell_count = 0
                if self.dwell_progress != 0.0:
                    self.dwell_progress = 0.0
                    self._render_ct_display()

            # 3. StableGesture Debouncer Execution
            triggered = self.gesture_debouncer.update(detected_gesture)
            if detected_gesture is not None and dwell_norm == 0.0:
                dwell_norm = self.gesture_debouncer.progress

            if triggered == 2:
                self.cycle_window()
            elif triggered == 3:
                self.bookmark_current_slice()
            elif triggered == 4:
                self.speak_vitals_summary()
            elif triggered == 5:
                self.toggle_gesture_lock()

            self.vision_panel.set_gesture_status(gesture_text, gesture_color, dwell_norm * 100.0)

            rgb_cam = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.vision_panel.set_camera_frame(rgb_cam)

        except Exception as e:
            print(f"[OR-SterileAssist v2] Camera loop exception: {e}")

    # -------------------------------------------------------------------------
    # 60 FPS Physiological Waveforms & Vitals / Alarms (16ms + 1000ms)
    # -------------------------------------------------------------------------
    def _waveform_loop(self) -> None:
        if not self.is_running:
            return
        now = time.monotonic()
        dt = max(0.005, min(0.05, now - self._last_wave_time))
        self._last_wave_time = now

        self.vitals_panel.step_waveforms(dt)

        rem_sec = max(0.0, self.alarm_silenced_until - now)
        sev = str(self.vitals.get("alarm_severity", "none")).lower()
        flash_phase = False
        if sev in ("medium", "high") and rem_sec <= 0.0:
            flash_phase = (int(now * 2.0) % 2) == 1

        self.vitals_panel.update_vitals(
            self.vitals,
            silence_remaining_sec=rem_sec,
            flash_phase=flash_phase,
        )

    def _vitals_monitor_loop(self) -> None:
        if not self.is_running:
            return
        self.clock_lbl.setText(datetime.datetime.now().strftime("%H:%M:%S"))

        now = time.monotonic()
        silenced = now < self.alarm_silenced_until
        sev = str(self.vitals.get("alarm_severity", "none")).lower()

        if sev in ("medium", "high") and not silenced and not self.speech.is_speaking():
            pattern = ALARMS.get(sev)
            if pattern is not None:
                self._play_sound(sequence(pattern, volume=0.22), half_duplex=True)

    def _set_scenario(self, scenario: str, log_change: bool = True) -> None:
        key = scenario.lower().strip()
        if key == "normal":
            self.vitals.update({
                "hr": 74,
                "spo2": 98,
                "bp_sys": 118,
                "bp_dia": 76,
                "temp": 36.8,
                "alarm_severity": "none",
            })
            log_txt = "Clinical scenario: Normal Baseline (SpO2 98%)"
        elif key == "mild":
            self.vitals.update({
                "hr": 94,
                "spo2": 91,
                "bp_sys": 136,
                "bp_dia": 88,
                "temp": 37.1,
                "alarm_severity": "medium",
            })
            log_txt = "Clinical scenario: Mild Hypoxia (SpO2 91% - IEC Medium Alarm)"
        elif key == "critical":
            self.vitals.update({
                "hr": 122,
                "spo2": 84,
                "bp_sys": 158,
                "bp_dia": 102,
                "temp": 37.5,
                "alarm_severity": "high",
            })
            log_txt = "Clinical scenario: CRITICAL DESATURATION (SpO2 84% - IEC High Alarm)"
        else:
            return

        for s_key, btn in self.vitals_panel._scenario_buttons.items():
            btn.setProperty("active", "true" if s_key == key else "false")
            repolish(btn)

        rem_sec = max(0.0, self.alarm_silenced_until - time.monotonic())
        self.vitals_panel.update_vitals(self.vitals, silence_remaining_sec=rem_sec, flash_phase=False)

        if log_change:
            self.log_message(log_txt)

    def silence_alarm(self) -> None:
        self.alarm_silenced_until = time.monotonic() + 120.0
        self._play_sound(sequence(EARCONS["ok"]))
        self.vitals_panel.update_vitals(self.vitals, silence_remaining_sec=120.0, flash_phase=False)
        self.log_message("IEC 60601-1-8 Alarm silenced for 120 seconds.")

    def _on_alarm_test(self, severity: str) -> None:
        pattern = ALARMS.get(severity)
        if pattern is not None:
            self._play_sound(sequence(pattern), half_duplex=True)
            self.log_message(f"IEC 60601-1-8 Test Alarm triggered: {severity.upper()}")

    def _on_pulse_ox_toggle(self, enabled: bool) -> None:
        self.pulse_ox_active = bool(enabled)
        if self.pulse_ox_active:
            self.log_message("SpO2 Pulse Sonification ENABLED.")
            self._on_pulse_ox_tick()
        else:
            self.pulse_ox_timer.stop()
            self.log_message("SpO2 Pulse Sonification DISABLED.")

    def _on_pulse_ox_tick(self) -> None:
        if not self.is_running or not self.pulse_ox_active:
            return
        hr = max(30, int(self.vitals.get("hr", 74)))
        interval_ms = int(60000.0 / hr)

        if not self.headless and not self.speech.is_speaking():
            chirp = pulse_ox_tone(self.vitals["spo2"], duration=0.065, volume=0.20)
            try:
                sd.play(chirp, SAMPLE_RATE)
            except Exception:
                pass

        self.pulse_ox_timer.start(interval_ms)

    def speak_vitals_summary(self) -> None:
        lang = "id" if self.voice_mode == "ID_ONLINE" else "en"
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
        self._say(text, sound=sequence(EARCONS["attention"]), lang=lang)
        self.log_message("Spoke Patient Vitals Summary.")

    # -------------------------------------------------------------------------
    # Speech Polling & Voice Command Dispatch (50ms)
    # -------------------------------------------------------------------------
    def _speech_poll_loop(self) -> None:
        if not self.is_running:
            return
        try:
            for _ in range(64):
                try:
                    event = self.speech.events_queue.get_nowait()
                except queue.Empty:
                    break
                etype = event.get("type")

                if etype == "meter":
                    level = float(event.get("dbfs", -60.0))
                    thresh = float(event.get("threshold", self.speech.threshold_db))
                    self.speech_panel.set_audio_level(level, thresh)

                    if event.get("speaking"):
                        self.speech_panel.set_half_duplex_state("TTS SPEAKING")
                    elif not event.get("mic_enabled"):
                        self.speech_panel.set_half_duplex_state("MIC OFF")
                    else:
                        self.speech_panel.set_half_duplex_state("LISTENING")

                elif etype == "calibration_done":
                    thresh = float(event.get("threshold_db", -44.0))
                    self.speech_panel.set_audio_level(-60.0, thresh)
                    self.speech_panel.set_heard_text(event.get("msg", "Calibration done."))
                    self._play_sound(sequence(EARCONS["berhasil"]))

                elif etype == "command":
                    cmd_id = event.get("cmd", "")
                    matched_text = event.get("text", "")
                    conf = float(event.get("conf", 0.9))
                    reply = event.get("reply", "")
                    self.speech_panel.set_heard_text(f'"{matched_text}" (Conf: {conf:.2f}) -> {cmd_id}')
                    self.log_message(f'Voice Command executed: "{matched_text}" -> {cmd_id}')
                    self._execute_voice_command(cmd_id, reply)

                elif etype == "dictation":
                    text = event.get("text", "")
                    self.speech_panel.set_heard_text(f'Dictated: "{text}"')
                    self.log_dictation(text)
                    self._play_sound(sequence(EARCONS["ok"]))

                elif etype == "unrecognized":
                    text = event.get("text", "")
                    self.speech_panel.set_heard_text(f'Unrecognized: "{text}"')

                elif etype == "status":
                    self.speech_panel.set_heard_text(event.get("msg", ""))
        except Exception as e:
            print(f"[OR-SterileAssist v2] Speech poll error: {e}")

    def _execute_voice_command(self, cmd_id: str, spoken_reply: str) -> None:
        lang = "id" if self.voice_mode == "ID_ONLINE" else "en"

        if cmd_id == "NEXT_SLICE":
            self.next_slice()
            self._say(spoken_reply or "Iris berikutnya.", lang=lang)

        elif cmd_id == "PREV_SLICE":
            self.prev_slice()
            self._say(spoken_reply or "Iris sebelumnya.", lang=lang)

        elif cmd_id == "WINDOW_BRAIN":
            self.set_window("Brain")
            self._say(spoken_reply or "Jendela otak.", lang=lang)

        elif cmd_id == "WINDOW_BONE":
            self.set_window("Bone")
            self._say(spoken_reply or "Jendela tulang.", lang=lang)

        elif cmd_id == "WINDOW_SUBDURAL":
            self.set_window("Subdural")
            self._say(spoken_reply or "Jendela subdural.", lang=lang)

        elif cmd_id == "BOOKMARK_SLICE":
            self.bookmark_current_slice()

        elif cmd_id == "MARK_LESION":
            self.mark_current_lesion()

        elif cmd_id == "CLEAR_MARKERS":
            self.clear_markers_on_slice()
            self._say(spoken_reply or "Tanda dihapus.", lang="id" if lang == "id" else "en")

        elif cmd_id == "PATIENT_VITALS":
            self.speak_vitals_summary()

        elif cmd_id == "HEART_RATE":
            hr_txt = (
                f"Detak jantung pasien {self.vitals['hr']} kali per menit."
                if lang == "id"
                else f"Patient heart rate is {self.vitals['hr']} beats per minute."
            )
            self._say(hr_txt, lang=lang)

        elif cmd_id == "OXYGEN":
            spo2_txt = (
                f"Saturasi oksigen {self.vitals['spo2']} persen."
                if lang == "id"
                else f"Oxygen saturation is {self.vitals['spo2']} percent."
            )
            self._say(spo2_txt, lang=lang)

        elif cmd_id == "BLOOD_PRESSURE":
            bp_txt = (
                f"Tekanan darah {self.vitals['bp_sys']} per {self.vitals['bp_dia']} milimeter merkuri."
                if lang == "id"
                else f"Blood pressure is {self.vitals['bp_sys']} over {self.vitals['bp_dia']} millimeters of mercury."
            )
            self._say(bp_txt, lang=lang)

        elif cmd_id == "SILENCE_ALARM":
            self.silence_alarm()
            silence_txt = "Alarm diheningkan dua menit." if lang == "id" else "Alarm silenced for two minutes."
            self._say(silence_txt, lang=lang)

        elif cmd_id == "PAUSE_CAMERA":
            if not self.gesture_locked:
                self.toggle_gesture_lock()

        elif cmd_id == "RESUME_CAMERA":
            if self.gesture_locked:
                self.toggle_gesture_lock()

        elif cmd_id == "JUMP_DEFAULT":
            self.jump_to_default_slice()

        elif cmd_id == "HELP":
            self._say(spoken_reply, lang=lang)

    def _on_voice_mode_change(self, mode: str) -> None:
        self.voice_mode = mode
        self.speech.set_mode(mode)
        self.log_message(f"ASR Mode changed to: {mode}")

    def _on_toggle_mic(self) -> None:
        is_on = self.speech.toggle_mic()
        self.speech_panel.set_mic_enabled(is_on)
        self.speech_panel.set_half_duplex_state("LISTENING" if is_on else "MIC OFF")

    # -------------------------------------------------------------------------
    # Surgical Logging & Report Export
    # -------------------------------------------------------------------------
    def log_message(self, msg: str) -> None:
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.speech_panel.append_log(f"[{ts}] {msg}")

    def log_dictation(self, text: str) -> None:
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.speech_panel.append_log(f'[{ts} DICTATION] "{text}"')

    def speak_notes_summary(self) -> None:
        lang = "id" if self.voice_mode == "ID_ONLINE" else "en"
        n_bm = len(self.bookmarks)
        total_markers = sum(len(m) for m in self.markers.values())
        if lang == "id":
            summary = (
                f"Ringkasan operasi: {n_bm} iris disimpan dalam bookmark, "
                f"dan {total_markers} target lesi telah ditandai."
            )
        else:
            summary = f"Surgical summary: {n_bm} slices bookmarked, and {total_markers} lesion targets marked."
        self._say(summary, lang=lang)

    def export_report(self) -> None:
        if self.headless:
            return
        content = self.speech_panel.get_log_text().strip()
        default_filename = f"OR_Surgical_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Intra-Operative Report",
            default_filename,
            "Text files (*.txt);;All files (*.*)",
        )
        if not file_path:
            return
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
                f.write("\n\n=== END OF SURGICAL REPORT ===\n")
            QMessageBox.information(self, "Export Success", f"Surgical report saved to:\n{file_path}")
            self.speech.play_sound(sequence(EARCONS["berhasil"]))
            self.log_message(f"Report exported: {os.path.basename(file_path)}")
        except Exception as exc:
            QMessageBox.warning(self, "Export Failed", f"Could not save file:\n{exc}")

    # -------------------------------------------------------------------------
    # Guarded Keyboard Shortcuts
    # -------------------------------------------------------------------------
    def _is_text_focused(self) -> bool:
        fw = QApplication.focusWidget()
        return isinstance(fw, (QPlainTextEdit, QTextEdit, QLineEdit))

    def _handle_shortcut_key(
        self,
        key: int,
        modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier,
    ) -> bool:
        if self._is_text_focused():
            return False
        if modifiers & (
            Qt.KeyboardModifier.ControlModifier
            | Qt.KeyboardModifier.AltModifier
            | Qt.KeyboardModifier.MetaModifier
        ):
            return False

        if key in (Qt.Key.Key_Left, Qt.Key.Key_Down, Qt.Key.Key_P):
            self.prev_slice()
            return True
        if key in (Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_N):
            self.next_slice()
            return True
        if key == Qt.Key.Key_W:
            self.cycle_window()
            return True
        if key == Qt.Key.Key_B:
            self.bookmark_current_slice()
            return True
        if key == Qt.Key.Key_M:
            self.mark_current_lesion()
            return True
        if key == Qt.Key.Key_C:
            self.clear_markers_on_slice()
            return True
        if key == Qt.Key.Key_A:
            self.silence_alarm()
            return True
        if key == Qt.Key.Key_V:
            self.speak_vitals_summary()
            return True
        if key == Qt.Key.Key_Space:
            self.toggle_gesture_lock()
            return True
        return False

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Type.KeyPress and isinstance(event, QKeyEvent):
            if self.isActiveWindow() or watched is self or (
                hasattr(watched, "window") and watched.window() is self
            ):
                if self._handle_shortcut_key(event.key(), event.modifiers()):
                    event.accept()
                    return True
        return super().eventFilter(watched, event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if self._handle_shortcut_key(event.key(), event.modifiers()):
            event.accept()
            return
        super().keyPressEvent(event)

    # -------------------------------------------------------------------------
    # Graceful Shutdown
    # -------------------------------------------------------------------------
    def shutdown(self) -> None:
        if not self.is_running:
            return
        print("[OR-SterileAssist v2] Shutting down workstation...")
        self.is_running = False
        self.pulse_ox_active = False

        for timer_name in ("cam_timer", "wave_timer", "speech_timer", "vitals_timer", "pulse_ox_timer"):
            timer = getattr(self, timer_name, None)
            if timer is not None:
                try:
                    timer.stop()
                except Exception:
                    pass

        if getattr(self, "_key_filter_installed", False):
            qapp = QApplication.instance()
            if qapp is not None:
                try:
                    qapp.removeEventFilter(self)
                except Exception:
                    pass
            self._key_filter_installed = False

        if hasattr(self, "speech") and self.speech is not None:
            try:
                self.speech.stop()
            except Exception:
                pass
            for th_name in ("speaker_thread", "mic_thread"):
                th = getattr(self.speech, th_name, None)
                if th is not None and th.is_alive():
                    try:
                        th.join(timeout=0.5)
                    except Exception:
                        pass

        for tracker_attr, inner_attr in (("hand_tracker", "landmarker"), ("face_tracker", "detector")):
            tracker = getattr(self, tracker_attr, None)
            if tracker is not None:
                try:
                    if hasattr(tracker, "close"):
                        tracker.close()
                    else:
                        inner = getattr(tracker, inner_attr, None)
                        if inner is not None and hasattr(inner, "close"):
                            inner.close()
                except Exception:
                    pass
                setattr(self, tracker_attr, None)

        if hasattr(self, "cap") and self.cap is not None:
            try:
                if self.cap.isOpened():
                    self.cap.release()
            except Exception:
                pass
            self.cap = None

    def closeEvent(self, event) -> None:
        self.shutdown()
        super().closeEvent(event)


def run_self_check() -> int:
    """Runs automated headless verification of all v2 subsystems and exits 0."""
    print("[OR-SterileAssist v2] Running automated self-check...")
    app = QApplication.instance() or QApplication(sys.argv)
    win = ORSterileAssistV2Window(headless=True)
    try:
        # Step waveforms & verify core navigation/clinical actions
        for _ in range(10):
            win._waveform_loop()
        win.next_slice()
        win.prev_slice()
        win.cycle_window()
        win.bookmark_current_slice()
        win.mark_current_lesion((0.52, 0.48))
        win._set_scenario("mild")
        win._waveform_loop()
        win.silence_alarm()
        win.switch_ct_source()
        win.switch_ct_source()
        app.processEvents()
        print("[OR-SterileAssist v2] All subsystems verified cleanly.")
    finally:
        win.close()
        win.shutdown()
        win.deleteLater()
        app.processEvents()
    print("[OR-SterileAssist v2] Self-check completed successfully (exit code 0).")
    return 0


def main() -> None:
    if "--self-check" in sys.argv or "--test" in sys.argv:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        sys.exit(run_self_check())

    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(APP_QSS)
    win = ORSterileAssistV2Window(headless=False)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
