"""vision_hud.py - Sterile Vision Telemetry Panel for Hand Gestures & Surgeon Presence.

Provides frameless 4:3 RGB camera preview with offline fallback, surgeon presence banner,
active gesture telemetry card with vector dwell bar, 6-row surgical gesture reference,
and sterile lock / face auto-lock controls.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .theme import (
    BG_CARD,
    BG_INPUT,
    BG_VIEWPORT,
    BORDER_HAIRLINE,
    COLOR_AMBER,
    COLOR_CYAN,
    COLOR_DANGER,
    COLOR_NOMINAL,
    TEXT_MUTED,
    TEXT_PRIMARY,
    repolish,
    tabular_font,
    ui_font,
)


class _CameraPreviewLabel(QLabel):
    """Frameless 4:3 camera preview label rendering direct QImage(Format_RGB888) without resize loops."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._qimage: QImage | None = None
        self.setMinimumSize(310, 232)
        self.setFixedHeight(248)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_frame(self, rgb_np: np.ndarray | None) -> None:
        if rgb_np is None or rgb_np.size == 0:
            if self._qimage is not None:
                self._qimage = None
                self.update()
            return
        arr = np.ascontiguousarray(rgb_np, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[2] != 3:
            return
        h, w, _ = arr.shape
        qimg = QImage(arr.data, w, h, w * 3, QImage.Format.Format_RGB888)
        self._qimage = qimg.copy()
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        w, h = self.width(), self.height()
        painter.fillRect(0, 0, w, h, QColor(BG_VIEWPORT))

        if self._qimage is not None:
            # Maintain 4:3 / native aspect ratio centered in frame
            scaled = self._qimage.scaled(
                w - 2,
                h - 2,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            dx = (w - scaled.width()) // 2
            dy = (h - scaled.height()) // 2
            painter.drawImage(dx, dy, scaled)
        else:
            painter.setFont(tabular_font(9, bold=True))
            painter.setPen(QColor(TEXT_MUTED))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "[CAMERA OFFLINE]\nNo Sterile Vision Feed",
            )

        painter.setPen(QPen(QColor(BORDER_HAIRLINE), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(0, 0, w - 1, h - 1)


class _VectorDwellBar(QWidget):
    """Crisp vector horizontal dwell progress bar with percentage ticks."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pct: float = 0.0
        self._color: str = COLOR_CYAN
        self.setFixedHeight(14)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_progress(self, dwell_pct: float, color_hex: str | None = None) -> None:
        pct = max(0.0, min(100.0, float(dwell_pct)))
        col = color_hex or (COLOR_NOMINAL if pct >= 99.0 else COLOR_AMBER if pct > 0.0 else COLOR_CYAN)
        if abs(pct - self._pct) < 0.25 and col == self._color:
            return
        self._pct = pct
        self._color = col
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w = float(self.width())
        h = float(self.height())

        painter.setPen(QPen(QColor(BORDER_HAIRLINE), 1))
        painter.setBrush(QColor(BG_INPUT))
        painter.drawRoundedRect(QRectF(0.5, 0.5, w - 1.0, h - 1.0), 3.0, 3.0)

        if self._pct > 0.0:
            fill_w = max(4.0, (w - 4.0) * (self._pct / 100.0))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(self._color))
            painter.drawRoundedRect(QRectF(2.0, 2.0, fill_w, h - 4.0), 2.0, 2.0)

        # Subtle quartile ticks
        painter.setPen(QPen(QColor(BG_CARD), 1))
        for frac in (0.25, 0.50, 0.75):
            tx = w * frac
            painter.drawLine(int(tx), 2, int(tx), int(h - 2))


class VisionHudPanel(QFrame):
    """Middle workstation column: camera feed, surgeon presence, gesture telemetry, and lock controls."""

    toggle_lock_requested = Signal()
    face_autolock_toggled = Signal(bool)

    _GESTURE_ROWS = (
        ("PINCH", "Scroll CT slices Up / Down"),
        ("1 FINGER", "Laser pointer & 1.4s dwell ROI"),
        ("2 FINGERS", "Cycle window (Brain / Bone / Subdural)"),
        ("3 FINGERS", "Bookmark current CT slice"),
        ("4 FINGERS", "Voice summary of patient vitals"),
        ("5 FINGERS", "Toggle sterile gesture lock"),
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")

        self._last_face_state: tuple[str, bool] | None = None
        self._last_gesture_state: tuple[str, str, int] | None = None
        self._last_lock_state: tuple[bool, str] | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # Header Row
        hdr = QHBoxLayout()
        hdr.setSpacing(6)
        title = QLabel("STERILE VISION TELEMETRY")
        title.setFont(ui_font(10, bold=True))
        title.setStyleSheet(f"color: {TEXT_PRIMARY}; letter-spacing: 0.6px;")
        hdr.addWidget(title)
        hdr.addStretch(1)

        self.status_pill = QLabel("[ACTIVE]")
        self.status_pill.setFont(tabular_font(8, bold=True))
        hdr.addWidget(self.status_pill)
        root.addLayout(hdr)

        # Frameless 4:3 Camera Preview (fixed height eliminates vertical letterboxing)
        self.camera_preview = _CameraPreviewLabel()
        root.addWidget(self.camera_preview)

        # Surgeon Presence Banner
        presence_card = QFrame()
        presence_card.setObjectName("Card")
        pres_lay = QHBoxLayout(presence_card)
        pres_lay.setContentsMargins(10, 8, 10, 8)
        pres_lay.setSpacing(6)
        pres_title = QLabel("SURGEON PRESENCE:")
        pres_title.setFont(tabular_font(8, bold=True))
        pres_title.setStyleSheet(f"color: {TEXT_MUTED};")
        pres_lay.addWidget(pres_title)

        self.face_status_lbl = QLabel("[CHECKING]")
        self.face_status_lbl.setFont(tabular_font(8, bold=True))
        pres_lay.addWidget(self.face_status_lbl, stretch=1)
        root.addWidget(presence_card)

        # Active Gesture Telemetry Card
        gest_card = QFrame()
        gest_card.setObjectName("Card")
        gest_lay = QVBoxLayout(gest_card)
        gest_lay.setContentsMargins(10, 10, 10, 10)
        gest_lay.setSpacing(7)

        g_top = QHBoxLayout()
        g_hdr = QLabel("ACTIVE GESTURE TELEMETRY")
        g_hdr.setFont(tabular_font(8, bold=True))
        g_hdr.setStyleSheet(f"color: {TEXT_MUTED};")
        g_top.addWidget(g_hdr)
        g_top.addStretch(1)

        self.dwell_pct_lbl = QLabel("DWELL: 0%")
        self.dwell_pct_lbl.setFont(tabular_font(8, bold=True))
        self.dwell_pct_lbl.setStyleSheet(f"color: {COLOR_CYAN};")
        g_top.addWidget(self.dwell_pct_lbl)
        gest_lay.addLayout(g_top)

        self.gesture_mode_lbl = QLabel("[IDLE] READY")
        self.gesture_mode_lbl.setFont(tabular_font(11, bold=True))
        self.gesture_mode_lbl.setStyleSheet(
            f"color: {COLOR_CYAN}; font-family: 'Cascadia Mono', 'Consolas', monospace; "
            "font-size: 11pt; font-weight: 700;"
        )
        gest_lay.addWidget(self.gesture_mode_lbl)

        self.dwell_bar = _VectorDwellBar()
        gest_lay.addWidget(self.dwell_bar)
        root.addWidget(gest_card)

        # 6-Row Surgical Gesture Reference Table
        ref_card = QFrame()
        ref_card.setObjectName("Card")
        ref_lay = QVBoxLayout(ref_card)
        ref_lay.setContentsMargins(10, 10, 10, 10)
        ref_lay.setSpacing(6)

        ref_hdr = QLabel("SURGICAL GESTURE REFERENCE")
        ref_hdr.setFont(tabular_font(8, bold=True))
        ref_hdr.setStyleSheet(f"color: {TEXT_MUTED};")
        ref_lay.addWidget(ref_hdr)

        table = QGridLayout()
        table.setContentsMargins(0, 2, 0, 0)
        table.setHorizontalSpacing(10)
        table.setVerticalSpacing(6)

        for row_idx, (code, desc) in enumerate(self._GESTURE_ROWS):
            code_lbl = QLabel(code)
            code_lbl.setFont(tabular_font(8, bold=True))
            code_lbl.setStyleSheet(f"color: {COLOR_CYAN};")
            desc_lbl = QLabel(desc)
            desc_lbl.setFont(ui_font(8, bold=False))
            desc_lbl.setStyleSheet(f"color: {TEXT_PRIMARY};")
            table.addWidget(code_lbl, row_idx, 0)
            table.addWidget(desc_lbl, row_idx, 1)

        table.setColumnStretch(1, 1)
        ref_lay.addLayout(table)
        root.addWidget(ref_card)
        root.addStretch(1)

        # Sterile Lock & Face Auto-Lock Controls
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(6)
        self.lock_btn = QPushButton("Pause Gesture Lock (Space)")
        self.lock_btn.setFont(ui_font(8, bold=True))
        self.lock_btn.clicked.connect(self.toggle_lock_requested.emit)
        ctrl_row.addWidget(self.lock_btn, stretch=1)

        self.face_autolock_chk = QCheckBox("Face Auto-Lock")
        self.face_autolock_chk.setFont(ui_font(8, bold=False))
        self.face_autolock_chk.setChecked(True)
        self.face_autolock_chk.toggled.connect(self.face_autolock_toggled.emit)
        ctrl_row.addWidget(self.face_autolock_chk)
        root.addLayout(ctrl_row)

        self.set_lock_state(False, "")
        self.set_face_status("[PRESENT] Sterile Operator Verified", False)
        self.set_gesture_status("[IDLE] GESTURES ACTIVE", COLOR_CYAN, 0.0)

    def set_camera_frame(self, rgb_np: np.ndarray | None) -> None:
        """Updates the frameless RGB888 camera preview or shows offline banner when None."""
        self.camera_preview.set_frame(rgb_np)

    def set_face_status(self, text: str, locked: bool) -> None:
        """Updates surgeon presence banner text and semantic color."""
        state = (str(text), bool(locked))
        if state == self._last_face_state:
            return
        self._last_face_state = state

        color = COLOR_DANGER if locked else COLOR_NOMINAL
        self.face_status_lbl.setText(text)
        self.face_status_lbl.setStyleSheet(f"color: {color};")

    def set_gesture_status(self, mode_text: str, color_hex: str, dwell_pct: float) -> None:
        """Updates active gesture telemetry label and vector dwell progress bar."""
        pct_int = max(0, min(100, int(round(float(dwell_pct)))))
        state = (str(mode_text), str(color_hex), pct_int)
        if state == self._last_gesture_state:
            return
        self._last_gesture_state = state

        self.gesture_mode_lbl.setText(mode_text)
        self.gesture_mode_lbl.setStyleSheet(
            f"color: {color_hex}; font-family: 'Cascadia Mono', 'Consolas', monospace; "
            "font-size: 11pt; font-weight: 700;"
        )
        self.dwell_pct_lbl.setText(f"DWELL: {pct_int:3d}%")
        bar_col = COLOR_NOMINAL if pct_int >= 99 else COLOR_AMBER if pct_int > 0 else color_hex
        self.dwell_pct_lbl.setStyleSheet(f"color: {bar_col};")
        self.dwell_bar.set_progress(float(dwell_pct), bar_col)

    def set_lock_state(self, locked: bool, reason: str = "") -> None:
        """Updates top status pill ([ACTIVE], [PAUSED], [AUTO-LOCKED]) and lock button."""
        reason_str = str(reason or "").strip()
        state = (bool(locked), reason_str)
        if state == self._last_lock_state:
            return
        self._last_lock_state = state

        if locked:
            is_auto = "auto" in reason_str.lower() or "face" in reason_str.lower()
            pill_txt = "[AUTO-LOCKED]" if is_auto else "[PAUSED]"
            color = COLOR_DANGER if is_auto else COLOR_AMBER
            bg = "rgba(255, 59, 48, 0.14)" if is_auto else "rgba(255, 179, 0, 0.14)"

            self.lock_btn.setText("Resume Gesture Lock (Space)")
            self.lock_btn.setProperty("nominal", "true")
            self.lock_btn.setProperty("warn", "false")
        else:
            pill_txt = "[ACTIVE]"
            color = COLOR_NOMINAL
            bg = "rgba(0, 230, 118, 0.12)"

            self.lock_btn.setText("Pause Gesture Lock (Space)")
            self.lock_btn.setProperty("nominal", "false")
            self.lock_btn.setProperty("warn", "true")

        repolish(self.lock_btn)
        self.status_pill.setText(pill_txt)
        self.status_pill.setStyleSheet(
            f"color: {color}; background-color: {bg}; "
            f"border: 1px solid {color}; border-radius: 3px; padding: 2px 7px;"
        )
