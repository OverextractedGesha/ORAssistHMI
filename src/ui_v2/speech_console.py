"""speech_console.py - Bilingual Voice Command, dBFS Audio Meter & Clinical Log Console.

Provides half-duplex status badge, segmented ASR mode radio pills, custom-painted dBFS
threshold meter, live transcript chip, and Escape-aware QPlainTextEdit surgical log.
"""
from __future__ import annotations

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QKeyEvent, QPainter, QPen
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .theme import (
    BG_CARD,
    BG_INPUT,
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


class AudioMeterBar(QWidget):
    """Custom-painted dBFS level meter with green/amber/red zones and yellow threshold line."""

    def __init__(self, min_db: float = -80.0, max_db: float = 0.0, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._min_db = float(min_db)
        self._max_db = float(max_db)
        self._dbfs = -60.0
        self._thresh_db = -44.0
        self.setFixedHeight(16)
        self.setMinimumWidth(110)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_levels(self, dbfs: float, threshold_db: float) -> None:
        """Updates current dBFS loudness and noise-floor trigger threshold."""
        new_db = max(self._min_db, min(self._max_db, float(dbfs)))
        new_th = max(self._min_db, min(self._max_db, float(threshold_db)))
        if abs(new_db - self._dbfs) < 0.2 and abs(new_th - self._thresh_db) < 0.2:
            return
        self._dbfs = new_db
        self._thresh_db = new_th
        self.update()

    def _db_to_frac(self, db: float) -> float:
        span = max(1.0, self._max_db - self._min_db)
        return max(0.0, min(1.0, (db - self._min_db) / span))

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        w = float(self.width())
        h = float(self.height())

        # Matte trough
        painter.fillRect(0, 0, int(w), int(h), QColor(BG_INPUT))
        painter.setPen(QPen(QColor(BORDER_HAIRLINE), 1))
        painter.drawRect(0, 0, int(w) - 1, int(h) - 1)

        inner_w = max(1.0, w - 4.0)
        level_frac = self._db_to_frac(self._dbfs)
        fill_px = level_frac * inner_w

        # Zone breakpoints: Green [-80..-18], Amber [-18..-6], Red [-6..0]
        amber_start = self._db_to_frac(-18.0) * inner_w
        red_start = self._db_to_frac(-6.0) * inner_w

        zones = (
            (0.0, min(fill_px, amber_start), COLOR_NOMINAL),
            (amber_start, min(fill_px, red_start), COLOR_AMBER),
            (red_start, fill_px, COLOR_DANGER),
        )
        for z0, z1, col in zones:
            if z1 > z0:
                painter.fillRect(QRectF(2.0 + z0, 3.0, z1 - z0, h - 6.0), QColor(col))

        # Crisp vertical yellow threshold marker line
        th_x = 2.0 + self._db_to_frac(self._thresh_db) * inner_w
        painter.setPen(QPen(QColor(COLOR_AMBER), 2))
        painter.drawLine(int(round(th_x)), 1, int(round(th_x)), int(h) - 2)


class _SurgicalLogEdit(QPlainTextEdit):
    """QPlainTextEdit that yields focus back to workstation shortcuts when Escape is pressed."""

    escape_pressed = Signal()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.clearFocus()
            win = self.window()
            if win is not None:
                win.setFocus(Qt.FocusReason.ShortcutFocusReason)
            self.escape_pressed.emit()
            event.accept()
            return
        super().keyPressEvent(event)


class SpeechConsolePanel(QFrame):
    """Voice command controls, ASR mode pills, dBFS meter, transcript chip, and surgical log."""

    asr_mode_changed = Signal(str)
    mic_toggled = Signal()
    calibrate_requested = Signal()
    speak_log_requested = Signal()
    export_log_requested = Signal()
    escape_pressed = Signal()

    _ASR_MODES = (
        ("ID_ONLINE", "ID Online"),
        ("EN_OFFLINE", "EN Offline"),
        ("DICTATION_ID", "Dictation ID"),
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")

        self._last_hd_state: str | None = None
        self._last_dbfs_key: tuple[int, int] | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(7)

        # Header Row
        hdr = QHBoxLayout()
        hdr.setSpacing(6)
        title = QLabel("VOICE COMMAND & SURGICAL LOG")
        title.setFont(ui_font(10, bold=True))
        title.setStyleSheet(f"color: {TEXT_PRIMARY}; letter-spacing: 0.6px;")
        hdr.addWidget(title)
        hdr.addStretch(1)

        self.hd_badge = QLabel("[LISTENING]")
        self.hd_badge.setFont(tabular_font(8, bold=True))
        hdr.addWidget(self.hd_badge)
        root.addLayout(hdr)

        # Segmented ASR Mode Radio Pills
        mode_row = QHBoxLayout()
        mode_row.setSpacing(4)
        mode_lbl = QLabel("ASR")
        mode_lbl.setFont(tabular_font(8, bold=True))
        mode_lbl.setStyleSheet(f"color: {TEXT_MUTED};")
        mode_row.addWidget(mode_lbl)

        self._asr_buttons: dict[str, QPushButton] = {}
        for mode_id, label in self._ASR_MODES:
            btn = QPushButton(label)
            btn.setFont(ui_font(8, bold=True))
            btn.clicked.connect(lambda _=False, m=mode_id: self._on_mode_clicked(m))
            self._asr_buttons[mode_id] = btn
            mode_row.addWidget(btn, stretch=1)
        root.addLayout(mode_row)

        # Mic Toggle + Calibrate + AudioMeterBar + Tabular Readouts
        meter_row = QHBoxLayout()
        meter_row.setSpacing(5)

        self.mic_btn = QPushButton("Mic ON")
        self.mic_btn.setFont(tabular_font(8, bold=True))
        self.mic_btn.setProperty("nominal", "true")
        repolish(self.mic_btn)
        self.mic_btn.clicked.connect(self.mic_toggled.emit)
        meter_row.addWidget(self.mic_btn)

        self.calibrate_btn = QPushButton("Calibrate (2s)")
        self.calibrate_btn.setFont(ui_font(8, bold=True))
        self.calibrate_btn.clicked.connect(self.calibrate_requested.emit)
        meter_row.addWidget(self.calibrate_btn)

        self.meter_bar = AudioMeterBar()
        meter_row.addWidget(self.meter_bar, stretch=1)

        self.dbfs_lbl = QLabel("-60.0 dBFS")
        self.dbfs_lbl.setFont(tabular_font(8, bold=True))
        self.dbfs_lbl.setStyleSheet(f"color: {TEXT_MUTED};")
        self.dbfs_lbl.setMinimumWidth(74)
        self.dbfs_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        meter_row.addWidget(self.dbfs_lbl)

        self.thresh_lbl = QLabel("Thresh: -44 dB")
        self.thresh_lbl.setFont(tabular_font(8, bold=True))
        self.thresh_lbl.setStyleSheet(f"color: {COLOR_AMBER};")
        self.thresh_lbl.setMinimumWidth(92)
        self.thresh_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        meter_row.addWidget(self.thresh_lbl)
        root.addLayout(meter_row)

        # Live Heard Transcript Chip
        self.heard_lbl = QLabel('HEARD: Ready for commands ("iris berikutnya", "bookmark", "vitals")')
        self.heard_lbl.setFont(tabular_font(8, bold=False))
        self.heard_lbl.setStyleSheet(
            f"background-color: {BG_CARD}; color: {COLOR_CYAN}; "
            f"border: 1px solid {BORDER_HAIRLINE}; border-radius: 4px; padding: 4px 8px;"
        )
        root.addWidget(self.heard_lbl)

        # Log Header + Actions
        log_hdr = QHBoxLayout()
        log_hdr.setSpacing(6)
        log_title = QLabel("INTRA-OPERATIVE CLINICAL LOG (Esc = Exit Focus)")
        log_title.setFont(tabular_font(8, bold=True))
        log_title.setStyleSheet(f"color: {TEXT_MUTED};")
        log_hdr.addWidget(log_title)
        log_hdr.addStretch(1)

        self.speak_log_btn = QPushButton("Speak Log")
        self.speak_log_btn.setFont(ui_font(8, bold=True))
        self.speak_log_btn.setProperty("active", "true")
        repolish(self.speak_log_btn)
        self.speak_log_btn.clicked.connect(self.speak_log_requested.emit)
        log_hdr.addWidget(self.speak_log_btn)

        self.export_btn = QPushButton("Export (.txt)")
        self.export_btn.setFont(ui_font(8, bold=True))
        self.export_btn.clicked.connect(self.export_log_requested.emit)
        log_hdr.addWidget(self.export_btn)
        root.addLayout(log_hdr)

        # Clinical Surgical Log Editor
        self.log_edit = _SurgicalLogEdit()
        self.log_edit.setFont(tabular_font(8, bold=False))
        self.log_edit.setMinimumHeight(110)
        self.log_edit.escape_pressed.connect(self.escape_pressed.emit)
        root.addWidget(self.log_edit, stretch=1)

        self.set_asr_mode("ID_ONLINE")
        self.set_half_duplex_state("LISTENING")
        self.set_audio_level(-60.0, -44.0)

    def _on_mode_clicked(self, mode_id: str) -> None:
        self.set_asr_mode(mode_id)
        self.asr_mode_changed.emit(mode_id)

    def set_asr_mode(self, mode_id: str) -> None:
        """Highlights the active ASR mode radio pill."""
        norm = mode_id.strip().upper()
        for key, btn in self._asr_buttons.items():
            btn.setProperty("active", "true" if key == norm else "false")
            repolish(btn)

    def set_half_duplex_state(self, state: str) -> None:
        """Updates the half-duplex badge ([LISTENING], [TTS SPEAKING], [MIC OFF])."""
        upper = state.strip().upper().strip("[]")
        if upper == self._last_hd_state:
            return
        self._last_hd_state = upper

        if "TTS" in upper or "SPEAK" in upper:
            text, color, bg = "[TTS SPEAKING]", COLOR_AMBER, "rgba(255, 179, 0, 0.14)"
        elif "OFF" in upper or "MUTED" in upper:
            text, color, bg = "[MIC OFF]", COLOR_DANGER, "rgba(255, 59, 48, 0.14)"
        else:
            text, color, bg = "[LISTENING]", COLOR_NOMINAL, "rgba(0, 230, 118, 0.12)"

        self.hd_badge.setText(text)
        self.hd_badge.setStyleSheet(
            f"color: {color}; background-color: {bg}; "
            f"border: 1px solid {color}; border-radius: 3px; padding: 2px 7px;"
        )

    def set_mic_enabled(self, enabled: bool) -> None:
        """Updates mic toggle button text and semantic style."""
        if enabled:
            self.mic_btn.setText("Mic ON")
            self.mic_btn.setProperty("nominal", "true")
            self.mic_btn.setProperty("danger", "false")
        else:
            self.mic_btn.setText("Mic OFF")
            self.mic_btn.setProperty("nominal", "false")
            self.mic_btn.setProperty("danger", "true")
        repolish(self.mic_btn)

    def set_audio_level(self, dbfs: float, thresh_db: float) -> None:
        """Updates AudioMeterBar and tabular dBFS / threshold readouts."""
        self.meter_bar.set_levels(dbfs, thresh_db)
        key = (int(round(float(dbfs) * 2.0)), int(round(float(thresh_db))))
        if key == self._last_dbfs_key:
            return
        self._last_dbfs_key = key
        self.dbfs_lbl.setText(f"{float(dbfs):6.1f} dBFS")
        self.thresh_lbl.setText(f"Thresh: {int(round(float(thresh_db))):d} dB")

    def set_heard_text(self, text: str) -> None:
        """Updates the live heard transcript chip."""
        self.heard_lbl.setText(f"HEARD: {text}")

    def append_log(self, line: str) -> None:
        """Appends a clinical log line and keeps cursor scrolled to bottom."""
        self.log_edit.appendPlainText(line)

    def set_log_text(self, text: str) -> None:
        """Replaces the surgical log text cleanly."""
        self.log_edit.setPlainText(text)

    def get_log_text(self) -> str:
        """Returns full surgical log text."""
        return self.log_edit.toPlainText()
