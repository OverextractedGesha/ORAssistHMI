"""waveform_monitor.py - 60 FPS ICU Sweeping Waveforms & IEC 60601-1-8 Vitals Monitor Panel.

Implements Philips IntelliVue-inspired ECG Lead II and Arterial Plethysmograph traces
with phosphor erase-bar gap, tabular clinical parameter blocks, and IEC alarm styling.
"""
from __future__ import annotations

import math

from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
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


def _spo2_pitch_hz(spo2: float) -> float:
    clamped = max(50.0, min(100.0, float(spo2)))
    return 880.0 * (2.0 ** (-(100.0 - clamped) / 12.0))


class WaveformTraceWidget(QWidget):
    """60 FPS ICU sweeping oscilloscope trace with erase-bar gap and glow head."""

    def __init__(self, mode: str = "ecg", label: str | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.mode = mode.lower().strip()
        if self.mode not in ("ecg", "pleth"):
            self.mode = "ecg"

        self.label = label or ("ECG II" if self.mode == "ecg" else "PLETH SpO2")
        self.trace_color = COLOR_NOMINAL if self.mode == "ecg" else COLOR_CYAN
        self.buffer_len = 300
        self.erase_gap = 14
        self.samples: list[float | None] = [None] * self.buffer_len
        self.sweep_idx: int = 0
        self._phase: float = 0.0

        self.setMinimumHeight(86)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    @staticmethod
    def _gauss(phase: float, center: float, width: float) -> float:
        d = (phase - center) / width
        return math.exp(-0.5 * d * d)

    def _synthesize_sample(self, phase: float, spo2_pct: float) -> float:
        if self.mode == "ecg":
            # Realistic P-Q-R-S-T complex normalized roughly in [-0.30, +1.05]
            p = 0.14 * self._gauss(phase, 0.16, 0.032)
            q = -0.14 * self._gauss(phase, 0.265, 0.011)
            r = 1.00 * self._gauss(phase, 0.295, 0.013)
            s = -0.26 * self._gauss(phase, 0.325, 0.013)
            t = 0.28 * self._gauss(phase, 0.48, 0.055)
            return p + q + r + s + t

        # Arterial plethysmogram: systolic upstroke + dicrotic notch + diastolic wave
        sys_peak = 0.88 * self._gauss(phase, 0.25, 0.068)
        diastolic = 0.36 * self._gauss(phase, 0.48, 0.095)
        notch = -0.12 * self._gauss(phase, 0.365, 0.025)
        amp_scale = max(0.35, min(1.0, (float(spo2_pct) - 60.0) / 40.0))
        return max(0.0, (sys_peak + diastolic + notch) * amp_scale)

    def step_sample(self, dt_sec: float, hr_bpm: float, spo2_pct: float) -> None:
        """Advances cardiac phase and writes next sample into the circular sweep buffer."""
        dt = max(0.001, min(0.1, float(dt_sec)))
        bpm = max(30.0, min(220.0, float(hr_bpm)))
        self._phase = (self._phase + dt * (bpm / 60.0)) % 1.0

        val = self._synthesize_sample(self._phase, spo2_pct)
        self.samples[self.sweep_idx] = val

        # Maintain erase-bar gap ahead of sweep head
        for offset in range(1, self.erase_gap + 1):
            clear_idx = (self.sweep_idx + offset) % self.buffer_len
            self.samples[clear_idx] = None

        self._last_head_idx = self.sweep_idx
        self.sweep_idx = (self.sweep_idx + 1) % self.buffer_len
        self.update()

    def _val_to_y(self, val: float, h: float) -> float:
        if self.mode == "ecg":
            # Map [-0.35, +1.10] -> [h - 10, 12]
            norm = (val + 0.35) / 1.45
        else:
            # Map [-0.05, +1.00] -> [h - 10, 14]
            norm = (val + 0.05) / 1.05
        norm = max(0.0, min(1.0, norm))
        return (h - 10.0) - norm * max(1.0, h - 22.0)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        w = float(self.width())
        h = float(self.height())

        # Matte zero-luminance oscilloscope canvas
        painter.fillRect(0, 0, int(w), int(h), QColor(BG_VIEWPORT))

        # Subtle oscilloscope grid
        grid_pen = QPen(QColor("#0F1822"), 1)
        painter.setPen(grid_pen)
        for gx in range(25, int(w), 25):
            painter.drawLine(gx, 0, gx, int(h))
        for gy in range(20, int(h), 20):
            painter.drawLine(0, gy, int(w), gy)

        # Hairline frame
        painter.setPen(QPen(QColor(BORDER_HAIRLINE), 1))
        painter.drawRect(0, 0, int(w) - 1, int(h) - 1)

        # Trace path split across erase-bar gap
        base_color = QColor(self.trace_color)
        trace_pen = QPen(base_color, 1.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        painter.setPen(trace_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        path = QPainterPath()
        drawing = False
        dx = w / max(1, self.buffer_len - 1)

        for i, val in enumerate(self.samples):
            if val is None:
                if drawing:
                    painter.drawPath(path)
                    path = QPainterPath()
                    drawing = False
                continue
            px = i * dx
            py = self._val_to_y(val, h)
            if not drawing:
                path.moveTo(px, py)
                drawing = True
            else:
                path.lineTo(px, py)

        if drawing:
            painter.drawPath(path)

        # Sweep head glow dot
        head_idx = (self.sweep_idx - 1) % self.buffer_len
        head_val = self.samples[head_idx]
        if head_val is not None:
            hx = head_idx * dx
            hy = self._val_to_y(head_val, h)
            glow = QColor(base_color)
            glow.setAlpha(85)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(glow)
            painter.drawEllipse(QPointF(hx, hy), 5.0, 5.0)
            painter.setBrush(QColor("#FFFFFF"))
            painter.drawEllipse(QPointF(hx, hy), 2.0, 2.0)

        # Lead label overlay
        painter.setFont(tabular_font(8, bold=True))
        painter.setPen(base_color)
        painter.drawText(8, 15, self.label)


class _ParamCard(QFrame):
    """Single high-contrast clinical parameter block (HR, SpO2, NIBP, TEMP)."""

    def __init__(
        self,
        title: str,
        unit: str,
        limits: str,
        accent_hex: str,
        value_pt: int = 24,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("Card")
        self._accent = accent_hex
        self._value_pt = int(value_pt)
        self.setMinimumHeight(92)
        self.setMinimumWidth(154)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(2)

        top = QHBoxLayout()
        top.setSpacing(4)
        self.title_lbl = QLabel(title)
        self.title_lbl.setFont(tabular_font(8, bold=True))
        self.title_lbl.setStyleSheet(f"color: {accent_hex}; font-size: 9pt; font-weight: 700;")
        top.addWidget(self.title_lbl)

        top.addStretch(1)

        self.limits_lbl = QLabel(limits)
        self.limits_lbl.setFont(tabular_font(8, bold=False))
        self.limits_lbl.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 8pt;")
        top.addWidget(self.limits_lbl)
        layout.addLayout(top)

        val_row = QHBoxLayout()
        val_row.setSpacing(4)
        self.value_lbl = QLabel("--")
        self.value_lbl.setFont(tabular_font(self._value_pt, bold=True))
        self.value_lbl.setStyleSheet(
            f"color: {accent_hex}; font-family: 'Cascadia Mono', 'Consolas', monospace; "
            f"font-size: {self._value_pt}pt; font-weight: 700;"
        )
        val_row.addWidget(self.value_lbl)

        self.unit_lbl = QLabel(unit)
        self.unit_lbl.setFont(tabular_font(8, bold=True))
        self.unit_lbl.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 8pt; font-weight: 600; padding-bottom: 3px;")
        self.unit_lbl.setAlignment(Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft)
        val_row.addWidget(self.unit_lbl)
        val_row.addStretch(1)
        layout.addLayout(val_row)

        self.sub_lbl = QLabel("")
        self.sub_lbl.setFont(tabular_font(8, bold=False))
        self.sub_lbl.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 8pt;")
        self.sub_lbl.setVisible(False)
        layout.addWidget(self.sub_lbl)

    def set_subtext(self, text: str) -> None:
        if text:
            self.sub_lbl.setText(text)
            self.sub_lbl.setVisible(True)
        else:
            self.sub_lbl.setVisible(False)

    def set_alarm_border(self, color_hex: str | None) -> None:
        if color_hex:
            self.setStyleSheet(f"QFrame#Card {{ background-color: {BG_CARD}; border: 1px solid {color_hex}; }}")
        else:
            self.setStyleSheet("")


class VitalsMonitorPanel(QFrame):
    """Intra-operative patient monitor with 60 FPS waveforms, IEC 60601-1-8 alarms, and controls."""

    silence_requested = Signal()
    scenario_requested = Signal(str)
    sonification_toggled = Signal(bool)
    alarm_test_requested = Signal(str)
    speak_vitals_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self._last_state_key: tuple | None = None
        self._last_hr: float = 74.0
        self._last_spo2: float = 98.0

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # Header Row
        hdr = QHBoxLayout()
        hdr.setSpacing(6)
        title = QLabel("INTRA-OPERATIVE PATIENT MONITOR")
        title.setFont(ui_font(10, bold=True))
        title.setStyleSheet(f"color: {TEXT_PRIMARY}; letter-spacing: 0.6px;")
        hdr.addWidget(title)

        self.priority_banner = QLabel("[NOMINAL]")
        self.priority_banner.setFont(tabular_font(8, bold=True))
        hdr.addWidget(self.priority_banner)
        hdr.addStretch(1)

        self.mute_timer_lbl = QLabel("MUTE: --")
        self.mute_timer_lbl.setFont(tabular_font(8, bold=True))
        self.mute_timer_lbl.setStyleSheet(f"color: {TEXT_MUTED};")
        hdr.addWidget(self.mute_timer_lbl)

        self.silence_btn = QPushButton("[MUTE 120s]")
        self.silence_btn.setFont(tabular_font(8, bold=True))
        self.silence_btn.setProperty("warn", "true")
        repolish(self.silence_btn)
        self.silence_btn.clicked.connect(self.silence_requested.emit)
        hdr.addWidget(self.silence_btn)
        root.addLayout(hdr)

        # Split Grid: Waveforms Left, Parameter Blocks Right
        split = QHBoxLayout()
        split.setSpacing(8)

        waves_col = QVBoxLayout()
        waves_col.setSpacing(6)
        self.ecg_trace = WaveformTraceWidget(mode="ecg", label="ECG II")
        self.pleth_trace = WaveformTraceWidget(mode="pleth", label="PLETH SpO2")
        waves_col.addWidget(self.ecg_trace, stretch=1)
        waves_col.addWidget(self.pleth_trace, stretch=1)
        split.addLayout(waves_col, stretch=4)

        params_grid = QGridLayout()
        params_grid.setSpacing(6)
        self.hr_card = _ParamCard("HR", "bpm", "50-120", COLOR_NOMINAL, value_pt=24)
        self.spo2_card = _ParamCard("SpO2", "%", "90-100", COLOR_CYAN, value_pt=24)
        self.nibp_card = _ParamCard("NIBP", "mmHg", "90-140", TEXT_PRIMARY, value_pt=19)
        self.temp_card = _ParamCard("TEMP", "\u00b0C", "36.0-38.0", COLOR_AMBER, value_pt=22)

        params_grid.addWidget(self.hr_card, 0, 0)
        params_grid.addWidget(self.spo2_card, 0, 1)
        params_grid.addWidget(self.nibp_card, 1, 0)
        params_grid.addWidget(self.temp_card, 1, 1)
        split.addLayout(params_grid, stretch=5)

        root.addLayout(split, stretch=1)

        # Footer Row 1: Scenarios + Speak Vitals
        foot1 = QHBoxLayout()
        foot1.setSpacing(6)

        self._scenario_buttons: dict[str, QPushButton] = {}
        for key, label in (
            ("normal", "Normal 98%"),
            ("mild", "Mild Hypoxia 91%"),
            ("critical", "Critical Desat 84%"),
        ):
            btn = QPushButton(label)
            btn.setFont(ui_font(8, bold=True))
            btn.clicked.connect(lambda _=False, k=key: self.scenario_requested.emit(k))
            self._scenario_buttons[key] = btn
            foot1.addWidget(btn)

        foot1.addStretch(1)

        self.speak_vitals_btn = QPushButton("Speak Vitals (V)")
        self.speak_vitals_btn.setFont(ui_font(8, bold=True))
        self.speak_vitals_btn.setProperty("active", "true")
        repolish(self.speak_vitals_btn)
        self.speak_vitals_btn.clicked.connect(self.speak_vitals_requested.emit)
        foot1.addWidget(self.speak_vitals_btn)
        root.addLayout(foot1)

        # Footer Row 2: IEC 60601-1-8 Test Buttons + SpO2 Sonification
        foot2 = QHBoxLayout()
        foot2.setSpacing(6)
        iec_lbl = QLabel("IEC 60601-1-8 TEST:")
        iec_lbl.setFont(tabular_font(8, bold=True))
        iec_lbl.setStyleSheet(f"color: {TEXT_MUTED};")
        foot2.addWidget(iec_lbl)

        for sev, text, prop in (
            ("low", "Low", "active"),
            ("medium", "Med", "warn"),
            ("high", "High", "danger"),
        ):
            btn = QPushButton(text)
            btn.setFont(tabular_font(8, bold=True))
            btn.setProperty(prop, "true")
            repolish(btn)
            btn.clicked.connect(lambda _=False, s=sev: self.alarm_test_requested.emit(s))
            foot2.addWidget(btn)

        foot2.addStretch(1)

        self.sonification_chk = QCheckBox("SpO2 Sonification")
        self.sonification_chk.setFont(ui_font(8, bold=False))
        self.sonification_chk.toggled.connect(self.sonification_toggled.emit)
        foot2.addWidget(self.sonification_chk)
        root.addLayout(foot2)

        self.update_vitals(
            {"hr": 74, "spo2": 98, "bp_sys": 118, "bp_dia": 76, "temp": 36.8, "alarm_severity": "none"},
            silence_remaining_sec=0.0,
            flash_phase=False,
        )

    def step_waveforms(self, dt_sec: float) -> None:
        """Convenience 60 FPS tick updating both ECG II and PLETH SpO2 waveforms."""
        self.ecg_trace.step_sample(dt_sec, self._last_hr, self._last_spo2)
        self.pleth_trace.step_sample(dt_sec, self._last_hr, self._last_spo2)

    def update_vitals(
        self,
        vitals_dict: dict,
        silence_remaining_sec: float = 0.0,
        flash_phase: bool = False,
    ) -> None:
        """Updates numeric parameter cards, mute countdown, and IEC alarm styling with change caching."""
        hr = int(round(float(vitals_dict.get("hr", 74))))
        spo2 = int(round(float(vitals_dict.get("spo2", 98))))
        bp_sys = int(round(float(vitals_dict.get("bp_sys", 118))))
        bp_dia = int(round(float(vitals_dict.get("bp_dia", 76))))
        temp = round(float(vitals_dict.get("temp", 36.8)), 1)
        severity = str(vitals_dict.get("alarm_severity", "none")).lower().strip()

        self._last_hr = float(hr)
        self._last_spo2 = float(spo2)

        rem_sec = max(0, int(math.ceil(float(silence_remaining_sec))))
        flash = bool(flash_phase)

        state_key = (hr, spo2, bp_sys, bp_dia, temp, severity, rem_sec, flash)
        if state_key == self._last_state_key:
            return
        self._last_state_key = state_key

        # Numeric Cards
        self.hr_card.value_lbl.setText(f"{hr:d}")
        self.spo2_card.value_lbl.setText(f"{spo2:d}")
        self.spo2_card.set_subtext(f"Pitch: {_spo2_pitch_hz(spo2):.0f} Hz")
        self.nibp_card.value_lbl.setText(f"{bp_sys}/{bp_dia}")
        map_val = int(round(bp_dia + (bp_sys - bp_dia) / 3.0))
        self.nibp_card.set_subtext(f"MAP: {map_val} mmHg")
        self.temp_card.value_lbl.setText(f"{temp:.1f}")

        # Mute countdown badge
        if rem_sec > 0:
            mins, secs = divmod(rem_sec, 60)
            self.mute_timer_lbl.setText(f"MUTE: {mins:02d}:{secs:02d}")
            self.mute_timer_lbl.setStyleSheet(f"color: {COLOR_AMBER};")
        else:
            self.mute_timer_lbl.setText("MUTE: --")
            self.mute_timer_lbl.setStyleSheet(f"color: {TEXT_MUTED};")

        # IEC 60601-1-8 Priority Banner & Card Alarm Borders
        if severity == "high":
            banner_txt = "[HIGH PRIORITY: DESATURATION]"
            fg = "#080C10" if flash else COLOR_DANGER
            bg = COLOR_DANGER if flash else "rgba(255, 59, 48, 0.16)"
            border = COLOR_DANGER
            self.spo2_card.set_alarm_border(COLOR_DANGER if flash else None)
            self.hr_card.set_alarm_border(None)
        elif severity == "medium":
            banner_txt = "[MED PRIORITY ALARM]"
            fg = "#080C10" if flash else COLOR_AMBER
            bg = COLOR_AMBER if flash else "rgba(255, 179, 0, 0.16)"
            border = COLOR_AMBER
            self.spo2_card.set_alarm_border(COLOR_AMBER if flash else None)
            self.hr_card.set_alarm_border(None)
        elif severity == "low":
            banner_txt = "[LOW PRIORITY ALARM]"
            fg = COLOR_CYAN
            bg = "rgba(0, 229, 255, 0.12)"
            border = COLOR_CYAN
            self.spo2_card.set_alarm_border(None)
            self.hr_card.set_alarm_border(None)
        else:
            banner_txt = "[NOMINAL]"
            fg = COLOR_NOMINAL
            bg = "rgba(0, 230, 118, 0.12)"
            border = COLOR_NOMINAL
            self.spo2_card.set_alarm_border(None)
            self.hr_card.set_alarm_border(None)

        self.priority_banner.setText(banner_txt)
        self.priority_banner.setStyleSheet(
            f"color: {fg}; background-color: {bg}; "
            f"border: 1px solid {border}; border-radius: 3px; padding: 2px 7px;"
        )
