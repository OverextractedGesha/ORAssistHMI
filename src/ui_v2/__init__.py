"""ui_v2 - OR Obsidian Medical Workstation PySide6 UI Component Package."""

from .dicom_viewport import CTPanel, DicomViewportWidget, SliceRailWidget
from .speech_console import AudioMeterBar, SpeechConsolePanel
from .theme import (
    APP_QSS,
    BG_CARD,
    BG_HOVER,
    BG_INPUT,
    BG_PANEL,
    BG_ROOT,
    BG_VIEWPORT,
    BORDER_HAIRLINE,
    BORDER_SUBTLE,
    COLOR_AMBER,
    COLOR_CYAN,
    COLOR_DANGER,
    COLOR_NOMINAL,
    TEXT_DIM,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    repolish,
    tabular_font,
    ui_font,
)
from .vision_hud import VisionHudPanel
from .waveform_monitor import VitalsMonitorPanel, WaveformTraceWidget

__all__ = [
    "APP_QSS",
    "AudioMeterBar",
    "BG_CARD",
    "BG_HOVER",
    "BG_INPUT",
    "BG_PANEL",
    "BG_ROOT",
    "BG_VIEWPORT",
    "BORDER_HAIRLINE",
    "BORDER_SUBTLE",
    "COLOR_AMBER",
    "COLOR_CYAN",
    "COLOR_DANGER",
    "COLOR_NOMINAL",
    "CTPanel",
    "DicomViewportWidget",
    "SliceRailWidget",
    "SpeechConsolePanel",
    "TEXT_DIM",
    "TEXT_MUTED",
    "TEXT_PRIMARY",
    "TEXT_SECONDARY",
    "VisionHudPanel",
    "VitalsMonitorPanel",
    "WaveformTraceWidget",
    "repolish",
    "tabular_font",
    "ui_font",
]
