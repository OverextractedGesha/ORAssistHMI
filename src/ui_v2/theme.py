"""theme.py - 'OR Obsidian' Medical Workstation Design System for PySide6.

Ultra-clean dark matte surfaces, strict clinical semantic palette (Philips IntelliVue +
Brainlab/OHIF PACS inspired), tabular monospace font helpers, and flat hairline QSS.
"""
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QWidget

# ---------------------------------------------------------------------------
# Matte Surface Palette
# ---------------------------------------------------------------------------
BG_ROOT = "#080C10"
BG_PANEL = "#0E141C"
BG_CARD = "#141C27"
BG_INPUT = "#0A1017"
BG_HOVER = "#1A2534"
BG_VIEWPORT = "#000000"

BORDER_HAIRLINE = "#1C2634"
BORDER_SUBTLE = "#283648"

TEXT_PRIMARY = "#E6EDF5"
TEXT_SECONDARY = "#B0BEC5"
TEXT_MUTED = "#7D8B9D"
TEXT_DIM = "#4D5B6D"

# ---------------------------------------------------------------------------
# Strict Clinical Semantic Colors ONLY
# ---------------------------------------------------------------------------
COLOR_NOMINAL = "#00E676"   # ECG / Nominal / Active
COLOR_CYAN = "#00E5FF"      # SpO2 / Navigation / Laser Pointer
COLOR_AMBER = "#FFB300"     # Warning / Medium Alarm / Bookmark / Dwell
COLOR_DANGER = "#FF3B30"    # High Priority Alarm / Lock / Critical


def tabular_font(size: int = 10, bold: bool = False) -> QFont:
    """Returns a tabular monospace QFont (Cascadia Mono / Consolas) so numerals never jump."""
    font = QFont("Consolas", size)
    font.setFamilies(["Cascadia Mono", "Consolas", "Courier New"])
    font.setStyleHint(QFont.StyleHint.Monospace)
    font.setFixedPitch(True)
    font.setBold(bold)
    return font


def ui_font(size: int = 9, bold: bool = False) -> QFont:
    """Returns a crisp clinical sans-serif QFont for static UI labels."""
    font = QFont("Segoe UI", size)
    font.setFamilies(["Segoe UI", "Inter", "Arial"])
    font.setBold(bold)
    return font


def repolish(widget: QWidget) -> None:
    """Forces Qt stylesheet re-evaluation after dynamic property changes."""
    style = widget.style()
    if style is not None:
        style.unpolish(widget)
        style.polish(widget)
    widget.update()


# ---------------------------------------------------------------------------
# Master PySide6 Stylesheet ("OR Obsidian")
# ---------------------------------------------------------------------------
APP_QSS = f"""
QMainWindow {{
    background-color: {BG_ROOT};
    color: {TEXT_PRIMARY};
}}

QWidget {{
    color: {TEXT_PRIMARY};
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 9pt;
}}

QLabel {{
    background-color: transparent;
    border: none;
}}

QFrame#Panel {{
    background-color: {BG_PANEL};
    border: 1px solid {BORDER_HAIRLINE};
    border-radius: 6px;
}}

QFrame#Card {{
    background-color: {BG_CARD};
    border: 1px solid {BORDER_HAIRLINE};
    border-radius: 5px;
}}

QPushButton {{
    background-color: {BG_CARD};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_HAIRLINE};
    border-radius: 4px;
    padding: 5px 10px;
    font-weight: 600;
    min-height: 20px;
}}

QPushButton:hover {{
    background-color: {BG_HOVER};
    border-color: {BORDER_SUBTLE};
}}

QPushButton:pressed {{
    background-color: {BG_INPUT};
}}

QPushButton:disabled {{
    background-color: {BG_INPUT};
    color: {TEXT_DIM};
    border-color: {BORDER_HAIRLINE};
}}

QPushButton[active="true"] {{
    background-color: rgba(0, 229, 255, 0.14);
    color: {COLOR_CYAN};
    border: 1px solid {COLOR_CYAN};
}}

QPushButton[active="true"]:hover {{
    background-color: rgba(0, 229, 255, 0.22);
}}

QPushButton[danger="true"] {{
    background-color: rgba(255, 59, 48, 0.15);
    color: {COLOR_DANGER};
    border: 1px solid {COLOR_DANGER};
}}

QPushButton[danger="true"]:hover {{
    background-color: rgba(255, 59, 48, 0.24);
}}

QPushButton[warn="true"] {{
    background-color: rgba(255, 179, 0, 0.14);
    color: {COLOR_AMBER};
    border: 1px solid {COLOR_AMBER};
}}

QPushButton[warn="true"]:hover {{
    background-color: rgba(255, 179, 0, 0.22);
}}

QPushButton[nominal="true"] {{
    background-color: rgba(0, 230, 118, 0.14);
    color: {COLOR_NOMINAL};
    border: 1px solid {COLOR_NOMINAL};
}}

QPushButton[nominal="true"]:hover {{
    background-color: rgba(0, 230, 118, 0.22);
}}

QCheckBox {{
    spacing: 6px;
    color: {TEXT_PRIMARY};
    background-color: transparent;
}}

QCheckBox::indicator {{
    width: 14px;
    height: 14px;
    border-radius: 3px;
    border: 1px solid {BORDER_SUBTLE};
    background-color: {BG_INPUT};
}}

QCheckBox::indicator:hover {{
    border-color: {COLOR_CYAN};
}}

QCheckBox::indicator:checked {{
    background-color: {COLOR_CYAN};
    border-color: {COLOR_CYAN};
}}

QPlainTextEdit {{
    background-color: {BG_INPUT};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_HAIRLINE};
    border-radius: 4px;
    padding: 6px;
    selection-background-color: #163A4A;
    selection-color: {TEXT_PRIMARY};
}}

QPlainTextEdit:focus {{
    border-color: {COLOR_CYAN};
}}

QSlider::groove:horizontal {{
    height: 4px;
    background-color: {BORDER_HAIRLINE};
    border-radius: 2px;
}}

QSlider::sub-page:horizontal {{
    background-color: {COLOR_CYAN};
    border-radius: 2px;
}}

QSlider::handle:horizontal {{
    background-color: {COLOR_CYAN};
    border: 1px solid {BG_ROOT};
    width: 12px;
    height: 12px;
    margin: -4px 0;
    border-radius: 6px;
}}

QSlider::groove:vertical {{
    width: 4px;
    background-color: {BORDER_HAIRLINE};
    border-radius: 2px;
}}

QSlider::sub-page:vertical {{
    background-color: {COLOR_CYAN};
    border-radius: 2px;
}}

QSlider::handle:vertical {{
    background-color: {COLOR_CYAN};
    border: 1px solid {BG_ROOT};
    width: 12px;
    height: 12px;
    margin: 0 -4px;
    border-radius: 6px;
}}

QProgressBar {{
    background-color: {BG_INPUT};
    border: 1px solid {BORDER_HAIRLINE};
    border-radius: 3px;
    color: {TEXT_PRIMARY};
    text-align: center;
    min-height: 10px;
    max-height: 14px;
}}

QProgressBar::chunk {{
    background-color: {COLOR_CYAN};
    border-radius: 2px;
}}

QScrollBar:vertical {{
    background: {BG_INPUT};
    width: 8px;
    margin: 0;
}}

QScrollBar::handle:vertical {{
    background: {BORDER_SUBTLE};
    min-height: 20px;
    border-radius: 4px;
}}

QScrollBar::handle:vertical:hover {{
    background: {TEXT_MUTED};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}
"""
