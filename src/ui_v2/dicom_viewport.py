"""dicom_viewport.py - PACS-grade CT DICOM Viewport, Slice Scrubber Rail & Navigation Panel.

Implements 'OR Obsidian' zero-luminance CT rendering with calibrated millimeter scale bar,
anatomical orientation tags, surgical ROI reticles, touchless dwell laser reticle, and
segmented windowing controls.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
from PySide6.QtCore import QPointF, QRect, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFontMetrics, QImage, QPainter, QPen
from PySide6.QtWidgets import (
    QFrame,
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
    BG_PANEL,
    BG_VIEWPORT,
    BORDER_HAIRLINE,
    COLOR_AMBER,
    COLOR_CYAN,
    COLOR_DANGER,
    COLOR_NOMINAL,
    TEXT_DIM,
    TEXT_MUTED,
    TEXT_PRIMARY,
    repolish,
    tabular_font,
    ui_font,
)

_UNSET = object()


class SliceRailWidget(QWidget):
    """Custom-painted vertical slice scrubber with lesion span, bookmark ticks, and cyan playhead."""

    slice_changed = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedWidth(28)
        self.setMinimumHeight(160)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

        self._current_slice: int = 0
        self._total_slices: int = 60
        self._bookmarks: set[int] = set()
        self._lesion_span: tuple[int, int] | None = None
        self._dragging: bool = False

    def set_lesion_span(self, span: tuple[int, int] | None) -> None:
        """Sets an optional (start_slice, end_slice) span to highlight along the track."""
        if span != self._lesion_span:
            self._lesion_span = span
            self.update()

    def set_state(
        self,
        current_slice: int,
        total_slices: int,
        bookmarks: Iterable[int] | None = None,
        lesion_span: tuple[int, int] | None | object = _UNSET,
    ) -> None:
        """Updates rail state and schedules repaint only when values change."""
        total = max(1, int(total_slices))
        cur = max(0, min(total - 1, int(current_slice)))
        bm_set = set(int(b) for b in bookmarks) if bookmarks is not None else set()
        new_span = self._lesion_span if lesion_span is _UNSET else lesion_span

        if (
            cur == self._current_slice
            and total == self._total_slices
            and bm_set == self._bookmarks
            and new_span == self._lesion_span
        ):
            return

        self._current_slice = cur
        self._total_slices = total
        self._bookmarks = bm_set
        self._lesion_span = new_span  # type: ignore[assignment]
        self.update()

    def _track_bounds(self) -> tuple[int, int]:
        pad = 12
        top = pad
        bottom = max(top + 1, self.height() - pad)
        return top, bottom

    def _slice_to_y(self, slice_idx: int) -> float:
        top, bottom = self._track_bounds()
        if self._total_slices <= 1:
            return float(top)
        frac = max(0.0, min(1.0, slice_idx / (self._total_slices - 1)))
        return top + frac * (bottom - top)

    def _y_to_slice(self, y: float) -> int:
        top, bottom = self._track_bounds()
        frac = (y - top) / max(1, bottom - top)
        frac = max(0.0, min(1.0, frac))
        return int(round(frac * (self._total_slices - 1)))

    def _emit_for_y(self, y: float) -> None:
        target = self._y_to_slice(y)
        if target != self._current_slice:
            self._current_slice = target
            self.update()
            self.slice_changed.emit(target)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._emit_for_y(event.position().y())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._dragging:
            self._emit_for_y(event.position().y())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._dragging:
            self._dragging = False
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event) -> None:
        dy = event.angleDelta().y()
        if dy != 0:
            step = 1 if dy > 0 else -1
            target = max(0, min(self._total_slices - 1, self._current_slice + step))
            if target != self._current_slice:
                self._current_slice = target
                self.update()
                self.slice_changed.emit(target)
        event.accept()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        w = self.width()
        h = self.height()

        # Background + hairline frame
        painter.fillRect(0, 0, w, h, QColor(BG_INPUT))
        painter.setPen(QPen(QColor(BORDER_HAIRLINE), 1))
        painter.drawRect(0, 0, w - 1, h - 1)

        top, bottom = self._track_bounds()
        cx = w / 2.0

        # Central vertical track
        track_rect = QRectF(cx - 2.0, top, 4.0, bottom - top)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(BORDER_HAIRLINE))
        painter.drawRoundedRect(track_rect, 2.0, 2.0)

        # Optional lesion span highlight
        if self._lesion_span is not None and self._total_slices > 1:
            s0, s1 = self._lesion_span
            y0 = self._slice_to_y(min(s0, s1))
            y1 = self._slice_to_y(max(s0, s1))
            span_h = max(4.0, y1 - y0)
            span_rect = QRectF(cx - 4.0, y0, 8.0, span_h)
            painter.setBrush(QColor(255, 59, 48, 70))
            painter.setPen(QPen(QColor(COLOR_DANGER), 1))
            painter.drawRoundedRect(span_rect, 2.0, 2.0)

        # Bookmark amber ticks
        painter.setPen(QPen(QColor(COLOR_AMBER), 2))
        for bm in sorted(self._bookmarks):
            if 0 <= bm < self._total_slices:
                by = self._slice_to_y(bm)
                painter.drawLine(QPointF(4.0, by), QPointF(w - 4.0, by))

        # Cyan playhead handle
        py = self._slice_to_y(self._current_slice)
        handle_rect = QRectF(3.0, py - 4.0, w - 6.0, 8.0)
        painter.setPen(QPen(QColor(COLOR_CYAN), 1))
        painter.setBrush(QColor(COLOR_CYAN))
        painter.drawRoundedRect(handle_rect, 2.5, 2.5)

        painter.setPen(QPen(QColor(BG_VIEWPORT), 1.5))
        painter.drawLine(QPointF(6.0, py), QPointF(w - 6.0, py))


class DicomViewportWidget(QWidget):
    """Hardware-friendly QPainter PACS CT canvas with DICOM HUD, scale bar, and reticles."""

    slice_wheeled = Signal(int)
    cursor_moved = Signal(float, float)
    clicked_uv = Signal(float, float)
    cursor_left = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(420, 420)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._qimage: QImage | None = None
        self._img_width: int = 380
        self._metadata: dict = {}
        self._slice_idx: int = 0
        self._total_slices: int = 60
        self._window_name: str = "Brain"
        self._wl: float = 40.0
        self._ww: float = 80.0
        self._cursor_uv: tuple[float, float] | None = None
        self._cursor_hu: float | None = None
        self._markers: list[tuple[float, float, str, float]] = []
        self._is_bookmarked: bool = False
        self._dwell_progress: float = 0.0
        self._cursor_inside: bool = False

    def update_view(
        self,
        slice_uint8: np.ndarray | None,
        metadata: dict | None,
        slice_idx: int,
        total_slices: int,
        window_name: str,
        wl: float,
        ww: float,
        cursor_uv: tuple[float, float] | None = None,
        cursor_hu: float | None = None,
        markers: Sequence[tuple[float, float, str, float]] | None = None,
        is_bookmarked: bool = False,
        dwell_progress: float = 0.0,
    ) -> None:
        """Updates the CT slice image and all PACS overlay telemetry."""
        if slice_uint8 is not None and slice_uint8.size > 0:
            arr = np.ascontiguousarray(slice_uint8, dtype=np.uint8)
            if arr.ndim == 2:
                h, w = arr.shape
                self._img_width = int(w)
                qimg = QImage(arr.data, w, h, w, QImage.Format.Format_Grayscale8)
                self._qimage = qimg.copy()
            else:
                self._qimage = None
        else:
            self._qimage = None

        self._metadata = dict(metadata) if metadata else {}
        self._slice_idx = int(slice_idx)
        self._total_slices = max(1, int(total_slices))
        self._window_name = str(window_name)
        self._wl = float(wl)
        self._ww = float(ww)
        self._cursor_uv = tuple(cursor_uv) if cursor_uv is not None else None  # type: ignore[assignment]
        self._cursor_hu = float(cursor_hu) if cursor_hu is not None else None
        self._markers = list(markers) if markers else []
        self._is_bookmarked = bool(is_bookmarked)
        self._dwell_progress = max(0.0, min(1.0, float(dwell_progress)))
        self.update()

    def _ct_rect(self) -> QRect:
        """Returns the centered square rect where the CT slice is rendered inside 24px HUD margins."""
        pad_x = 22
        pad_y = 24
        avail_w = max(1, self.width() - 2 * pad_x)
        avail_h = max(1, self.height() - 2 * pad_y)
        side = min(avail_w, avail_h)
        x = (self.width() - side) // 2
        y = pad_y + (avail_h - side) // 2
        return QRect(x, y, side, side)

    def _pos_to_uv(self, pos: QPointF) -> tuple[float, float] | None:
        rect = self._ct_rect()
        if rect.width() <= 0 or rect.height() <= 0 or not rect.contains(pos.toPoint()):
            return None
        u = (pos.x() - rect.left()) / rect.width()
        v = (pos.y() - rect.top()) / rect.height()
        return (max(0.0, min(1.0, float(u))), max(0.0, min(1.0, float(v))))

    def mouseMoveEvent(self, event) -> None:
        uv = self._pos_to_uv(event.position())
        if uv is not None:
            self._cursor_inside = True
            self.cursor_moved.emit(uv[0], uv[1])
        elif self._cursor_inside:
            self._cursor_inside = False
            self.cursor_left.emit()
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            uv = self._pos_to_uv(event.position())
            if uv is not None:
                self.clicked_uv.emit(uv[0], uv[1])
                event.accept()
                return
        super().mousePressEvent(event)

    def leaveEvent(self, event) -> None:
        if self._cursor_inside:
            self._cursor_inside = False
            self.cursor_left.emit()
        super().leaveEvent(event)

    def wheelEvent(self, event) -> None:
        dy = event.angleDelta().y()
        if dy != 0:
            self.slice_wheeled.emit(1 if dy > 0 else -1)
        event.accept()

    def _draw_pill_label(
        self,
        painter: QPainter,
        x: float,
        y: float,
        text: str,
        fg_color: str,
        border_color: str | None = None,
        bg_rgba: tuple[int, int, int, int] = (8, 12, 16, 205),
    ) -> None:
        fm = QFontMetrics(painter.font())
        rect_txt = fm.boundingRect(text)
        pw = rect_txt.width() + 10
        ph = rect_txt.height() + 4
        box = QRectF(x, y, pw, ph)
        painter.setBrush(QColor(*bg_rgba))
        painter.setPen(QPen(QColor(border_color), 1) if border_color else Qt.PenStyle.NoPen)
        painter.drawRoundedRect(box, 3.0, 3.0)
        painter.setPen(QColor(fg_color))
        painter.drawText(box, Qt.AlignmentFlag.AlignCenter, text)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        w, h = self.width(), self.height()
        painter.fillRect(0, 0, w, h, QColor(BG_VIEWPORT))

        ct_rect = self._ct_rect()

        # 1. Grayscale CT slice scaled 1:1
        if self._qimage is not None:
            scaled = self._qimage.scaled(
                ct_rect.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            painter.drawImage(ct_rect.topLeft(), scaled)
        else:
            painter.setFont(tabular_font(10, bold=True))
            painter.setPen(QColor(TEXT_MUTED))
            painter.drawText(ct_rect, Qt.AlignmentFlag.AlignCenter, "NO CT VOLUME LOADED")

        # Subtle viewport hairline frame
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(BORDER_HAIRLINE), 1))
        painter.drawRect(0, 0, w - 1, h - 1)

        # 2. Edge anatomical orientation markers [A], [P], [R], [L] inside dedicated HUD margins
        painter.setFont(tabular_font(8, bold=True))
        fm_orient = QFontMetrics(painter.font())
        orient_ph = fm_orient.height() + 4
        orient_specs = [
            ("[A]", (w - (fm_orient.horizontalAdvance("[A]") + 10)) / 2.0, 4.0),
            ("[P]", (w - (fm_orient.horizontalAdvance("[P]") + 10)) / 2.0, h - orient_ph - 4.0),
            ("[R]", 6.0, (h - orient_ph) / 2.0),
            ("[L]", w - (fm_orient.horizontalAdvance("[L]") + 10) - 6.0, (h - orient_ph) / 2.0),
        ]
        for code, ox, oy in orient_specs:
            self._draw_pill_label(painter, ox, oy, code, COLOR_CYAN, bg_rgba=(8, 12, 16, 175))

        # 3. Surgical ROI markers
        for u, v, label, hu in self._markers:
            mx = ct_rect.left() + max(0.0, min(1.0, float(u))) * ct_rect.width()
            my = ct_rect.top() + max(0.0, min(1.0, float(v))) * ct_rect.height()

            amber_pen = QPen(QColor(COLOR_AMBER), 1.6)
            painter.setPen(amber_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPointF(mx, my), 8.0, 8.0)
            painter.drawLine(QPointF(mx - 13.0, my), QPointF(mx - 4.0, my))
            painter.drawLine(QPointF(mx + 4.0, my), QPointF(mx + 13.0, my))
            painter.drawLine(QPointF(mx, my - 13.0), QPointF(mx, my - 4.0))
            painter.drawLine(QPointF(mx, my + 4.0), QPointF(mx, my + 13.0))

            painter.setBrush(QColor(COLOR_AMBER))
            painter.drawEllipse(QPointF(mx, my), 1.8, 1.8)

            painter.setFont(tabular_font(8, bold=True))
            tag_text = f"{label} | {float(hu):+.0f} HU"
            fm_tag = QFontMetrics(painter.font())
            tag_w = fm_tag.horizontalAdvance(tag_text) + 12
            tx = mx + 12.0 if (mx + 12.0 + tag_w) < (w - 8) else max(8.0, mx - 12.0 - tag_w)
            ty = max(8.0, min(h - 28.0, my - 10.0))
            self._draw_pill_label(painter, tx, ty, tag_text, COLOR_AMBER, border_color=COLOR_AMBER)

        # 4. Active touchless laser pointer reticle + dwell ring
        if self._cursor_uv is not None:
            cu, cv = self._cursor_uv
            lx = ct_rect.left() + max(0.0, min(1.0, float(cu))) * ct_rect.width()
            ly = ct_rect.top() + max(0.0, min(1.0, float(cv))) * ct_rect.height()

            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(COLOR_CYAN), 1.5))
            painter.drawLine(QPointF(lx - 14.0, ly), QPointF(lx - 5.0, ly))
            painter.drawLine(QPointF(lx + 5.0, ly), QPointF(lx + 14.0, ly))
            painter.drawLine(QPointF(lx, ly - 14.0), QPointF(lx, ly - 5.0))
            painter.drawLine(QPointF(lx, ly + 5.0), QPointF(lx, ly + 14.0))

            painter.setBrush(QColor(COLOR_CYAN))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(lx, ly), 2.0, 2.0)

            if self._dwell_progress > 0.0:
                arc_r = 18.0
                arc_rect = QRectF(lx - arc_r, ly - arc_r, arc_r * 2.0, arc_r * 2.0)
                span_16 = -int(round(self._dwell_progress * 360.0 * 16.0))

                # Soft glow under-arc
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(QColor(0, 230, 118, 70), 5.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                painter.drawArc(arc_rect, 90 * 16, span_16)

                # Sharp nominal green arc
                painter.setPen(QPen(QColor(COLOR_NOMINAL), 2.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                painter.drawArc(arc_rect, 90 * 16, span_16)

                pct_text = f"{int(round(self._dwell_progress * 100.0)):d}%"
                painter.setFont(tabular_font(8, bold=True))
                self._draw_pill_label(
                    painter,
                    min(w - 48.0, lx + 20.0),
                    min(h - 26.0, ly + 14.0),
                    pct_text,
                    COLOR_NOMINAL,
                    border_color=COLOR_NOMINAL,
                )

        # 5. DICOM Corner HUDs
        self._paint_corner_huds(painter, w, h, ct_rect)

    def _paint_corner_huds(self, painter: QPainter, w: int, h: int, ct_rect: QRect) -> None:
        margin = 10
        top_y = 26
        painter.setFont(tabular_font(8, bold=False))
        fm = QFontMetrics(painter.font())
        line_h = fm.height() + 2

        patient_id = str(self._metadata.get("patient_id", "OR-702 (Phantom)"))
        series_desc = str(self._metadata.get("series_desc", "CRANIAL CT"))

        # Top-Left HUD (positioned below the 24px [A] top margin so it never overlaps [A])
        tl_lines = [
            (patient_id, TEXT_PRIMARY, True),
            ("MODALITY: CT CRANIAL", TEXT_MUTED, False),
            (f"SLICE: {self._slice_idx:02d} / {self._total_slices:02d}", COLOR_CYAN, True),
        ]
        for i, (txt, col, bold) in enumerate(tl_lines):
            painter.setFont(tabular_font(8, bold=bold))
            fm_l = QFontMetrics(painter.font())
            painter.setPen(QColor(col))
            painter.drawText(margin, top_y + fm_l.ascent() + i * line_h, txt)

        # Top-Right HUD
        wl_str = f"WL: {int(round(self._wl))}  WW: {int(round(self._ww))} [{self._window_name.upper()}]"
        tr_lines = [
            (series_desc, TEXT_PRIMARY, True),
            (wl_str, TEXT_MUTED, False),
        ]
        for i, (txt, col, bold) in enumerate(tr_lines):
            painter.setFont(tabular_font(8, bold=bold))
            fm_r = QFontMetrics(painter.font())
            tw = fm_r.horizontalAdvance(txt)
            painter.setPen(QColor(col))
            painter.drawText(w - margin - tw, top_y + fm_r.ascent() + i * line_h, txt)

        if self._is_bookmarked:
            painter.setFont(tabular_font(8, bold=True))
            bm_tag = "[BOOKMARKED]"
            fm_b = QFontMetrics(painter.font())
            bw = fm_b.horizontalAdvance(bm_tag) + 10
            self._draw_pill_label(
                painter,
                w - margin - bw,
                top_y + 2 * line_h + 2,
                bm_tag,
                COLOR_AMBER,
                border_color=COLOR_AMBER,
            )

        # Bottom-Left HUD (compact cursor readout <= ~38% width, never overlapping centered [P])
        painter.setFont(tabular_font(7, bold=True))
        fm_cur = QFontMetrics(painter.font())
        cur_ph = fm_cur.height() + 4
        if self._cursor_uv is not None:
            u, v = self._cursor_uv
            hu_txt = f"{self._cursor_hu:+.0f}" if self._cursor_hu is not None else "--"
            pos_txt = f"CURSOR ({u:.2f}, {v:.2f})  {hu_txt} HU"
            pos_col = COLOR_CYAN
        else:
            hu_txt = f"{self._cursor_hu:+.0f}" if self._cursor_hu is not None else "--"
            pos_txt = f"CURSOR (0.50, 0.50)  {hu_txt} HU"
            pos_col = TEXT_MUTED
        self._draw_pill_label(painter, margin, h - cur_ph - 4, pos_txt, pos_col)

        # Bottom-Right calibrated 50 mm scale bar
        spacing_raw = self._metadata.get("spacing_mm", [1.0, 0.65, 0.65])
        try:
            mm_per_px = float(spacing_raw[1]) if len(spacing_raw) > 1 else 0.65
        except (TypeError, ValueError, IndexError):
            mm_per_px = 0.65
        if not math.isfinite(mm_per_px) or mm_per_px <= 0.0:
            mm_per_px = 0.65

        img_w = max(1, self._img_width or int(self._metadata.get("width", 380)))
        px_per_mm = (ct_rect.width() / float(img_w)) / mm_per_px
        bar_px = max(20.0, min(ct_rect.width() * 0.40, 50.0 * px_per_mm))

        bx1 = float(w - margin)
        bx0 = bx1 - bar_px
        by = float(h - margin - 4)

        painter.setPen(QPen(QColor(COLOR_CYAN), 1.6))
        painter.drawLine(QPointF(bx0, by), QPointF(bx1, by))
        painter.drawLine(QPointF(bx0, by - 5.0), QPointF(bx0, by + 2.0))
        painter.drawLine(QPointF(bx1, by - 5.0), QPointF(bx1, by + 2.0))

        # 10 mm sub-ticks
        step_px = bar_px / 5.0
        if step_px >= 6.0:
            for k in range(1, 5):
                tx = bx0 + k * step_px
                painter.drawLine(QPointF(tx, by - 3.0), QPointF(tx, by))

        painter.setFont(tabular_font(8, bold=True))
        fm_s = QFontMetrics(painter.font())
        scale_lbl = "50 mm"
        sw = fm_s.horizontalAdvance(scale_lbl)
        painter.setPen(QColor(COLOR_CYAN))
        painter.drawText(int(bx0 + (bar_px - sw) / 2.0), int(by - 8), scale_lbl)


class CTPanel(QFrame):
    """Complete left workstation column: CT header, viewport + slice rail, windowing, and ROI controls."""

    load_nifti_requested = Signal()
    switch_source_requested = Signal()
    window_selected = Signal(str)
    prev_slice_requested = Signal()
    next_slice_requested = Signal()
    bookmark_toggled = Signal()
    mark_roi_requested = Signal()
    clear_roi_requested = Signal()
    default_slice_requested = Signal()
    slice_changed = Signal(int)

    _WINDOW_OPTIONS = ("Brain", "Bone", "Subdural")

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # Header Row
        hdr = QHBoxLayout()
        hdr.setSpacing(6)
        title = QLabel("NEURO CT NAVIGATION")
        title.setFont(ui_font(10, bold=True))
        title.setStyleSheet(f"color: {TEXT_PRIMARY}; letter-spacing: 0.6px;")
        hdr.addWidget(title)

        # Retained off-layout for state/test inspection; source status is displayed in top window bar
        self.source_badge = QLabel("SYNTHETIC", self)
        self.source_badge.setFont(tabular_font(8, bold=True))
        self.source_badge.hide()
        hdr.addStretch(1)

        self.load_nifti_btn = QPushButton("Load NIfTI")
        self.load_nifti_btn.setFont(ui_font(8, bold=True))
        self.load_nifti_btn.setMinimumWidth(88)
        self.load_nifti_btn.clicked.connect(self.load_nifti_requested.emit)
        hdr.addWidget(self.load_nifti_btn)

        self.switch_source_btn = QPushButton("Switch CT")
        self.switch_source_btn.setFont(ui_font(8, bold=True))
        self.switch_source_btn.setMinimumWidth(84)
        self.switch_source_btn.clicked.connect(self.switch_source_requested.emit)
        hdr.addWidget(self.switch_source_btn)
        root.addLayout(hdr)

        # Viewport + Slice Rail
        view_row = QHBoxLayout()
        view_row.setSpacing(6)
        self.viewport = DicomViewportWidget()
        self.slice_rail = SliceRailWidget()
        view_row.addWidget(self.viewport, stretch=1)
        view_row.addWidget(self.slice_rail)
        root.addLayout(view_row, stretch=1)

        self.slice_rail.slice_changed.connect(self.slice_changed.emit)

        # Segmented Window Pills
        win_row = QHBoxLayout()
        win_row.setSpacing(4)
        win_lbl = QLabel("WINDOW")
        win_lbl.setFont(tabular_font(8, bold=True))
        win_lbl.setStyleSheet(f"color: {TEXT_MUTED};")
        win_row.addWidget(win_lbl)

        self._window_buttons: dict[str, QPushButton] = {}
        for win_name in self._WINDOW_OPTIONS:
            btn = QPushButton(win_name.upper())
            btn.setFont(tabular_font(8, bold=True))
            btn.clicked.connect(lambda _=False, w=win_name: self.window_selected.emit(w))
            self._window_buttons[win_name] = btn
            win_row.addWidget(btn, stretch=1)
        root.addLayout(win_row)

        # Tactile Navigation & ROI Controls
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(4)
        self.prev_btn = self._make_ctrl_btn("Prev", self.prev_slice_requested.emit)
        self.next_btn = self._make_ctrl_btn("Next", self.next_slice_requested.emit)
        self.bookmark_btn = self._make_ctrl_btn("Bookmark", self.bookmark_toggled.emit)
        self.mark_roi_btn = self._make_ctrl_btn("Mark ROI", self.mark_roi_requested.emit)
        self.clear_btn = self._make_ctrl_btn("Clear", self.clear_roi_requested.emit)
        self.default_slice_btn = self._make_ctrl_btn("Default Slice", self.default_slice_requested.emit)

        for b in (
            self.prev_btn,
            self.next_btn,
            self.bookmark_btn,
            self.mark_roi_btn,
            self.clear_btn,
            self.default_slice_btn,
        ):
            ctrl_row.addWidget(b)
        root.addLayout(ctrl_row)

        # Bookmark Summary Chip
        bm_row = QHBoxLayout()
        bm_row.setSpacing(6)
        self.bookmark_summary_lbl = QLabel("BOOKMARKS: None")
        self.bookmark_summary_lbl.setFont(tabular_font(8, bold=False))
        self.bookmark_summary_lbl.setStyleSheet(
            f"background-color: {BG_CARD}; color: {TEXT_MUTED}; "
            f"border: 1px solid {BORDER_HAIRLINE}; border-radius: 4px; padding: 3px 8px;"
        )
        bm_row.addWidget(self.bookmark_summary_lbl, stretch=1)
        root.addLayout(bm_row)

        self.set_source_badge(False)
        self.set_active_window("Brain")

    def _make_ctrl_btn(self, text: str, slot) -> QPushButton:
        btn = QPushButton(text)
        btn.setFont(ui_font(8, bold=True))
        btn.clicked.connect(slot)
        return btn

    def set_source_badge(self, is_real: bool, custom_text: str | None = None) -> None:
        """Updates the CT source badge style ('REAL CRANIAL CT' vs 'SYNTHETIC')."""
        if is_real:
            label = custom_text or "REAL CRANIAL CT"
            color = COLOR_NOMINAL
            bg = "rgba(0, 230, 118, 0.12)"
        else:
            label = custom_text or "SYNTHETIC"
            color = COLOR_AMBER
            bg = "rgba(255, 179, 0, 0.12)"
        self.source_badge.setText(label)
        self.source_badge.setStyleSheet(
            f"color: {color}; background-color: {bg}; "
            f"border: 1px solid {color}; border-radius: 3px; padding: 2px 6px;"
        )

    def set_active_window(self, window_name: str) -> None:
        """Highlights the active CT window pill button."""
        target = window_name.strip().lower()
        for name, btn in self._window_buttons.items():
            is_act = name.lower() == target
            btn.setProperty("active", "true" if is_act else "false")
            repolish(btn)

    def set_bookmarks_summary(self, bookmarks: Iterable[int], current_slice: int | None = None) -> None:
        """Updates the bookmark summary chip and bookmark button highlight."""
        sorted_bm = sorted(set(int(b) for b in bookmarks))
        if sorted_bm:
            joined = ", ".join(str(b) for b in sorted_bm)
            self.bookmark_summary_lbl.setText(f"BOOKMARKS ({len(sorted_bm)}): {joined}")
            self.bookmark_summary_lbl.setStyleSheet(
                f"background-color: {BG_CARD}; color: {COLOR_AMBER}; "
                f"border: 1px solid {BORDER_HAIRLINE}; border-radius: 4px; padding: 3px 8px;"
            )
        else:
            self.bookmark_summary_lbl.setText("BOOKMARKS: None")
            self.bookmark_summary_lbl.setStyleSheet(
                f"background-color: {BG_CARD}; color: {TEXT_MUTED}; "
                f"border: 1px solid {BORDER_HAIRLINE}; border-radius: 4px; padding: 3px 8px;"
            )

        if current_slice is not None:
            is_bm = int(current_slice) in sorted_bm
            self.bookmark_btn.setProperty("warn", "true" if is_bm else "false")
            repolish(self.bookmark_btn)
