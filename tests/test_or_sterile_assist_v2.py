"""test_or_sterile_assist_v2.py - Automated Verification & Offscreen UI Tests for v2.0.

Tests:
1. Real clinical CT loader (`load_real_ct`), HU dynamic range (<= -900, >= +600),
   radiological bounds, and `window_slice_grayscale()`.
2. Headless `ORSterileAssistV2Window` (QT_QPA_PLATFORM=offscreen):
   - Real CT startup, header badges, geometry (1380x860, min 1280x800)
   - Source switching Real <-> Synthetic Phantom
   - Scenario transitions (Normal 98%, Mild Hypoxia 91%, Critical Desat 84%)
   - 60 FPS ECG II & PLETH SpO2 waveform trace advancement
   - Slice navigation, window cycling, bookmarking, ROI marking & clearing
   - IEC 60601-1-8 alarm silencing & mute countdown
   - Text-focus shortcut guard & Escape focus return
   - Clean shutdown and offscreen 1380x860 UI render capture
"""
from __future__ import annotations

import gc
import os
import sys
import tempfile
import time
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if sys.platform == "win32" and os.path.isdir(r"C:\Windows\Fonts"):
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC_DIR = os.path.join(_ROOT_DIR, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from ct_phantom import N_SLICES, SIZE as CT_SIZE, WINDOWS, load_real_ct, window_slice_grayscale
from or_sterile_assist_v2 import ORSterileAssistV2Window

PREVIEW_PATH = (
    r"C:\Users\LENOVO\.gemini\antigravity\brain\f84d638a-2362-46ac-a038-4973fdc41e2c"
    r"\scratch\v2_ui_preview.png"
)


def _ensure_qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


class TestRealCTLoaderAndWindowing(unittest.TestCase):
    """Validates real clinical CT asset loading, HU bounds, and grayscale windowing."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.volume, cls.meta = load_real_ct()

    def test_real_ct_shape_and_dtype(self) -> None:
        self.assertEqual(self.volume.ndim, 3)
        self.assertEqual(self.volume.dtype, np.float32)
        n_slices, h, w = self.volume.shape
        self.assertGreaterEqual(n_slices, 30)
        self.assertEqual(h, w)
        self.assertTrue(self.meta["is_real"])
        self.assertEqual(self.meta["patient_id"], "OR-702 (Clinical Lesion CT)")
        self.assertEqual(self.meta["default_slice"], 101)
        self.assertEqual(self.meta["lesion_span"], (41, 136))
        self.assertIsInstance(self.meta["lesion_marker_uv"], tuple)
        self.assertEqual(len(self.meta["lesion_marker_uv"]), 2)
        self.assertIsInstance(self.meta["lesion_label"], str)
        self.assertTrue(self.meta["lesion_label"])
        self.assertEqual(self.meta["n_slices"], n_slices)
        self.assertEqual(self.meta["height"], h)
        self.assertEqual(self.meta["width"], w)

    def test_real_ct_hounsfield_range_and_bounds(self) -> None:
        min_hu = float(np.min(self.volume))
        max_hu = float(np.max(self.volume))
        self.assertLessEqual(min_hu, -900.0)
        self.assertGreaterEqual(max_hu, 600.0)

        default_slice = int(self.meta["default_slice"])
        self.assertGreaterEqual(default_slice, 0)
        self.assertLess(default_slice, self.volume.shape[0])

        spacing = self.meta.get("spacing_mm", [])
        self.assertEqual(len(spacing), 3)
        for s in spacing:
            self.assertGreater(float(s), 0.0)

    def test_window_slice_grayscale(self) -> None:
        mid_slice = self.volume[int(self.meta["default_slice"])]
        for win_name, (wl, ww) in WINDOWS.items():
            gray = window_slice_grayscale(mid_slice, wl, ww)
            self.assertEqual(gray.shape, mid_slice.shape)
            self.assertEqual(gray.dtype, np.uint8)
            self.assertEqual(int(gray.min()), 0)
            self.assertGreaterEqual(int(gray.max()), 250)


class TestORSterileAssistV2Window(unittest.TestCase):
    """Headless PySide6 workstation integration tests."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _ensure_qapp()

    def setUp(self) -> None:
        self.win = ORSterileAssistV2Window(headless=True)
        self.app.processEvents()

    def tearDown(self) -> None:
        if self.win is not None:
            self.win.close()
            self.win.shutdown()
            self.win.deleteLater()
            self.win = None
            self.app.processEvents()

    def test_startup_real_ct_and_geometry(self) -> None:
        self.assertEqual(self.win.windowTitle(), "OR-SterileAssist v2 — Medical Workstation (ETS HMI)")
        self.assertEqual(self.win.minimumWidth(), 1280)
        self.assertEqual(self.win.minimumHeight(), 800)
        self.assertEqual(self.win.width(), 1380)
        self.assertEqual(self.win.height(), 860)

        self.assertTrue(self.win.ct_meta["is_real"])
        self.assertEqual(self.win.ct_meta["patient_id"], "OR-702 (Clinical Lesion CT)")
        self.assertEqual(self.win.current_slice, 101)
        self.assertEqual(self.win.current_slice, int(self.win.ct_meta["default_slice"]))
        self.assertEqual(self.win.lesion_span, (41, 136))
        self.assertEqual(self.win.bookmarks, {101})
        default_markers = self.win.markers.get(101, [])
        self.assertEqual(len(default_markers), 1)
        _, _, marker_label, marker_hu = default_markers[0]
        self.assertEqual(marker_label, self.win.ct_meta["lesion_label"])
        self.assertGreaterEqual(marker_hu, 45.0)
        self.assertLessEqual(marker_hu, 120.0)
        log_text = self.win.notes_edit.toPlainText()
        self.assertIn("Slice 101", log_text)
        self.assertIn(marker_label, log_text)
        self.assertEqual(self.win.header_source_badge.text(), "REAL CRANIAL CT")
        self.assertEqual(self.win.ct_panel.source_badge.text(), "REAL CRANIAL CT")
        self.assertEqual(self.win.version_badge.text(), "v2.0")

    def test_source_switching_real_and_synthetic(self) -> None:
        real_slices = self.win.n_slices

        # Toggle Real -> Synthetic Phantom
        self.win.switch_ct_source()
        self.assertFalse(self.win.ct_meta["is_real"])
        self.assertEqual(self.win.n_slices, N_SLICES)
        self.assertEqual(self.win.ct_volume.shape, (N_SLICES, CT_SIZE, CT_SIZE))
        self.assertEqual(self.win.current_slice, 38)
        self.assertEqual(self.win.lesion_span, (34, 43))
        self.assertIn(38, self.win.bookmarks)
        self.assertEqual(len(self.win.markers.get(38, [])), 1)
        self.assertEqual(self.win.header_source_badge.text(), "SYNTHETIC PHANTOM")
        self.assertEqual(self.win.ct_panel.source_badge.text(), "SYNTHETIC PHANTOM")

        # Toggle Synthetic Phantom -> Real CT
        self.win.switch_ct_source()
        self.assertTrue(self.win.ct_meta["is_real"])
        self.assertEqual(self.win.n_slices, real_slices)
        self.assertEqual(self.win.current_slice, 101)
        self.assertEqual(self.win.lesion_span, (41, 136))
        self.assertEqual(self.win.bookmarks, {101})
        self.assertEqual(len(self.win.markers.get(101, [])), 1)
        self.assertEqual(self.win.header_source_badge.text(), "REAL CRANIAL CT")

    def test_scenario_transitions_and_alarm_silence(self) -> None:
        # Normal 98%
        self.win._set_scenario("normal")
        self.assertEqual(self.win.vitals["spo2"], 98)
        self.assertEqual(self.win.vitals["alarm_severity"], "none")
        self.assertEqual(self.win.vitals_panel.priority_banner.text(), "[NOMINAL]")

        # Mild Hypoxia 91%
        self.win._set_scenario("mild")
        self.assertEqual(self.win.vitals["spo2"], 91)
        self.assertEqual(self.win.vitals["alarm_severity"], "medium")
        self.assertEqual(self.win.vitals_panel.priority_banner.text(), "[MED PRIORITY ALARM]")
        self.assertEqual(self.win.vitals_panel.spo2_card.value_lbl.text().strip(), "91")

        # Critical Desat 84%
        self.win._set_scenario("critical")
        self.assertEqual(self.win.vitals["spo2"], 84)
        self.assertEqual(self.win.vitals["alarm_severity"], "high")
        self.assertEqual(self.win.vitals_panel.priority_banner.text(), "[HIGH PRIORITY: DESATURATION]")
        self.assertEqual(self.win.vitals_panel.hr_card.value_lbl.text().strip(), "122")

        # Silence alarm 120s
        self.win.silence_alarm()
        self.assertGreater(self.win.alarm_silenced_until, time.monotonic() + 110.0)
        self.assertIn("MUTE: 0", self.win.vitals_panel.mute_timer_lbl.text())

    def test_waveform_trace_advancement(self) -> None:
        ecg_before = self.win.vitals_panel.ecg_trace.sweep_idx
        pleth_before = self.win.vitals_panel.pleth_trace.sweep_idx

        for _ in range(15):
            self.win._waveform_loop()
            time.sleep(0.006)

        ecg_after = self.win.vitals_panel.ecg_trace.sweep_idx
        pleth_after = self.win.vitals_panel.pleth_trace.sweep_idx
        self.assertNotEqual(ecg_after, ecg_before)
        self.assertNotEqual(pleth_after, pleth_before)
        self.assertTrue(any(s is not None for s in self.win.vitals_panel.ecg_trace.samples))
        self.assertTrue(any(s is not None for s in self.win.vitals_panel.pleth_trace.samples))

    def test_slice_nav_window_bookmark_and_roi_markers(self) -> None:
        start = self.win.current_slice
        self.win.next_slice()
        self.assertEqual(self.win.current_slice, min(self.win.n_slices - 1, start + 1))

        self.win.prev_slice()
        self.assertEqual(self.win.current_slice, start)

        self.win.set_slice(10)
        self.assertEqual(self.win.current_slice, 10)
        self.win.jump_to_default_slice()
        self.assertEqual(self.win.current_slice, int(self.win.ct_meta["default_slice"]))

        # Window cycling
        self.assertEqual(self.win.current_window, "Brain")
        self.win.cycle_window()
        self.assertEqual(self.win.current_window, "Bone")
        self.win.cycle_window()
        self.assertEqual(self.win.current_window, "Subdural")
        self.win.cycle_window()
        self.assertEqual(self.win.current_window, "Brain")

        # Bookmark toggling on slice 25
        self.win.set_slice(25)
        self.assertNotIn(25, self.win.bookmarks)
        self.win.bookmark_current_slice()
        self.assertIn(25, self.win.bookmarks)
        self.win.bookmark_current_slice()
        self.assertNotIn(25, self.win.bookmarks)

        # ROI marking & clearing
        self.win.mark_current_lesion((0.55, 0.45))
        self.assertEqual(len(self.win.markers.get(25, [])), 1)
        self.win.clear_markers_on_slice()
        self.assertEqual(len(self.win.markers.get(25, [])), 0)

    def test_shortcut_focus_guard_and_escape(self) -> None:
        self.win.show()
        self.app.processEvents()

        # When log_edit has focus, shortcuts must be suppressed
        self.win.notes_edit.setFocus()
        self.app.processEvents()
        self.assertTrue(self.win._is_text_focused())

        s0 = self.win.current_slice
        self.assertFalse(self.win._handle_shortcut_key(Qt.Key.Key_Right))
        self.assertEqual(self.win.current_slice, s0)

        # Pressing Escape in log_edit returns focus outside text widget
        esc_ev = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier)
        self.win.notes_edit.keyPressEvent(esc_ev)
        self.app.processEvents()
        self.assertFalse(self.win._is_text_focused())

        # Now shortcut works
        self.assertTrue(self.win._handle_shortcut_key(Qt.Key.Key_Right))
        self.assertEqual(self.win.current_slice, s0 + 1)

        # Ctrl+C must NOT clear markers
        self.win.mark_current_lesion((0.5, 0.5))
        s_cur = self.win.current_slice
        self.assertEqual(len(self.win.markers.get(s_cur, [])), 1)
        self.assertFalse(self.win._handle_shortcut_key(Qt.Key.Key_C, Qt.KeyboardModifier.ControlModifier))
        self.assertEqual(len(self.win.markers.get(s_cur, [])), 1)

    def test_qimage_lifetime_safety(self) -> None:
        slice_arr = np.full((64, 64), 128, dtype=np.uint8)
        self.win.ct_panel.viewport.update_view(
            slice_uint8=slice_arr,
            metadata={"width": 64, "spacing_mm": [1.0, 0.5, 0.5]},
            slice_idx=0,
            total_slices=1,
            window_name="Brain",
            wl=40.0,
            ww=80.0,
        )
        del slice_arr

        cam_arr = np.full((120, 160, 3), 200, dtype=np.uint8)
        self.win.vision_panel.set_camera_frame(cam_arr)
        del cam_arr
        gc.collect()

        ct_qimg = self.win.ct_panel.viewport._qimage
        cam_qimg = self.win.vision_panel.camera_preview._qimage
        self.assertIsNotNone(ct_qimg)
        self.assertIsNotNone(cam_qimg)
        self.assertEqual(ct_qimg.pixelColor(10, 10).red(), 128)
        self.assertEqual(cam_qimg.pixelColor(10, 10).red(), 200)

    def test_nifti_error_handling_and_volume_bounds_switch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_nii = os.path.join(tmpdir, "corrupt.nii")
            with open(bad_nii, "wb") as f:
                f.write(b"not a valid nifti file")

            # 1. Corrupt NIfTI triggers clean QMessageBox.warning without crashing
            self.win.headless = False
            try:
                with (
                    mock.patch.object(QFileDialog, "getOpenFileName", return_value=(bad_nii, "")),
                    mock.patch.object(QMessageBox, "warning") as warn_mock,
                ):
                    self.win.prompt_load_nifti()
                self.assertEqual(warn_mock.call_count, 1)
                self.assertIn("NIfTI Load Failed", warn_mock.call_args[0][1])

                # 2. Missing nibabel triggers clean QMessageBox.warning without crashing
                with (
                    mock.patch.dict(sys.modules, {"nibabel": None}),
                    mock.patch.object(QFileDialog, "getOpenFileName", return_value=(bad_nii, "")),
                    mock.patch.object(QMessageBox, "warning") as warn_mock,
                ):
                    self.win.prompt_load_nifti()
                self.assertEqual(warn_mock.call_count, 1)
            finally:
                self.win.headless = True

            # 3. Valid smaller NIfTI (12 slices) -> switch to Synthetic (60 slices) safely
            import nibabel as nib

            small_nii = os.path.join(tmpdir, "small_head.nii.gz")
            data = np.full((32, 48, 12), -1000.0, dtype=np.float32)
            data[8:24, 10:38, 3:9] = 40.0
            nib.save(nib.Nifti1Image(data, np.eye(4)), small_nii)

            self.win.load_nifti_file(small_nii)
            self.assertEqual(self.win.n_slices, 12)
            self.assertEqual(self.win.current_slice, 6)
            self.assertIsNone(self.win.lesion_span)
            self.assertEqual(self.win.markers, {})
            self.win.set_slice(11)
            self.win.bookmark_current_slice()
            self.win.mark_current_lesion((0.5, 0.5))

            # Switch Real -> Synthetic (60 slices) without IndexError
            self.win.switch_ct_source()
            self.assertFalse(self.win.ct_meta["is_real"])
            self.assertEqual(self.win.n_slices, N_SLICES)
            self.assertEqual(self.win.current_slice, 38)
            self.assertEqual(self.win.bookmarks, {38})
            self.assertNotIn(11, self.win.markers)

    def test_clean_shutdown_stops_timers_trackers_and_speech(self) -> None:
        self.win.close()
        self.app.processEvents()

        self.assertFalse(self.win.is_running)
        for tname in ("cam_timer", "wave_timer", "speech_timer", "vitals_timer", "pulse_ox_timer"):
            self.assertFalse(getattr(self.win, tname).isActive())
        self.assertIsNone(self.win.hand_tracker)
        self.assertIsNone(self.win.face_tracker)
        self.assertIsNone(self.win.cap)
        self.assertFalse(self.win.speech.running)
        # Idempotent second shutdown call
        self.win.shutdown()

    def test_render_offscreen_preview_screenshot(self) -> None:
        os.makedirs(os.path.dirname(PREVIEW_PATH), exist_ok=True)
        self.win.show()
        self.win.resize(1380, 860)

        # Advance waveforms for realistic live ECG & PLETH traces in preview
        for _ in range(180):
            self.win.vitals_panel.step_waveforms(0.016)
        self.win._camera_loop()
        self.app.processEvents()

        pixmap = self.win.grab()
        self.assertEqual(pixmap.width(), 1380)
        self.assertEqual(pixmap.height(), 860)
        saved = pixmap.save(PREVIEW_PATH, "PNG")
        self.assertTrue(saved)
        self.assertTrue(os.path.isfile(PREVIEW_PATH))
        self.assertGreater(os.path.getsize(PREVIEW_PATH), 10_000)


if __name__ == "__main__":
    unittest.main()
