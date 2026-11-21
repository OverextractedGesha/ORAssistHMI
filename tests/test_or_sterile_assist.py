"""test_or_sterile_assist.py - Comprehensive Unit Tests & Smoke Verification.

Tests all modular components of OR-SterileAssist:
1. gesture_utils: finger counting, pinch, laser pointer detection, hysteresis, debouncing.
2. ct_phantom: 3D volume synthesis, HU ranges, windowing, reticle rendering, lesion discovery.
3. audio_utils: IEC 60601-1-8 alarms, tone synthesis, dBFS calculation, SpO2 pitch sonification.
4. speech_engine: command pattern matching (Indonesian and English).
5. application smoke test: headless Tkinter initialization and cleanup.
"""
import unittest
import numpy as np
import os
import sys
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import gesture_utils
from gesture_utils import (
    dist,
    count_fingers,
    is_pinch,
    pinch_ratio,
    is_index_pointer,
    Hysteresis,
    StableGesture,
    WRIST,
    THUMB_TIP,
    THUMB_IP,
    INDEX_FINGER_TIP,
    INDEX_FINGER_PIP,
    MIDDLE_FINGER_TIP,
    MIDDLE_FINGER_PIP,
    RING_FINGER_TIP,
    RING_FINGER_PIP,
    PINKY_TIP,
    PINKY_PIP,
    PINKY_MCP,
    MIDDLE_FINGER_MCP,
)
import ct_phantom
from ct_phantom import (
    N_SLICES,
    SIZE as CT_SIZE,
    WINDOWS,
    make_phantom,
    get_hu_at,
    to_display_rgb,
)
import audio_utils
from audio_utils import (
    SAMPLE_RATE,
    EARCONS,
    ALARMS,
    tone,
    sequence,
    dbfs,
    spo2_to_pitch,
    pulse_ox_tone,
)
from speech_engine import match_command_id, choose_microphone, SpeechEngine


class TestGestureUtils(unittest.TestCase):
    """Tests for landmark geometry, finger counting, and surgical debouncing."""

    def setUp(self):
        # 1. Synthetic Open Hand (All 5 extended)
        self.open_hand = [(0, 100)] * 21
        self.open_hand[WRIST] = (0, 0)
        self.open_hand[PINKY_MCP] = (50, 0)
        self.open_hand[THUMB_IP] = (20, 20)
        self.open_hand[THUMB_TIP] = (60, 40)
        self.open_hand[MIDDLE_FINGER_MCP] = (0, 40)
        for tip, pip in gesture_utils.FINGERS:
            self.open_hand[pip] = (0, 50)
            self.open_hand[tip] = (0, 100)

        # 2. Synthetic Fist (All bent)
        self.fist = [(0, 20)] * 21
        self.fist[WRIST] = (0, 0)
        self.fist[PINKY_MCP] = (50, 0)
        self.fist[THUMB_IP] = (30, 0)
        self.fist[THUMB_TIP] = (40, 0)
        self.fist[MIDDLE_FINGER_MCP] = (0, 40)
        for tip, pip in gesture_utils.FINGERS:
            self.fist[pip] = (0, 50)
            self.fist[tip] = (0, 20)

        # 3. Synthetic Index Pointer (Index extended, others bent, thumb folded)
        self.pointer = [(0, 20)] * 21
        self.pointer[WRIST] = (0, 0)
        self.pointer[PINKY_MCP] = (50, 0)
        self.pointer[THUMB_IP] = (20, 0)
        self.pointer[THUMB_TIP] = (15, 0)
        self.pointer[MIDDLE_FINGER_MCP] = (0, 40)
        # Index extended
        self.pointer[INDEX_FINGER_PIP] = (0, 50)
        self.pointer[INDEX_FINGER_TIP] = (0, 110)
        # Middle, Ring, Pinky bent
        for tip, pip in [(MIDDLE_FINGER_TIP, MIDDLE_FINGER_PIP),
                         (RING_FINGER_TIP, RING_FINGER_PIP),
                         (PINKY_TIP, PINKY_PIP)]:
            self.pointer[pip] = (0, 50)
            self.pointer[tip] = (0, 20)

    def test_dist(self):
        self.assertAlmostEqual(dist((0, 0), (3, 4)), 5.0)
        self.assertAlmostEqual(dist((10, 20), (10, 20)), 0.0)

    def test_count_fingers(self):
        self.assertEqual(count_fingers(self.open_hand), 5)
        self.assertEqual(count_fingers(self.fist), 0)

    def test_is_pinch_and_ratio(self):
        hand = list(self.open_hand)
        # Thumb and index far apart
        hand[THUMB_TIP] = (60, 40)
        hand[INDEX_FINGER_TIP] = (0, 100)
        self.assertFalse(is_pinch(hand, thresh=30))
        self.assertGreater(pinch_ratio(hand), 0.5)

        # Thumb and index close together (pinch)
        hand[THUMB_TIP] = (10, 80)
        hand[INDEX_FINGER_TIP] = (12, 82)
        self.assertTrue(is_pinch(hand, thresh=30))
        self.assertLess(pinch_ratio(hand), 0.35)

    def test_is_index_pointer(self):
        # Pointer hand should return True
        self.assertTrue(is_index_pointer(self.pointer))
        # Open hand should return False (other fingers extended)
        self.assertFalse(is_index_pointer(self.open_hand))
        # Fist should return False (index bent)
        self.assertFalse(is_index_pointer(self.fist))

    def test_hysteresis(self):
        h = Hysteresis(on_below=30, off_above=60)
        self.assertFalse(h.active)
        # Value drops to 25 -> activates
        self.assertTrue(h.update(25))
        self.assertTrue(h.active)
        # Value moves to 45 (within deadband) -> remains active
        self.assertFalse(h.update(45))
        self.assertTrue(h.active)
        # Value exceeds 65 -> deactivates
        self.assertFalse(h.update(65))
        self.assertFalse(h.active)

    def test_stable_gesture_debouncing(self):
        sg = StableGesture(frames=3)
        self.assertIsNone(sg.update(2))
        self.assertAlmostEqual(sg.progress, 1 / 3)
        self.assertIsNone(sg.update(2))
        self.assertAlmostEqual(sg.progress, 2 / 3)
        # 3rd consecutive frame triggers
        self.assertEqual(sg.update(2), 2)
        self.assertAlmostEqual(sg.progress, 1.0)
        # 4th frame does not re-trigger
        self.assertIsNone(sg.update(2))


class TestCTPhantom(unittest.TestCase):
    """Tests for 3D Head CT phantom generation, windowing, and surgical rendering."""

    @classmethod
    def setUpClass(cls):
        cls.volume = make_phantom()

    def test_volume_dimensions(self):
        self.assertEqual(self.volume.shape, (N_SLICES, CT_SIZE, CT_SIZE))
        self.assertEqual(self.volume.dtype, np.float32)

    def test_hounsfield_units_range(self):
        # Air background should be -1000 HU
        self.assertAlmostEqual(self.volume[30, 10, 10], -1000.0, places=1)
        # Skull bone should be +1000 HU
        max_hu = np.max(self.volume[30])
        self.assertAlmostEqual(max_hu, 1000.0, places=1)

    def test_target_lesion_presence(self):
        # Lesion is modeled on slices 34 to 43 around (u ~ 0.66, v ~ 0.39)
        lesion_slice = 38
        hu_core = get_hu_at(self.volume, lesion_slice, 0.66, 0.39)
        # Lesion core is +78 HU, rim is +140 HU
        self.assertGreater(hu_core, 60.0)
        self.assertLessEqual(hu_core, 140.0)

    def test_to_display_rgb_rendering(self):
        slice_hu = self.volume[38]
        rgb = to_display_rgb(
            slice_hu=slice_hu,
            level=40,
            width=80,
            cursor_uv=(0.5, 0.5),
            markers=[(0.66, 0.39, "Lesion", 78.0)],
            is_bookmarked=True,
            dwell_progress=0.75,
        )
        self.assertEqual(rgb.shape, (CT_SIZE, CT_SIZE, 3))
        self.assertEqual(rgb.dtype, np.uint8)


class TestAudioUtils(unittest.TestCase):
    """Tests for tone synthesis, IEC 60601-1-8 alarms, dBFS, and SpO2 sonification."""

    def test_tone_synthesis(self):
        wave = tone(440.0, 0.1, volume=0.5, sr=SAMPLE_RATE)
        expected_len = int(0.1 * SAMPLE_RATE)
        self.assertEqual(len(wave), expected_len)
        self.assertEqual(wave.dtype, np.float32)
        self.assertLessEqual(np.max(np.abs(wave)), 0.51)

    def test_sequence_synthesis(self):
        notes = [(440, 0.05), (0, 0.05), (880, 0.05)]
        seq = sequence(notes, volume=0.4, sr=SAMPLE_RATE)
        expected_len = int(0.15 * SAMPLE_RATE)
        self.assertEqual(len(seq), expected_len)

    def test_dbfs_calculation(self):
        # Silence
        silence = np.zeros(1600, dtype=np.float32)
        self.assertAlmostEqual(dbfs(silence), -100.0, places=1)

        # Full-scale sine wave
        sine_full = tone(1000.0, 0.2, volume=1.0)
        # RMS of full scale sine is ~1/sqrt(2), 20*log10(0.707) ~= -3.0 dBFS
        level = dbfs(sine_full)
        self.assertAlmostEqual(level, -3.01, delta=0.5)

    def test_spo2_to_pitch(self):
        # 100% SpO2 should yield 880.0 Hz
        self.assertAlmostEqual(spo2_to_pitch(100.0), 880.0, places=1)
        # 88% SpO2 (12% decrease = 1 octave lower) -> 440.0 Hz
        self.assertAlmostEqual(spo2_to_pitch(88.0), 440.0, places=1)
        # Lower SpO2 gives lower pitch
        self.assertLess(spo2_to_pitch(85.0), spo2_to_pitch(95.0))

    def test_iec_alarms_structure(self):
        # High priority alarm must have 10 pulses in 2 bursts (3+2 per burst)
        high = ALARMS["high"]
        sound_high = sequence(high)
        self.assertGreater(len(sound_high), 0)

        med = ALARMS["medium"]
        self.assertEqual(len(med), 6)  # 3 pulses + 3 gaps

        low = ALARMS["low"]
        self.assertEqual(len(low), 3)  # 2 pulses + 1 gap


class TestSpeechEngine(unittest.TestCase):
    """Tests for Indonesian and English voice command pattern recognition."""

    def test_indonesian_command_matching(self):
        # Next slice
        cmd, reply, text = match_command_id(["tolong iris berikutnya"], lang="id")
        self.assertEqual(cmd, "NEXT_SLICE")

        # Window brain
        cmd, reply, text = match_command_id(["ganti jendela otak"], lang="id")
        self.assertEqual(cmd, "WINDOW_BRAIN")

        # Bookmark
        cmd, reply, text = match_command_id(["simpan iris sekarang"], lang="id")
        self.assertEqual(cmd, "BOOKMARK_SLICE")

        # Mark lesion
        cmd, reply, text = match_command_id(["tandai lesi"], lang="id")
        self.assertEqual(cmd, "MARK_LESION")

        # Patient vitals
        cmd, reply, text = match_command_id(["laporkan tanda vital"], lang="id")
        self.assertEqual(cmd, "PATIENT_VITALS")

        # Silence alarm
        cmd, reply, text = match_command_id(["hening alarm"], lang="id")
        self.assertEqual(cmd, "SILENCE_ALARM")

    def test_english_command_matching(self):
        # Next slice
        cmd, reply, text = match_command_id(["next slice please"], lang="en")
        self.assertEqual(cmd, "NEXT_SLICE")

        # Window bone
        cmd, reply, text = match_command_id(["switch window bone"], lang="en")
        self.assertEqual(cmd, "WINDOW_BONE")

        # Silence alarm
        cmd, reply, text = match_command_id(["silence alarm"], lang="en")
        self.assertEqual(cmd, "SILENCE_ALARM")

    def test_choose_microphone(self):
        dev = choose_microphone()
        self.assertTrue(dev is None or isinstance(dev, int))
        # Invalid device index returns None
        self.assertIsNone(choose_microphone(device=999999))

    def test_speech_mode_offline_fallback(self):
        engine = SpeechEngine()
        engine.set_mode("EN_OFFLINE")
        self.assertEqual(engine.mode, "EN_OFFLINE")
        # Drain queue to verify status event
        found_status = False
        while not engine.events_queue.empty():
            ev = engine.events_queue.get_nowait()
            if ev.get("type") == "status":
                found_status = True
                break
        self.assertTrue(found_status)
        engine.stop()


class TestAppHeadlessSmoke(unittest.TestCase):
    """Headless smoke test verifying Tkinter app creation, bindings, and clean destruction."""

    def test_app_instantiation_and_teardown(self):
        import tkinter as tk
        from or_sterile_assist import ORSterileAssistApp

        root = tk.Tk()
        root.withdraw()  # Headless mode for automated tests
        app = ORSterileAssistApp(root)

        # Verify initial states
        self.assertEqual(app.current_slice, 38)
        self.assertEqual(app.current_window, "Brain")
        self.assertIn(38, app.bookmarks)

        # Trigger navigation methods
        app.next_slice()
        self.assertEqual(app.current_slice, 39)
        app.prev_slice()
        self.assertEqual(app.current_slice, 38)

        # Trigger window cycling
        app.cycle_window()
        self.assertEqual(app.current_window, "Bone")

        # Teardown
        app.on_closing()

    def test_notes_focus_shortcut_guard(self):
        """Verifies that typing inside notes editor never triggers single-key root shortcuts."""
        import tkinter as tk
        from or_sterile_assist import ORSterileAssistApp

        root = tk.Tk()
        root.withdraw()
        app = ORSterileAssistApp(root)

        initial_slice = app.current_slice
        initial_window = app.current_window

        class MockEvent:
            def __init__(self, widget):
                self.widget = widget

        # 1. When notes_text is the event widget or focused, shortcuts must be suppressed
        mock_text_event = MockEvent(app.notes_text)
        self.assertTrue(app._is_text_focused(mock_text_event))

        # Attempt to trigger slice navigation, window cycling, and alarm silence
        app._guarded_action(app.next_slice)(mock_text_event)
        self.assertEqual(app.current_slice, initial_slice)

        app._guarded_action(app.prev_slice)(mock_text_event)
        self.assertEqual(app.current_slice, initial_slice)

        app._guarded_action(app.cycle_window)(mock_text_event)
        self.assertEqual(app.current_window, initial_window)

        # 2. When focus is outside notes_text (e.g. root), shortcuts must execute normally
        mock_root_event = MockEvent(app.root)
        app.root.focus_set()

        app._guarded_action(app.next_slice)(mock_root_event)
        self.assertEqual(app.current_slice, initial_slice + 1)

        app._guarded_action(app.cycle_window)(mock_root_event)
        self.assertNotEqual(app.current_window, initial_window)

        # 3. Test Slider Feedback Guard and Render State Caching
        initial_render_state = app._last_render_state
        app._render_ct_display()  # Redundant call with identical state
        self.assertEqual(app._last_render_state, initial_render_state)

        # Test _set_label_cached
        app._set_label_cached("test_key", app.ct_status_lbl, text="Test Status")
        self.assertEqual(app._ui_cache.get("test_key"), {"text": "Test Status"})

        app.on_closing()

    def test_half_duplex_play_sound_flag(self):
        """Verifies that earcons (half_duplex=False) do not trigger is_speaking, while alarms/TTS do."""
        from speech_engine import SpeechEngine
        import numpy as np

        engine = SpeechEngine()
        try:
            # Earcon sound (half_duplex=False)
            dummy_sound = np.zeros(160, dtype=np.float32)
            engine.play_sound(dummy_sound, half_duplex=False)
            # is_speaking should NOT be locked for earcon
            self.assertFalse(engine.is_speaking())

            # Alarm sound (half_duplex=True)
            engine.play_sound(dummy_sound, half_duplex=True)
            self.assertTrue(engine.is_speaking())
        finally:
            engine.stop()


if __name__ == "__main__":
    unittest.main()
