"""speech_engine.py - Thread-Safe Audio, Speech Recognition & Text-to-Speech Engine.

Features:
- Half-Duplex protection: tracks speaking state to mute recognition during earcons & speech synthesis.
- Multi-engine TTS: pyttsx3 (offline SAPI5), edge-tts (online ID), Piper TTS (offline ID if onnx available).
- Multi-mode ASR:
    * "ID_ONLINE": Google Web Speech (id-ID) with Indonesian surgical commands.
    * "EN_OFFLINE": Vosk offline Kaldi recognizer with surgical grammar (or Google en-US fallback).
    * "DICTATION_ID": Free-form clinical dictation for intra-operative text logging (id-ID).
    * "DICTATION_EN": Free-form clinical dictation for intra-operative text logging (en-US).
- 2-second ambient noise calibration for adaptive VAD threshold.
- Thread-safe queues for GUI event communication (never blocks Tkinter mainloop).
"""
import io
import json
import os
import sys
import queue
import tempfile
import threading
import time
import urllib.request
import zipfile
import numpy as np
import sounddevice as sd
import soundfile as sf
import speech_recognition as sr
import pyttsx3
try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

from audio_utils import SAMPLE_RATE, dbfs

# Piper TTS Indonesian Model Details
PIPER_MODEL_NAME = "id_ID-news_tts-medium"
PIPER_MODEL_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/id/id_ID/news_tts/medium/" + PIPER_MODEL_NAME

# Vosk English Model Details
VOSK_MODEL_NAME = "vosk-model-small-en-us-0.15"
VOSK_MODEL_URL = "https://alphacephei.com/vosk/models/" + VOSK_MODEL_NAME + ".zip"

# Constrained English surgical command vocabulary
EN_GRAMMAR = [
    "next slice",
    "previous slice",
    "window brain",
    "window bone",
    "window subdural",
    "bookmark slice",
    "mark lesion",
    "clear markers",
    "patient vitals",
    "heart rate",
    "oxygen",
    "blood pressure",
    "silence alarm",
    "pause camera",
    "resume camera",
    "help",
]

# Indonesian surgical commands mapping: [[keywords], COMMAND_ID, spoken_feedback]
ID_COMMAND_PATTERNS = [
    (["iris berikutnya", "berikutnya", "naik", "maju"], "NEXT_SLICE", "Iris berikutnya."),
    (["iris sebelumnya", "sebelumnya", "turun", "mundur"], "PREV_SLICE", "Iris sebelumnya."),
    (["jendela otak", "mode otak", "otak"], "WINDOW_BRAIN", "Jendela otak diaktifkan."),
    (["jendela tulang", "mode tulang", "tulang"], "WINDOW_BONE", "Jendela tulang diaktifkan."),
    (["jendela subdural", "mode subdural", "subdural"], "WINDOW_SUBDURAL", "Jendela subdural diaktifkan."),
    (["simpan iris", "bookmark", "tanda iris", "simpan"], "BOOKMARK_SLICE", "Iris berhasil disimpan."),
    (["tandai lesi", "tandai target", "tandai", "lesi"], "MARK_LESION", "Target lesi ditandai."),
    (["hapus tanda", "hapus marker", "bersihkan tanda"], "CLEAR_MARKERS", "Tanda target dibersihkan."),
    (["tanda vital", "kondisi pasien", "status pasien", "vital"], "PATIENT_VITALS", "Membacakan ringkasan tanda vital."),
    (["detak jantung", "nadi", "jantung"], "HEART_RATE", "Detak jantung normal."),
    (["saturasi", "oksigen"], "OXYGEN", "Saturasi oksigen dipantau."),
    (["tekanan darah", "tensi"], "BLOOD_PRESSURE", "Tekanan darah pasien."),
    (["hening alarm", "matikan alarm", "senyap", "diam"], "SILENCE_ALARM", "Alarm diheningkan dua menit."),
    (["jeda kamera", "kunci gestur", "pause kamera"], "PAUSE_CAMERA", "Pelacakan gestur dijeda."),
    (["lanjut kamera", "resume kamera", "aktifkan kamera", "buka gestur"], "RESUME_CAMERA", "Pelacakan gestur aktif."),
    (["bantuan", "tolong", "menu"], "HELP", "Perintah suara: iris berikutnya, jendela otak, bookmark, tandai lesi, atau tanda vital."),
]

EN_COMMAND_PATTERNS = [
    (["next slice", "next"], "NEXT_SLICE", "Next slice."),
    (["previous slice", "previous", "prev"], "PREV_SLICE", "Previous slice."),
    (["window brain", "brain window", "brain"], "WINDOW_BRAIN", "Brain window."),
    (["window bone", "bone window", "bone"], "WINDOW_BONE", "Bone window."),
    (["window subdural", "subdural window", "subdural"], "WINDOW_SUBDURAL", "Subdural window."),
    (["bookmark slice", "bookmark", "save slice"], "BOOKMARK_SLICE", "Slice bookmarked."),
    (["mark lesion", "mark target", "mark"], "MARK_LESION", "Lesion target marked."),
    (["clear markers", "clear marker", "clear"], "CLEAR_MARKERS", "Markers cleared."),
    (["patient vitals", "vitals summary", "vitals"], "PATIENT_VITALS", "Reading patient vitals."),
    (["heart rate", "pulse"], "HEART_RATE", "Heart rate status."),
    (["oxygen", "saturation", "spo2"], "OXYGEN", "Oxygen saturation status."),
    (["blood pressure", "bp"], "BLOOD_PRESSURE", "Blood pressure status."),
    (["silence alarm", "mute alarm", "silence"], "SILENCE_ALARM", "Alarm silenced for two minutes."),
    (["pause camera", "lock gestures", "pause"], "PAUSE_CAMERA", "Gesture tracking paused."),
    (["resume camera", "unlock gestures", "resume"], "RESUME_CAMERA", "Gesture tracking resumed."),
    (["help", "commands"], "HELP", "Commands: next slice, window brain, bookmark slice, mark lesion, patient vitals."),
]


def match_command_id(texts, lang="id"):
    """Matches a list of recognized alternative strings against command patterns."""
    patterns = ID_COMMAND_PATTERNS if lang == "id" else EN_COMMAND_PATTERNS
    for text in texts:
        t_low = text.lower().strip()
        for keywords, cmd_id, reply in patterns:
            for kw in keywords:
                if kw in t_low:
                    return cmd_id, reply, text
    return None, None, texts[0] if texts else ""


def choose_microphone(device=None):
    """Safely selects and validates an audio input device for sounddevice.
    
    Args:
        device: Optional device index (int) or device name substring (str).
                If None, queries system default or first available input device.
                
    Returns:
        int: Valid device index with >= 1 input channels, or None if unavailable.
    """
    try:
        devices = sd.query_devices()
        if not devices:
            return None

        # 1. User-specified integer device index
        if isinstance(device, int):
            if 0 <= device < len(devices):
                dev_info = devices[device]
                if dev_info.get("max_input_channels", 0) > 0:
                    return device
            return None

        # 2. User-specified string substring
        if isinstance(device, str):
            device_lower = device.lower()
            for idx, dev in enumerate(devices):
                if device_lower in dev.get("name", "").lower() and dev.get("max_input_channels", 0) > 0:
                    return idx

        # 3. Default input device query
        default_dev = sd.default.device
        if default_dev is not None:
            if isinstance(default_dev, (list, tuple)) and len(default_dev) > 0:
                def_in = default_dev[0]
            elif isinstance(default_dev, int):
                def_in = default_dev
            else:
                def_in = None

            if def_in is not None and 0 <= def_in < len(devices):
                if devices[def_in].get("max_input_channels", 0) > 0:
                    return def_in

        # 4. Fallback: First device with input channels
        for idx, dev in enumerate(devices):
            if dev.get("max_input_channels", 0) > 0:
                return idx

    except Exception as e:
        print(f"[speech_engine] Warning querying audio input devices: {e}")

    return None


class SpeechEngine:
    """Manages audio output (TTS, earcons, alarms) and input (ASR, VAD, calibration) safely in worker threads."""

    def __init__(self, models_dir=None, input_device=None):
        if getattr(sys, "frozen", False):
            base_dir = os.path.dirname(sys.executable)
            meipass = getattr(sys, "_MEIPASS", base_dir)
            self.models_dir = models_dir or os.path.join(meipass, "models")
            if not os.path.isdir(self.models_dir):
                self.models_dir = os.path.join(base_dir, "models")
        else:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            if models_dir is not None:
                self.models_dir = models_dir
            elif os.path.isdir(os.path.join(base_dir, "..", "models")):
                self.models_dir = os.path.abspath(os.path.join(base_dir, "..", "models"))
            else:
                self.models_dir = os.path.join(base_dir, "models")

        os.makedirs(self.models_dir, exist_ok=True)
        self.input_device = input_device

        # Queues
        self.jobs_queue = queue.Queue()       # Output: (type, data)
        self.events_queue = queue.Queue()     # Input -> GUI events
        self.mic_queue = queue.Queue()        # Raw audio blocks

        # State flags
        self.speaking = False
        self.running = True
        self.mic_enabled = True
        self.mode = "ID_ONLINE"               # "ID_ONLINE", "EN_OFFLINE", "DICTATION_ID", "DICTATION_EN"
        self.threshold_db = -44.0             # Default speech threshold
        self.is_calibrating = False
        self.calibration_samples = []

        # Recognizer instances
        self.sr_recognizer = sr.Recognizer()
        self.vosk_recognizer = None
        self.vosk_model = None
        self.piper_voice = None

        # Try to locate existing Vosk model
        self._init_vosk_if_available()
        # Try to locate existing Piper model
        self._init_piper_if_available()

        # Start Speaker thread
        self.speaker_thread = threading.Thread(target=self._speaker_worker, daemon=True, name="SpeakerWorker")
        self.speaker_thread.start()

        # Start Microphone thread
        self.mic_stream = None
        self.mic_thread = threading.Thread(target=self._mic_worker, daemon=True, name="MicWorker")
        self.mic_thread.start()

    def _init_vosk_if_available(self):
        """Attempts to load Vosk model if present on disk."""
        candidate_paths = [
            os.path.join(self.models_dir, VOSK_MODEL_NAME),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), VOSK_MODEL_NAME),
            os.path.join(r"C:\Users\LENOVO\Documents\5th Semester\HMI\models", VOSK_MODEL_NAME),
        ]
        if getattr(sys, "frozen", False):
            base_dir = os.path.dirname(sys.executable)
            meipass = getattr(sys, "_MEIPASS", base_dir)
            candidate_paths.insert(0, os.path.join(meipass, "models", VOSK_MODEL_NAME))
            candidate_paths.insert(0, os.path.join(meipass, VOSK_MODEL_NAME))
            candidate_paths.insert(0, os.path.join(base_dir, "models", VOSK_MODEL_NAME))
            candidate_paths.insert(0, os.path.join(base_dir, VOSK_MODEL_NAME))

        for p in candidate_paths:
            if os.path.isdir(p):
                try:
                    import vosk
                    vosk.SetLogLevel(-1)
                    self.vosk_model = vosk.Model(p)
                    grammar = json.dumps(EN_GRAMMAR + ["[unk]"])
                    self.vosk_recognizer = vosk.KaldiRecognizer(self.vosk_model, SAMPLE_RATE, grammar)
                    self.vosk_recognizer.SetWords(True)
                    print(f"[speech_engine] Vosk model loaded from {p}")
                    return
                except Exception as e:
                    print(f"[speech_engine] Error loading Vosk: {e}")

    def _init_piper_if_available(self):
        """Attempts to load Piper Indonesian voice if onnx model exists."""
        candidate_paths = [
            os.path.join(self.models_dir, PIPER_MODEL_NAME + ".onnx"),
            os.path.join(r"C:\Users\LENOVO\Documents\5th Semester\HMI\models", PIPER_MODEL_NAME + ".onnx"),
        ]
        if getattr(sys, "frozen", False):
            base_dir = os.path.dirname(sys.executable)
            meipass = getattr(sys, "_MEIPASS", base_dir)
            candidate_paths.insert(0, os.path.join(meipass, "models", PIPER_MODEL_NAME + ".onnx"))
            candidate_paths.insert(0, os.path.join(meipass, PIPER_MODEL_NAME + ".onnx"))
            candidate_paths.insert(0, os.path.join(base_dir, "models", PIPER_MODEL_NAME + ".onnx"))
            candidate_paths.insert(0, os.path.join(base_dir, PIPER_MODEL_NAME + ".onnx"))
        for p in candidate_paths:
            if os.path.isfile(p):
                try:
                    from piper import PiperVoice
                    self.piper_voice = PiperVoice.load(p)
                    print(f"[speech_engine] Piper voice loaded from {p}")
                    return
                except Exception as e:
                    print(f"[speech_engine] Error loading Piper: {e}")

    def is_speaking(self) -> bool:
        """Returns True if the engine is currently playing speech or alarms requiring Half-Duplex protection."""
        if self.speaking:
            return True
        with self.jobs_queue.mutex:
            for item in self.jobs_queue.queue:
                j_type = item[0]
                if j_type == "SAY":
                    return True
                elif j_type == "PLAY_SOUND" and len(item) > 1:
                    payload = item[1]
                    if isinstance(payload, tuple) and len(payload) > 1 and payload[1]:
                        return True
        return False

    def say(self, text: str, sound=None, lang: str = "id", rate: int = 160):
        """Queues a text-to-speech request with optional introductory sound."""
        if text or sound is not None:
            self.jobs_queue.put(("SAY", (text, sound, lang, rate)))

    def play_sound(self, sound_array: np.ndarray, half_duplex: bool = False):
        """Queues a raw sound array to play.
        
        Args:
            sound_array: 1D float32 audio waveform.
            half_duplex: If True, sets self.speaking = True during playback to mute mic and lock gestures
                         (used for long IEC alarms). If False, plays without muting gestures (used for earcons).
        """
        if sound_array is not None and len(sound_array) > 0:
            self.jobs_queue.put(("PLAY_SOUND", (sound_array, half_duplex)))

    def set_mode(self, mode: str):
        """Switches recognition mode ('ID_ONLINE', 'EN_OFFLINE', 'DICTATION_ID', 'DICTATION_EN')."""
        self.mode = mode
        if mode == "EN_OFFLINE" and self.vosk_recognizer is None:
            self.events_queue.put({
                "type": "status",
                "msg": "Vosk offline model not installed. Using online Google ASR (en-US) fallback without freezing.",
            })
        else:
            self.events_queue.put({"type": "status", "msg": f"Voice mode: {mode}"})

    def start_calibration(self):
        """Begins 2-second ambient noise calibration."""
        self.calibration_samples = []
        self.is_calibrating = True
        self.events_queue.put({"type": "status", "msg": "Calibrating mic (2s) - please remain quiet..."})

    def toggle_mic(self) -> bool:
        """Toggles microphone listening on/off."""
        self.mic_enabled = not self.mic_enabled
        status = "ON" if self.mic_enabled else "OFF"
        self.events_queue.put({"type": "status", "msg": f"Microphone is now {status}"})
        return self.mic_enabled

    def _speak_pyttsx3(self, text: str, rate: int = 160):
        """Speaks text using a freshly instantiated pyttsx3 engine in worker thread (safe on Windows)."""
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", rate)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
            del engine
        except Exception as e:
            print(f"[speech_engine] pyttsx3 error: {e}")

    def _speak_edge_tts(self, text: str):
        """Speaks Indonesian text using Microsoft Edge Neural TTS ('id-ID-GadisNeural')."""
        import edge_tts
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tmp_path = f.name
            edge_tts.Communicate(text, "id-ID-GadisNeural").save_sync(tmp_path)
            audio, rate = sf.read(tmp_path, dtype="float32")
            sd.play(audio, rate)
            sd.wait()
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    def _speaker_worker(self):
        """Worker thread processing all TTS speech and earcon sounds sequentially."""
        while self.running:
            try:
                job_type, payload = self.jobs_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                if job_type == "PLAY_SOUND":
                    sound_array, half_duplex = payload if isinstance(payload, tuple) else (payload, False)
                    if half_duplex:
                        self.speaking = True
                    try:
                        sd.play(sound_array, SAMPLE_RATE)
                        sd.wait()
                    finally:
                        if half_duplex:
                            self.speaking = False

                elif job_type == "SAY":
                    self.speaking = True
                    text, sound, lang, rate = payload
                    # Play introductory sound if given
                    if sound is not None and len(sound) > 0:
                        sd.play(sound, SAMPLE_RATE)
                        sd.wait()

                    if text and text.strip():
                        if lang == "id":
                            # Try Piper -> Edge-TTS -> Pyttsx3
                            spoken = False
                            if self.piper_voice is not None:
                                try:
                                    from piper.config import SynthesisConfig
                                    buffer = io.BytesIO()
                                    with sf.SoundFile(buffer, "w", samplerate=SAMPLE_RATE, channels=1, format="WAV") as f:
                                        self.piper_voice.synthesize_wav(text, f, SynthesisConfig(length_scale=160.0 / rate))
                                    buffer.seek(0)
                                    audio, audio_rate = sf.read(buffer, dtype="float32")
                                    sd.play(audio, audio_rate)
                                    sd.wait()
                                    spoken = True
                                except Exception as ep:
                                    print(f"[speech_engine] Piper TTS failed: {ep}")

                            if not spoken:
                                try:
                                    self._speak_edge_tts(text)
                                    spoken = True
                                except Exception as ee:
                                    print(f"[speech_engine] Edge-TTS failed: {ee}")

                            if not spoken:
                                self._speak_pyttsx3(text, rate)
                        else:
                            # English TTS
                            self._speak_pyttsx3(text, rate)

            except Exception as e:
                print(f"[speech_engine] speaker_worker error: {e}")
            finally:
                self.speaking = False

    def _mic_callback(self, indata, frames, t, status):
        """Audio stream input callback."""
        if status:
            pass
        self.mic_queue.put(bytes(indata))

    def _mic_worker(self):
        """Microphone acquisition and Speech Recognition worker loop."""
        chosen_dev = choose_microphone(self.input_device)
        if chosen_dev is None:
            print("[speech_engine] Warning: No audio input device found. Microphone disabled.")
            self.events_queue.put({"type": "status", "msg": "No microphone detected (Touchless/Key/Mouse active)"})
        else:
            try:
                self.mic_stream = sd.RawInputStream(
                    samplerate=SAMPLE_RATE,
                    blocksize=1600,  # 0.1 s at 16 kHz
                    device=chosen_dev,
                    dtype="int16",
                    channels=1,
                    callback=self._mic_callback,
                )
                self.mic_stream.start()
                print(f"[speech_engine] Microphone stream started successfully on device #{chosen_dev}.")
            except Exception as e:
                print(f"[speech_engine] Warning: Could not open microphone stream on device #{chosen_dev}: {e}")
                self.events_queue.put({"type": "status", "msg": f"Microphone unavailable: {e}"})
                self.mic_stream = None

        audio_buffer = []
        silent_blocks = 0
        PAUSE_BLOCKS = 7  # 0.7s pause signifies end of speech phrase

        while self.running:
            try:
                pcm = self.mic_queue.get(timeout=0.2)
            except queue.Empty:
                if self.mic_stream is None:
                    self.events_queue.put({
                        "type": "meter",
                        "dbfs": -60.0,
                        "threshold": self.threshold_db,
                        "speaking": self.is_speaking(),
                        "loud": False,
                        "mic_enabled": False,
                    })
                continue

            samples = np.frombuffer(pcm, dtype=np.int16) / 32768.0
            level = dbfs(samples)
            is_loud = level > self.threshold_db

            # 1. Noise calibration mode
            if self.is_calibrating:
                self.calibration_samples.append(level)
                if len(self.calibration_samples) >= 20:  # ~2 seconds
                    max_ambient = max(self.calibration_samples)
                    self.threshold_db = float(np.clip(max_ambient + 12.0, -55.0, -25.0))
                    self.is_calibrating = False
                    self.events_queue.put({
                        "type": "calibration_done",
                        "threshold_db": self.threshold_db,
                        "msg": f"Calibration complete: Threshold = {self.threshold_db:.1f} dBFS (Ambient: {max_ambient:.1f})"
                    })
                continue

            # Emit level meter event
            self.events_queue.put({
                "type": "meter",
                "dbfs": level,
                "threshold": self.threshold_db,
                "speaking": self.is_speaking(),
                "loud": is_loud,
                "mic_enabled": self.mic_enabled,
            })

            # Strict Half-Duplex guard: Ignore microphone when app is speaking or mic is disabled
            if self.is_speaking() or not self.mic_enabled:
                audio_buffer.clear()
                silent_blocks = 0
                if self.vosk_recognizer is not None:
                    self.vosk_recognizer.Reset()
                continue

            # 2. Vosk Offline Command Mode (if loaded and mode == "EN_OFFLINE")
            if self.mode == "EN_OFFLINE" and self.vosk_recognizer is not None:
                if self.vosk_recognizer.AcceptWaveform(pcm):
                    res = json.loads(self.vosk_recognizer.Result())
                    text = res.get("text", "").strip()
                    if text:
                        words = res.get("result", [])
                        confs = [w.get("conf", 1.0) for w in words if w.get("word") != "[unk]"]
                        avg_conf = (sum(confs) / len(confs)) if confs else 0.8
                        if avg_conf >= 0.55:
                            cmd_id, reply, matched_text = match_command_id([text], lang="en")
                            if cmd_id:
                                self.events_queue.put({
                                    "type": "command",
                                    "cmd": cmd_id,
                                    "reply": reply,
                                    "text": text,
                                    "conf": avg_conf,
                                    "mode": "EN_OFFLINE",
                                })
                continue

            # 3. VAD Buffer Collection for Online / Dictation Modes (or Vosk fallback)
            if not audio_buffer and not is_loud:
                continue

            audio_buffer.append(pcm)
            if is_loud:
                silent_blocks = 0
            else:
                silent_blocks += 1

            # End of speech phrase detected
            if silent_blocks >= PAUSE_BLOCKS and len(audio_buffer) >= 3:
                full_pcm = b"".join(audio_buffer)
                audio_buffer.clear()
                silent_blocks = 0

                # Process in a sub-worker thread to avoid stalling mic loop
                threading.Thread(
                    target=self._process_phrase,
                    args=(full_pcm, self.mode),
                    daemon=True,
                ).start()

    def _process_phrase(self, pcm_bytes: bytes, current_mode: str):
        """Transcribes an accumulated audio buffer via SpeechRecognition (Google ASR)."""
        audio_data = sr.AudioData(pcm_bytes, SAMPLE_RATE, 2)
        lang_code = "id-ID" if ("ID" in current_mode) else "en-US"

        try:
            if current_mode in ("ID_ONLINE", "EN_OFFLINE"):
                result = self.sr_recognizer.recognize_google(audio_data, language=lang_code, show_all=True)
                if not result or "alternative" not in result or len(result["alternative"]) == 0:
                    self.events_queue.put({"type": "status", "msg": "Speech heard, but not understood."})
                    return

                alternatives = [a["transcript"] for a in result["alternative"]]
                best = result["alternative"][0]
                conf = best.get("confidence", 0.88)
                cmd_lang = "id" if ("ID" in current_mode) else "en"
                cmd_id, reply, matched = match_command_id(alternatives, lang=cmd_lang)

                if cmd_id:
                    self.events_queue.put({
                        "type": "command",
                        "cmd": cmd_id,
                        "reply": reply,
                        "text": matched,
                        "conf": conf,
                        "mode": current_mode,
                    })
                else:
                    self.events_queue.put({
                        "type": "unrecognized",
                        "text": alternatives[0],
                        "conf": conf,
                        "msg": f'Unrecognized voice: "{alternatives[0]}"',
                    })

            elif current_mode in ("DICTATION_ID", "DICTATION_EN"):
                text = self.sr_recognizer.recognize_google(audio_data, language=lang_code)
                if text and text.strip():
                    self.events_queue.put({
                        "type": "dictation",
                        "text": text.strip(),
                        "lang": lang_code,
                    })

        except sr.UnknownValueError:
            self.events_queue.put({"type": "status", "msg": "(Speech not understood)"})
        except sr.RequestError as e:
            self.events_queue.put({"type": "status", "msg": f"Google Speech API offline/error: {e}"})
        except Exception as e:
            self.events_queue.put({"type": "status", "msg": f"ASR error: {e}"})

    def stop(self):
        """Stops background threads and releases audio resources."""
        self.running = False
        if self.mic_stream:
            try:
                self.mic_stream.stop()
                self.mic_stream.close()
            except Exception:
                pass
            self.mic_stream = None
        try:
            sd.stop()
        except Exception:
            pass
