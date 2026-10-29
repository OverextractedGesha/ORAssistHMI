# OR-SterileAssist — Touchless Intra-Operative CT & Vital Monitor

**Course:** Human-Machine Interface (HMI) — ETS Project  
**Platform:** Windows 10/11, Python 3.14 (Virtual Environment `.venv`)  
**Core Technologies:** Tkinter (Event-driven GUI), OpenCV & MediaPipe Tasks (Computer Vision), SoundDevice & NumPy (Real-Time Audio Synthesis), SpeechRecognition & Vosk (Bilingual ASR), Edge-TTS & SAPI5 pyttsx3 (TTS).

---

## 1. Project Overview & Clinical Problem

In modern operating rooms (OR) and interventional neurosurgical suites, neurosurgeons must frequently consult intra-operative CT/MRI scans and patient physiological monitors while performing sterile procedures. Operating conventional input devices (mice, keyboards, touchscreens) violates the surgical sterile field, necessitating glove changes or verbal delegation to circulating nurses—both of which cause delays, miscommunication, and infection risks.

**OR-SterileAssist** provides an intra-operative touchless multimodal workstation combining:
1. **Touchless CT Navigation & Surgical Target Marking**: Natural hand gestures and a 1-finger virtual surgical laser pointer with dwell progress feedback to browse slices and mark pathological lesions without contact.
2. **Standardized IEC 60601-1-8 Medical Alarms & SpO2 Pitch Sonification**: Real-time auditory display of patient vital status and a regulatory 2-minute alarm silence countdown.
3. **Bilingual Speech Recognition (Indonesian & English)**: Hands-free command execution and intra-operative clinical notes dictation with strict half-duplex acoustic protection.
4. **Surgeon Presence & Sterile Zone Auto-Lock**: BlazeFace-based distance detection to automatically freeze touchless input when the surgeon steps back or is not in an optimal operating position.
5. **Redundant Physical Controls**: Full mouse and keyboard shortcuts ensuring zero single-point-of-failure in critical medical environments.

---

## 2. System Architecture & Modular Structure

```
Project ETS HMI/
├── trackers.py              # MediaPipe Tasks wrappers: FaceTracker, HandTracker, draw_hand()
├── gesture_utils.py         # Geometric landmark math, finger count, pinch, laser pointer, StableGesture
├── ct_phantom.py            # 3D Head CT phantom (60x380x380 HU), clinical windowing, reticle rendering
├── audio_utils.py           # Sine tone synthesis, dBFS calculation, Earcons, IEC 60601-1-8 alarms, SpO2 pitch
├── speech_engine.py         # Thread-safe ASR & TTS engine (Half-Duplex, Google online, Vosk offline, pyttsx3)
├── or_sterile_assist.py     # Main Tkinter event-driven surgical application
├── test_or_sterile_assist.py# Automated unit test suite & headless smoke tests
└── requirements.txt         # Project dependencies
```

---

## 3. Surgical Gesture Mapping

| Gesture | Pose / Dynamic | Action in Workstation | Auditory / Visual Feedback |
| :--- | :--- | :--- | :--- |
| **🤏 Pinch** | Thumb & Index pinch (`ratio < 0.28`), move hand up/down | Scroll CT slices forward / backward | Live slice scrolling + subtle click earcon |
| **☝ 1 Finger** | Single extended index finger, others curled (`is_index_pointer`) | **Virtual Surgical Laser Pointer**: Hover on CT | Cyan surgical crosshair at pointing position |
| **☝ Dwell** | Hold laser pointer steady on target for 1.4s (42 frames) | **Mark Surgical Target Lesion** | Circular green progress ring -> Target reticle + `marker` earcon + TTS confirmation |
| **✌ 2 Fingers** | Hold 2 fingers extended for 0.5s (15 frames) | **Cycle CT Window** (Brain ➔ Bone ➔ Subdural) | `ok` earcon + window badge update |
| **🤟 3 Fingers** | Hold 3 fingers extended for 0.5s (15 frames) | **Bookmark Current Slice** | `bookmark` chime + badge on CT slice |
| **🖖 4 Fingers** | Hold 4 fingers extended for 0.5s (15 frames) | **Speak Patient Vitals Summary** | `attention` earcon + synthesized speech report |
| **🖐 5 Fingers** | Open palm (5 fingers) held for 0.5s (15 frames) | **Toggle Sterile Gesture Lock** (Pause/Resume) | Yellow/Green status badge + voice announcement |

---

## 4. Voice Commands & Multimodal Speech Engine

The speech engine operates in multiple user-selectable modes:
- **`ID Online (Google)`**: Indonesian surgical commands using Google Web Speech (`id-ID`) with `truststore` SSL certificate security.
- **`EN Offline (Vosk)`**: English offline commands using Vosk / Kaldi with constrained grammar and confidence thresholding (`MIN_CONF = 0.6`).
- **`Dictation (ID)`**: Free-form clinical speech-to-text logging directly into the Intra-Operative Surgical Log.

### Indonesian Voice Commands
- `"iris berikutnya"`, `"naik"` ➔ Navigate to next slice
- `"iris sebelumnya"`, `"turun"` ➔ Navigate to previous slice
- `"jendela otak"`, `"jendela tulang"`, `"jendela subdural"` ➔ Switch CT windowing preset
- `"simpan iris"`, `"bookmark"` ➔ Bookmark current CT slice
- `"tandai lesi"`, `"tandai target"` ➔ Mark lesion at current cursor
- `"hapus tanda"`, `"bersihkan tanda"` ➔ Clear markers on slice
- `"tanda vital"`, `"status pasien"` ➔ Read patient vitals summary
- `"detak jantung"`, `"saturasi"`, `"tekanan darah"` ➔ Read specific vital sign
- `"hening alarm"`, `"matikan alarm"` ➔ Silence alarm for 2 minutes
- `"jeda kamera"`, `"lanjut kamera"` ➔ Toggle gesture lock
- `"bantuan"` ➔ Spoken voice command guide

---

## 5. Biomedical Engineering & Ergonomics Standards

### A. IEC 60601-1-8 Medical Equipment Alarms
Standardized medical auditory alarm bursts are synthesized via NumPy:
- **Low Priority**: 2 slow 440 Hz pulses (`[(440, 0.20), (0, 0.15), (440, 0.20)]`).
- **Medium Priority**: 3 pulses at 660 Hz (`[(660, 0.20), (0, 0.10)] * 3`).
- **High Priority**: 10 pulses arranged in two 3+2 bursts at 880 Hz separated by inter-burst pauses.
- **Alarm Silence (2 Minutes)**: Implements standard regulatory 120-second mute timer with dynamic UI countdown and keyboard shortcut (`A`).

### B. Pulse Oximetry Sonification (Pitch = SpO2)
Auditory display maps blood oxygen saturation directly to pitch:
$$\text{Pitch (Hz)} = 880.0 \times 2^{-\frac{100 - \text{SpO}_2}{12}}$$
At 100% $\text{SpO}_2$, pitch is 880 Hz (A5). Drops by exact musical semitones as oxygenation decreases, providing immediate peripheral acoustic awareness to the surgical team without requiring screen gaze.

### C. Midas Touch Mitigation Mechanisms
To prevent accidental actions caused by natural hand gestures or bystander movements:
1. **`StableGesture` Debouncing**: Demands consistent pose detection across 15 consecutive frames (~0.5 s) before triggering actions.
2. **Laser Pointer Dwell Circle**: Marking lesions requires holding pointer within a 4.5% tolerance region for 1.4 seconds with real-time circular visual progress arc feedback.
3. **Sterile Gesture Lock**: Manual toggle via 5 fingers or Spacebar to pause gesture parsing during surgical hand adjustments.
4. **Face Presence Auto-Lock**: BlazeFace tracking checks if the surgeon is present within an optimal distance (20%–55% of frame width); pauses gestures if surgeon steps away.
5. **Strict Half-Duplex Protection**: While earcons, alarms, or TTS speech output is active, microphone recognition and gesture speech triggers are muted to eliminate acoustic echo loops.

---

## 6. How to Run & Verify

### Double-Click Launchers (Siap Pakai)
For ease of presentation and immediate setup on any Windows machine:
1. **`1_SETUP_DEPENDENCIES.bat`**: Automatic dependency installer & virtual environment setup script.
2. **`2_JALANKAN_APLIKASI.bat`**: Console-enabled launcher (ideal for viewing live tracking logs and frame rates).
3. **`3_JALANKAN_TANPA_CMD.vbs`**: Silent VBScript launcher (`wscript.exe`) that starts `pythonw.exe` with **zero console flash**.
4. **`PETUNJUK_PENGGUNAAN_TEMAN.txt`**: Complete step-by-step Indonesian guide for group members.

### Running via Terminal / PowerShell
```powershell
.venv\Scripts\python.exe or_sterile_assist.py
```

### Running Automated Unit Tests
Verify all 22 modular unit tests:
```powershell
.venv\Scripts\python.exe -m unittest test_or_sterile_assist.py -v
```
Expected output:
```
Ran 22 tests in ~11s
OK
```

---

## 7. Redundant Keyboard & Mouse Controls

| Key / Mouse Action | Function |
| :--- | :--- |
| **`◀` / `Down` / `P`** | Previous slice |
| **`▶` / `Up` / `N`** | Next slice |
| **`W`** | Cycle window (Brain / Bone / Subdural) |
| **`B`** | Bookmark current slice |
| **`M`** | Mark lesion at cursor position |
| **`C`** | Clear markers on current slice |
| **`V`** | Speak patient vitals summary |
| **`A`** | Silence active alarm for 2 minutes |
| **`Space`** | Toggle Sterile Gesture Lock |
| **`Escape` (inside Notes)** | Return focus from notes editor back to root |
| **Mouse Wheel on CT** | Scrub slices directly |
| **Mouse Move on CT** | Inspect Hounsfield Units (HU) under crosshair |
| **Mouse Click on CT** | Place surgical target marker (restores keyboard focus) |
| **Scrub Slider** | Scrub through volume slices 0–59 |

> **Note on Text Focus Protection:** When keyboard focus is inside the Clinical Notes & Dictation editor, single-key shortcuts (`W`, `B`, `M`, `C`, `V`, `A`, `P`, `N`, `Space`, Arrows) are automatically guarded to prevent accidental triggering during note entry. Pressing `Escape` or clicking the CT canvas returns focus to the main window so shortcuts function immediately.
