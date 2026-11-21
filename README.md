# OR-SterileAssist v2 — Touchless Intra-Operative CT & Vital Monitor

**Course:** Human-Machine Interface (HMI) — ETS Project  
**Platform:** Windows 10/11, Python 3.10+ (Virtual Environment `.venv`)  
**Core Technologies:** PySide6 (v2 "OR Obsidian" Medical Workstation) + Tkinter (v1 Legacy Fallback), OpenCV & MediaPipe Tasks (Hand & BlazeFace Vision), NiBabel & NumPy (Real `.npz` / `.nii` CT Volumes), SoundDevice (60 FPS Audio/Pulse-Ox & IEC 60601-1-8 Alarms), SpeechRecognition & Vosk (Bilingual Half-Duplex ASR), Edge-TTS & SAPI5 `pyttsx3` (TTS).

---

## 1. Overview: v2 Medical Workstation + Preserved v1

**OR-SterileAssist v2** upgrades the touchless neurosurgical workstation into a dark-room **PySide6 ("OR Obsidian")** clinical interface while preserving the original **v1 Tkinter** application as a lightweight fallback:

1. **Real Clinical Cranial Lesion CT & NIfTI Support**: Launches directly onto a real cranial CT volume (`assets/real_head_ct.npz`, 149 slices of `224x224` axial HU data) centered on **Slice 101/149** with a pre-calibrated focal hyperdense lesion marker (span slices 41–136), plus instant switching to the 60-slice Synthetic Phantom or external `.nii` / `.nii.gz` NIfTI volumes.
2. **PACS-Grade Vector DICOM Viewport**: Calibrated millimeter scale bar, slice rail with bookmark/lesion ticks, live HU cursor readout, and 1-finger virtual surgical laser pointer with circular dwell progress.
3. **60 FPS Physiological Waveforms & IEC 60601-1-8 Alarms**: Real-time sweeping ECG II and PLETH SpO2 traces, synchronized pulse-oximeter pitch sonification, clinical scenario simulation (`normal`, `mild`, `critical`), and a regulatory 120-second alarm silence countdown.
4. **Bilingual Half-Duplex Voice Control & Clinical Dictation**: Indonesian online commands (`ID Online`), English offline Vosk grammar (`EN Offline`), and free-form Indonesian intra-operative dictation (`Dictation ID`) with `.txt` export and strict acoustic echo protection.
5. **Surgeon Presence & Midas Touch Mitigation**: BlazeFace auto-lock when the surgeon steps outside the sterile camera zone, 15-frame `StableGesture` debouncing, 42-frame (~1.4 s) laser dwell marking, and full redundant mouse/keyboard controls.

---

## 2. Directory Structure

```text
OR_SterileAssist_SiapPakai/
├── assets/
│   ├── real_head_ct.npz             # Bundled Real Clinical Cranial Lesion CT (149x224x224 HU)
│   └── CT_LICENSE.txt               # Apache 2.0 dataset license notice
├── models/
│   ├── blaze_face_short_range.tflite# Offline MediaPipe BlazeFace model
│   └── hand_landmarker.task         # Offline MediaPipe Hand Landmarker model
├── scripts/
│   ├── OR_SterileAssist.spec        # PyInstaller spec for v1
│   └── OR_SterileAssist_v2.spec     # PyInstaller spec for v2
├── src/
│   ├── ui_v2/                       # PySide6 "OR Obsidian" modular UI components
│   │   ├── __init__.py
│   │   ├── dicom_viewport.py        # Vector CT viewport, scale bar, slice rail, laser reticle
│   │   ├── speech_console.py        # ASR mode selector, dBFS meter, clinical log & export
│   │   ├── theme.py                 # Dark-room surgical palette & QSS styling
│   │   ├── vision_hud.py            # Webcam HUD, sterile lock & dwell indicators
│   │   └── waveform_monitor.py      # 60 FPS ECG II & PLETH SpO2 waveforms + alarm banner
│   ├── audio_utils.py               # IEC 60601-1-8 alarms, earcons, dBFS & SpO2 pitch synthesis
│   ├── ct_phantom.py                # Real CT (.npz), NIfTI (.nii/.nii.gz) & synthetic phantom loader
│   ├── gesture_utils.py             # Landmark math, finger count, pinch clutch, StableGesture
│   ├── or_sterile_assist.py         # Legacy v1 Tkinter workstation (preserved)
│   ├── or_sterile_assist_v2.py      # Main v2 PySide6 medical workstation (--self-check supported)
│   ├── speech_engine.py             # Half-duplex bilingual ASR/TTS engine
│   └── trackers.py                  # MediaPipe FaceTracker & HandTracker wrappers
├── tests/
│   ├── test_or_sterile_assist.py    # 22 unit & headless smoke tests (core + v1)
│   └── test_or_sterile_assist_v2.py # 13 offscreen PySide6 & real CT integration tests (v2)
├── 1_SETUP_DEPENDENCIES.bat         # Step 1: Creates .venv and installs requirements.txt
├── 2_JALANKAN_APLIKASI_V2.bat       # Step 2: Launches v2 PySide6 workstation (with console)
├── 2_JALANKAN_APLIKASI_V1.bat       # Step 2 (fallback): Launches v1 Tkinter workstation
├── 3_JALANKAN_V2_TANPA_CMD.vbs      # Step 2 (silent): Launches v2 via pythonw.exe (no CMD window)
├── PETUNJUK_PENGGUNAAN.txt          # Quickstart guide in Indonesian
├── VOICE_COMMANDS_TUTORIAL.txt      # Complete ID/EN voice command & calibration reference
├── requirements.txt                 # Python dependencies (incl. PySide6, nibabel)
└── README.md
```

---

## 3. Quickstart (Windows)

### Step 1: One-Time Dependency Setup
Double-click **`1_SETUP_DEPENDENCIES.bat`**.  
It detects Python 3.10+, creates a local `.venv`, upgrades `pip`, and installs `requirements.txt` (including `PySide6` and `nibabel`).

### Step 2: Launch the Workstation
- **v2 PySide6 Workstation (Recommended)**: Double-click **`2_JALANKAN_APLIKASI_V2.bat`** (with console logs) or **`3_JALANKAN_V2_TANPA_CMD.vbs`** (silent, no CMD window).
- **v1 Legacy Tkinter Fallback**: Double-click **`2_JALANKAN_APLIKASI_V1.bat`**.

Or launch directly from PowerShell / Command Prompt:
```powershell
.venv\Scripts\python.exe src\or_sterile_assist_v2.py
```

---

## 4. Surgical Hand Gestures (6 Gestures)

| Gesture | Pose / Dynamic | Workstation Action | Feedback |
| :--- | :--- | :--- | :--- |
| **Pinch Clutch** | Thumb & Index pinch (`ratio < 0.28`), move hand up/down | Scroll CT slices forward / backward | Live slice scrubbing + `click` earcon |
| **1 Finger (Pointer + Dwell)** | Single extended index finger; hold steady for 1.4 s (42 frames) | Virtual surgical laser pointer; dwell marks lesion ROI | Cyan crosshair + circular dwell progress ring + target reticle |
| **2 Fingers** | Hold 2 fingers extended for 0.5 s (15 frames) | Cycle CT window (`Brain` ➔ `Bone` ➔ `Subdural`) | `ok` earcon + window badge update |
| **3 Fingers** | Hold 3 fingers extended for 0.5 s (15 frames) | Bookmark current CT slice | `bookmark` chime + slice rail tick |
| **4 Fingers** | Hold 4 fingers extended for 0.5 s (15 frames) | Speak patient vitals summary | `attention` earcon + spoken HR/SpO2/BP |
| **5 Fingers (Open Palm)** | Hold all 5 fingers open for 0.5 s (15 frames) | Toggle Sterile Gesture Lock (Pause / Resume) | Lock status badge + voice confirmation |

---

## 5. Voice Commands (Indonesian & English)

Switch modes directly in the Speech Console (`ID Online`, `EN Offline`, `Dictation ID`). Full details are in [VOICE_COMMANDS_TUTORIAL.txt](VOICE_COMMANDS_TUTORIAL.txt).

| Command ID | Indonesian (`ID Online`) | English (`EN Offline`) | Action |
| :--- | :--- | :--- | :--- |
| `NEXT_SLICE` | `"iris berikutnya"`, `"berikutnya"`, `"naik"`, `"maju"` | `"next slice"`, `"next"` | Next CT slice |
| `PREV_SLICE` | `"iris sebelumnya"`, `"sebelumnya"`, `"turun"`, `"mundur"` | `"previous slice"`, `"previous"`, `"prev"` | Previous CT slice |
| `WINDOW_BRAIN` | `"jendela otak"`, `"mode otak"`, `"otak"` | `"window brain"`, `"brain window"`, `"brain"` | Brain window (`L:40 W:80`) |
| `WINDOW_BONE` | `"jendela tulang"`, `"mode tulang"`, `"tulang"` | `"window bone"`, `"bone window"`, `"bone"` | Bone window (`L:500 W:2000`) |
| `WINDOW_SUBDURAL` | `"jendela subdural"`, `"mode subdural"`, `"subdural"` | `"window subdural"`, `"subdural window"`, `"subdural"` | Subdural window (`L:70 W:200`) |
| `BOOKMARK_SLICE` | `"simpan iris"`, `"bookmark"`, `"tanda iris"`, `"simpan"` | `"bookmark slice"`, `"bookmark"`, `"save slice"` | Bookmark active slice |
| `MARK_LESION` | `"tandai lesi"`, `"tandai target"`, `"tandai"`, `"lesi"` | `"mark lesion"`, `"mark target"`, `"mark"` | Mark lesion at cursor |
| `CLEAR_MARKERS` | `"hapus tanda"`, `"hapus marker"`, `"bersihkan tanda"` | `"clear markers"`, `"clear marker"`, `"clear"` | Clear markers on active slice |
| `PATIENT_VITALS` | `"tanda vital"`, `"kondisi pasien"`, `"status pasien"`, `"vital"` | `"patient vitals"`, `"vitals summary"`, `"vitals"` | Speak full vitals readout |
| `HEART_RATE` | `"detak jantung"`, `"nadi"`, `"jantung"` | `"heart rate"`, `"pulse"` | Speak heart rate |
| `OXYGEN` | `"saturasi"`, `"oksigen"` | `"oxygen"`, `"saturation"`, `"spo2"` | Speak SpO2 saturation |
| `BLOOD_PRESSURE` | `"tekanan darah"`, `"tensi"` | `"blood pressure"`, `"bp"` | Speak arterial NIBP |
| `SILENCE_ALARM` | `"hening alarm"`, `"matikan alarm"`, `"senyap"`, `"diam"` | `"silence alarm"`, `"mute alarm"`, `"silence"` | Mute alarm for 120 seconds |
| `PAUSE_CAMERA` | `"jeda kamera"`, `"kunci gestur"`, `"pause kamera"` | `"pause camera"`, `"lock gestures"`, `"pause"` | Lock gesture input |
| `RESUME_CAMERA` | `"lanjut kamera"`, `"resume kamera"`, `"aktifkan kamera"`, `"buka gestur"` | `"resume camera"`, `"unlock gestures"`, `"resume"` | Unlock gesture input |
| `HELP` | `"bantuan"`, `"tolong"`, `"menu"` | `"help"`, `"commands"` | Spoken command guide |

---

## 6. Redundant Keyboard & Mouse Controls

| Input | Action |
| :--- | :--- |
| **`◀` / `Down` / `P`** | Previous CT slice |
| **`▶` / `Up` / `N`** | Next CT slice |
| **`W`** | Cycle CT window (`Brain` ➔ `Bone` ➔ `Subdural`) |
| **`B`** | Bookmark current slice |
| **`M`** | Mark surgical target lesion at cursor |
| **`C`** | Clear markers on current slice |
| **`V`** | Speak patient vitals summary |
| **`A`** | Silence active alarm for 2 minutes (IEC 60601-1-8) |
| **`Space`** | Toggle Sterile Gesture Lock |
| **`Esc` (in Clinical Log)** | Return focus from text editor to workstation shortcuts |
| **Mouse Wheel / Move / Click on CT** | Scroll slices, inspect live HU, and place lesion markers |

---

## 7. Automated Tests & Self-Check

Run all core/v1 unit tests, v2 PySide6 offscreen integration tests, and the headless workstation self-check:

```powershell
.venv\Scripts\python.exe -m unittest tests\test_or_sterile_assist.py -v
.venv\Scripts\python.exe -m unittest tests\test_or_sterile_assist_v2.py -v
.venv\Scripts\python.exe src\or_sterile_assist_v2.py --self-check
```
