# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

block_cipher = None

datas = [
    ('../models/blaze_face_short_range.tflite', 'models'),
    ('../models/hand_landmarker.task', 'models'),
    ('../assets/real_head_ct.npz', 'assets'),
    ('../assets/CT_LICENSE.txt', 'assets'),
]
binaries = []
hiddenimports = [
    'ui_v2',
    'ui_v2.theme',
    'ui_v2.dicom_viewport',
    'ui_v2.waveform_monitor',
    'ui_v2.vision_hud',
    'ui_v2.speech_console',
    'pyttsx3.drivers',
    'pyttsx3.drivers.sapi5',
    'comtypes',
    'comtypes.stream',
    'cv2',
    'sounddevice',
    'soundfile',
    'speech_recognition',
    'truststore',
    'edge_tts',
    'vosk',
    'numpy',
    'nibabel',
]

for pkg in [
    'mediapipe',
    '_sounddevice_data',
    '_soundfile_data',
    'speech_recognition',
    'truststore',
    'edge_tts',
    'vosk',
    'nibabel',
]:
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception as e:
        print(f"Warning collecting {pkg}: {e}")

excludes = [
    'tkinter',
    'flet',
    'piper',
    'piper_tts',
    'onnxruntime',
    'torch',
    'transformers',
    'scipy',
    'pandas',
    'IPython',
    'jupyter',
    'notebook',
    'pytest',
    'PyQt5',
    'PyQt6',
    'PySide2',
]

a = Analysis(
    ['../src/or_sterile_assist_v2.py'],
    pathex=['../src'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='OR_SterileAssist_v2',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='OR_SterileAssist_v2',
)
