"""audio_utils.py - Audio Synthesis, Earcons, IEC 60601-1-8 Alarms & SpO2 Sonification.

Implements tone synthesis, decibel full-scale (dBFS) calculation,
auditory earcons for surgical actions, standardized medical equipment alarms,
and physiological pulse oximetry sonification.
"""
import numpy as np

SAMPLE_RATE = 16000  # 16 kHz standardized for voice & audio synthesis

# Auditory Earcons: List of (frequency_hz, duration_seconds); freq 0 = silence
EARCONS = {
    "berhasil": [(523, 0.10), (659, 0.10), (784, 0.20)],          # Ascending C-E-G (Success)
    "ok": [(660, 0.10), (880, 0.15)],                             # Confirmation chirp
    "error": [(440, 0.15), (330, 0.25)],                          # Descending tone (Action blocked/error)
    "attention": [(880, 0.08), (0, 0.06), (880, 0.08)],           # Double beep (Listening / Detected)
    "bookmark": [(587, 0.08), (880, 0.12)],                       # D5-A5 bookmark chime
    "marker": [(784, 0.08), (1046, 0.14)],                        # G5-C6 surgical marker target ping
}

# IEC 60601-1-8 Standard Medical Alarm Patterns
PULSE_HIGH = [(880, 0.12), (0, 0.06)]                                        # Single high priority pulse
HIGH_BURST = PULSE_HIGH * 3 + [(0, 0.25)] + PULSE_HIGH * 2 + [(0, 0.60)]    # 3+2 pulse burst with inter-burst gap
ALARMS = {
    "low": [(440, 0.20), (0, 0.15), (440, 0.20)],                             # 2 slow 440 Hz pulses
    "medium": [(660, 0.20), (0, 0.10)] * 3,                                    # 3 medium 660 Hz pulses
    "high": HIGH_BURST + HIGH_BURST,                                           # 10 pulses in two 3+2 bursts (880 Hz)
}


def tone(freq: float, seconds: float, volume: float = 0.3, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Generates a sinusoidal audio waveform with 10 ms smooth fade-in/fade-out to eliminate clicking."""
    if seconds <= 0:
        return np.zeros(0, dtype=np.float32)
    t = np.arange(int(seconds * sr)) / sr
    wave = volume * np.sin(2 * np.pi * freq * t)
    fade = min(len(wave) // 2, int(0.01 * sr))
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade)
        wave[:fade] = wave[:fade] * ramp
        wave[-fade:] = wave[-fade:] * ramp[::-1]
    return wave.astype(np.float32)


def sequence(notes, volume: float = 0.3, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Combines a list of (frequency, duration) tuples into a single audio waveform."""
    parts = []
    for freq, sec in notes:
        if freq == 0:
            parts.append(np.zeros(int(sec * sr), dtype=np.float32))
        else:
            parts.append(tone(freq, sec, volume, sr))
    if not parts:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(parts)


def dbfs(block) -> float:
    """Calculates root-mean-square loudness in dBFS: 0 dBFS is full-scale, <= -100 dBFS is silence."""
    x = np.asarray(block, dtype=np.float64)
    if len(x) == 0:
        return -100.0
    rms = np.sqrt(np.mean(x * x))
    level = 20.0 * np.log10(rms + 1e-10)
    if level < -100.0:
        level = -100.0
    return float(level)


def spo2_to_pitch(spo2: float) -> float:
    """Converts blood oxygen saturation (SpO2 percentage) to auditory pitch in Hz.
    
    IEC sonification rule: 100% SpO2 = 880 Hz (A5). Drops by semitones as SpO2 decreases.
    Formula: 880.0 * 2 ** (-(100.0 - spo2) / 12.0)
    """
    clamped_spo2 = max(50.0, min(100.0, float(spo2)))
    pitch = 880.0 * (2.0 ** (-(100.0 - clamped_spo2) / 12.0))
    return float(pitch)


def pulse_ox_tone(spo2: float, duration: float = 0.07, volume: float = 0.25) -> np.ndarray:
    """Generates a single pulse oximeter sonification beep for the given SpO2 value."""
    freq = spo2_to_pitch(spo2)
    return tone(freq, duration, volume=volume, sr=SAMPLE_RATE)
