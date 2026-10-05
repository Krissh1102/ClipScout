"""Low-level media signals: audio extraction, loudness curve, scene cuts."""

from __future__ import annotations

import re
import subprocess
import wave
from pathlib import Path

import numpy as np


def probe_duration(video: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return float(out)


def extract_audio(video: Path, wav: Path) -> None:
    """Mono 16 kHz WAV, the format Whisper wants."""
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", str(wav)],
        check=True,
    )


def _zscore(x: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return x
    sd = x.std()
    return (x - x.mean()) / (sd if sd > 1e-9 else 1.0)


def audio_energy(wav: Path) -> np.ndarray:
    """Per-second log-RMS loudness, z-scored. Streams the file so long videos stay cheap."""
    out: list[float] = []
    with wave.open(str(wav)) as w:
        sr = w.getframerate()
        while True:
            frames = w.readframes(sr)
            if not frames:
                break
            x = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(x * x))) if len(x) else 0.0
            out.append(np.log(rms + 1e-6))
    return _zscore(np.array(out, dtype=np.float32))


def scene_times(video: Path, threshold: float = 0.3, fps: int = 2) -> list[float]:
    """Timestamps (seconds) of visual cuts. Frames are downscaled and sampled for speed."""
    vf = f"fps={fps},scale=320:-2,select='gt(scene,{threshold})',showinfo"
    p = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(video), "-an", "-vf", vf, "-f", "null", "-"],
        capture_output=True,
        text=True,
    )
    return [float(m) for m in re.findall(r"pts_time:([0-9.]+)", p.stderr)]
