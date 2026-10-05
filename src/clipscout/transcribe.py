"""Speech-to-text with word-level-ish segment timestamps (local, via faster-whisper)."""

from __future__ import annotations

from pathlib import Path

from .scoring import Segment


def transcribe(audio: Path, model_size: str = "small", language: str | None = None, device: str = "auto") -> list[Segment]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:  # pragma: no cover
        raise SystemExit("faster-whisper is not installed. Run: pip install clipscout  (or use --no-transcribe)") from e

    model = WhisperModel(model_size, device=device, compute_type="auto")
    segments, _info = model.transcribe(str(audio), language=language, vad_filter=True)
    return [Segment(start=float(s.start), end=float(s.end), text=s.text.strip()) for s in segments if s.text.strip()]
