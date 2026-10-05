"""Cut clips out of the source video with ffmpeg."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


def slugify(text: str, max_len: int = 40) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return s[:max_len] or "clip"


def cut(video: Path, start: float, end: float, out: Path, fast: bool = False) -> None:
    """Accurate re-encode by default; `fast` stream-copies (instant, but snaps to keyframes)."""
    dur = max(0.1, end - start)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.3f}", "-i", str(video), "-t", f"{dur:.3f}"]
    if fast:
        cmd += ["-c", "copy", "-avoid_negative_ts", "make_zero"]
    else:
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-c:a", "aac", "-movflags", "+faststart"]
    cmd.append(str(out))
    subprocess.run(cmd, check=True)
