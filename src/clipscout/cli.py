"""Command-line interface."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from . import __version__
from .pipeline import run


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="clipscout", description="Find and clip the key moments of a long video.")
    p.add_argument("video", type=Path)
    p.add_argument("-o", "--out", type=Path, default=None, help="output folder (default: <video>_clips)")
    p.add_argument("-n", "--num-clips", type=int, default=8)
    p.add_argument("--min-len", type=float, default=20, help="min clip seconds")
    p.add_argument("--max-len", type=float, default=60, help="max clip seconds")
    p.add_argument("--whisper-model", default="small", help="tiny/base/small/medium/large-v3")
    p.add_argument("--language", default=None, help="e.g. en, hi (default: auto-detect)")
    p.add_argument("--llm", dest="llm", action="store_true", default=None, help="force Claude re-ranking (needs ANTHROPIC_API_KEY)")
    p.add_argument("--no-llm", dest="llm", action="store_false", help="disable Claude re-ranking")
    p.add_argument("--no-transcribe", action="store_true", help="signals only (audio energy + scene cuts); no Whisper")
    p.add_argument("--fast", action="store_true", help="stream-copy cuts: instant, but may be off by a keyframe")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="print the picks without writing clips")
    p.add_argument("--version", action="version", version=f"clipscout {__version__}")
    a = p.parse_args(argv)

    if not a.video.exists():
        p.error(f"file not found: {a.video}")
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            p.error(f"{tool} not found on PATH; install ffmpeg first")
    if a.min_len >= a.max_len:
        p.error("--min-len must be smaller than --max-len")

    out = a.out or a.video.with_name(a.video.stem + "_clips")
    out.mkdir(parents=True, exist_ok=True)
    run(
        a.video,
        out,
        n=a.num_clips,
        min_len=a.min_len,
        max_len=a.max_len,
        whisper_model=a.whisper_model,
        language=a.language,
        use_llm=a.llm,
        do_transcribe=not a.no_transcribe,
        fast=a.fast,
        use_cache=not a.no_cache,
        dry_run=a.dry_run,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
