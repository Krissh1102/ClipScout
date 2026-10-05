"""End-to-end pipeline: video in, ranked clips + manifest out."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from . import clipper, signals
from .scoring import Moment, Segment, build_signal, heuristic_moments, select_top, snap, window_score

HEURISTIC_PENALTY = 0.6  # LLM-picked moments outrank pure-signal ones


def run(
    video: Path,
    out_dir: Path,
    n: int = 8,
    min_len: float = 20.0,
    max_len: float = 60.0,
    whisper_model: str = "small",
    language: str | None = None,
    use_llm: bool | None = None,
    do_transcribe: bool = True,
    fast: bool = False,
    use_cache: bool = True,
    dry_run: bool = False,
    log=print,
) -> list[Moment]:
    video = Path(video)
    out_dir = Path(out_dir)
    work = out_dir / ".work"
    work.mkdir(parents=True, exist_ok=True)

    duration = signals.probe_duration(video)
    log(f"Video: {video.name} ({duration / 60:.1f} min)")

    # 1. audio + loudness
    energy_f = work / "energy.npy"
    if use_cache and energy_f.exists():
        energy = np.load(energy_f)
    else:
        log("Extracting audio…")
        wav = work / "audio.wav"
        signals.extract_audio(video, wav)
        energy = signals.audio_energy(wav)
        np.save(energy_f, energy)

    # 2. scene cuts
    scenes_f = work / "scenes.json"
    if use_cache and scenes_f.exists():
        scenes = json.loads(scenes_f.read_text())
    else:
        log("Detecting scene changes…")
        scenes = signals.scene_times(video)
        scenes_f.write_text(json.dumps(scenes))

    # 3. transcript
    segments: list[Segment] = []
    if do_transcribe:
        tr_f = work / f"transcript.{whisper_model}.json"
        if use_cache and tr_f.exists():
            segments = [Segment(**d) for d in json.loads(tr_f.read_text())]
        else:
            from .transcribe import transcribe

            log(f"Transcribing with Whisper ({whisper_model})…")
            segments = transcribe(work / "audio.wav", whisper_model, language)
            tr_f.write_text(json.dumps([s.__dict__ for s in segments]))
        log(f"Transcript: {len(segments)} segments")

    # 4. score
    sig = build_signal(duration, energy, segments, scenes)
    pool = heuristic_moments(segments, sig, duration, min_len, max_len)
    for m in pool:
        m.score *= HEURISTIC_PENALTY

    if use_llm is None:
        use_llm = bool(os.environ.get("ANTHROPIC_API_KEY")) and bool(segments)
    if use_llm and segments:
        from .llm import find_moments

        log("Asking Claude to pick highlights…")
        for d in find_moments(segments, n, min_len, max_len):
            a, b = snap(d["start"], d["end"], segments)
            if not (min_len * 0.7 <= b - a <= max_len * 1.3):
                continue
            llm_part = max(0.0, min(1.0, d["score"] / 10))
            score = 0.7 * llm_part + 0.3 * window_score(sig, a, b)
            pool.append(Moment(a, b, score, d["title"], d["reason"], "llm"))

    picks = select_top(pool, n)
    if not picks:
        log("No moments found.")
        return []

    # 5. cut
    manifest = []
    for rank, m in enumerate(picks, 1):
        name = f"clip{rank:02d}_{clipper.slugify(m.title)}.mp4"
        entry = {"rank": rank, "file": name, **m.to_dict()}
        manifest.append(entry)
        log(f"#{rank} {m.start:7.1f}s–{m.end:7.1f}s  score={m.score:.2f} [{m.source}]  {m.title}")
        if not dry_run:
            clipper.cut(video, m.start, m.end, out_dir / name, fast=fast)

    (out_dir / "manifest.json").write_text(json.dumps({"video": str(video), "duration": duration, "clips": manifest}, indent=2))
    log(f"Done. Output in {out_dir}/")
    return picks
