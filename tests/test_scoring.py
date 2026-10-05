import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from clipscout.scoring import (
    Segment,
    build_signal,
    candidate_windows,
    heuristic_moments,
    select_top,
    snap,
    text_score,
)


def make_segments():
    # 10 minutes of 5-second sentences
    return [Segment(i * 5.0, i * 5.0 + 5.0, f"Sentence number {i}.") for i in range(120)]


def test_text_score_prefers_hooks():
    assert text_score("Here's the biggest mistake! Why 90 percent fail?") > text_score("and then we went to the store.")


def test_candidates_respect_length_bounds():
    segs = make_segments()
    for a, b in candidate_windows(segs, 600, 20, 60):
        assert 20 <= b - a <= 60


def test_select_top_is_non_overlapping_and_sorted_by_score():
    segs = make_segments()
    energy = np.zeros(600)
    energy[200:230] = 3  # loud stretch
    sig = build_signal(600, energy, segs, [])
    picks = select_top(heuristic_moments(segs, sig, 600, 20, 60), 5)
    assert 1 <= len(picks) <= 5
    for i, a in enumerate(picks):
        for b in picks[i + 1 :]:
            assert min(a.end, b.end) <= max(a.start, b.start)
    assert picks[0].start < 230 and picks[0].end > 200  # best pick hits the loud stretch
    assert [m.score for m in picks] == sorted((m.score for m in picks), reverse=True)


def test_no_transcript_uses_grid():
    wins = candidate_windows([], 300, 20, 60)
    assert wins and all(b - a >= 20 for a, b in wins)


def test_snap_moves_to_segment_boundaries():
    segs = make_segments()
    assert snap(12.3, 41.8, segs) == (10.0, 40.0)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_end_to_end_signals_only(tmp_path: Path):
    video = tmp_path / "t.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=duration=90:size=320x180:rate=15",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=90",
            "-af", "volume='if(between(t,40,60),1.0,0.05)':eval=frame",
            "-c:v", "libx264", "-c:a", "aac", "-shortest", str(video),
        ],
        check=True,
    )
    from clipscout.pipeline import run

    out = tmp_path / "out"
    out.mkdir()
    picks = run(video, out, n=2, min_len=10, max_len=25, do_transcribe=False, use_llm=False, log=lambda *_: None)
    assert picks
    assert (out / "manifest.json").exists()
    assert list(out.glob("clip*.mp4"))
    assert picks[0].end > 40 and picks[0].start < 60  # loud section wins
