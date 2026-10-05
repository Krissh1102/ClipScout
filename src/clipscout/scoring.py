"""Scoring: turn raw signals into ranked, non-overlapping moments."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class Moment:
    start: float
    end: float
    score: float
    title: str = ""
    reason: str = ""
    source: str = "signals"

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        d = asdict(self)
        d["duration"] = round(self.duration, 2)
        return d


_HOOKS = [
    re.compile(
        r"\b(secret|truth|mistake|never|always|biggest|worst|best|key|important|crazy|insane|"
        r"incredible|amazing|shocking|surprising|wow|huge|critical)\b",
        re.I,
    ),
    re.compile(
        r"\b(here'?s (the|why|how|what)|the (reason|problem|lesson)|what (if|most)|"
        r"let me tell you|you won'?t believe|turns out|nobody (tells|talks))\b",
        re.I,
    ),
]
_LAUGH = re.compile(r"\b(ha ?ha+|lol|laugh(s|ing|ter)?)\b|\[laughter\]", re.I)


def text_score(text: str) -> float:
    """Cheap 0..1 'this sounds like a highlight' heuristic."""
    hits = sum(len(p.findall(text)) for p in _HOOKS)
    s = 0.25 * min(hits, 3)
    if "?" in text:
        s += 0.15
    if "!" in text:
        s += 0.15
    if re.search(r"\d", text):
        s += 0.1
    if _LAUGH.search(text):
        s += 0.2
    return min(s, 1.0)


def build_signal(duration: float, energy, segments: list[Segment], scenes: list[float]) -> np.ndarray:
    """Per-second interest curve in 0..1 combining loudness, transcript hooks and cut density."""
    T = max(1, int(np.ceil(duration)))

    e = np.zeros(T)
    has_e = energy is not None and len(energy) > 0
    if has_e:
        m = min(T, len(energy))
        e[:m] = np.clip(np.asarray(energy)[:m], 0, 3) / 3

    tx = np.zeros(T)
    for s in segments:
        a = int(s.start)
        if a >= T:
            continue
        b = min(T, max(a + 1, int(np.ceil(s.end))))
        tx[a:b] = np.maximum(tx[a:b], text_score(s.text))

    sc = np.zeros(T)
    for t in scenes:
        i = int(t)
        if 0 <= i < T:
            sc[i] += 1
    sc = np.clip(np.convolve(sc, np.ones(10), mode="same") / 3, 0, 1)

    w_e = 0.4 if has_e else 0.0
    w_t = 0.45 if segments else 0.0
    w_s = 0.15 if scenes else 0.0
    total = w_e + w_t + w_s
    if total == 0:
        return np.zeros(T)
    sig = (w_e * e + w_t * tx + w_s * sc) / total
    return np.convolve(sig, np.ones(3) / 3, mode="same")


def window_score(sig: np.ndarray, start: float, end: float) -> float:
    a = max(0, int(start))
    b = max(a + 1, int(np.ceil(end)))
    w = sig[a:b]
    if len(w) == 0:
        return 0.0
    return float(0.7 * w.mean() + 0.3 * w.max())


def candidate_windows(segments: list[Segment], duration: float, min_len: float, max_len: float) -> list[tuple[float, float]]:
    """Windows that start on a segment boundary and prefer to end at a sentence end."""
    out: set[tuple[float, float]] = set()

    if not segments:  # silent / untranscribed: fixed grid
        step = max(5.0, min_len / 4)
        length = (min_len + max_len) / 2
        t = 0.0
        while t + min_len <= duration:
            out.add((t, min(duration, t + length)))
            t += step
        return sorted(out)

    for i, s in enumerate(segments):
        start = s.start
        first = best = None
        for j in range(i, len(segments)):
            end = segments[j].end
            dur = end - start
            if dur > max_len:
                break
            if dur >= min_len:
                if first is None:
                    first = end
                if segments[j].text.rstrip().endswith((".", "?", "!")):
                    best = end
                    break
        end = best or first
        if end is None and s.end - s.start > max_len:
            end = start + max_len
        if end is None or start + min_len > duration + 1e-6:
            continue
        out.add((start, min(end, duration)))
    return sorted(out)


def snap(start: float, end: float, segments: list[Segment]) -> tuple[float, float]:
    """Snap an LLM-proposed range to the nearest transcript boundaries."""
    if not segments:
        return start, end
    s = min((x.start for x in segments), key=lambda t: abs(t - start))
    e = min((x.end for x in segments), key=lambda t: abs(t - end))
    return (s, e) if e > s else (start, end)


def heuristic_moments(segments: list[Segment], sig: np.ndarray, duration: float, min_len: float, max_len: float) -> list[Moment]:
    moments = []
    for a, b in candidate_windows(segments, duration, min_len, max_len):
        text = " ".join(s.text for s in segments if s.start >= a - 0.01 and s.end <= b + 0.01)
        title = " ".join(text.split()[:8]) or f"Moment at {a:.0f}s"
        moments.append(Moment(a, b, window_score(sig, a, b), title, "energy / hook words / cut density", "signals"))
    return moments


def select_top(moments: list[Moment], n: int) -> list[Moment]:
    """Greedy best-first pick of n non-overlapping moments, returned in score order."""
    chosen: list[Moment] = []
    for m in sorted(moments, key=lambda m: -m.score):
        if all(min(m.end, c.end) - max(m.start, c.start) <= 0 for c in chosen):
            chosen.append(m)
        if len(chosen) >= n:
            break
    return chosen
