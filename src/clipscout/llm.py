"""Optional LLM pass: ask Claude to pick self-contained highlights from the transcript."""

from __future__ import annotations

import json
import math
import os

from .scoring import Segment

DEFAULT_MODEL = os.environ.get("CLIPSCOUT_MODEL", "claude-sonnet-5-5")
CHUNK_SECONDS = 900  # 15 min of transcript per request keeps long videos within context

PROMPT = """You are an expert video editor choosing short highlight clips from a transcript.
Each line is "[start_seconds] text".

Pick up to {k} moments. Rules:
- Each moment must be between {min_len} and {max_len} seconds long.
- It must make sense on its own: begin at a natural start, end at a natural stop.
- Prefer hooks, key insights, emotional peaks, punchlines, surprising claims, strong conclusions.
- Moments must not overlap.
- Use start/end values taken from the timestamps in the transcript.

Reply with ONLY a JSON array, no prose. Each item:
{{"start": <seconds>, "end": <seconds>, "score": <1-10>, "title": "<=8 words", "reason": "<one sentence>"}}

Transcript:
{transcript}
"""


def _chunks(segments: list[Segment], seconds: int = CHUNK_SECONDS) -> list[list[Segment]]:
    out: list[list[Segment]] = []
    cur: list[Segment] = []
    base = segments[0].start if segments else 0.0
    for s in segments:
        if s.start - base > seconds and cur:
            out.append(cur)
            cur, base = [], s.start
        cur.append(s)
    if cur:
        out.append(cur)
    return out


def _parse(text: str) -> list[dict]:
    a, b = text.find("["), text.rfind("]")
    if a < 0 or b < a:
        return []
    try:
        data = json.loads(text[a : b + 1])
    except json.JSONDecodeError:
        return []
    return [d for d in data if isinstance(d, dict) and {"start", "end"} <= d.keys()]


def find_moments(segments: list[Segment], n: int, min_len: float, max_len: float, model: str | None = None) -> list[dict]:
    """Return raw moment dicts (start, end, score, title, reason). Needs ANTHROPIC_API_KEY."""
    try:
        import anthropic
    except ImportError as e:  # pragma: no cover
        raise SystemExit("Install the LLM extra: pip install 'clipscout[llm]'") from e

    client = anthropic.Anthropic()
    chunks = _chunks(segments)
    k = max(3, math.ceil(n * 2 / max(1, len(chunks))))  # oversample; we re-rank later
    found: list[dict] = []
    for chunk in chunks:
        transcript = "\n".join(f"[{s.start:.1f}] {s.text}" for s in chunk)
        msg = client.messages.create(
            model=model or DEFAULT_MODEL,
            max_tokens=2000,
            messages=[{"role": "user", "content": PROMPT.format(k=k, min_len=min_len, max_len=max_len, transcript=transcript)}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        for d in _parse(text):
            try:
                found.append(
                    {
                        "start": float(d["start"]),
                        "end": float(d["end"]),
                        "score": float(d.get("score", 5)),
                        "title": str(d.get("title", ""))[:80],
                        "reason": str(d.get("reason", ""))[:200],
                    }
                )
            except (TypeError, ValueError):
                continue
    return found
