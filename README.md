# ClipScout

An open-source agent that watches a long video for you, finds the key moments, and cuts them into ready-to-share clips.

**How it works**

1. **Listens**: extracts audio and measures loudness second by second.
2. **Reads**: transcribes locally with [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (no API needed).
3. **Watches**: detects scene cuts with ffmpeg.
4. **Scores** every candidate window on energy, hook words, questions/exclamations, laughter and cut density.
5. **(Optional) Asks Claude** to pick self-contained highlights from the transcript, then blends that with the signal score.
6. **Clips**: picks the top non-overlapping moments and cuts them with ffmpeg, writing a `manifest.json` alongside.

Works fully offline without an API key. Add one for smarter picks.

## Install

Requires Python 3.9+ and [ffmpeg](https://ffmpeg.org/download.html) on your PATH.

```bash
git clone https://github.com/<you>/clipscout && cd clipscout
pip install -e .            # core
pip install -e '.[llm]'     # + Claude re-ranking
```

## Use

```bash
clipscout my_podcast.mp4                       # 8 clips, 20–60 s each -> my_podcast_clips/
clipscout talk.mp4 -n 12 --min-len 30 --max-len 90
clipscout talk.mp4 --dry-run                   # just print the picks
clipscout talk.mp4 --no-transcribe             # audio energy + scene cuts only
clipscout talk.mp4 --fast                      # stream-copy cuts (instant, keyframe-accurate only)
clipscout talk.mp4 --language hi --whisper-model medium

export ANTHROPIC_API_KEY=sk-ant-...            # enables Claude re-ranking automatically
clipscout talk.mp4 --no-llm                    # ...or switch it off
```

Output:

```
my_podcast_clips/
  clip01_the-biggest-mistake-i-made.mp4
  clip02_...
  manifest.json      # rank, start, end, score, title, reason, source
```

Re-runs reuse cached audio/transcript in `.work/`, so tweaking `-n` or clip length is instant. Use `--no-cache` to redo everything.

Set `CLIPSCOUT_MODEL` to choose which Claude model does the re-ranking.

## Notes

- Long videos: audio is streamed, the scene scan is downscaled, and transcripts are sent to Claude in 15-minute chunks.
- Only transcript text is sent to the API, never the video or audio.
- Re-encoding (default) gives frame-accurate cuts; `--fast` is quicker but may start a moment early.

## Roadmap

- 9:16 vertical reframing with face tracking, burned-in captions
- Speaker-change and audience-reaction signals
- Visual scoring via a vision model
- Resume-able batch mode for whole folders

## Contributing

Issues and PRs welcome. `pip install -e '.[dev]' && pytest`.

## License

MIT
