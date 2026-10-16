"""End-to-end engine benchmark: stream a file at real-time pace (like a microphone) and measure
time-to-text per segment = wall time from the VAD finalizing a segment to the segment leaving the
reorder buffer (includes LID, ASR, MT and reorder wait). Also reports peak RSS of the whole engine.

Usage: python scripts/benchmark_engine.py data/demo/mixed_en_hi_ta.wav --langs hi en ta --target en --tier int8
Writes results/engine-bench-<tier>-<target>.json.
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np

from indicstt.audio import AudioSource, load_audio
from indicstt.engine import Engine, EngineConfig

ROOT = Path(__file__).resolve().parent.parent


def _status_mb(key: str) -> int:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(key + ":"):
                return int(line.split()[1]) // 1024
    return -1


class RealtimeSource(AudioSource):
    """Feeds a buffer in 32 ms chunks at real-time pace, like a live microphone."""

    def __init__(self, audio: np.ndarray, chunk: int = 512):
        self.audio, self.chunk = audio, chunk

    def chunks(self):
        t0 = time.perf_counter()
        for i in range(0, len(self.audio), self.chunk):
            due = t0 + i / 16000
            delay = due - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            yield self.audio[i : i + self.chunk]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--langs", nargs="+", default=["hi", "en", "ta"])
    ap.add_argument("--target", default="en")
    ap.add_argument("--tier", default="int8")
    ap.add_argument("--mode", default="rnnt")
    args = ap.parse_args()

    t = time.perf_counter()
    eng = Engine(EngineConfig(languages=args.langs, target_language=args.target, asr_tier=args.tier,
                              asr_mode=args.mode))
    load_s = time.perf_counter() - t
    audio = load_audio(ROOT / args.file)
    rows = []
    for seg in eng.stream(RealtimeSource(audio)):
        ttt = (time.time() - seg.timings["finalized_at"]) * 1000
        mt = seg.translation is not None
        rows.append({"seq": seg.seq, "lang": seg.language, "audio_s": round((seg.end_ms - seg.start_ms) / 1000, 2),
                     "mt": mt, "time_to_text_ms": round(ttt), "flags": seg.flags})
        print(rows[-1], flush=True)
    t_same = [r["time_to_text_ms"] for r in rows if not r["mt"]]
    t_mt = [r["time_to_text_ms"] for r in rows if r["mt"]]
    res = {
        "tier": args.tier, "mode": args.mode, "languages": args.langs, "target": args.target,
        "audio_s": round(len(audio) / 16000, 1), "load_and_warmup_s": round(load_s, 1),
        "segments": rows,
        "time_to_text_ms_same_language": {"median": int(np.median(t_same)), "max": max(t_same)} if t_same else None,
        "time_to_text_ms_with_mt": {"median": int(np.median(t_mt)), "max": max(t_mt)} if t_mt else None,
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
        "end_rss_private_mb": _status_mb("RssAnon"), "end_rss_file_mb": _status_mb("RssFile"),
        "command": f"python scripts/benchmark_engine.py {args.file} --langs {' '.join(args.langs)} "
                   f"--target {args.target} --tier {args.tier} --mode {args.mode}",
    }
    out = ROOT / "results" / f"engine-bench-{args.tier}-{args.mode}-{args.target}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("segments", "command")}, indent=1))


if __name__ == "__main__":
    main()
