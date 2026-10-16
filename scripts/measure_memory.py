"""Measure RAM per pipeline component, loaded the way the engine loads it, and per-stage latency
from a real-time run. Feeds the dashboard's "How does this work?" page.

Usage: python scripts/measure_memory.py [--tier int8]
Writes results/stage-profile-<tier>.json:
  memory: per component, private (RssAnon) and file-backed (RssFile) MB added when loaded + run once
  latency: per stage, median ms per segment over the real-time demo run
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def mem() -> dict[str, int]:
    d = {}
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(("RssAnon:", "RssFile:")):
                d[line.split(":")[0]] = int(line.split()[1]) // 1024
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="int8")
    args = ap.parse_args()
    md = ROOT / "models"
    rows, prev = [], {"RssAnon": 0, "RssFile": 0}  # first step = whole process (Python, numpy, ORT)

    def step(key: str):
        nonlocal prev
        cur = mem()
        rows.append({"component": key, "private_mb": cur["RssAnon"] - prev["RssAnon"],
                     "file_mb": cur["RssFile"] - prev["RssFile"]})
        prev = cur

    import onnxruntime  # noqa: F401

    from indicstt import engine as E
    step("runtime")
    rng = np.random.default_rng(0)
    speech = (rng.standard_normal(16000 * 10) * 0.05).astype(np.float32)

    from indicstt.vad import SileroVAD
    vad = SileroVAD(str(md / "silero-vad/onnx/model.onnx"))
    for i in range(0, len(speech) - 512, 512):
        vad(speech[i : i + 512])
    step("vad")
    from indicstt.lid import AmberNetLID
    lid = AmberNetLID(md / "lid-ambernet", threads=4, onnx_name="ambernet.int8.onnx")
    lid.scores(speech)
    step("lid")
    from indicstt.asr import IndicConformerASR
    asr = IndicConformerASR(md / f"indic-conformer-{args.tier}", threads=4, languages=["hi", "ta"])
    enc = asr.encode(speech)
    step("encoder")
    asr.decode(enc, "hi", "rnnt")
    asr.decode(enc, "hi", "ctc")
    step("decoder")
    from indicstt.english import WhisperEnglish
    WhisperEnglish(md / "whisper-tiny-en", threads=4).transcribe(speech)
    step("english")
    from indicstt.mt import Translator
    tr = Translator(md, threads=4)
    tr.translate("मुझे मदद चाहिए, पाँच लोग पुल के पास फंसे हैं", "hi", "en")
    step("mt")

    # latency: real-time stream of the demo clip through the full engine
    from scripts_bench import RealtimeSource  # type: ignore[import-not-found]

    from indicstt.audio import load_audio

    eng = E.Engine(E.EngineConfig(languages=["hi", "en", "ta"], target_language="en", asr_tier=args.tier))
    stages: dict[str, list[float]] = {}
    for seg in eng.stream(RealtimeSource(load_audio(ROOT / "data/demo/mixed_en_hi_ta.wav"))):
        t = dict(seg.timings)
        fin = t.pop("finalized_at", None)
        if fin:
            t["time_to_text"] = (time.time() - fin) * 1000
        for k, v in t.items():
            stages.setdefault(f"{k}:{'en' if seg.language == 'en' else 'indic'}", []).append(v)
    latency = {k: round(float(np.median(v)), 1) for k, v in sorted(stages.items())}
    out = {"tier": args.tier, "memory": rows, "latency_median_ms": latency,
           "command": f"python scripts/measure_memory.py --tier {args.tier}",
           "machine": "i7-14650HX laptop, CPU only"}
    (ROOT / "results" / f"stage-profile-{args.tier}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import importlib

    sys.modules["scripts_bench"] = importlib.import_module("benchmark_engine")
    main()
