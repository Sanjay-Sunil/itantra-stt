"""Concatenate smoke clips (level-matched, 1 s gaps) into a code-switched demo file.

Usage: python scripts/make_demo_clip.py en:0 hi:0 ta:1 en:2 --out data/demo/mixed.wav
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from indicstt.audio import load_audio

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("parts", nargs="+", help="lang:index into data/smoke/<lang>")
    ap.add_argument("--out", default="data/demo/mixed.wav")
    ap.add_argument("--max-seconds", type=float, help="trim each part to this length")
    args = ap.parse_args()
    gap = np.zeros(16000, dtype=np.float32)
    pieces, refs = [], []
    for part in args.parts:
        lang, idx = part.split(":")
        rec = json.loads(sorted((ROOT / "data/smoke" / lang).glob("*.json"))[int(idx)].read_text())
        a = load_audio(ROOT / rec["audio"])
        if args.max_seconds:
            a = a[: int(args.max_seconds * 16000)]
        start = sum(len(x) for x in pieces) / 16000
        pieces += [a * (0.5 / max(1e-6, float(np.abs(a).max()))), gap]
        refs.append({"language": lang, "text": rec["text"], "start_s": round(start, 2),
                     "end_s": round(start + len(a) / 16000, 2), "trimmed": bool(args.max_seconds)})
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, np.concatenate(pieces), 16000)
    out.with_suffix(".json").write_text(json.dumps(refs, ensure_ascii=False, indent=1))
    print(out)


if __name__ == "__main__":
    main()
