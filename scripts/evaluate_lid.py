"""LID accuracy on the smoke set (AmberNet ONNX), optionally with an IndicConformer CTC tie-break.

Usage: python scripts/evaluate_lid.py --onnx ambernet.int8.onnx --name lid-int8 [--candidates hi mr or ...]
Candidates = the languages a user selected; accuracy is top-1 among them. Writes results/<name>.json.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from indicstt.audio import load_audio
from indicstt.lid import AmberNetLID

ROOT = Path(__file__).resolve().parent.parent
ALL = ["hi", "gu", "mr", "kn", "ml", "ta", "te", "or", "bn", "en"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="ambernet.onnx")
    ap.add_argument("--name", required=True)
    ap.add_argument("--candidates", nargs="*", default=ALL)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    lid = AmberNetLID(ROOT / "models/lid-ambernet", threads=args.threads, onnx_name=args.onnx)
    per, conf, ms = {}, Counter(), []
    for lang in args.candidates:
        ok = n = 0
        for p in sorted((ROOT / "data/smoke" / lang).glob("*.json"))[: args.n]:
            audio = load_audio(ROOT / json.loads(p.read_text())["audio"])
            t = time.perf_counter()
            pred, _ = lid.identify(audio, allowed=args.candidates)
            ms.append((time.perf_counter() - t) * 1000)
            ok += pred == lang
            n += 1
            conf[(lang, pred)] += 1
        per[lang] = round(100 * ok / n, 1)
        print(lang, per[lang], flush=True)
    res = {"name": args.name, "onnx": args.onnx, "candidates": args.candidates, "accuracy": per,
           "mean_accuracy": round(sum(per.values()) / len(per), 2),
           "avg_ms": round(sum(ms) / len(ms)), "confusions": {f"{a}->{b}": c for (a, b), c in conf.items() if a != b},
           "command": f"python scripts/evaluate_lid.py --onnx {args.onnx} --name {args.name} "
                      f"--candidates {' '.join(args.candidates)}"}
    (ROOT / "results" / f"{args.name}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("mean_accuracy", "avg_ms", "confusions")}))


if __name__ == "__main__":
    main()
