"""Language decision as the engine makes it (AmberNet INT8 + IndicConformer CTC check), per speaker profile.

Each smoke clip is run through the engine stages as one segment; the engine picks among the profile's languages.
Usage: python scripts/evaluate_engine_lid.py --profiles or,bn,hi hi,mr,en ta,te,en --tier int8
Writes results/engine-lid-<tier>.json.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from indicstt.audio import load_audio
from indicstt.engine import EngineConfig, _Stages
from indicstt.segment import Segment

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", nargs="+", default=["or,bn,hi", "hi,mr,en", "ta,te,en", "or,en"])
    ap.add_argument("--tier", default="int8")
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()
    out = {"tier": args.tier, "profiles": {}}
    for prof in args.profiles:
        langs = prof.split(",")
        st = _Stages(EngineConfig(languages=langs, target_language=None, asr_tier=args.tier))
        acc, conf, ms = {}, Counter(), []
        for lang in langs:
            ok = n = 0
            for p in sorted((ROOT / "data/smoke" / lang).glob("*.json"))[: args.n]:
                seg = Segment(seq=0, start_ms=0, end_ms=0, audio=load_audio(ROOT / json.loads(p.read_text())["audio"]))
                t = time.perf_counter()
                lid = st.lid(seg)
                enc = st.encode(seg) if lid[0] != "en" else None
                st.decode(seg, lid, enc)
                ms.append(seg.timings.get("lid", 0) + seg.timings.get("lid_ctc", 0))
                ok += seg.language == lang
                n += 1
                if seg.language != lang:
                    conf[f"{lang}->{seg.language}"] += 1
            acc[lang] = round(100 * ok / n, 1)
        out["profiles"][prof] = {"accuracy": acc, "mean": round(sum(acc.values()) / len(acc), 2),
                                 "confusions": dict(conf), "avg_lid_ms": round(sum(ms) / len(ms))}
        print(prof, out["profiles"][prof], flush=True)
    out["command"] = f"python scripts/evaluate_engine_lid.py --profiles {' '.join(args.profiles)} --tier {args.tier}"
    (ROOT / "results" / f"engine-lid-{args.tier}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
