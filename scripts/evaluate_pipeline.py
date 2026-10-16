"""Code-switched pipeline evaluation.

For each speaker profile, builds mixed recordings (4 smoke clips from the profile's languages, level-matched,
1 s gaps), runs the full engine (VAD → LID ‖ encoder → decode, no MT) and compares:
  pipeline WER  — engine output vs the concatenated references
  oracle WER    — the same clips transcribed one by one with the true language given (no VAD, no LID)
The gap is what segmentation + language identification cost (agent-kit A10 acceptance: within 0.5 points).

Usage: python scripts/evaluate_pipeline.py --profiles hi,en ta,te,en or,bn,hi --clips 5 --tier int8
Writes results/pipeline-<tier>.json.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from indicstt.audio import load_audio
from indicstt.engine import Engine, EngineConfig
from indicstt.evaluation import edit_distance, normalize

ROOT = Path(__file__).resolve().parent.parent


def load_part(lang: str, idx: int) -> tuple[np.ndarray, str]:
    rec = json.loads(sorted((ROOT / "data/smoke" / lang).glob("*.json"))[idx].read_text())
    a = load_audio(ROOT / rec["audio"])
    return a * (0.5 / max(1e-6, float(np.abs(a).max()))), rec["text"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", nargs="+", default=["hi,en", "ta,te,en", "or,bn,hi", "gu,mr,hi", "kn,ml,en"])
    ap.add_argument("--clips", type=int, default=5)
    ap.add_argument("--parts", type=int, default=4)
    ap.add_argument("--tier", default="int8")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lid-ctc", default="auto", choices=["auto", "always"])
    ap.add_argument("--name", help="results file suffix (default: tier)")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    gap = np.zeros(16000, dtype=np.float32)
    out = {"tier": args.tier, "profiles": {}}
    for prof in args.profiles:
        langs = prof.split(",")
        counts = {x: len(list((ROOT / "data/smoke" / x).glob("*.json"))) for x in langs}
        eng = Engine(EngineConfig(languages=langs, target_language=None, asr_tier=args.tier, lid_ctc=args.lid_ctc))
        st = eng.stages
        p_err = p_n = o_err = 0
        lang_ok = lang_n = 0
        for _ in range(args.clips):
            parts = [(lang, rng.randrange(counts[lang])) for lang in (rng.choice(langs) for _ in range(args.parts))]
            audios, refs = zip(*(load_part(lang, i) for lang, i in parts), strict=True)
            mixed = np.concatenate([x for a in audios for x in (a, gap)])
            segs = eng.transcribe_audio(mixed)
            ref_words = normalize(" ".join(refs)).split()
            hyp_words = normalize(" ".join(s.text for s in segs)).split()
            p_err += edit_distance(ref_words, hyp_words)
            p_n += len(ref_words)
            # language attribution: each segment's language vs the part it lies in
            bounds, t = [], 0.0
            for a in audios:
                bounds.append((t, t + len(a) / 16000))
                t += len(a) / 16000 + 1.0
            for s in segs:
                mid = (s.start_ms + s.end_ms) / 2000
                truth = next((parts[k][0] for k, (b0, b1) in enumerate(bounds) if b0 <= mid <= b1 + 0.5), None)
                if truth:
                    lang_n += 1
                    lang_ok += s.language == truth
            # oracle: each part alone, true language, no VAD/LID
            hyps = []
            for (lang, _), a in zip(parts, audios, strict=True):
                if lang == "en":
                    hyps.append(st.english_model.transcribe(a)[0])
                else:
                    hyps.append(st.indic_model.decode(st.indic_model.encode(a), lang))
            o_err += edit_distance(ref_words, normalize(" ".join(hyps)).split())
        r = {"pipeline_wer": round(100 * p_err / p_n, 2), "oracle_wer": round(100 * o_err / p_n, 2),
             "segment_language_accuracy": round(100 * lang_ok / max(1, lang_n), 1), "segments": lang_n,
             "words": p_n}
        r["gap"] = round(r["pipeline_wer"] - r["oracle_wer"], 2)
        out["profiles"][prof] = r
        print(prof, r, flush=True)
    out["command"] = (f"python scripts/evaluate_pipeline.py --profiles {' '.join(args.profiles)} "
                      f"--clips {args.clips} --parts {args.parts} --tier {args.tier} --seed {args.seed} --lid-ctc {args.lid_ctc}")
    (ROOT / "results" / f"pipeline-{args.name or args.tier}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
