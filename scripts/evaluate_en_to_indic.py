"""English → Indic: dedicated en-indic model vs the indic-indic model fed English (FLEURS, parallel sentences).

Source: FLEURS English sentence (data/smoke/en_refs.json); reference: the same sentence id in the Indic
language, raw transcription (data/smoke/indic_raw_refs.json). Scores: SacreBLEU and chrF++ per language.
Usage: python scripts/evaluate_en_to_indic.py → results/en-to-indic-models.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import sacrebleu

from indicstt.mt import IndicTransONNX

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    en = json.loads((ROOT / "data/smoke/en_refs.json").read_text())
    refs = json.loads((ROOT / "data/smoke/indic_raw_refs.json").read_text())
    models = {"en-indic (dedicated)": "mt-en-indic", "indic-indic (fed English)": "mt-indic-indic"}
    out: dict = {"per_language": {}, "overall": {}}
    all_ref: list[str] = []
    all_hyp: dict[str, list[str]] = {m: [] for m in models}
    ms: dict[str, list[float]] = {m: [] for m in models}
    loaded = {m: IndicTransONNX(ROOT / "models" / d, threads=4) for m, d in models.items()}
    for lang, items in refs.items():
        ref = list(items.values())
        row = {"n": len(ref)}
        for m, mt in loaded.items():
            hyp = []
            for i in items:
                t = time.perf_counter()
                hyp.append(mt.translate(en[i], "en", lang))
                ms[m].append((time.perf_counter() - t) * 1000)
            row[m] = {"bleu": round(sacrebleu.corpus_bleu(hyp, [ref]).score, 1),
                      "chrf++": round(sacrebleu.corpus_chrf(hyp, [ref], word_order=2).score, 1)}
            all_hyp[m] += hyp
        all_ref += ref
        out["per_language"][lang] = row
        print(lang, row, flush=True)
    for m in models:
        out["overall"][m] = {"bleu": round(sacrebleu.corpus_bleu(all_hyp[m], [all_ref]).score, 1),
                             "chrf++": round(sacrebleu.corpus_chrf(all_hyp[m], [all_ref], word_order=2).score, 1),
                             "ms_per_sentence": round(sum(ms[m]) / len(ms[m]))}
    out["command"] = "python scripts/evaluate_en_to_indic.py"
    (ROOT / "results/en-to-indic-models.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out["overall"], indent=1))


if __name__ == "__main__":
    main()
