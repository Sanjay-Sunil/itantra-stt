"""Translation quality into English on FLEURS smoke clips (FLEURS sentences are parallel across languages).

Per language, two inputs are translated by IndicTrans2 (indic-en 200M INT8) and scored against the English
FLEURS reference with SacreBLEU and chrF++:
  text   — the gold Indic transcript (MT quality alone)
  speech — our ASR output (IndicConformer INT8, RNN-T, true language): the full speech → English path
Needs data/smoke/en_refs.json (English references by FLEURS id).

Usage: python scripts/evaluate_mt.py [--langs hi ta ...] [--tier int8]
Writes results/mt-to-en-<tier>.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import sacrebleu

from indicstt.asr import SUPPORTED, IndicConformerASR
from indicstt.audio import load_audio
from indicstt.mt import Translator

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="*", default=SUPPORTED)
    ap.add_argument("--tier", default="int8")
    args = ap.parse_args()
    refs_en = json.loads((ROOT / "data/smoke/en_refs.json").read_text())
    asr = IndicConformerASR(ROOT / f"models/indic-conformer-{args.tier}", threads=8)
    mt = Translator(ROOT / "models", threads=8)
    out = {"tier": args.tier, "per_language": {}}
    all_ref, all_text, all_speech, ms = [], [], [], []
    for lang in args.langs:
        ref, hyp_text, hyp_speech = [], [], []
        for p in sorted((ROOT / "data/smoke" / lang).glob("*.json")):
            if p.stem not in refs_en:
                continue
            rec = json.loads(p.read_text())
            ref.append(refs_en[p.stem])
            hyp_text.append(mt.translate(rec["text"], lang, "en")[0])
            asr_text = asr.decode(asr.encode(load_audio(ROOT / rec["audio"])), lang, "rnnt")
            t = time.perf_counter()
            hyp_speech.append(mt.translate(asr_text, lang, "en")[0])
            ms.append((time.perf_counter() - t) * 1000)
        r = {
            "n": len(ref),
            "text_bleu": round(sacrebleu.corpus_bleu(hyp_text, [ref]).score, 1),
            "text_chrf++": round(sacrebleu.corpus_chrf(hyp_text, [ref], word_order=2).score, 1),
            "speech_bleu": round(sacrebleu.corpus_bleu(hyp_speech, [ref]).score, 1),
            "speech_chrf++": round(sacrebleu.corpus_chrf(hyp_speech, [ref], word_order=2).score, 1),
            "example": {"ref": ref[0], "text": hyp_text[0], "speech": hyp_speech[0]},
        }
        out["per_language"][lang] = r
        all_ref += ref
        all_text += hyp_text
        all_speech += hyp_speech
        print(lang, {k: v for k, v in r.items() if k != "example"}, flush=True)
    out["overall"] = {
        "text_bleu": round(sacrebleu.corpus_bleu(all_text, [all_ref]).score, 1),
        "text_chrf++": round(sacrebleu.corpus_chrf(all_text, [all_ref], word_order=2).score, 1),
        "speech_bleu": round(sacrebleu.corpus_bleu(all_speech, [all_ref]).score, 1),
        "speech_chrf++": round(sacrebleu.corpus_chrf(all_speech, [all_ref], word_order=2).score, 1),
        "mt_ms_per_sentence": round(sum(ms) / len(ms)),
    }
    out["command"] = f"python scripts/evaluate_mt.py --langs {' '.join(args.langs)} --tier {args.tier}"
    (ROOT / "results" / f"mt-to-en-{args.tier}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(out["overall"])


if __name__ == "__main__":
    main()
