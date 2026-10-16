"""Score the IndicConformer runner on the smoke set: WER/CER per language and decode mode, speed, peak RSS.

Usage: python scripts/evaluate_asr.py --model models/indic-conformer --name fp32 [--langs hi ta] [--threads 8]
Writes results/asr-<name>.json.
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np

from indicstt.asr import SUPPORTED, IndicConformerASR
from indicstt.audio import load_audio
from indicstt.evaluation import cer_counts, wer_counts

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/indic-conformer")
    ap.add_argument("--name", required=True)
    ap.add_argument("--langs", nargs="*", default=SUPPORTED)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()

    t = time.perf_counter()
    asr = IndicConformerASR(ROOT / args.model, threads=args.threads)
    load_s = time.perf_counter() - t
    modes = ["ctc", "rnnt"]
    res: dict = {"name": args.name, "model": args.model, "threads": args.threads, "load_s": round(load_s, 1),
                 "per_language": {}}
    tot_audio = tot_enc = 0.0
    tot_dec = dict.fromkeys(modes, 0.0)
    for lang in args.langs:
        recs = [json.loads(p.read_text()) for p in sorted((ROOT / "data/smoke" / lang).glob("*.json"))[: args.n]]
        c = {m: [0, 0, 0, 0] for m in modes}  # word err, words, char err, chars
        for r in recs:
            audio = load_audio(ROOT / r["audio"])
            tot_audio += len(audio) / 16000
            t = time.perf_counter()
            enc = asr.encode(audio)
            tot_enc += time.perf_counter() - t
            for m in modes:
                t = time.perf_counter()
                hyp = asr.decode(enc, lang, m)
                tot_dec[m] += time.perf_counter() - t
                we, wn = wer_counts(r["text"], hyp)
                ce, cn = cer_counts(r["text"], hyp)
                for i, v in enumerate((we, wn, ce, cn)):
                    c[m][i] += v
        res["per_language"][lang] = {m: {"wer": round(100 * v[0] / v[1], 2), "cer": round(100 * v[2] / v[3], 2)}
                                     for m, v in c.items()}
        print(lang, {m: res["per_language"][lang][m]["wer"] for m in modes}, flush=True)
    for m in modes:
        res[f"mean_wer_{m}"] = round(float(np.mean([v[m]["wer"] for v in res["per_language"].values()])), 2)
    res["audio_s"] = round(tot_audio, 1)
    res["rtf_encoder"] = round(tot_enc / tot_audio, 4)
    res["rtf"] = {m: round((tot_enc + tot_dec[m]) / tot_audio, 4) for m in modes}
    res["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    res["model_mb_on_disk"] = round(sum(f.stat().st_size for f in (ROOT / args.model).rglob("*") if f.is_file()) / 1e6)
    res["command"] = "python scripts/evaluate_asr.py " + " ".join(
        f"--{k} {' '.join(v) if isinstance(v, list) else v}" for k, v in vars(args).items())
    out = ROOT / "results" / f"asr-{args.name}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "per_language"}, indent=1))


if __name__ == "__main__":
    main()
