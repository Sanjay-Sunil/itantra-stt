"""Build the smoke evaluation set: N clips per language from FLEURS test (CC-BY-4.0).

Writes data/smoke/<lang>/<id>.wav (16 kHz mono) and data/smoke/manifest.jsonl
with {"audio", "text", "language"} per line.
Usage: python scripts/build_smoke_set.py [--n 20] [--langs hi ta ...]
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import HfFileSystem

from indicstt.audio import to_mono_16k

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "smoke"
FLEURS = {"hi": "hi_in", "gu": "gu_in", "mr": "mr_in", "kn": "kn_in", "ml": "ml_in",
          "ta": "ta_in", "te": "te_in", "or": "or_in", "bn": "bn_in", "en": "en_us"}


def build(lang: str, n: int) -> list[dict]:
    d = OUT / lang
    d.mkdir(parents=True, exist_ok=True)
    existing = sorted(d.glob("*.json"))
    if len(existing) >= n:
        return [json.loads(p.read_text()) for p in existing[:n]]
    fs = HfFileSystem()
    f = fs.open(f"datasets/google/fleurs@~parquet/{FLEURS[lang]}/test/0000.parquet", "rb", block_size=1 << 22)
    rows = []
    for batch in pq.ParquetFile(f).iter_batches(batch_size=n, columns=["id", "audio", "transcription"]):
        for r in batch.to_pylist():
            audio, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
            wav = d / f"{r['id']}.wav"
            sf.write(wav, to_mono_16k(audio, sr), 16000)
            rec = {"audio": str(wav.relative_to(ROOT)), "text": r["transcription"], "language": lang}
            wav.with_suffix(".json").write_text(json.dumps(rec, ensure_ascii=False))
            rows.append(rec)
        break
    return rows[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--langs", nargs="*", default=list(FLEURS))
    args = ap.parse_args()
    all_rows = []
    for lang in args.langs:
        rows = build(lang, args.n)
        print(f"{lang}: {len(rows)} clips", flush=True)
        all_rows += rows
    with (OUT / "manifest.jsonl").open("w") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
