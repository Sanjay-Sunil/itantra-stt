# Code-switched pipeline and translation (measured 2026-10-16)

Laptop i7-14650HX, CPU only, INT8 models, RNN-T decoding. FLEURS smoke clips (18–20 per language).

## Mixed-language recordings: full pipeline vs oracle

`python scripts/evaluate_pipeline.py` → `results/pipeline-int8.json`. Per profile: 5 recordings, each 4 random
clips from the profile's languages (level-matched, 1 s gaps). Pipeline = engine (VAD → LID ‖ encoder → decode).
Oracle = same clips transcribed one by one with the true language (no VAD, no LID).

| Profile | Pipeline WER | Oracle WER | Gap | Segment language correct | Words |
|---|---|---|---|---|---|
| hi, en | 9.16 | 8.59 | +0.57 | 100% (29) | 524 |
| ta, te, en | 17.37 | 15.14 | +2.23 | 93.9% (33) | 403 |
| or, bn, hi | 15.71 | 14.38 | +1.33 | 100% (29) | 452 |
| gu, mr, hi | 13.67 | 13.67 | 0.00 | 100% (34) | 512 |
| kn, ml, en | 14.80 | 14.80 | 0.00 | 96.0% (25) | 358 |

Agent-kit A10 target is a gap ≤ 0.5: met for gu/mr/hi and kn/ml/en, close for hi/en, not met for or/bn/hi and
ta/te/en. Segment-level language errors and VAD cuts mid-word are the two sources.

## Translation into English (FLEURS is parallel across languages)

`python scripts/evaluate_mt.py` → `results/mt-to-en-int8.json`. IndicTrans2 indic-en dist-200M INT8.
"text" = gold Indic transcript translated (MT alone); "speech" = our ASR output translated (full path).

| Lang | n | BLEU text | BLEU speech | chrF++ text | chrF++ speech |
|---|---|---|---|---|---|
| hi | 20 | 37.1 | 33.8 | 64.3 | 61.2 |
| gu | 20 | 37.9 | 31.1 | 63.3 | 58.7 |
| mr | 20 | 33.6 | 23.9 | 60.2 | 54.1 |
| kn | 19 | 34.5 | 30.2 | 61.6 | 58.7 |
| ml | 20 | 40.0 | 34.4 | 64.0 | 61.3 |
| ta | 20 | 32.1 | 22.1 | 55.7 | 51.1 |
| te | 20 | 32.8 | 32.3 | 62.0 | 60.2 |
| or | 20 | 27.0 | 26.2 | 54.2 | 53.9 |
| bn | 18 | 34.5 | 31.4 | 61.3 | 59.3 |
| **all** | 177 | **34.4** | **29.4** | **60.7** | **57.6** |

MT latency ≈ 0.9 s per FLEURS sentence (long, ~20 words) here; conversational segments are shorter.
These are BLEU against human English references, not the docs' "87.13 SacreBLEU vs FP32 oracle" metric
(which measures INT8-vs-FP32 agreement, a different thing).

## Language ID with all 10 languages as candidates

`python scripts/evaluate_engine_lid.py --profiles hi,gu,mr,kn,ml,ta,te,or,bn,en --tier int8` → `results/engine-lid-int8.json`
(whole FLEURS clips, 18–20 per language): **99.5%** — 100% on all 9 Indian languages including Odia,
95% English (1 clip → Bengali). ~0.5 s LID per clip (AmberNet + CTC check over 9 languages).

Dashboard "Language test" on `data/demo/all10_languages.wav` (7 s per language, VAD-chunked): 12/13 chunks
correct; the miss is a 1 s fragment at the end of the Telugu part. Very short chunks are the weak spot.
