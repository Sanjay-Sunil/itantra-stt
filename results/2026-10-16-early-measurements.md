# Early measurements (2026-10-16)

Machine: i7-14650HX laptop, CPU only, onnxruntime 1.31. Smoke set: first 20 FLEURS test clips per language
(`python scripts/build_smoke_set.py`). Small sample — indicative, not a benchmark.

| Component | Metric | Value | How |
|---|---|---|---|
| whisper-tiny.en ONNX int8 (44 MB) | WER, FLEURS en smoke (20 clips) | 10.0% | ad-hoc script in session, lowercase + punctuation-stripped |
| whisper-tiny.en ONNX int8 | RTF (4 threads) | 0.073 | same |
| AmberNet LID ONNX fp32 (116 MB) | Feature parity vs NeMo | MAE 9e-7, logits max diff 6e-6 | `.venv-nemo/bin/python scripts/export_ambernet.py` |
| AmberNet LID | Top-1 accuracy, full segment, hi/gu/mr/kn/ml/ta (119 clips) | 99.2% | ad-hoc script; 9 target classes |
| AmberNet LID | Accuracy with 2 s / 3 s / 4 s crop | 61.3% / 76.5% / 87.4% | same |
| AmberNet LID | Latency, ~10 s clip, 4 threads | ~630 ms avg | same |
| IndicTrans2 ONNX int8 | Warm latency, one short sentence | ~140–200 ms | ad-hoc |

AmberNet has no Odia class (VoxLingua107).
