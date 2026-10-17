# Browser engine (measured 2026-10-17)

The engine ported to JavaScript (`web/lib/`) running the same INT8 ONNX models in ONNX Runtime Web 1.30
(WebAssembly, SIMD, multi-threaded). Laptop i7-14650HX, CPU only, 8 WebAssembly threads.
Browser runs: Chromium 152 (headless), `web/test/browser_e2e.mjs`. Node runs use the same `onnxruntime-web` package.

## Parity with the Python engine

| Check | Result | Command |
|---|---|---|
| Log-mel features (NeMo, IndicConformer, Whisper) | max abs diff 0 / 0 / 1.2e-7 | `node test/feat.mjs` |
| Language ID top-1, 30 smoke clips | 30/30 agree | `node test/parity_asr.mjs <ref>` |
| ASR transcripts, 27 Indian clips × CTC/RNN-T + 3 English | 47/57 identical, 1.3% WER between JS and Python | same |
| IndicProcessor pre/postprocessing, 499 FLEURS sentences | 499/499 and 499/499 identical | `node test/parity_text.mjs <ref>` |
| Translation token ids / outputs, 29 sentences | 29/29 / 16/29 identical (rest are rewordings) | `node test/parity_mt.mjs <ref>` |
| Whole pipeline, all10_languages.wav | 13/13 segments, languages 13/13, transcripts 11/13 identical | `node test/pipeline.mjs …` |

Remaining differences come from the WebAssembly INT8 kernels: on the same input the MT encoder output differs from
native ONNX Runtime by 0.13 (max abs, values up to 22.7, ≈0.5%), and native ORT gives identical output with graph
optimisations on or off, so the port itself is not the cause. That is enough to flip some greedy choices.

## Accuracy

ASR word error rate on the smoke set (20 FLEURS clips per language, RNN-T, INT8), same normalisation as
`src/indicstt/evaluation.py`. Raw: `browser-asr.json` (`node test/eval_wer.mjs rnnt`).

| Lang | Browser (ONNX Runtime Web) | Python engine (native, `quantization.md`) |
|---|---|---|
| hi | 8.92 | 8.92 |
| gu | 17.82 | 17.59 |
| mr | 16.67 | 15.86 |
| kn | 17.82 | 18.75 |
| ml | 26.73 | 27.06 |
| ta | 26.32 | 27.19 |
| te | 22.03 | 21.47 |
| or | 21.87 | 21.33 |
| bn | 13.48 | 13.30 |
| **mean** | **19.07** | **19.05** |

Same accuracy: per-language differences go both ways (from −0.93 kn to +0.81 mr; ±1 point is within noise for 20 clips) and the mean moves by 0.02 points.
Browser runtime in Node, 8 threads: RTF 0.294 for encoder + RNN-T decode (native: 0.128).

## Speed and compute (Chromium, 8 threads, models from Cache Storage)

| Clip | Speech | Compute | Real-time factor | GMAC |
|---|---|---|---|---|
| mixed_en_hi_ta.wav (en, hi, ta → en) | 49.5 s | 20.1 s | 0.41 | 434 |
| all10_languages.wav (10 languages → en) | 61.9 s | 38.0 s | 0.61 | 664 |

Per stage on the mixed clip: speech encoder 252 GMAC in 7.8 s, language ID 110 GMAC in 2.3 s, Whisper encoder
37 GMAC in 1.4 s, translation 31 GMAC in 2.1 s; throughput 26 GMAC/s. The Whisper encoder always sees a padded
30 s window (18.5 GMAC per call), which is why short English parts cost as much as long ones.

Model load (en + hi + ta, 15 ONNX graphs, 1,011 MB): 8.3 s from the local server, 7.8 s from Cache Storage on a
later visit (Hindi only, 11 graphs: 6.4 s). With the network switched off after loading (`OFFLINE=1`), the clip transcribes and translates the
same; requests sent while transcribing: 0.

MAC counts: `scripts/count_macs.py` profiles each model with ONNX Runtime (graph optimisations off), sums
Conv / MatMul / Gemm / LSTM MACs from the real node shapes, and fits them in the input length (`web/compute.json`).
Speech encoder: 79.5 M MACs per 10 ms mel frame ≈ 7.9 GMAC per second of audio.
