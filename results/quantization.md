# IndicConformer quantization (measured 2026-10-16)

Machine: i7-14650HX laptop, CPU only, onnxruntime 1.31, 8 intra-op threads.
Test set: first 20 FLEURS test clips per language (`scripts/build_smoke_set.py`), 2,264 s of audio total.
Small sample: ±1 WER point is within noise for a single language.
Commands: `python scripts/quantize.py --tier <tier>` then
`python scripts/evaluate_asr.py --model models/indic-conformer-<tier> --name <tier> --threads 8`.
Raw numbers: `results/asr-*.json`.

## Shipped tiers — WER % (RNN-T greedy, the engine's default decoder)

| Lang | FP32 ref | INT8 (high-end) | INT4-mixed (low-end) |
|---|---|---|---|
| hi | 8.16 | 8.92 | 8.92 |
| gu | 16.93 | 17.59 | 17.82 |
| mr | — | 15.86 | 17.74 |
| kn | — | 18.75 | 19.38 |
| ml | — | 27.06 | 25.08 |
| ta | — | 27.19 | 26.02 |
| te | — | 21.47 | 22.03 |
| or | — | 21.33 | 22.40 |
| bn | — | 13.30 | 15.16 |
| **mean** | | **19.05** | **19.39** |
| CTC mean (fast path / language scoring) | | 19.96 | 20.43 |

FP32 reference was measured for hi and gu only (run stopped early; FP32 is never shipped).
Gate (agent-kit D12, ≤ 2 points vs FP32): INT8 passes on hi (+0.76) and gu (+0.66); other languages
not gated for lack of an FP32 number.

| | INT8 | INT4-mixed | FP32 (reference) |
|---|---|---|---|
| ASR model on disk | 675 MB | 535 MB | 2,557 MB |
| Peak RSS, eval process | 1,143 MB | 1,018 MB | — |
| RTF encoder + RNN-T decode | 0.128 | 0.151 | — |
| Load time | 2.6 s | 4.2 s | — |

INT4-mixed is smaller but not faster on this x86 CPU (MatMulNBits kernels); speed on ARM phones is unmeasured.

## What each tier is

- **INT8**: per-channel dynamic INT8 on every encoder MatMul and Conv, except the 24 depthwise convs
  (≈0.2M of 607M weights, ≈1 MB) which stay FP32. CTC head, prediction LSTM, joint nets and per-language
  heads: per-channel dynamic INT8.
- **INT4-mixed**: as INT8, plus the FFN MatMuls of layers 2–23 (88 nodes, 400M weights) as block-wise INT4
  (MatMulNBits, block 32, symmetric). Layers 0–1 and attention stay INT8 (docs/compression §2.3).

## How we got there (Hindi RNN-T WER, FP32 = 8.16)

| Variant | hi | gu | Size |
|---|---|---|---|
| Dynamic INT8, per-tensor, everything | 11.57 | 19.15 | 672 MB |
| …same encoder + FP32 heads | 12.14 | 18.93 | — |
| Block-wise 8-bit MatMulNBits encoder | 14.23 | 18.49 | 738 MB |
| Per-channel INT8, everything | 10.63 | 17.82 | 674 MB |
| Per-channel, leave pre_encode FP32 | 11.20 | | 682 MB |
| Per-channel, leave attention FP32 | 13.47 | | 1,051 MB |
| Per-channel, leave FFN FP32 | 14.61 | | 1,881 MB |
| Per-channel, leave all conv-module convs FP32 | 7.97 | | 902 MB |
| Per-channel, leave pointwise convs FP32 | 12.14 | 18.04 | 901 MB |
| **Per-channel, leave depthwise convs FP32 (shipped INT8)** | **8.92** | **17.59** | **675 MB** |

Findings: quantizing the small heads costs nothing measurable; the loss came from ORT's dynamic INT8
depthwise convolution. Per-channel scales help (+1 point). Upstream's RNN-T prediction network must be fed
per-language token ids; feeding union-vocab ids gives 568% WER.

## Language ID (AmberNet, 9 classes, 20 clips each)

| | Accuracy | Notes |
|---|---|---|
| FP32 (116 MB) | 98.89% | errors: mr→hi, en→bn · 272 ms per 8 s audio (4 threads) |
| Dynamic INT8 (29 MB) | 98.33% | 877 ms per 8 s audio — 3x *slower* (ConvInteger kernels); rejected |
| **Static QDQ INT8, per-channel (30 MB), shipped** | **99.44%** | **103 ms per 8 s audio**; error: mr→hi |

Static INT8 is calibrated on 3 smoke clips per language, which overlap the 20 evaluation clips — accuracy
slightly optimistic. Commands: `python scripts/quantize.py --lid`,
`python scripts/evaluate_lid.py --onnx ambernet.int8.onnx --name lid-int8-static --candidates hi gu mr kn ml ta te bn en`.
AmberNet has no Odia class; Odia relies on the IndicConformer CTC language check (not yet measured).
