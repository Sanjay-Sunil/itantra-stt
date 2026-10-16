# Memory optimisation (measured 2026-10-16)

Laptop i7-14650HX, CPU only, onnxruntime 1.31. RSS from /proc/self/status. "Private" = RssAnon (heap, not
reclaimable); "file" = RssFile (memory-mapped model files: page cache, shared between processes, the OS can
evict and re-read them under pressure).

## Baseline: cost per component (one process, each loaded and run once)

| Component | RSS added |
|---|---|
| Python + onnxruntime + numpy | 93 MB |
| Silero VAD | 16 MB |
| AmberNet LID INT8 (30 MB file) | 126 MB |
| IndicConformer INT8 loaded (675 MB on disk) | 516 MB |
| … + one 20 s segment (activations, arena) | 169 MB |
| whisper-tiny.en INT8 (44 MB file) | 279 MB |
| MT indic-en INT8 (258 MB file) | 496 MB |
| MT indic-indic INT8 (372 MB file) | 470 MB |
| MT en-indic INT8 (304 MB file) | 400 MB |
| **All loaded** | **2,598 MB** |

## A1. MT: share decoder weights, no pre-packing

IndicTrans2 bundles open `decoder_model` and `decoder_with_past_model`. All ~100 MB of the second graph's weights
are the same bytes of `decoder_shared.onnx.data` under different names. ONNX Runtime pre-packs weights per
session, so each graph held a private packed copy.
Fix (`mt._SharedWeights`): feed external initializers as memory-mapped arrays, one per distinct byte range,
to every session, and disable pre-packing so sessions use those buffers directly.

| MT indic-en | Private | File | Total RSS | ms / sentence |
|---|---|---|---|---|
| Before | 454 MB | 85 MB | 489 MB | 464 |
| Shared mmap + pre-packing kept | 463 MB | 226 MB | 640 MB | 378 |
| **Shared mmap, no pre-packing (shipped)** | **241 MB** | 226 MB | **417 MB** | 484 |

Translations: 31/32 identical on FLEURS hi/ta/or/bn sentences; the 1 difference is an equally valid rewording
(different float summation order). `INDICSTT_MT_PREPACK=1` restores pre-packing.

## A3. Per-model ORT settings (pre-packing, arena)

`logs/model_mem_probe.py`, 20 s (encoder) / 8 s (others) random input, fresh process per row. Private MB after run.

| Model | default | arena off | no pre-pack | no pre-pack + arena off | Shipped |
|---|---|---|---|---|---|
| IndicConformer encoder | 727 (+101 file), 1075 ms | 564, 1124 ms | 231 (+608 file), 1133 ms | **68 (+608 file), 1181 ms** | no pre-pack + arena off |
| AmberNet LID | 150, 107 ms | **121, 115 ms** | 115, 203 ms | 125, 205 ms | arena off |
| Whisper encoder | 127, 203 ms | **55, 197 ms** | 125, 180 ms | 74, 187 ms | arena off |
| CTC head | 52 | **41** | 55 | 52 | arena off |

Outputs identical in every configuration. ORT already memory-maps external-data files; pre-packing is what
copied the encoder's weights into private memory. Per-token loop graphs (Whisper decoder, RNN-T
prediction/joint) keep the arena: without it the Whisper decoder took 1,361 ms vs 390 ms per clip.

## Whole engine, real-time benchmark (`scripts/benchmark_engine.py data/demo/mixed_en_hi_ta.wav --tier int8`)

| | Before | After A1 + A3 (3 runs) |
|---|---|---|
| Peak RSS | 1,851 MB | 1,655–1,761 MB |
| Private RSS at end | — | 793–867 MB (plus ~840 MB file-backed, evictable) |
| Time-to-text, Indic + MT (median) | 0.81 s | 0.96–1.05 s |
| Time-to-text, English (median) | 1.23 s | 1.24–1.36 s |

Trade-off: −100 to −200 MB peak RSS (and most weights now evictable page cache) for ~+0.2 s on translated
segments, mainly MT without pre-packing and the encoder without pre-packing. Run-to-run noise is ~±0.1 s.
