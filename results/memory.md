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
