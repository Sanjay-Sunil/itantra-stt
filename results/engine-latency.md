# Engine latency (measured 2026-10-16)

Laptop i7-14650HX, CPU only. Input `data/demo/mixed_en_hi_ta.wav` (62.7 s: English, Hindi, English, Tamil FLEURS
clips, level-matched, 1 s gaps) streamed at real-time pace in 32 ms chunks, like a microphone.
Speaker languages hi, en, ta; everything delivered in English. RNN-T decoding, 2 ASR workers, 4 threads per model.
Time-to-text = VAD finalizes a segment (end of speech + 0.4 s silence) → segment leaves the reorder buffer.

| | INT8 | INT4-mixed |
|---|---|---|
| Time-to-text, Hindi/Tamil → English (ASR + MT), median / max | 0.81 s / 1.50 s | 1.09 s / 1.65 s |
| Time-to-text, English (Whisper, no MT), median / max | 1.23 s / 1.29 s | 1.25 s / 1.37 s |
| Peak RSS, whole engine (VAD, LID, IndicConformer, Whisper, 1 MT model) | 1.85 GB | 1.73 GB |
| Load + warm-up | 6.8 s | 7.3 s |

Docs projection for high-end phones (INT8, NPU): ~1.2 s same-language, ~1.45–1.55 s with MT; peak RAM 0.95–1.15 GB.
Not comparable directly (laptop CPU vs phone NPU); RAM is above the projection — the MT model and Whisper are
loaded alongside the 600M (not 300M-distilled) encoder.

Change that mattered: when LID decides English, the engine now cancels the in-flight Indic encoder run
(onnxruntime RunOptions.terminate). English time-to-text 1.77 s → 1.23 s.

Commands: `python scripts/benchmark_engine.py data/demo/mixed_en_hi_ta.wav --tier int8` (and `--tier int4mixed`).
