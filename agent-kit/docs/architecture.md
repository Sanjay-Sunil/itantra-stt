# Architecture: indic-stt

This is the single specification agents build from. Numbers marked **projection** come from planning documents and have not been measured; replace them with measured values in `docs/results/` as modules land.

## 1. Scope

- Fully offline, on-device speech-to-text for Hindi, Gujarati, Marathi, Kannada, Malayalam, Tamil, Telugu, Odia, Bengali and English, spoken in any order, including code-switching between segments.
- Output: one transcript, in speaking order, optionally translated into the receiver's language.
- Shipped as a library plus model packs. Users install only the languages and tier they need.
- Device tiers:
  - **High-end:** INT8 models; NPU or DSP-capable chip; code-switching across the 2–3 languages a user selects.
  - **Low-end:** INT4 mixed-precision models (translation held at INT8); CPU-only; code-switching for up to 2 languages, falling back to one language under memory pressure (target, not yet validated).
- Original pre-compression low-end design, kept for reference only: one Vakyansh wav2vec2 model per language, single language, no code-switching, about 160 MB RAM.

## 2. Pipeline

```
mic / file
  → VAD thread ──(A) segment queue──┐
                                     ├─ LID ─────────┐
                                     └─ ASR encoder ─┴─(C) join: language token + decode ─(D) ASR worker pool
  → language = target? ── yes ───────────────────────────────────────────────┐
                       └─ no ─(E) MT queue ─(F) MT worker pool ──────────────┤
  → (G) reorder buffer (release N after N−1) → segment records → caller / networking layer
```

### 2.1 VAD
- Silero VAD (conv + GRU, about 2 MB). Not compressed further.
- 512-sample windows (32 ms at 16 kHz). Debounced state machine, not a single-window threshold.
- Settings: start threshold, end threshold, minimum speech, minimum silence, edge padding, maximum segment length (default 30 s, forced cut).
- Runs on its own thread and never waits on downstream stages. Stamps each segment with a sequence number at finalization.
- Known limit: no speaker separation; overlapping speakers form one region.

### 2.2 LID
- NVIDIA NeMo TitaNet-LID, about 29M parameters. Final classification layer restricted (later fine-tuned) to the 10 target languages.
- Runs inline at segment finalization, no pool of its own, at the same time as the ASR encoder.
- Always covers all 10 languages, even when the user selected fewer, so an unexpected language is detected.
- If the detected language has no installed head: use the best-scoring installed language and set flag `unexpected_language`.

### 2.3 ASR
- `ai4bharat/indic-conformer-600m-multilingual`: Conformer encoder shared across the Indic languages, hybrid CTC / RNN-T decoding, per-language decode heads. MIT license; the Hugging Face repository is gated.
- The encoder does not need the language, which is why it runs in parallel with LID.
- v0.1: separate per-language heads, CTC first, RNN-T next.
- Later (A16): one merged head; a language-ID token is prepended to the decoder input; output projection over the union vocabulary, sliced at install time to the user's languages.
- English: separate whisper-tiny.en (8-bit), loaded only if the user selects English.
- 2–3 ASR workers: the main compute allocation.

### 2.4 Translation (optional plugin)
- Runs only when segment language ≠ target language. Peer-to-peer mode: target set per call.
- v0.1: published IndicTrans2 distilled 320M, ONNX INT8 (497 MB published; SacreBLEU 87.13 and 72.36% exact match against its FP32 version, per the model author). Optional vocabulary pruning to 10 languages (projection: about 380–400 MB).
- INT8 is the floor: the published 4-bit export is larger (712 MB), slower and lower quality (SacreBLEU 71.64).
- Later (A19): fresh distillation from the 1B teacher on 10 languages (projection: 230–340 MB at INT8).

### 2.5 Reorder buffer
- Holds finished segments; releases segment N only after N−1. Keyed by the VAD sequence number.
- A stuck segment is released after a timeout with flag `timeout`, so one failure cannot freeze the transcript.
- Trade-off: a little latency (waiting for the slowest recent segment) for in-order output.

## 3. Concurrency points

| Point | What | Kind |
| --- | --- | --- |
| A | VAD thread → segment queue | Queue 1 |
| B | LID and ASR encoder start together on one segment | Parallel 1 |
| C | Decoding starts when LID and encoder are both done | Join |
| D | 2–3 ASR workers process different segments | Parallel 2 |
| E | Only segments whose language ≠ target enter the MT queue | Queue 2 + skip |
| F | MT worker pool | Pool |
| G | Reorder buffer | Queue 3, exit gate |

Queues are bounded. When work piles up, emit a `falling_behind` event; never drop audio silently.

## 4. Interfaces (contracts)

### 4.1 Segment record

| Field | Type | Meaning |
| --- | --- | --- |
| seq | int | Sequence number from VAD finalization |
| start_ms, end_ms | int | Position in the input stream |
| language | str | Detected language code, e.g. `ta` |
| lid_confidence | float 0–1 | LID confidence |
| text | str | Transcript in the spoken language |
| translation | str or null | Text in the target language when MT ran |
| target_language | str or null | Target used for translation |
| tokens | list or null | Token timestamps and confidences |
| timings | dict | Per-stage milliseconds: vad, lid, encoder, decode, mt, wait_reorder |
| flags | list[str] | `unexpected_language`, `forced_cut`, `timeout`, `low_confidence` |

### 4.2 Pack manifest

```json
{
  "pack_id": "asr-head-ta",
  "version": "1.0.0",
  "kind": "vad | lid | asr-encoder | asr-decode-head | asr-english | mt",
  "languages": ["ta"],
  "tier": "int8 | int4mixed",
  "min_engine": "0.1.0",
  "files": [{"path": "head_ta.onnx", "sha256": "<hash>", "bytes": 0}],
  "onnx_ops": ["<every operator used>"],
  "license": "<upstream license>",
  "attribution": "<required credit text>",
  "metrics": {"wer": {"ta": null}, "test_set": "<name>", "measured_with": "<command>"}
}
```

Pack kinds: a base pack per tier (VAD + LID + encoder), one decode-head pack per language (or a sliced merged head), an English pack, an MT pack. The base APK equivalent ships VAD, LID and the encoder; the rest download on demand for the languages a user picks.

### 4.3 Model I/O contracts

| Graph | Input | Output |
| --- | --- | --- |
| VAD | 512 PCM samples + recurrent state | Speech probability + state |
| LID | Segment features | 10 language scores |
| ASR encoder | Log-mel features + length | Per-frame states + length |
| Prediction network | Previous token, language token, state | Prediction output + state |
| Joint network | Encoder frame + prediction output | Token scores including blank |
| English ASR | Whisper-format log-mel features | Tokens |
| Translation | Source tokens with language tags | Target tokens |

A model change that keeps its contract ships as a new pack without an engine release.

## 5. Compression plan

| Technique | Applies to | Mechanism |
| --- | --- | --- |
| On-demand packs | Decode heads, English, MT | Download only the user's languages |
| Domain-restricted distillation | ASR encoder (600M → 300M), LID backbone, MT | Student trained on teacher outputs, 10 target languages only |
| Vocabulary pruning | MT (fallback path) | Remove embedding and output rows for untargeted languages |
| Mixed-precision quantization | ASR encoder, LID | Embeddings, first 1–2 layers and attention Q/K/V at INT8; mid/late FFN at block-wise INT4 (MatMulNBits); QAT on INT4 layers |
| Merged decode head | ASR heads | One head with a language token instead of separate heads |
| Minimal ONNX Runtime build | Runtime binary | Only the operators the shipped models use |
| Weight clustering | Any quantized model, optional | k-means codebook after quantization; adopt only if the gate passes |

Size targets (**projection**, single-user profile: 2–3 sender languages, all Indic):

| Component | FP32 original | High-end INT8 | Low-end INT4 mixed |
| --- | --- | --- | --- |
| VAD | 4 MB | 2 MB | 2 MB |
| LID | 180 MB | 55–65 MB (fallback 90 MB) | 40–45 MB (fallback 70 MB) |
| ASR encoder | 2,400 MB (600M) | 310 MB (300M student) | 210–220 MB |
| ASR decode heads | 180 MB (9 heads) | 12 MB merged | 7–9 MB merged |
| English ASR (if selected) | — | about 40 MB | about 25 MB |
| Translation | 1,870 MB | 230–340 MB (fallback about 390 MB) | same, held at INT8 |

The 600M encoder at INT8 is about 620 MB; that is what v0.1 ships.

## 6. Evaluation targets (projection)

| Metric | High-end (INT8) | Low-end (INT4 mixed) |
| --- | --- | --- |
| STT WER, average case | 15–18% | 17–20% |
| STT WER, worst case (30 s continuous) | 19–23% | 21–26% |
| LID error rate | < 1% | < 1.5% |
| ASR real-time factor | 0.12–0.18 | 0.28–0.40 |
| Time to text, same language | about 1.2 s | about 2.7 s |
| Time to text, with translation | about 1.45–1.55 s | about 2.8–2.9 s |
| Peak RAM, full pipeline | 0.95–1.15 GB | 0.62–0.77 GB |
| Idle RAM / idle CPU | 100–120 MB / 1–3% | 85–100 MB / 3–6% |

Reference WER of the uncompressed 600M model at INT8: 13–15% (projection). Gate (D12): no language may lose more than 2 points against the measured FP32 baseline.

## 7. Known risks

- Merged head may lose accuracy on same-script pairs (Hindi/Marathi): keep separate heads as a supported pack type.
- The preprocessor must match upstream exactly; a mismatch silently raises WER. Parity tests in CI.
- INT4 kernels may be missing or slow on some mobile CPUs; INT8 pack is the per-device fallback.
- Android NNAPI is deprecated as of Android 15; do not build the high-end tier on it.
- LID may misroute during rapid code-switching; code-switched test audio is part of the smoke set.
