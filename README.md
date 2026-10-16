# iTantra STT engine

Offline, on-device speech-to-text for **Hindi, Gujarati, Marathi, Kannada, Malayalam, Tamil, Telugu, Odia,
Bengali and English**, spoken in any order, with every part delivered in **one language the listener chooses**.
This is the STT half of the iTantra AI pipeline (`docs/`). TTS, mesh networking and the app UI are other parts.

## Progress toward the final STT product

```
██████████████░░░░░░░░░░░░░░░░░░░░░░  39%   (10.5 of 27 milestones; done = 1, partial = ½)
```

| Phase | Milestones | Status |
|---|---|---|
| 1. Foundation | repo scaffold · model fetch + licences · evaluation harness | ◐ ◐ ● |
| 2. Models into ONNX | ASR runner + parity · LID · VAD · INT8 / INT4 quantization | ● ● ● ● |
| 3. Engine (reference) | audio I/O · scheduler + reorder · pipeline API · CLI · packs + model manager · translation + English · release v0.1 | ● ● ● ◐ ○ ● ○ |
| 4. Compression training | encoder 600M → 300M · merged decode head · QAT for INT4 · LID distillation · MT distillation · weight clustering | ○ ○ ○ ○ ○ ○ |
| 5. Mobile | C++ core · Python on core · Android library · Flutter plugin · iOS · on-device benchmarks · docs + release | ○ ○ ○ ○ ○ ○ ○ |

● done · ◐ partial · ○ not started. Milestones follow `agent-kit/PLAN.md` (A01–A27). LID distillation is
optional now: AmberNet stays the LID model (decision 2026-10-16), measured at 99.5% with all 10 languages.

## Current product vs final product

| | **Current (MVP, measured)** | **Final STT product (target, from `docs/`)** |
|---|---|---|
| Runs on | Laptop CPU (i7), Python reference engine | Android phones via Flutter; C++ core on ONNX Runtime Mobile (minimal-ops build) |
| Device tiers | INT8 and INT4-mixed models, same laptop | High-end: INT8 on NPU/DSP · Low-end: INT4-mixed on CPU only |
| Languages / code-switching | 10 languages, switching at pauses | 10 languages; high-end 2–3 per speaker, low-end up to 2 |
| VAD | Silero, 2 MB | Silero, 2 MB |
| Language ID | AmberNet INT8, 30 MB (+ ASR CTC check for Odia) | AmberNet INT8 (kept); docs' target was a distilled 10-language LID, 55–65 / 40–45 MB |
| ASR encoder | IndicConformer **600M**, 653 MB INT8 / 511 MB INT4-mixed | Distilled **300M**: 310 MB INT8 / 210–220 MB INT4-mixed + QAT |
| ASR decode heads | Separate per-language heads, loaded on demand | One merged head over the user's 2–3 languages, 12 / 7–9 MB |
| English ASR | whisper-tiny.en INT8, 44 MB | whisper-tiny.en, ~40 / ~25 MB |
| Translation | IndicTrans2 distilled 200M/320M INT8: 258–372 MB per direction | Distilled to 10 languages: 230–340 MB, INT8 floor |
| STT models on disk, one user (2 speaker languages, 1 MT direction) | ~1.0–1.1 GB | high-end ~0.61–0.73 GB · low-end ~0.49–0.62 GB (sum of the docs' component targets) |
| Peak RAM | ~1.7 GB, all models of one profile + 1 MT direction | ≤ 0.95–1.15 GB high-end · ≤ 0.62–0.77 GB low-end (docs' figure, whole pipeline) |
| Idle RAM / CPU | not measured | 100–120 MB / 1–3% · 85–100 MB / 3–6% |
| ASR WER, average (9 Indic languages) | **19.1%** INT8 · 19.4% INT4-mixed | 15–18% high-end · 17–20% low-end |
| LID error | **0.5%** (all 10 candidates) | < 1% · < 1.5% |
| ASR real-time factor | 0.13 INT8 · 0.15 INT4-mixed (laptop CPU) | 0.12–0.18 · 0.28–0.40 (phones) |
| Speech end → text, same language | ~1.3 s (English) | ~1.2 s · ~2.7 s |
| Speech end → text, with translation | **~1.0 s** | ~1.45–1.55 s · ~2.8–2.9 s |
| Speech → English quality | BLEU 29.4 / chrF++ 57.6 | not specified (MT target: ≥ parent model, 87.13 SacreBLEU vs its FP32) |
| Packaging | Models in `models/`, fetched by script | Base pack in the app (VAD + LID + encoder); per-language heads, English and MT packs downloaded on demand |

Current numbers come from `results/` (FLEURS test clips, 18–20 per language). Final numbers are the docs'
projections, so treat them as targets, not measurements.

---

## The final STT product

### Pipeline

```
MIC (16 kHz) → VAD thread: Silero, 32 ms windows, debounced; numbered segments          ── (A) segment queue
  → per segment, in parallel:                                                           ── (B)
        ├─ LID: AmberNet ─────────────────────────────┐
        └─ ASR encoder (distilled 300M Conformer) ─────┴─ (C) join: language → merged head ── (D) ASR worker pool
  → language ≠ listener's language?  yes → MT (distilled IndicTrans2)                   ── (E) MT queue, (F) MT pool
                                     no  → skip
  → reorder buffer: segment N released after N−1                                        ── (G) exit gate
  → segment records → networking layer (encrypted, mesh)
```

All seven concurrency points are implemented and stress-tested in the reference engine.

### Components and compression

| Stage | Final model | Compression applied | Target size (high / low tier) |
|---|---|---|---|
| VAD | Silero VAD | none (already tiny) | 2 MB |
| LID | AmberNet (TitaNet family) | static INT8; Odia decided by the ASR CTC check | 30 MB (measured) |
| ASR encoder | IndicConformer, distilled 600M → 300M on the 10 languages | INT8 / INT4-mixed with QAT; depthwise convs kept FP32 (sensitivity scan) | 310 / 210–220 MB |
| Decode head | one merged RNN-T head with a language token, sliced to the user's languages | INT8 / INT4 output projection | 12 / 7–9 MB |
| English ASR | whisper-tiny.en | INT8 / further quantized | ~40 / ~25 MB |
| Translation | IndicTrans2 distilled on the 10 languages (from the 1B teacher) | INT8 (4-bit measured worse for this family) | 230–340 MB |
| Runtime | ONNX Runtime Mobile, only the operators the packs use | operator pruning | several MB smaller binary |

On-demand distribution: the app ships VAD, LID and the encoder. Decode heads, the English pack and MT packs
download for the languages a user picks. Weights are memory-mapped, so idle RAM stays near the VAD + LID
footprint.

### Interfaces

- **Segment record:** `seq`, `start_ms`, `end_ms`, `language`, `lid_confidence`, `text`, `translation`,
  `target_language`, `timings` (per stage), `flags` (`unexpected_language`, `forced_cut`, `timeout`,
  `low_confidence`), plus `lid_detail` (how the language was decided).
- **Pack manifest:** id, version, kind (vad / lid / asr-encoder / asr-decode-head / asr-english / mt), languages,
  tier, files with sha256, operator list, licence, measured metrics. See `agent-kit/docs/architecture.md`.

### What changes from the original docs, and why

| Docs | Final product | Why |
|---|---|---|
| TitaNet-LID fine-tuned to 10 languages | AmberNet + ASR CTC check | No public TitaNet-LID; AmberNet measured 99.5% on all 10 |
| English via GGUF whisper | whisper-tiny.en ONNX INT8 | One runtime for every model |
| INT8 everywhere | Depthwise convs FP32 (≈1 MB) | Measured +3.4 WER on Hindi when quantized |
| RNN-T prediction/joint FP32 | INT8 | No measured cost |
| — | LID cancels the encoder run when it hears English | English time-to-text 1.77 → 1.23 s |
| — | Boost-only gain control before VAD | Catches quiet speakers after loud ones |
| Code-switching mid-sentence | Switching at pauses | Language is decided per VAD segment |
| — | English→Indic stays a dedicated model | indic-indic fed English measured −10 chrF++ |

### Still to build for the final product

1. Encoder distillation 600M → 300M (training data and GPU days); the biggest lever for RAM and WER.
2. Merged decode head and QAT for INT4.
3. MT distillation or vocabulary pruning to 10 languages (42–59% of each MT model is vocabulary tables).
4. C++ core with a C API, Android library, Flutter plugin, minimal-ops ONNX Runtime build.
5. Packs and model manager (download, verify, memory-map, evict).
6. On-device benchmarks on a low-end CPU phone and a high-end NPU phone.

---

## Running the current product (MVP)

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[ui,dev]" pyarrow IndicTransToolkit onnx_ir
.venv/bin/python scripts/fetch_models.py                 # IndicConformer is gated: accept on Hugging Face first
python3 -m venv .venv-nemo && .venv-nemo/bin/pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
.venv-nemo/bin/pip install "nemo_toolkit[asr]" onnx onnxruntime
.venv-nemo/bin/python scripts/export_ambernet.py          # AmberNet → ONNX (one-off)
.venv/bin/python scripts/build_smoke_set.py               # FLEURS test clips (also used for LID calibration)
.venv/bin/python scripts/quantize.py --tier int8 --lid
.venv/bin/python scripts/quantize.py --tier int4mixed
.venv/bin/python scripts/make_demo_clip.py en:0 hi:0 en:1 ta:0 --out data/demo/mixed_en_hi_ta.wav
.venv/bin/streamlit run app/streamlit_app.py --server.address 0.0.0.0
```

The dashboard has three tabs: **Try it** (record, upload, examples, live mic; transcript cards in the chosen
language), **Language test** (VAD chunks tagged with their language on a timeline, all 10 languages) and
**How does this work?** (pipeline diagram with measured latency and RAM per stage).

```python
from indicstt.engine import Engine, EngineConfig
eng = Engine(EngineConfig(languages=["hi", "ta", "en"], target_language="en", asr_tier="int8", asr_mode="rnnt"))
for seg in eng.transcribe_file("clip.wav"):
    print(seg.seq, seg.language, seg.text, "→", seg.output_text)
```

Measurements and the commands that produced them: `results/` (`quantization.md`, `memory.md`,
`engine-latency.md`, `pipeline-and-translation.md`, `vad.md`). Work log: `PROGRESS.md`.
Tests: `.venv/bin/pytest -q` (add `INDICSTT_MODELS=1` for real-model tests).

### Licences

Code: Apache-2.0. Models: IndicConformer (MIT, gated), IndicTrans2 (MIT), whisper (MIT), Silero VAD (MIT),
AmberNet (NGC Terms of Use; check before redistributing). Test audio: FLEURS (CC-BY-4.0).
