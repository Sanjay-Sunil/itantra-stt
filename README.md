# indicstt: iTantra speech-to-text engine (MVP)

Offline, on-device speech-to-text for **Hindi, Gujarati, Marathi, Kannada, Malayalam, Tamil, Telugu, Odia,
Bengali and English**, spoken in any order, with every segment delivered in **one target language**
(machine translation is part of the STT path). All inference runs locally with ONNX Runtime on CPU, and
every shipped model is INT8 or lower.

This repository is the **STT engine only**: the "STT" half of the iTantra AI pipeline described in `docs/`
(`ai architecture.docx`, `compression methadologies.docx`, `evaluation.docx`, `iTantra STT Pipeline.pdf`).
TTS, the networking layer (BLE / Wi-Fi Direct), encryption and the Flutter app are out of scope. The
engine's output (ordered, translated segment records) is what the networking layer would consume.

- Design: `docs/` (full app) and `agent-kit/docs/architecture.md` (engine spec and contracts)
- Work log and decisions: `PROGRESS.md`
- Measurements: `results/` (only measured numbers; figures in `docs/` are projections)

---

## 1. Pipeline

As designed in the docs (STT pipeline, mid–high tier):

```
MIC / FILE (16 kHz mono)
  → VAD thread: Silero VAD, 512-sample windows, debounced state machine        ── (A) segment queue, seq numbers
  → per segment, in parallel:                                                   ── (B)
        ├─ LID: AmberNet (TitaNet family) ─────────────┐
        └─ IndicConformer shared encoder ──────────────┴─ (C) join: pick language, attach decode head
  → decode: RNN-T (or CTC) with the language's head; English → whisper-tiny.en    ── (D) ASR worker pool (2–3)
  → language ≠ target?  yes → IndicTrans2 MT worker                             ── (E) MT queue + skip, (F) MT pool
                        no  → skip MT
  → reorder buffer: release segment N only after N−1 (timeout → flag)           ── (G) exit gate
  → Segment records → caller / UI / networking layer
```

All seven concurrency points (A–G) from the docs are implemented in `src/indicstt/scheduler.py`.
Queues are bounded and report `falling_behind` rather than dropping audio. A stuck segment is released
after a timeout with the flag `timeout`, so one failure cannot freeze the transcript.

### Models

| Stage | Model | Format | Size on disk | Licence |
|---|---|---|---|---|
| VAD | Silero VAD v5 | ONNX FP32 (as docs: too small to compress) | 2 MB | MIT |
| LID | NVIDIA NeMo **AmberNet** (`langid_ambernet`, 28.9M params) | ONNX, static INT8 | 30 MB | NGC Terms of Use (check before redistributing) |
| Indic ASR | `ai4bharat/indic-conformer-600m-multilingual` | ONNX, INT8 (or INT4-mixed) | 675 MB (535 MB) | MIT (gated on HF) |
| English ASR | whisper-tiny.en | ONNX INT8 | 44 MB | MIT |
| MT Indic↔Indic | IndicTrans2 indic-indic dist-320M | ONNX INT8 | 372 MB | MIT |
| MT Indic→English | IndicTrans2 indic-en dist-200M | ONNX INT8 | 258 MB | MIT |
| MT English→Indic | IndicTrans2 en-indic dist-200M | ONNX INT8 | 304 MB | MIT |

MT models load lazily: only the directions a user's languages need are ever loaded.

### Segment record (output contract)

| Field | Meaning |
|---|---|
| `seq`, `start_ms`, `end_ms` | VAD sequence number and position in the stream |
| `language`, `lid_confidence` | Detected language and confidence |
| `text` | Transcript in the spoken language |
| `translation`, `target_language` | Text in the target language when MT ran (`output_text` gives whichever applies) |
| `timings` | Per-stage ms: `vad`, `lid`, `lid_ctc`, `encoder`, `decode`, `mt`, `wait_reorder` |
| `flags` | `unexpected_language`, `forced_cut`, `timeout`, `low_confidence`, `error`, `asr_unavailable` |

---

## 2. Changes from the docs, and why

The engine follows the docs' architecture. Where it differs, the reason is one of four: a model isn't
publicly available, a measurement showed something better, the step needs training compute beyond an MVP,
or it belongs to the app rather than the engine.

### Models

| Docs | This engine | Why |
|---|---|---|
| **TitaNet-LID**, final layer fine-tuned to 10 languages | **AmberNet** (NVIDIA's released TitaNet-family LID, 28.9M params, same size class), exported to ONNX with an exact numpy re-implementation of NeMo's features | No TitaNet-LID checkpoint is published (only TitaNet *speaker* models). AmberNet is NVIDIA's public LID model. Fine-tuning a 10-class head (TitaNet or AmberNet) is possible on the RTX 4060 and is an open decision |
| LID covers all 10 languages | AmberNet covers 9; **Odia** is decided by the ASR model (see "CTC language check" below) | AmberNet's training set (VoxLingua107) has no Odia |
| ASR encoder **distilled 600M → 300M** | **600M** encoder, quantized | Distillation needs a large training corpus and GPU days. It stays a projection |
| **Merged decode head** with language token | Separate per-language heads (as upstream ships them), loaded on demand for the user's languages | Merging needs training. The docs flag it as the step most in need of validation |
| MT: IndicTrans2 indic-indic 1B / fresh 10-language distillation | Published **distilled 320M** (Indic↔Indic) **plus 200M indic-en and en-indic** | The 320M export is the docs' own fallback. English needs the en models because indic-indic cannot translate to or from English |
| English: whisper-tiny.en **GGUF** Q8 | whisper-tiny.en **ONNX INT8** | One runtime (ONNX Runtime) for every model |
| Low tier: single language, ~160 MB Vakyansh model | Low tier = **INT4-mixed** of the same pipeline, all languages and code-switching | Same direction as `evaluation.docx` (compression unlocks code-switching on low-end) |

### Quantization (compression doc, technique 4)

| Docs | This engine | Why (measured, `results/quantization.md`) |
|---|---|---|
| Encoder INT8 ≈ 620 MB | INT8 encoder **653 MB** (whole ASR 675 MB) | — |
| INT8 everywhere | Per-channel dynamic INT8 on every MatMul and Conv, **except the 24 depthwise convs** (≈0.2M of 607M weights, ≈1 MB) which stay FP32 | ORT's dynamic INT8 depthwise convolution raised Hindi WER 8.16 → 11.57%. Found by a layer-group sensitivity scan; with this exception Hindi is 8.92% |
| INT4 on mid/late FFN, INT8 on early layers and attention, QAT on INT4 layers | INT4 block-wise (MatMulNBits, block 32) on FFN MatMuls of layers 2–23; layers 0–1 and attention INT8; **no QAT** | QAT needs training. Without it INT4-mixed costs +0.3 WER on average, so QAT isn't needed yet |
| RNN-T prediction/joint nets kept FP32 (explainer doc) | INT8 | Measured: no WER cost (≤ noise) |
| LID INT8 | **Static** (calibrated QDQ) INT8 | Dynamic INT8 was 3× *slower* than FP32 on conv layers. Static INT8 is 2.6× faster than FP32 |
| Vocabulary pruning of MT, weight clustering, minimal-ops ORT build | Not done | Optional size work, mainly relevant for an Android build |

### Engine behaviour added or changed

- **CTC language check.** The shared encoder's CTC head is scored under each selected language's vocabulary
  mask. That score decides between Indic languages when LID is unsure, and it decides Odia. If a selected
  language has no LID class, LID only decides English vs Indic. Odia+English profile: Odia 0% → 100% correct.
- **Encoder cancellation for English.** LID and the Indic encoder still start in parallel (point B). When LID
  decides English, the in-flight encoder run is cancelled (`onnxruntime.RunOptions.terminate`). English
  time-to-text: 1.77 s → 1.23 s.
- **VAD gain control.** A boost-only automatic gain (cap 10×, ~1 s release) sits on the VAD input. The ASR
  still gets raw audio. It catches a quiet speaker after a loud one, with no false segments on hiss or
  rumble (`results/vad.md`).
- **Code-switching granularity.** Language is decided per VAD segment, so switches are handled at pauses.
  The docs' "mid-sentence switching without a pause" is not handled: a segment gets one language.
- **RNN-T start token.** Upstream's `config.json` says SOS = 256, but its loader never reads that file. The
  effective value is 5632. Using 256 does not reproduce upstream output.
- **Runtime.** A Python reference engine on CPU (laptop as the edge device) with a Streamlit UI. The docs'
  ORT Mobile / Flutter / Android build is not part of this MVP.

---

## 3. Measured results

Laptop i7-14650HX, **CPU only**, INT8 unless noted. Test data: first 18–20 FLEURS test clips per language
(`scripts/build_smoke_set.py`). That's a small sample: ±1 WER point is noise for one language.

| Metric | Measured | Docs projection (high-end phone, INT8) |
|---|---|---|
| Indic ASR WER, RNN-T, mean of 9 languages | **19.05%** (INT4-mixed 19.39%) | 15–18% (300M distilled) |
| Per language (INT8) | hi 8.9 · bn 13.3 · mr 15.9 · gu 17.6 · kn 18.8 · or 21.3 · te 21.5 · ml 27.1 · ta 27.2 | — |
| INT8 vs FP32 (gate ≤ 2 points) | hi +0.76, gu +0.66 (FP32 measured for these two only) | — |
| English ASR WER | 10.0% | — |
| LID accuracy, AmberNet INT8, 9 classes | 99.4% (calibration overlaps test clips) | < 1% error |
| Language decision, 6 speaker profiles of 2–3 languages incl. Odia | 97.5–100% | — |
| Code-switched recordings, pipeline vs oracle WER | +0 to +2.2 points; segment language 94–100% | — |
| Speech → English, BLEU / chrF++ (text → English) | 29.4 / 57.6 (34.4 / 60.7) | — |
| Time-to-text after speech ends (real-time stream) | **0.81 s** with MT, **1.23 s** English | 1.45–1.55 s / ~1.2 s |
| ASR real-time factor (encoder + RNN-T) | 0.128 (INT4-mixed 0.151) | 0.12–0.18 |
| Peak RAM, whole engine (VAD, LID, ASR, Whisper, MT) | **1.85 GB** (INT4-mixed 1.73 GB) | 0.95–1.15 GB |

Where it falls short: RAM is above the projection, mainly because the encoder is 600M (not the 300M student)
and MT plus Whisper are resident. Tamil and Malayalam WER are high (~27%). INT4-mixed is smaller but not
faster on this x86 CPU; its speed on ARM is unmeasured. Nothing is measured on a phone yet.

Details: `results/quantization.md`, `results/engine-latency.md`, `results/pipeline-and-translation.md`,
`results/vad.md`. Each result file names the command that produced it.

---

## 4. Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[ui,dev]" pyarrow IndicTransToolkit onnx_ir

# 1. models (IndicConformer is gated: accept its conditions on Hugging Face first)
.venv/bin/python scripts/fetch_models.py

# 2. AmberNet LID → ONNX (one-off, needs NeMo in a separate env)
python3 -m venv .venv-nemo
.venv-nemo/bin/pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
.venv-nemo/bin/pip install "nemo_toolkit[asr]" onnx onnxruntime
.venv-nemo/bin/python scripts/export_ambernet.py

# 3. test clips (FLEURS, CC-BY-4.0), needed for LID calibration
.venv/bin/python scripts/build_smoke_set.py

# 4. quantize: ASR INT8 and INT4-mixed tiers, LID static INT8
.venv/bin/python scripts/quantize.py --tier int8 --lid
.venv/bin/python scripts/quantize.py --tier int4mixed

# 5. demo
.venv/bin/python scripts/make_demo_clip.py en:0 hi:0 en:1 ta:0 --out data/demo/mixed_en_hi_ta.wav
.venv/bin/streamlit run app/streamlit_app.py --server.address 0.0.0.0
```

The UI takes the device microphone (live), a recording from any browser on the same network (e.g. a phone),
or files and samples. Inference always stays on the edge device.

### Python API

```python
from indicstt.engine import Engine, EngineConfig
from indicstt.audio import MicSource

eng = Engine(EngineConfig(
    languages=["hi", "ta", "en"],   # languages the speaker may use
    target_language="en",           # every segment delivered in English (None = no MT)
    asr_tier="int8",                # "int8" (high-end) | "int4mixed" (low-end)
    asr_mode="rnnt",                # "rnnt" (accurate) | "ctc" (fast)
))
for seg in eng.transcribe_file("clip.wav"):          # or: eng.stream(MicSource())
    print(seg.seq, seg.language, seg.text, "→", seg.output_text, seg.timings)
```

Other options: `manual_language` (skip LID), `asr_workers`, `mt_workers`, `threads_per_model`, `lid_ctc`
(`auto` | `always`), and `vad` (`VADConfig`: thresholds, minimum speech/silence, padding, 30 s forced cut,
`agc`).

---

## 5. Repository layout

```
src/indicstt/
  audio.py       file / buffer / microphone input, resampling to 16 kHz mono
  vad.py         Silero VAD, gain control, debounced segmenter (chunk-size invariant)
  features.py    NeMo-compatible log-mel features in numpy (AmberNet, IndicConformer)
  lid.py         AmberNet LID runner
  asr.py         IndicConformer: shared encoder, CTC + RNN-T greedy decoding, CTC language scores
  english.py     whisper-tiny.en runner (log-mel + KV-cached greedy decoding)
  mt.py          IndicTrans2 runners, routed by direction, lazily loaded
  scheduler.py   concurrency points A–G, worker pools, reorder buffer
  engine.py      Engine / EngineConfig public API
  segment.py     Segment record
  evaluation.py  WER / CER with Indic-aware normalisation, regression gate
app/streamlit_app.py      edge demo UI
scripts/                  fetch_models, export_ambernet, quantize, build_smoke_set, make_demo_clip,
                          evaluate_asr, evaluate_lid, evaluate_engine_lid, evaluate_pipeline, evaluate_mt,
                          benchmark_engine
tests/                    pytest: VAD state machine, scheduler stress (1,000 randomized runs), engine (models)
results/                  measurements, each with its command
```

Tests: `.venv/bin/pytest -q` (no downloads needed); `INDICSTT_MODELS=1 .venv/bin/pytest -q` adds real-model tests.

## 6. Not done yet

- 600M → 300M encoder distillation, merged decode head, QAT, LID/MT distillation (training work)
- Fine-tuned 10-language LID (TitaNet or AmberNet head); decision pending
- Android / Flutter build, minimal-ops ONNX Runtime, on-phone benchmarks
- MT vocabulary pruning, weight clustering
- Tests on real speakers and real noise (all numbers so far use read FLEURS speech)
- FP32 baseline for 7 of 9 Indic languages (only hi and gu measured)
