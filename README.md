# indicstt — iTantra STT engine (MVP)

Offline, on-device speech-to-text for Hindi, Gujarati, Marathi, Kannada, Malayalam, Tamil, Telugu, Odia,
Bengali and English, with every segment delivered in one target language.

```
mic/file → Silero VAD → [AmberNet LID ‖ ASR encoder] → decode → IndicTrans2 MT (only if language ≠ target) → reorder buffer → segments
```

Design docs: `docs/` (full iTantra app), `agent-kit/docs/architecture.md` (engine spec). Work log: `PROGRESS.md`.

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[ui,dev]" pyarrow IndicTransToolkit
.venv/bin/python scripts/fetch_models.py           # models → models/
.venv/bin/python scripts/build_smoke_set.py        # FLEURS clips → data/smoke/
.venv/bin/python scripts/make_demo_clip.py en:0 hi:0 en:1 ta:0 --out data/demo/mixed_en_hi_ta.wav
.venv/bin/streamlit run app/streamlit_app.py --server.address 0.0.0.0
```

The IndicConformer ASR model is gated: accept its conditions at
https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual before fetching.

The AmberNet LID ONNX is produced once with NeMo: create `.venv-nemo` with `torch` (CPU) and `nemo_toolkit[asr]`,
then run `.venv-nemo/bin/python scripts/export_ambernet.py`.

## Python API

```python
from indicstt.engine import Engine, EngineConfig
eng = Engine(EngineConfig(languages=["hi", "ta", "en"], target_language="en"))
for seg in eng.transcribe_file("clip.wav"):
    print(seg.seq, seg.language, seg.text, seg.translation, seg.timings)
```

## Measured (laptop CPU, INT8, FLEURS smoke set — details in `results/`)

| | |
|---|---|
| Indic ASR WER (9 languages, RNN-T) | 19.1% mean (INT4-mixed 19.4%) |
| English ASR WER | 10.0% |
| LID, 2–3-language profiles incl. Odia | 97.5–100% |
| Code-switched pipeline vs oracle | +0 to +2.2 WER |
| Speech → English, BLEU / chrF++ | 29.4 / 57.6 |
| Time-to-text after speech ends | 0.81 s (with MT), 1.23 s (English) |
| Peak RAM, whole engine | 1.85 GB |
| Models on disk | ASR 675 MB, LID 30 MB, English 44 MB, MT 200–370 MB per direction |

Figures in `docs/` are projections; only `results/` holds measurements.
