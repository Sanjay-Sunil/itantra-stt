# PROGRESS

Running log for the iTantra STT engine MVP. Newest entries at the bottom of the log.
If work gets stuck, read **Decisions**, **Status** and the last log entries first.

## Scope (agreed with user, 2026-10-16)

- Build the **STT engine only** (`indicstt`). `docs/` describes the full iTantra app (TTS, networking, Flutter) — context only.
- MVP for a demo on an edge device: the engine runs fully offline on CPU; a **Streamlit UI** shows it working live.
- MT is part of STT: every segment is delivered in **one target language** chosen by the user.
- **English is in**: English speech recognition + translation to/from English.
- Test languages: all 10 from the docs (hi, gu, mr, kn, ml, ta, te, or, bn, en), smoke set from FLEURS.
- Process: one repo, frequent commits, this file. agent-kit is a reference, not followed module-by-module.

## Decisions

| # | Decision | Notes |
|---|---|---|
| 1 | VAD = Silero VAD ONNX (onnx-community/silero-vad) | 512-sample windows, debounced state machine |
| 2 | ASR = ai4bharat/indic-conformer-600m-multilingual (ONNX) | Shared encoder, CTC + per-language RNN-T heads. **Gated: user must accept on HF** |
| 3 | English ASR = whisper-tiny.en ONNX (onnx-community) | ~40 MB int8 |
| 4 | LID = NeMo AmberNet (TitaNet-family, NGC) exported to ONNX (+ CTC-mask scoring planned) | **OPEN — user asked about TitaNet-LID.** Options: A keep AmberNet, B TitaNet-Large encoder + 10-class head fine-tuned on FLEURS, C fine-tune AmberNet head to 10 classes incl Odia |
| 5 | MT Indic↔Indic = hari31416/indictrans2-indic-indic-dist-320M-ONNX-int8 | Published, open |
| 6 | MT Indic→En / En→Indic = hari31416/indictrans2-{indic-en,en-indic}-dist-200M-ONNX-int8 | Ungated ONNX exports of AI4Bharat models |
| 7 | Numbers: only measured values go into `results/`; doc figures stay "projection" | agent-kit rule |
| 8 | Segment record per agent-kit/docs/architecture.md §4.1 | contract |

## Blockers (needs the user)

- [ ] Accept access on HF for ai4bharat/indic-conformer-600m-multilingual (account `zxnjxd`). Currently 403. (MT English models no longer need it.)
- [ ] Decide LID option A/B/C (see Decisions #4).

## Status

| Part | State |
|---|---|
| Repo + venv (`.venv`, NeMo export env `.venv-nemo`) | done |
| Model fetch (`scripts/fetch_models.py`) | done except IndicConformer (gated) |
| FLEURS smoke set, 20 clips × 10 langs (`scripts/build_smoke_set.py`) | done → data/smoke/ |
| Audio input (file/buffer/mic) `audio.py` | done |
| VAD + segmenter `vad.py` | done, unit-tested (no real-model test yet) |
| Scheduler + reorder buffer `scheduler.py` | done, 1000-run stress test passes |
| MT `mt.py` (3 IndicTrans2 bundles) | done, smoke-tested |
| English ASR `english.py` | done, WER 10.0% on smoke |
| NeMo-parity features `features.py` | done |
| LID `lid.py` (AmberNet ONNX) | done; choice pending user |
| ASR runner (IndicConformer) | blocked on HF access |
| Engine API + Streamlit UI | todo |
| Quantization (INT8/INT4) + results | todo |

Measured numbers: `results/`.

## Environment

- Laptop: i7-14650HX (24 threads), 15 GB RAM, RTX 4060 8 GB, built-in mic. Python 3.11, venv at `.venv/`.
- Run: `.venv/bin/streamlit run app/streamlit_app.py` (once built).

## Log

- 2026-10-16 — Analysed docs/ and agent-kit/. Verified model availability on HF. Agreed scope with user. Created repo, venv, PROGRESS.md.
- 2026-10-16 — Built audio, VAD, scheduler, MT (all 3 directions via ungated hari31416 ONNX int8), Whisper English, AmberNet export with exact NeMo parity, FLEURS smoke set. Waiting on HF access + LID decision.
