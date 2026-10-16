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
| 4 | LID = NeMo AmberNet (TitaNet-family, NGC) exported to ONNX + CTC-mask scoring from IndicConformer | No public TitaNet-LID exists. AmberNet (VoxLingua107) lacks Odia → CTC scoring covers it |
| 5 | MT Indic↔Indic = hari31416/indictrans2-indic-indic-dist-320M-ONNX-int8 | Published, open |
| 6 | MT Indic→En / En→Indic = ai4bharat/indictrans2-{indic-en,en-indic}-dist-200M | **Gated**; we export to ONNX + INT8 ourselves |
| 7 | Numbers: only measured values go into `results/`; doc figures stay "projection" | agent-kit rule |
| 8 | Segment record per agent-kit/docs/architecture.md §4.1 | contract |

## Blockers (needs the user)

- [ ] Accept access on HF for: indic-conformer-600m-multilingual, indictrans2-indic-en-dist-200M, indictrans2-en-indic-dist-200M (account `zxnjxd`). Currently 403.

## Status

| Part | State |
|---|---|
| Repo + venv | done |
| Audio input (file/buffer/mic) | todo |
| VAD + segmenter | todo |
| Scheduler + reorder buffer | todo |
| Model fetch script | todo |
| ASR runner (IndicConformer) | blocked on HF access |
| English ASR (whisper-tiny.en) | todo |
| LID (AmberNet ONNX + CTC scoring) | todo |
| MT (IndicTrans2) | todo |
| Engine API | todo |
| Streamlit UI | todo |
| Quantization (INT8/INT4) + measured results | todo |

## Environment

- Laptop: i7-14650HX (24 threads), 15 GB RAM, RTX 4060 8 GB, built-in mic. Python 3.11, venv at `.venv/`.
- Run: `.venv/bin/streamlit run app/streamlit_app.py` (once built).

## Log

- 2026-10-16 — Analysed docs/ and agent-kit/. Verified model availability on HF. Agreed scope with user. Created repo, venv, PROGRESS.md.
