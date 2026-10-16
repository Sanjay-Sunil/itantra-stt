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
| 4 | LID = NeMo AmberNet (TitaNet-family, NGC) exported to ONNX (+ CTC-mask scoring planned) | **Deferred by user (revisit later); AmberNet in use.** Options: A keep AmberNet, B TitaNet-Large encoder + 10-class head fine-tuned on FLEURS, C fine-tune AmberNet head to 10 classes incl Odia |
| 5 | MT Indic↔Indic = hari31416/indictrans2-indic-indic-dist-320M-ONNX-int8 | Published, open |
| 6 | MT Indic→En / En→Indic = hari31416/indictrans2-{indic-en,en-indic}-dist-200M-ONNX-int8 | Ungated ONNX exports of AI4Bharat models |
| 7 | Numbers: only measured values go into `results/`; doc figures stay "projection" | agent-kit rule |
| 8 | Segment record per agent-kit/docs/architecture.md §4.1 | contract |

## Blockers (needs the user)

- [x] HF access to IndicConformer granted 2026-10-16; downloaded (2.4 GB FP32).
- [ ] (later) Decide LID option A/B/C (see Decisions #4) — user deferred.

## Status

| Part | State |
|---|---|
| Repo + venv (`.venv`, NeMo export env `.venv-nemo`) | done |
| Model fetch (`scripts/fetch_models.py`) | done except IndicConformer (gated) |
| FLEURS smoke set, 20 clips × 10 langs (`scripts/build_smoke_set.py`) | done → data/smoke/ |
| Audio input (file/buffer/mic) `audio.py` | done |
| VAD + segmenter `vad.py` | done; boost-only gain control (cap 10x) before Silero — results/vad.md |
| Scheduler + reorder buffer `scheduler.py` | done, 1000-run stress test passes |
| MT `mt.py` (3 IndicTrans2 bundles) | done, smoke-tested |
| English ASR `english.py` | done, WER 10.0% on smoke |
| NeMo-parity features `features.py` | done |
| LID `lid.py` (AmberNet ONNX, static INT8) | done; final model choice deferred by user |
| ASR runner `asr.py` (IndicConformer) | done; exact parity with upstream 9/9 (CTC and RNN-T) on hi/ta/or; numpy preprocessor MAE 1e-6 vs preprocessor.ts |
| WER/CER eval `evaluation.py`, `scripts/evaluate_asr.py` | done |
| Engine API `engine.py` | done; English path verified end-to-end; Indic segments tagged `asr_unavailable` until IndicConformer lands |
| Streamlit UI `app/streamlit_app.py` | done; verified via AppTest (8 segments, RTF 0.07) |
| Demo clip builder `scripts/make_demo_clip.py` | done → data/demo/ |
| Quantization `scripts/quantize.py` | done: ASR INT8 675 MB (mean WER 19.05%), INT4-mixed 535 MB (19.39%), LID static INT8 30 MB (99.44%). See results/quantization.md |
| Engine + UI on INT8 IndicConformer | done; tier + decoder selectable in sidebar |

Measured numbers: `results/`.

## Environment

- Laptop: i7-14650HX (24 threads), 15 GB RAM, RTX 4060 8 GB, built-in mic. Python 3.11, venv at `.venv/`.
- Run UI: `.venv/bin/streamlit run app/streamlit_app.py --server.address 0.0.0.0`
- Tests: `.venv/bin/pytest -q` (add `INDICSTT_MODELS=1` for real-model tests)

## Log

- 2026-10-16 — Analysed docs/ and agent-kit/. Verified model availability on HF. Agreed scope with user. Created repo, venv, PROGRESS.md.
- 2026-10-16 — Built audio, VAD, scheduler, MT (all 3 directions via ungated hari31416 ONNX int8), Whisper English, AmberNet export with exact NeMo parity, FLEURS smoke set. Waiting on HF access + LID decision.
- 2026-10-16 — Engine + Streamlit UI working. Known issue: Silero VAD misses a very quiet clip right after loud speech (60x level jump in a spliced test); demo clips are level-matched. Possible fix later: AGC before VAD.
- Next: IndicConformer runner (needs HF access) → CTC-mask language scoring (covers Odia) → INT8/INT4 quantization + measured WER/RTF/RAM table.
- 2026-10-16 — IndicConformer in. Gotchas: (1) upstream preprocessor uses reflect pad + T//hop+1 frames; (2) effective RNN-T SOS is 5632 (upstream ignores config.json's 256); (3) prediction net must get per-language token ids (union ids → 568% WER). User requirement: ship INT8 or lower for every model (FP32 = reference only).
- 2026-10-16 — Quantized everything to INT8 or lower (user requirement). Key finding: ORT dynamic INT8 on depthwise convs costs +3.4 WER (hi) → those 0.2M weights stay FP32; per-channel scales. LID: dynamic INT8 was 3x slower → static QDQ INT8 is 2.6x faster than FP32. Partial FP32 ref only for hi/gu (user stopped full FP32 run).
- 2026-10-16 — Odia LID fixed (English-vs-Indic decision when selected language has no AmberNet class): or+en profile 0% → 100% Odia; 6 profiles 97.5–100%. LID cancels the parallel Indic encoder when it says English (ORT RunOptions.terminate): English time-to-text 1.77 → 1.23 s. Real-time benchmark: INT8 time-to-text 0.81 s (Indic+MT), 1.23 s (English); peak RSS 1.85 GB. VAD gain control added; known VAD issue resolved.
- Next: real-voice code-switch test (needs user recording); optionally full FP32 baseline for remaining 7 languages; ORT minimal-ops build / Android are out of MVP scope.
