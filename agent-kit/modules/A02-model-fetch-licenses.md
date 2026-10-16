# A02 · Model fetch script and license record

**Wave:** 1 (Foundation)  
**Depends on:** A01  
**Needs from outside the agent:** HF_TOKEN with the IndicConformer access conditions accepted (human, D2)  
**Size:** S (about 0.3–1M tokens)

## Goal

One script that downloads every upstream model the plan uses, verifies it, and records its license.

## Read first

- CLAUDE.md, DECISIONS.md, STATUS.md
- docs/architecture.md (the sections this module touches)
- The STATUS.md entry for A01

## Build

- [ ] `scripts/fetch_models.py --set v01|all` reading HF_TOKEN from the environment; resumable; writes to `models/`
- [ ] Model set for v0.1: Silero VAD (ONNX), NeMo LID model (TitaNet/AmberNet), `ai4bharat/indic-conformer-600m-multilingual`; set `all` adds whisper-tiny.en and the IndicTrans2 distilled 320M ONNX export
- [ ] `models/manifest.lock.json`: source, revision, file list, sha256, bytes
- [ ] `LICENSES.md`: one row per model and dataset with license, redistribution yes/no, attribution text; mark unknowns as `to confirm`
- [ ] Inspect the IndicConformer repository after download and document its ONNX layout (encoder, CTC/RNN-T parts, per-language vocab) in `docs/upstream-indicconformer.md`

## Acceptance (all must pass before marking done)

- [ ] `python scripts/fetch_models.py --set v01` completes and re-running is a no-op
- [ ] sha256 of every file matches `manifest.lock.json`
- [ ] `LICENSES.md` has no empty cells

## Out of scope

- Converting or quantizing models

## When you finish

Update STATUS.md row A02: status, PR link, measured results (with the command), follow-ups. If blocked, set status to Blocked and write exactly what a human must provide.
