# A13 · Translation plugin and English path

**Wave:** 3 (Engine and v0.1 release)  
**Depends on:** A10  
**Needs from outside the agent:** Models downloaded (set `all`)  
**Size:** M (about 1–3M tokens)

## Goal

Optional translation into the receiver's language, and English transcription.

## Read first

- CLAUDE.md, DECISIONS.md, STATUS.md
- docs/architecture.md (the sections this module touches)
- The STATUS.md entry for A10

## Build

- [ ] IndicTrans2 distilled 320M INT8 ONNX runner with its pre/post-processing and language tags
- [ ] Called only when segment language ≠ target; peer-to-peer mode sets target per call
- [ ] Vocabulary pruning script to the 10 target languages (run if time allows; record size)
- [ ] English segments → whisper-tiny.en runner (only if the English pack is installed)
- [ ] SacreBLEU and exact-match evaluation against the FP32 reference, same method as the published benchmark

## Acceptance (all must pass before marking done)

- [ ] Skip logic unit-tested
- [ ] SacreBLEU recorded in docs/results
- [ ] Engine runs with MT disabled and no MT pack installed

## Out of scope

- Fresh MT distillation (A19)

## When you finish

Update STATUS.md row A13: status, PR link, measured results (with the command), follow-ups. If blocked, set status to Blocked and write exactly what a human must provide.
