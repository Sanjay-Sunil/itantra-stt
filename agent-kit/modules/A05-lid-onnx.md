# A05 · LID model export and runner

**Wave:** 2 (Models into ONNX)  
**Depends on:** A02, A03  
**Needs from outside the agent:** Models downloaded; NeMo export may need a machine with NeMo installed  
**Size:** M (about 1–3M tokens)

## Goal

A language identifier that runs in onnxruntime and returns one of the 10 target languages per segment.

## Read first

- CLAUDE.md, DECISIONS.md, STATUS.md
- docs/architecture.md (the sections this module touches)
- The STATUS.md entry for A02
- The STATUS.md entry for A03

## Build

- [ ] Export the NeMo LID model to ONNX; if export fails within the session, document why and fall back to running only the classifier restricted to the 10 classes
- [ ] Restrict output to the 10 target languages (mask other classes); return top-1 and confidence
- [ ] `indicstt.lid.LID.identify(audio) -> (lang, confidence)`
- [ ] Optional, CPU-feasible: fine-tune only the final classification layer on 10-language data if available (store the script even if not run)
- [ ] Measure LID accuracy and confusion matrix on the smoke set

## Acceptance (all must pass before marking done)

- [ ] Runs on a clip in under 100 ms on a laptop CPU (record measured value)
- [ ] Accuracy recorded in docs/results
- [ ] Unit tests with a fake model

## Out of scope

- Backbone distillation (A18)

## When you finish

Update STATUS.md row A05: status, PR link, measured results (with the command), follow-ups. If blocked, set status to Blocked and write exactly what a human must provide.
