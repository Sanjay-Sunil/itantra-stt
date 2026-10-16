# A03 · Evaluation harness

**Wave:** 1 (Foundation)  
**Depends on:** A01  
**Needs from outside the agent:** Dataset access for at least 3 languages (D7); downloads run where the network allows  
**Size:** M (about 1–3M tokens)

## Goal

One command that scores a checkpoint, an ONNX model, a pack or the full pipeline the same way.

## Read first

- CLAUDE.md, DECISIONS.md, STATUS.md
- docs/architecture.md (the sections this module touches)
- The STATUS.md entry for A01

## Build

- [ ] `indicstt.eval`: WER and CER with Indic-aware normalization (Unicode NFC, punctuation strip, configurable numerals)
- [ ] LID metrics: accuracy and confusion matrix; code-switched WER on segment-level references
- [ ] Speed and memory: real-time factor, per-stage latency from pipeline timestamps, peak RSS
- [ ] Dataset adapters producing a common manifest (audio path, text, language); start with a 20-clip smoke set per language for Hindi, Tamil and one more (D7), plus a short mixed-language recording
- [ ] `scripts/evaluate.py --target <onnx|pack|pipeline> --set smoke --out docs/results/<name>.json`
- [ ] Gate checker: compares a result file against a baseline and fails if any language loses more than the threshold in DECISIONS.md

## Acceptance (all must pass before marking done)

- [ ] Unit tests for WER/CER on hand-made examples, including Devanagari and Tamil text
- [ ] Smoke set manifest builds
- [ ] Gate checker fails on a synthetic regression and passes on equal results

## Out of scope

- Downloading full training corpora

## When you finish

Update STATUS.md row A03: status, PR link, measured results (with the command), follow-ups. If blocked, set status to Blocked and write exactly what a human must provide.
