# A15 · Encoder distillation 600M → 300M

**Wave:** 4 (Compression training (GPU))  
**Depends on:** A03, A04  
**Needs from outside the agent:** GPU machine and training data (human, D4)  
**Size:** L (about 3–6M tokens)

## Goal

A smaller student encoder trained only on the 10 target languages.

## Read first

- CLAUDE.md, DECISIONS.md, STATUS.md
- docs/architecture.md (the sections this module touches)
- The STATUS.md entry for A03
- The STATUS.md entry for A04

## Build

- [ ] Student config (fewer layers or narrower hidden size); document the choice
- [ ] Training code: soft targets from the teacher + intermediate-layer matching; resumable; logs to files
- [ ] Data pipeline restricted to the 10 languages, from datasets allowed in LICENSES.md
- [ ] Export the student through A04/A07 and evaluate with A03
- [ ] The agent writes and tests the code on a tiny dataset on CPU; a human launches the full run on the GPU machine

## Acceptance (all must pass before marking done)

- [ ] Tiny-data run completes on CPU
- [ ] Full-run config reviewed by a human before launch
- [ ] Student result recorded against the gate

## When you finish

Update STATUS.md row A15: status, PR link, measured results (with the command), follow-ups. If blocked, set status to Blocked and write exactly what a human must provide.
