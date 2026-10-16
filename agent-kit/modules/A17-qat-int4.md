# A17 · Quantization-aware training for INT4 layers

**Wave:** 4 (Compression training (GPU))  
**Depends on:** A07, A15  
**Needs from outside the agent:** GPU machine (human)  
**Size:** M (about 1–3M tokens)

## Goal

Recover WER lost when FFN layers go to INT4.

## Read first

- CLAUDE.md, DECISIONS.md, STATUS.md
- docs/architecture.md (the sections this module touches)
- The STATUS.md entry for A07
- The STATUS.md entry for A15

## Build

- [ ] Fake-quantization on the INT4 layers only; short fine-tune; export; re-evaluate

## Acceptance (all must pass before marking done)

- [ ] INT4-mixed tier passes the gate, or the gap is recorded

## When you finish

Update STATUS.md row A17: status, PR link, measured results (with the command), follow-ups. If blocked, set status to Blocked and write exactly what a human must provide.
