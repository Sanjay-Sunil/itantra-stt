# A07 · Quantization: INT8 and INT4 mixed

**Wave:** 2 (Models into ONNX)  
**Depends on:** A04, A05  
**Needs from outside the agent:** CPU only; more RAM speeds it up  
**Size:** M (about 1–3M tokens)

## Goal

Produce the two precision tiers and measure what each costs in WER.

## Read first

- CLAUDE.md, DECISIONS.md, STATUS.md
- docs/architecture.md (the sections this module touches)
- The STATUS.md entry for A04
- The STATUS.md entry for A05

## Build

- [ ] INT8 dynamic quantization for encoder, heads and LID
- [ ] Sensitivity scan: quantize one layer group at a time to INT4, record WER delta
- [ ] INT4 mixed per architecture.md: embeddings, first 1–2 encoder layers and attention Q/K/V stay INT8; mid/late FFN MatMuls go block-wise INT4 (MatMulNBits)
- [ ] `scripts/quantize.py --tier int8|int4mixed`
- [ ] Results table: size and smoke WER for FP32, INT8, INT4 mixed

## Acceptance (all must pass before marking done)

- [ ] Gate checker passes for INT8 against FP32
- [ ] INT4-mixed result recorded, pass or fail
- [ ] Sizes recorded from files on disk, not estimates

## Out of scope

- Quantization-aware training (A17)

## When you finish

Update STATUS.md row A07: status, PR link, measured results (with the command), follow-ups. If blocked, set status to Blocked and write exactly what a human must provide.
