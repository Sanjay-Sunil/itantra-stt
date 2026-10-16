# A20 · Weight clustering pass (optional)

**Wave:** 4 (Compression training (GPU))  
**Depends on:** A07  
**Needs from outside the agent:** CPU  
**Size:** S (about 0.3–1M tokens)

## Goal

Try k-means codebook compression after quantization.

## Read first

- CLAUDE.md, DECISIONS.md, STATUS.md
- docs/architecture.md (the sections this module touches)
- The STATUS.md entry for A07

## Build

- [ ] Apply per model; measure size and WER; adopt only where the gate passes

## Acceptance (all must pass before marking done)

- [ ] Result recorded; default stays off unless it passes

## When you finish

Update STATUS.md row A20: status, PR link, measured results (with the command), follow-ups. If blocked, set status to Blocked and write exactly what a human must provide.
