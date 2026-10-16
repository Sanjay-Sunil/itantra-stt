# A26 · Device benchmark harness

**Wave:** 5 (Portable core and platforms)  
**Depends on:** A23  
**Needs from outside the agent:** Physical phones: one CPU-only low-end, one high-end with NPU (human)  
**Size:** M (about 1–3M tokens)

## Goal

Replace every projected number with a measured one.

## Read first

- CLAUDE.md, DECISIONS.md, STATUS.md
- docs/architecture.md (the sections this module touches)
- The STATUS.md entry for A23

## Build

- [ ] On-device runner for RTF, time to text, round trip, peak/idle RAM, idle CPU, battery and heat over 30 minutes
- [ ] Results written to docs/results/devices/

## Acceptance (all must pass before marking done)

- [ ] Both tiers measured; architecture.md projections replaced

## When you finish

Update STATUS.md row A26: status, PR link, measured results (with the command), follow-ups. If blocked, set status to Blocked and write exactly what a human must provide.
