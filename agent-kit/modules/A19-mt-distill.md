# A19 · Fresh MT distillation (10 languages)

**Wave:** 4 (Compression training (GPU))  
**Depends on:** A13  
**Needs from outside the agent:** GPU machine and parallel text data (human)  
**Size:** L (about 3–6M tokens)

## Goal

A smaller translation model distilled from the 1B teacher, not from the 320M model.

## Read first

- CLAUDE.md, DECISIONS.md, STATUS.md
- docs/architecture.md (the sections this module touches)
- The STATUS.md entry for A13

## Build

- [ ] Teacher: IndicTrans2 1B; student sized for 230–340 MB at INT8
- [ ] Re-test INT8 vs INT4 on the new checkpoint; check every language, Odia and Malayalam first

## Acceptance (all must pass before marking done)

- [ ] SacreBLEU measured with the published method; replaces the placeholder 87.13 in docs

## When you finish

Update STATUS.md row A19: status, PR link, measured results (with the command), follow-ups. If blocked, set status to Blocked and write exactly what a human must provide.
