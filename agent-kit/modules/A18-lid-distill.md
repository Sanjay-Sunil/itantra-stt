# A18 · LID backbone distillation

**Wave:** 4 (Compression training (GPU))  
**Depends on:** A05  
**Needs from outside the agent:** GPU machine (human)  
**Size:** M (about 1–3M tokens)

## Goal

A smaller LID backbone trained on the 10 languages.

## Read first

- CLAUDE.md, DECISIONS.md, STATUS.md
- docs/architecture.md (the sections this module touches)
- The STATUS.md entry for A05

## Build

- [ ] Distil the backbone on 10-language data; test on rapid code-switching audio
- [ ] Keep the A05 model as the fallback pack

## Acceptance (all must pass before marking done)

- [ ] Error under 1% at INT8 and 1.5% at INT4 mixed, or the module records that the fallback stays

## When you finish

Update STATUS.md row A18: status, PR link, measured results (with the command), follow-ups. If blocked, set status to Blocked and write exactly what a human must provide.
