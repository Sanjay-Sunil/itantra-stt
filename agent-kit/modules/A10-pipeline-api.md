# A10 · Pipeline assembly and public API

**Wave:** 3 (Engine and v0.1 release)  
**Depends on:** A04, A05, A06, A08, A09  
**Needs from outside the agent:** Models downloaded  
**Size:** M (about 1–3M tokens)

## Goal

Wire the real stages into the scheduler and expose the API other people will use.

## Read first

- CLAUDE.md, DECISIONS.md, STATUS.md
- docs/architecture.md (the sections this module touches)
- The STATUS.md entry for A04
- The STATUS.md entry for A05
- The STATUS.md entry for A06
- The STATUS.md entry for A08
- The STATUS.md entry for A09

## Build

- [ ] `Engine(EngineConfig)` with languages, tier, target_language, workers, segment limits, mode
- [ ] `engine.transcribe_file(path) -> list[Segment]`; `engine.stream(source)` yields Segment events
- [ ] Segment record exactly as in architecture.md
- [ ] Policy when LID picks a language with no installed head: best installed language + `unexpected_language` flag
- [ ] Manual `--lang` mode that skips LID
- [ ] Per-stage timing written into each segment for the evaluation harness

## Acceptance (all must pass before marking done)

- [ ] End-to-end test on the mixed-language recording: in order, each segment tagged
- [ ] Pipeline WER equals A04 runner WER on pre-segmented smoke clips within 0.5 points
- [ ] API documented in docs/api.md

## Out of scope

- Translation (A13)

## When you finish

Update STATUS.md row A10: status, PR link, measured results (with the command), follow-ups. If blocked, set status to Blocked and write exactly what a human must provide.
