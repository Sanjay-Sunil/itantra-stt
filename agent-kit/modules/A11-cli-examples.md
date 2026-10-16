# A11 · CLI and examples

**Wave:** 3 (Engine and v0.1 release)  
**Depends on:** A10  
**Needs from outside the agent:** Nothing  
**Size:** S (about 0.3–1M tokens)

## Goal

The first thing a new user touches.

## Read first

- CLAUDE.md, DECISIONS.md, STATUS.md
- docs/architecture.md (the sections this module touches)
- The STATUS.md entry for A10

## Build

- [ ] `indicstt transcribe FILE [--lang] [--tier] [--json]`, `indicstt listen`, `indicstt models list|fetch|remove`, `indicstt bench FILE`
- [ ] examples/: transcribe a file, live captions, JSON to subtitles (SRT)

## Acceptance (all must pass before marking done)

- [ ] CLI smoke tests with a fake engine
- [ ] README quick start works copy-paste on a clean machine

## When you finish

Update STATUS.md row A11: status, PR link, measured results (with the command), follow-ups. If blocked, set status to Blocked and write exactly what a human must provide.
