# A22 · Python bindings on the C++ core

**Wave:** 5 (Portable core and platforms)  
**Depends on:** A21  
**Needs from outside the agent:** Nothing  
**Size:** M (about 1–3M tokens)

## Goal

Same Python API, now backed by the C++ core.

## Read first

- CLAUDE.md, DECISIONS.md, STATUS.md
- docs/architecture.md (the sections this module touches)
- The STATUS.md entry for A21

## Build

- [ ] Bindings (pybind11 or nanobind); wheels for Linux, macOS, Windows

## Acceptance (all must pass before marking done)

- [ ] All A10 tests pass against the new backend

## When you finish

Update STATUS.md row A22: status, PR link, measured results (with the command), follow-ups. If blocked, set status to Blocked and write exactly what a human must provide.
