# A21 · C++ core and C API

**Wave:** 5 (Portable core and platforms)  
**Depends on:** A10, A12  
**Needs from outside the agent:** Decision D5 (own core or sherpa-onnx)  
**Size:** L (about 3–6M tokens)

## Goal

One portable implementation of the engine for mobile and desktop.

## Read first

- CLAUDE.md, DECISIONS.md, STATUS.md
- docs/architecture.md (the sections this module touches)
- The STATUS.md entry for A10
- The STATUS.md entry for A12

## Build

- [ ] Port audio, VAD, LID, ASR, scheduler, reorder buffer and model manager to C++17 on onnxruntime
- [ ] Stable C API with opaque handles mirroring the Python API
- [ ] Golden tests: same transcripts as the Python reference on the smoke set
- [ ] If D5 = sherpa-onnx: build the Indic layer (LID routing, merged head, reorder, packs) on top of it instead

## Acceptance (all must pass before marking done)

- [ ] Golden tests pass on Linux and macOS
- [ ] No leaks under sanitizers in a 30-minute stream test

## When you finish

Update STATUS.md row A21: status, PR link, measured results (with the command), follow-ups. If blocked, set status to Blocked and write exactly what a human must provide.
