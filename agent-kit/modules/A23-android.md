# A23 · Android library and minimal runtime

**Wave:** 5 (Portable core and platforms)  
**Depends on:** A21  
**Needs from outside the agent:** Android SDK/NDK; a phone for manual checks (human)  
**Size:** L (about 3–6M tokens)

## Goal

An AAR with a Kotlin API, using a minimal-operator onnxruntime build.

## Read first

- CLAUDE.md, DECISIONS.md, STATUS.md
- docs/architecture.md (the sections this module touches)
- The STATUS.md entry for A21

## Build

- [ ] Minimal-ops ORT build from the onnx_ops lists in the pack manifests; confirm INT4 kernels present
- [ ] Kotlin wrapper, microphone capture, pack download
- [ ] Do not rely on NNAPI (deprecated as of Android 15); CPU/XNNPACK baseline, vendor execution providers optional

## Acceptance (all must pass before marking done)

- [ ] Instrumented test transcribes a bundled clip
- [ ] Binary size recorded

## When you finish

Update STATUS.md row A23: status, PR link, measured results (with the command), follow-ups. If blocked, set status to Blocked and write exactly what a human must provide.
