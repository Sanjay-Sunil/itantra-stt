# A06 · VAD and segmenter

**Wave:** 2 (Models into ONNX)  
**Depends on:** A01  
**Needs from outside the agent:** Silero VAD model (A02) for the integration test only  
**Size:** M (about 1–3M tokens)

## Goal

Turn a continuous 16 kHz stream into numbered speech segments.

## Read first

- CLAUDE.md, DECISIONS.md, STATUS.md
- docs/architecture.md (the sections this module touches)
- The STATUS.md entry for A01

## Build

- [ ] Silero VAD on 512-sample windows (32 ms at 16 kHz), keeping its recurrent state across windows
- [ ] Debounced state machine: start threshold, end threshold, minimum speech, minimum silence, edge padding — all in config
- [ ] Forced cut at max segment length (default 30 s)
- [ ] Each segment: monotonically increasing `seq`, `start_ms`, `end_ms`, PCM
- [ ] Works on files and on pushed chunks of any size

## Acceptance (all must pass before marking done)

- [ ] State-machine unit tests with synthetic score sequences (no model)
- [ ] Chunk-size invariance: same segments for 10 ms and 1 s chunks
- [ ] Integration test (marked `models`) on a real recording

## Out of scope

- Speaker separation

## When you finish

Update STATUS.md row A06: status, PR link, measured results (with the command), follow-ups. If blocked, set status to Blocked and write exactly what a human must provide.
