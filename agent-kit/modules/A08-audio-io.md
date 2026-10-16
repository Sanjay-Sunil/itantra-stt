# A08 · Audio input and microphone

**Wave:** 3 (Engine and v0.1 release)  
**Depends on:** A01  
**Needs from outside the agent:** A microphone for the manual check (human)  
**Size:** S (about 0.3–1M tokens)

## Goal

Audio from files, buffers and the microphone, delivered as 16 kHz mono chunks.

## Read first

- CLAUDE.md, DECISIONS.md, STATUS.md
- docs/architecture.md (the sections this module touches)
- The STATUS.md entry for A01

## Build

- [ ] File decoding (wav, flac, mp3, ogg/opus) and resampling to 16 kHz mono
- [ ] `AudioSource` interface with `FileSource`, `BufferSource`, `MicSource` (sounddevice, optional extra)
- [ ] Ring buffer feeding the VAD thread

## Acceptance (all must pass before marking done)

- [ ] Unit tests for resampling and chunking
- [ ] Manual: `python -m indicstt.audio.mic_check` records 5 s and reports level

## When you finish

Update STATUS.md row A08: status, PR link, measured results (with the command), follow-ups. If blocked, set status to Blocked and write exactly what a human must provide.
