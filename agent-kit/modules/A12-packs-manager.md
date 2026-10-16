# A12 · Pack format, builder and model manager

**Wave:** 3 (Engine and v0.1 release)  
**Depends on:** A07, A10  
**Needs from outside the agent:** A Hugging Face org to publish packs (human, D3)  
**Size:** M (about 1–3M tokens)

## Goal

Models ship as versioned packs; the engine installs and loads only what a user selects.

## Read first

- CLAUDE.md, DECISIONS.md, STATUS.md
- docs/architecture.md (the sections this module touches)
- The STATUS.md entry for A07
- The STATUS.md entry for A10

## Build

- [ ] Pack manifest schema from architecture.md, validated with JSON Schema
- [ ] `scripts/build_pack.py`: models + metadata → pack folder + manifest (sha256, bytes, onnx_ops, license, attribution, measured metrics)
- [ ] Pack catalogue: base pack per tier (VAD, LID, encoder), one decode-head pack per language, English pack, MT pack
- [ ] Model manager: download with resume, verify, cache, upgrade, remove; memory-map weights; load heads only for selected languages
- [ ] Low-end policy: cap code-switching at 2 languages; fall back to one under memory pressure
- [ ] Engine reads models only through the manager

## Acceptance (all must pass before marking done)

- [ ] Install, upgrade, remove tested against a local pack server
- [ ] Corrupted file detected by checksum
- [ ] Engine works offline after install

## Out of scope

- Hosting decisions beyond D3

## When you finish

Update STATUS.md row A12: status, PR link, measured results (with the command), follow-ups. If blocked, set status to Blocked and write exactly what a human must provide.
