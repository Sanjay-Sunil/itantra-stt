# A16 · Merged decode head

**Wave:** 4 (Compression training (GPU))  
**Depends on:** A15  
**Needs from outside the agent:** GPU machine (human)  
**Size:** L (about 3–6M tokens)

## Goal

One decode head steered by a language-ID token, sliceable to the user's languages.

## Read first

- CLAUDE.md, DECISIONS.md, STATUS.md
- docs/architecture.md (the sections this module touches)
- The STATUS.md entry for A15

## Build

- [ ] Shared prediction + joint networks with a language token prepended; output projection over the union vocabulary of 9 languages
- [ ] Install-time slicing of the output projection to the selected languages (avoids 120 combination packs)
- [ ] Per-language A/B against separate heads, same-script pairs first (Hindi/Marathi)

## Acceptance (all must pass before marking done)

- [ ] Every language within the gate against separate heads, or the module records which languages keep separate heads

## When you finish

Update STATUS.md row A16: status, PR link, measured results (with the command), follow-ups. If blocked, set status to Blocked and write exactly what a human must provide.
