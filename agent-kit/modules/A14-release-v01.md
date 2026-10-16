# A14 · Release v0.1

**Wave:** 3 (Engine and v0.1 release)  
**Depends on:** A11, A12  
**Needs from outside the agent:** Push access and a release decision (human)  
**Size:** S (about 0.3–1M tokens)

## Goal

A public, installable v0.1 with honest numbers.

## Read first

- CLAUDE.md, DECISIONS.md, STATUS.md
- docs/architecture.md (the sections this module touches)
- The STATUS.md entry for A11
- The STATUS.md entry for A12

## Build

- [ ] README: what it is, install, quick start, supported languages, limits
- [ ] docs/results.md: measured WER, RTF and peak RAM on the release machine, with commands
- [ ] Model cards for each pack; LICENSES.md complete
- [ ] Changelog, version tag, wheels built in CI

## Acceptance (all must pass before marking done)

- [ ] Fresh-environment install and quick start pass
- [ ] Every number in README links to a results file

## When you finish

Update STATUS.md row A14: status, PR link, measured results (with the command), follow-ups. If blocked, set status to Blocked and write exactly what a human must provide.
