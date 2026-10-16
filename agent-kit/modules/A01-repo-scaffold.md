# A01 · Repository scaffold

**Wave:** 1 (Foundation)  
**Depends on:** nothing  
**Needs from outside the agent:** Nothing  
**Size:** S (about 0.3–1M tokens)

## Goal

A clean Python package, test setup and CI that every later module builds on.

## Read first

- CLAUDE.md, DECISIONS.md, STATUS.md
- docs/architecture.md (the sections this module touches)

## Build

- [ ] `pyproject.toml` for package `indicstt` (src layout), Python 3.10+, extras: `[mic]`, `[mt]`, `[dev]`
- [ ] Directories: `src/indicstt/{audio,vad,lid,asr,mt,pipeline,packs,cli}`, `tests/`, `scripts/`, `docs/`, `logs/` (git-ignored), `models/` (git-ignored)
- [ ] pytest, ruff, a `models` pytest marker that is skipped unless `INDICSTT_MODELS` is set
- [ ] CI workflow: lint + unit tests on Linux, macOS, Windows
- [ ] LICENSE from DECISIONS.md, CONTRIBUTING.md, CODE_OF_CONDUCT.md, SECURITY.md, `.gitignore` that blocks weights, datasets and tokens
- [ ] Copy CLAUDE.md, PLAN.md, DECISIONS.md, STATUS.md and docs/architecture.md from this kit into the repo root

## Acceptance (all must pass before marking done)

- [ ] `pip install -e .[dev]` succeeds
- [ ] `pytest -q` passes (placeholder test)
- [ ] `ruff check .` is clean
- [ ] CI runs green on the PR

## Out of scope

- Any model code

## When you finish

Update STATUS.md row A01: status, PR link, measured results (with the command), follow-ups. If blocked, set status to Blocked and write exactly what a human must provide.
