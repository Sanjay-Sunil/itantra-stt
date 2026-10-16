# A09 · Scheduler, worker pools and reorder buffer

**Wave:** 3 (Engine and v0.1 release)  
**Depends on:** A01  
**Needs from outside the agent:** Nothing (built against fake stages)  
**Size:** M (about 1–3M tokens)

## Goal

The concurrency layer: queues A–G from architecture.md, testable without any model.

## Read first

- CLAUDE.md, DECISIONS.md, STATUS.md
- docs/architecture.md (the sections this module touches)
- The STATUS.md entry for A01

## Build

- [ ] VAD thread → segment queue (A)
- [ ] Per segment: LID and encoder start together (B); decode starts when both finish (C)
- [ ] ASR worker pool, size 2–3 from config (D)
- [ ] MT queue only for segments whose language differs from the target (E); MT worker pool (F)
- [ ] Reorder buffer keyed by `seq`, releases N after N−1 (G); timeout for a stuck segment emits it with flag `timeout`
- [ ] Bounded queues; `falling_behind` event instead of silent drops; clean cancel and shutdown

## Acceptance (all must pass before marking done)

- [ ] Stress test with fake stages of random duration: output always in `seq` order
- [ ] No deadlock in 1,000 randomized runs
- [ ] Timeout path tested

## Out of scope

- Real models

## When you finish

Update STATUS.md row A09: status, PR link, measured results (with the command), follow-ups. If blocked, set status to Blocked and write exactly what a human must provide.
