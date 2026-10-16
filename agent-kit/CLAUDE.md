# Agent instructions: indic-stt

You are building an open-source, offline speech-to-text engine for 10 Indian languages (Hindi, Gujarati, Marathi, Kannada, Malayalam, Tamil, Telugu, Odia, Bengali, English). It is a library plus downloadable model packs, not an app.

## Read before any work

1. `PLAN.md`: modules, order, dependencies
2. `DECISIONS.md`: choices already made. Do not reopen them; if one blocks you, say so in STATUS.md
3. `docs/architecture.md`: the pipeline spec and the interface contracts
4. `modules/<your module>.md`: your brief
5. `STATUS.md`: what is done, what is blocked

## Working rules

- One module per session. Do what its **Build** list says; put anything else under Follow-ups in STATUS.md.
- Work on branch `module/<id>-<slug>`. Open a PR when every acceptance check passes. Never push to `main`.
- Never invent numbers. WER, sizes, latency and memory go into `docs/results/` only when measured, together with the command that produced them. Figures marked "projection" in architecture.md stay labelled as projections until measured.
- The interfaces in architecture.md (segment record, pack manifest, model I/O contracts) are contracts. To change one, stop and write a proposal under "Interface change proposals" in STATUS.md.
- Download models only through `scripts/fetch_models.py`, which reads `HF_TOKEN` from the environment. Never commit weights, datasets, tokens or audio you were not given.
- If a step needs a GPU, a phone, a Mac, a microphone, a credential or a dataset you do not have: stop that step, set the module to Blocked in STATUS.md and write exactly what a human must provide. Continue with the parts that do not need it.
- Keep your context small: write long logs to `logs/` and read only the tail; run tests with `-q`; do not print model weights or large arrays.
- Before ending a session: update your STATUS.md row (status, PR, measured results, follow-ups) and commit.

## Code conventions

- Python 3.10+, package `indicstt` in `src/`, tests in `tests/` with pytest; ruff for lint; type hints.
- Unit tests must run on CPU without downloads: use tiny fake ONNX models or stubs. Tests that need real models carry `@pytest.mark.models` and run only when `INDICSTT_MODELS` is set.
- No global state. Models are loaded only through the model manager (A12) once it exists; before that, through a single loader function.
- Every stage records its own timing so the evaluation harness can report per-stage latency.
