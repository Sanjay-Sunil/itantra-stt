# Execution plan: indic-stt, built by agents

27 modules in 5 waves. Waves 1–3 (14 modules) produce v0.1: a Python package that transcribes files and live audio with published models. Waves 4–5 add the trained compression and the mobile platforms, and need hardware an agent cannot provide on its own.

## Constraints

| Constraint | What it means for the plan |
| --- | --- |
| Context window per session | One module per session. Everything an agent needs is in files (this plan, the module brief, architecture.md, STATUS.md), never in chat history |
| Token usage | Rough estimate: 12–35M tokens for v0.1, 35–90M for all 27 modules. Mostly re-reading files and test output, not writing code. Keep logs in files and tests quiet |
| Network | The agent's machine must reach huggingface.co, pypi.org and github.com. Some sandboxes block them |
| Gated model | IndicConformer needs a human to accept its access conditions and provide `HF_TOKEN` |
| Compute | v0.1 runs on a laptop CPU with 16 GB RAM or more. Distillation and QAT (wave 4) need a GPU machine and a human to launch the long runs |
| Hardware an agent cannot use | Microphone, phones, a Mac for iOS: a human runs those checks |
| Data | Training needs licensed Indic speech and parallel text; evaluation needs at least a smoke set per language |
| Time | Training runs take hours to days; agents write and test the code on tiny data, humans launch the full run |
| Honesty | Projected numbers stay labelled until measured; agents never fill results by estimate |

## Waves

| Wave | Modules | Can run in parallel | Human checkpoint at the end |
| --- | --- | --- | --- |
| 1 Foundation | A01 → A02, A03 | A02 and A03 after A01 | Approve LICENSES.md and the smoke set |
| 2 Models into ONNX | A04, A05, A06, A07 | A04, A05, A06 together; A07 after A04 and A05 | Check parity and INT8 results |
| 3 Engine and v0.1 | A08, A09, A10, A11, A12, A13, A14 | A08 and A09 can start in wave 1; A11, A12, A13 after A10 | Try the CLI with a microphone; approve the release |
| 4 Compression training | A15–A20 | A18, A19, A20 independent; A16, A17 after A15 | Approve each training config before the GPU run |
| 5 Platforms | A21–A27 | A23 and A25 after A21; A24 after A23 | Run benchmarks on real phones |

Shortest route to v0.1: A01 → (A02, A03, A06, A08, A09) → (A04, A05) → A07 → A10 → (A11, A12, A13) → A14.

## Modules

Size is a rough token estimate per module: S about 0.3–1M, M about 1–3M, L about 3–6M.

| Module | Wave | Depends on | Needs from outside | Size |
| --- | --- | --- | --- | --- |
| [A01 Repository scaffold](modules/A01-repo-scaffold.md) | 1 | — | Nothing | S |
| [A02 Model fetch script and license record](modules/A02-model-fetch-licenses.md) | 1 | A01 | HF_TOKEN with the IndicConformer access conditions accepted (human, D2) | S |
| [A03 Evaluation harness](modules/A03-eval-harness.md) | 1 | A01 | Dataset access for at least 3 languages (D7); downloads run where the network allows | M |
| [A04 IndicConformer ONNX runner and parity](modules/A04-asr-onnx-runner.md) | 2 | A02, A03 | Models downloaded (A02) | L |
| [A05 LID model export and runner](modules/A05-lid-onnx.md) | 2 | A02, A03 | Models downloaded; NeMo export may need a machine with NeMo installed | M |
| [A06 VAD and segmenter](modules/A06-vad-segmenter.md) | 2 | A01 | Silero VAD model (A02) for the integration test only | M |
| [A07 Quantization: INT8 and INT4 mixed](modules/A07-quantization.md) | 2 | A04, A05 | CPU only; more RAM speeds it up | M |
| [A08 Audio input and microphone](modules/A08-audio-io.md) | 3 | A01 | A microphone for the manual check (human) | S |
| [A09 Scheduler, worker pools and reorder buffer](modules/A09-scheduler-reorder.md) | 3 | A01 | Nothing (built against fake stages) | M |
| [A10 Pipeline assembly and public API](modules/A10-pipeline-api.md) | 3 | A04, A05, A06, A08, A09 | Models downloaded | M |
| [A11 CLI and examples](modules/A11-cli-examples.md) | 3 | A10 | Nothing | S |
| [A12 Pack format, builder and model manager](modules/A12-packs-manager.md) | 3 | A07, A10 | A Hugging Face org to publish packs (human, D3) | M |
| [A13 Translation plugin and English path](modules/A13-mt-english.md) | 3 | A10 | Models downloaded (set `all`) | M |
| [A14 Release v0.1](modules/A14-release-v01.md) | 3 | A11, A12 | Push access and a release decision (human) | S |
| [A15 Encoder distillation 600M → 300M](modules/A15-distill-encoder.md) | 4 | A03, A04 | GPU machine and training data (human, D4) | L |
| [A16 Merged decode head](modules/A16-merged-head.md) | 4 | A15 | GPU machine (human) | L |
| [A17 Quantization-aware training for INT4 layers](modules/A17-qat-int4.md) | 4 | A07, A15 | GPU machine (human) | M |
| [A18 LID backbone distillation](modules/A18-lid-distill.md) | 4 | A05 | GPU machine (human) | M |
| [A19 Fresh MT distillation (10 languages)](modules/A19-mt-distill.md) | 4 | A13 | GPU machine and parallel text data (human) | L |
| [A20 Weight clustering pass (optional)](modules/A20-weight-clustering.md) | 4 | A07 | CPU | S |
| [A21 C++ core and C API](modules/A21-cpp-core.md) | 5 | A10, A12 | Decision D5 (own core or sherpa-onnx) | L |
| [A22 Python bindings on the C++ core](modules/A22-python-on-core.md) | 5 | A21 | Nothing | M |
| [A23 Android library and minimal runtime](modules/A23-android.md) | 5 | A21 | Android SDK/NDK; a phone for manual checks (human) | L |
| [A24 Flutter plugin](modules/A24-flutter.md) | 5 | A23 | Flutter SDK | M |
| [A25 iOS framework](modules/A25-ios.md) | 5 | A21 | A Mac with Xcode (human, D6) | M |
| [A26 Device benchmark harness](modules/A26-device-bench.md) | 5 | A23 | Physical phones: one CPU-only low-end, one high-end with NPU (human) | M |
| [A27 Docs site, model cards and release automation](modules/A27-docs-release.md) | 5 | A14 | Hosting for the docs site | M |

## How to run it

1. Fill in DECISIONS.md (D1, D2 and D3 block everything).
2. Run A01 in a fresh session: "Read CLAUDE.md and PLAN.md, then do modules/A01-repo-scaffold.md."
3. For each next module, start a fresh session with the same prompt and that module's brief. Modules in the same row of the waves table can run as parallel sessions in separate git worktrees.
4. After each module, review its PR and STATUS.md row, then merge.
5. When a module is Blocked, provide what its STATUS.md entry asks for and rerun it.
