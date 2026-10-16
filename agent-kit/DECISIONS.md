# Decisions

Fill in the **Choice** column before the first agent session. Defaults are used where you leave a default in place. Agents treat this file as settled.

| # | Decision | Default | Choice | Blocks |
| --- | --- | --- | --- | --- |
| D1 | Where agents run | Your own machine with Claude Code, or a cloud environment that can reach huggingface.co, pypi.org and github.com | | Everything |
| D2 | Hugging Face access | You accept the IndicConformer access conditions on Hugging Face and set `HF_TOKEN` in the agent's environment | | A02 onward |
| D3 | Project name, GitHub repo, Hugging Face org for packs | Package name `indicstt` until named | | A01, A12, A14 |
| D4 | Scope of this run | Waves 1–3 (v0.1) now; waves 4–5 later | | Waves 4–5 |
| D5 | Engine core | Own Python reference now; evaluate sherpa-onnx at A21 | | A21 |
| D6 | Platforms | Android + Flutter; iOS only if a Mac with Xcode is available | | A23–A25 |
| D7 | First languages to validate, and allowed datasets | Hindi, Tamil, Bengali; datasets marked redistributable or evaluation-allowed in LICENSES.md | | A03 |
| D8 | Is English a translation target? | No: Indic-to-Indic only (the indic-indic IndicTrans2 model does not cover English) | | A13 |
| D9 | Review cadence | A human reviews and merges each module's PR before dependent modules start | | All |
| D10 | ASR model | IndicConformer 600M for v0.1; compare AI4Bharat's newer Indic-Transcribe after v0.1 | | A04 |
| D11 | Code license | Apache-2.0 | | A01 |
| D12 | Gate threshold | No smoke language may lose more than 2 points of absolute WER against the FP32 baseline | | A03, A07, A15–A19 |
| D13 | GPU machine and budget for training | None yet: wave 4 agents write and test training code on tiny data only | | A15–A19 |
