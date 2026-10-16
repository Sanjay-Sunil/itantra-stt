# A04 · IndicConformer ONNX runner and parity

**Wave:** 2 (Models into ONNX)  
**Depends on:** A02, A03  
**Needs from outside the agent:** Models downloaded (A02)  
**Size:** L (about 3–6M tokens)

## Goal

Run the shared Conformer encoder and the per-language decode heads in onnxruntime, with accuracy equal to the upstream reference.

## Read first

- CLAUDE.md, DECISIONS.md, STATUS.md
- docs/architecture.md (the sections this module touches)
- The STATUS.md entry for A02
- The STATUS.md entry for A03

## Build

- [ ] Preprocessor: reproduce the upstream log-mel features exactly; feature parity test against the upstream code
- [ ] Encoder session wrapper with a clean interface: features in, per-frame states out (see Model I/O contracts)
- [ ] CTC greedy decoding per language; RNN-T greedy decoding over prediction + joint networks
- [ ] Tokenizer/detokenizer per language from the upstream vocab files
- [ ] `indicstt.asr.ASR.transcribe(audio, lang, mode='ctc'|'rnnt')`
- [ ] Parity script: our runner vs upstream on the smoke set; record WER for both

## Acceptance (all must pass before marking done)

- [ ] Feature parity within 1e-3 mean absolute error
- [ ] WER difference to upstream under 0.2 points per smoke language (record in docs/results)
- [ ] Unit tests run with a tiny fake ONNX model, no download

## Out of scope

- Quantization
- Merged head

## When you finish

Update STATUS.md row A04: status, PR link, measured results (with the command), follow-ups. If blocked, set status to Blocked and write exactly what a human must provide.
