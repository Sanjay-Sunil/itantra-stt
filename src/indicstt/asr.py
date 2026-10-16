"""IndicConformer 600M multilingual ASR (ONNX): shared encoder + per-language CTC / RNN-T heads.

Layout (ai4bharat/indic-conformer-600m-multilingual, assets/):
  encoder.onnx        log-mel (1, 80, F) → (1, 1024, T)          shared across languages
  ctc_decoder.onnx    (1, 1024, T) → (1, T, 5633) union vocab; language_masks.json picks 257 per language
  rnnt_decoder.onnx   prediction network (2-layer LSTM, 640)
  joint_enc / joint_pred / joint_pre_net    shared joint network
  joint_post_net_<lang>.onnx                per-language output projection (257 = 256 tokens + blank)

A quantized tier lives in a sibling directory with the same file names (see scripts/quantize.py);
`variant` selects it, e.g. models/indic-conformer-int8/assets/encoder.onnx.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .features import INDIC_CONFORMER_MEL, log_mel

BLANK = 256  # per-language blank (257-way heads)
# Start token = union-vocab blank (5633 - 1). Upstream's config.json says 256, but its from_pretrained()
# never reads config.json, so the effective upstream value is the IndicASRConfig default 5632.
SOS = 5632
MAX_SYMBOLS = 10
SUPPORTED = ["hi", "gu", "mr", "kn", "ml", "ta", "te", "or", "bn"]


def _log_softmax(x: np.ndarray) -> np.ndarray:
    m = x.max(axis=-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=-1, keepdims=True))


@dataclass
class Encoded:
    out: np.ndarray  # (1, 1024, T)
    length: int
    _ctc: np.ndarray | None = None  # (T, 5633) raw CTC logprobs, computed once and shared by all languages


class IndicConformerASR:
    def __init__(self, model_dir: str | Path, threads: int = 4, mode: str = "rnnt",
                 languages: list[str] | None = None):
        import onnxruntime as ort

        self.dir = Path(model_dir)
        a = self.dir / "assets"
        if not (a / "encoder.onnx").exists():
            raise FileNotFoundError(
                f"IndicConformer not found in {self.dir}: accept access at "
                "https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual then run "
                "`python scripts/fetch_models.py --only indic-conformer`"
            )
        self.mode = mode
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        small = ort.SessionOptions()
        small.intra_op_num_threads = 1  # tiny per-token graphs: threading overhead dominates
        prov = ["CPUExecutionProvider"]

        def sess(name, opts=so):
            return ort.InferenceSession(str(a / f"{name}.onnx"), opts, providers=prov)

        self.encoder = sess("encoder")
        self.ctc = sess("ctc_decoder")
        self.pred = sess("rnnt_decoder", small)
        self.joint_enc = sess("joint_enc")
        self.joint_pred = sess("joint_pred", small)
        self.joint_pre = sess("joint_pre_net", small)
        self.vocab: dict[str, list[str]] = json.loads((a / "vocab.json").read_text())
        masks = json.loads((a / "language_masks.json").read_text())
        self.mask_idx = {k: np.flatnonzero(np.asarray(v, dtype=bool)) for k, v in masks.items()}
        self._post: dict[str, object] = {}
        self._post_lock = threading.Lock()
        self._sess, self._small = sess, small
        for lang in languages or []:  # on-demand heads: load only the user's languages
            if lang in SUPPORTED:
                self._post_net(lang)
        out_names = [o.name for o in self.pred.get_outputs()]
        self._pred_out = out_names

    def _post_net(self, lang: str):
        with self._post_lock:
            if lang not in self._post:
                self._post[lang] = self._sess(f"joint_post_net_{lang}", self._small)
            return self._post[lang]

    # ---------------------------------------------------------------- encoder (shared)
    def encode(self, audio: np.ndarray) -> Encoded:
        feats = log_mel(audio, INDIC_CONFORMER_MEL)
        out, length = self.encoder.run(
            ["outputs", "encoded_lengths"],
            {"audio_signal": feats[None], "length": np.array([feats.shape[1]], dtype=np.int64)},
        )
        return Encoded(out=out, length=int(length[0]))

    def _ctc_logprobs(self, enc: Encoded) -> np.ndarray:
        if enc._ctc is None:
            enc._ctc = self.ctc.run(["logprobs"], {"encoder_output": enc.out})[0][0, : enc.length]
        return enc._ctc

    # ---------------------------------------------------------------- language scoring
    def ctc_language_scores(self, enc: Encoded, langs: list[str]) -> dict[str, float]:
        """Mean per-frame greedy log-probability under each language's vocabulary mask (higher = better fit).

        Used as an acoustic-LID tie-breaker and for Odia, which AmberNet does not cover.
        """
        lp = self._ctc_logprobs(enc)
        out = {}
        for lang in langs:
            if lang in self.mask_idx:
                sub = _log_softmax(lp[:, self.mask_idx[lang]])
                best = sub.max(axis=-1)
                nonblank = sub.argmax(axis=-1) != BLANK
                # score only frames that emit tokens; blank frames say little about the language
                out[lang] = float(best[nonblank].mean()) if nonblank.any() else float(best.mean())
        return out

    # ---------------------------------------------------------------- decoding
    def decode(self, enc: Encoded, lang: str, mode: str | None = None) -> str:
        mode = mode or self.mode
        return self._rnnt(enc, lang) if mode == "rnnt" else self._ctc_decode(enc, lang)

    def _detok(self, ids, lang: str) -> str:
        v = self.vocab[lang]
        return "".join(v[i] for i in ids).replace("▁", " ").strip()

    def _ctc_decode(self, enc: Encoded, lang: str) -> str:
        sub = self._ctc_logprobs(enc)[:, self.mask_idx[lang]]
        path = sub.argmax(axis=-1)
        keep = np.ones(len(path), dtype=bool)
        keep[1:] = path[1:] != path[:-1]
        return self._detok([int(i) for i in path[keep] if i != BLANK], lang)

    def _rnnt(self, enc: Encoded, lang: str) -> str:
        post = self._post_net(lang)
        f_all = self.joint_enc.run(["output"], {"input": enc.out[:, :, : enc.length].transpose(0, 2, 1)})[0]
        h = np.zeros((2, 1, 640), dtype=np.float32)
        c = np.zeros((2, 1, 640), dtype=np.float32)
        hyp = [SOS]  # ids fed to the prediction network
        out: list[int] = []  # per-language ids for detokenisation
        g_cache = None  # prediction-net output for (hyp[-1], state); reused across blank frames

        def predict(token, h, c):
            g, _, h2, c2 = self.pred.run(self._pred_out, {
                "targets": np.array([[token]], dtype=np.int32), "target_length": np.array([1], dtype=np.int32),
                "states.1": h, "onnx::Slice_3": c})
            return self.joint_pred.run(["output"], {"input": g.transpose(0, 2, 1)})[0], h2, c2

        for t in range(f_all.shape[1]):
            f = f_all[:, t : t + 1, :]
            for _ in range(MAX_SYMBOLS):
                if g_cache is None:
                    g_cache = predict(hyp[-1], h, c)
                g, h2, c2 = g_cache
                z = self.joint_pre.run(["output"], {"input": f + g})[0]
                tok = int(post.run(["output"], {"input": z})[0].reshape(-1).argmax())
                if tok == BLANK:
                    break
                hyp.append(tok)  # per-language id, as upstream (union ids measured: 568% WER on hi)
                out.append(tok)
                h, c = h2, c2
                g_cache = None
        return self._detok(out, lang)
