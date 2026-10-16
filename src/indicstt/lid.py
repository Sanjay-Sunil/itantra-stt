"""Language identification: NeMo AmberNet (TitaNet-family, 29M params) exported to ONNX.

AmberNet is trained on VoxLingua107, which has no Odia. Its output is restricted to our languages
it knows; Odia (and same-script ambiguity such as hi/mr) is resolved by ASR CTC scoring in the engine.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .features import log_mel
from .ortutil import plain_options


class AmberNetLID:
    def __init__(self, model_dir: str | Path, threads: int = 2, onnx_name: str = "ambernet.onnx"):
        import onnxruntime as ort

        d = Path(model_dir)
        so = plain_options(threads, prepack=True, arena=False)  # arena off: −29 MB, same speed
        self.sess = ort.InferenceSession(str(d / onnx_name), so, providers=["CPUExecutionProvider"])
        self.labels: list[str] = json.loads((d / "labels.json").read_text())

    def scores(self, audio: np.ndarray) -> tuple[dict[str, float], float]:
        """Softmax over the languages this model knows, restricted to our set. Returns (scores, ms)."""
        t = time.perf_counter()
        feats = log_mel(audio)
        logits = self.sess.run(None, {"features": feats[None],
                                      "length": np.array([feats.shape[1]], dtype=np.int64)})[0][0]
        p = np.exp(logits - logits.max())
        p /= p.sum()
        return dict(zip(self.labels, p.tolist(), strict=True)), (time.perf_counter() - t) * 1000

    def identify(self, audio: np.ndarray, allowed: list[str] | None = None) -> tuple[str, float]:
        s, _ = self.scores(audio)
        if allowed:
            known = {k: v for k, v in s.items() if k in allowed}
            if known:
                total = sum(known.values())
                s = {k: v / total for k, v in known.items()}
        lang = max(s, key=s.get)
        return lang, s[lang]
