"""IndicConformer 600M multilingual ASR (ONNX): shared encoder + per-language CTC / RNN-T heads.

Placeholder until the gated model is downloadable: the runner is written against the repo's ONNX
layout (assets/encoder.onnx, ctc_decoder.onnx, rnnt_decoder.onnx, joint_*.onnx, language_masks.json).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class IndicConformerASR:
    def __init__(self, model_dir: str | Path, threads: int = 4):
        self.dir = Path(model_dir)
        if not (self.dir / "assets" / "encoder.onnx").exists():
            raise FileNotFoundError(
                "IndicConformer not downloaded: accept access at "
                "https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual then run "
                "`python scripts/fetch_models.py --only indic-conformer`"
            )
        raise NotImplementedError("IndicConformer runner: pending model inspection")

    def encode(self, audio: np.ndarray):
        raise NotImplementedError

    def decode(self, enc, lang: str) -> str:
        raise NotImplementedError

    def ctc_language_scores(self, enc, langs: list[str]) -> dict[str, float]:
        raise NotImplementedError
