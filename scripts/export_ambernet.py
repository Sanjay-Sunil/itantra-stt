"""Export NeMo AmberNet (langid_ambernet) to ONNX: raw 16 kHz audio → logits over our languages.

Needs the NeMo env: .venv-nemo/bin/python scripts/export_ambernet.py
Writes models/lid-ambernet/ambernet.onnx and labels.json (only target languages present in the model).
"""

import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "models" / "lid-ambernet"
TARGET = ["hi", "gu", "mr", "kn", "ml", "ta", "te", "or", "bn", "en"]


def main() -> None:
    import nemo.collections.asr as nemo_asr

    m = nemo_asr.models.EncDecSpeakerLabelModel.restore_from(str(OUT / "ambernet.nemo"), map_location="cpu")
    m.eval()
    m.preprocessor.featurizer.dither = 0.0
    m.preprocessor.featurizer.pad_to = 0
    labels = list(m.cfg.train_ds.labels)
    keep = [labels.index(c) for c in TARGET if c in labels]
    kept_codes = [labels[i] for i in keep]

    class Wrapper(torch.nn.Module):
        """features (1, 80, F) + length → logits over our languages."""

        def __init__(self):
            super().__init__()
            self.m = m
            self.register_buffer("idx", torch.tensor(keep, dtype=torch.long))

        def forward(self, features, length):
            enc, elen = self.m.encoder(audio_signal=features, length=length)
            logits, _ = self.m.decoder(encoder_output=enc, length=elen)
            return torch.index_select(logits, 1, self.idx)

    w = Wrapper().eval()
    f = torch.randn(1, 80, 300)
    path = OUT / "ambernet.onnx"
    torch.onnx.export(w, (f, torch.tensor([300])), str(path), input_names=["features", "length"],
                      output_names=["logits"], dynamic_axes={"features": {2: "F"}},
                      opset_version=17, dynamo=False)
    (OUT / "labels.json").write_text(json.dumps(kept_codes))

    import sys

    import onnxruntime as ort

    sys.path.insert(0, str(ROOT / "src"))
    from indicstt.features import log_mel

    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    for secs in (1.3, 3, 7):
        n = int(16000 * secs)
        audio = (rng.standard_normal(n) * 0.05 + 0.3 * np.sin(np.arange(n) * 0.05)).astype(np.float32)
        x = torch.from_numpy(audio)[None]
        with torch.no_grad():
            ref_feats, flen = m.preprocessor(input_signal=x, length=torch.tensor([n]))
            ref_feats = ref_feats[:, :, : int(flen[0])]
            ref = w(ref_feats, flen).numpy()
        ours = log_mel(audio)
        got = s.run(None, {"features": ours[None], "length": np.array([ours.shape[1]], dtype=np.int64)})[0]
        print(f"{secs}s: feature MAE {np.abs(ours - ref_feats[0].numpy()).mean():.2e} "
              f"shape {ours.shape} vs {tuple(ref_feats.shape[1:])}; logits max diff {np.abs(ref - got).max():.2e}")
    print("labels:", kept_codes, "size MB:", round(path.stat().st_size / 1e6, 1))


if __name__ == "__main__":
    main()
