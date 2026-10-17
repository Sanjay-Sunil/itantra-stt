"""Count multiply-accumulates (MACs) per model run from ONNX Runtime profiles, for the browser compute panel.

Runs each model with graph optimisations off and profiling on, reads the real tensor shapes of every Conv /
MatMul / Gemm / LSTM node (including the INT8 ConvInteger / MatMulInteger forms), and fits MACs as a polynomial
in the input length. Writes web/compute.json:

    {model: {"unit": what x counts, "coef": [c0, c1, c2], "params": weight count}}   MACs(x) = c0 + c1·x + c2·x²

    python scripts/count_macs.py
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "models"
OUT = ROOT / "web" / "compute.json"


def node_macs(op: str, ins: list[list[int]], outs: list[list[int]]) -> int:
    prod = lambda s: int(np.prod(s)) if s else 1  # noqa: E731
    if op in ("Conv", "ConvInteger") and len(ins) > 1:
        return prod(outs[0]) * prod(ins[1][1:])
    if op in ("MatMul", "MatMulInteger") and ins and ins[0]:
        return prod(outs[0]) * ins[0][-1]
    if op == "Gemm" and len(ins) > 1:
        n = outs[0][-1]
        return prod(outs[0]) * (prod(ins[1]) // max(1, n))
    if op == "LSTM" and len(ins) > 2:
        t, _, i = ins[0]
        dirs, four_h, h = ins[2]
        return t * dirs * four_h * (i + h)
    return 0


def profile_macs(path: Path, feeds: dict) -> int:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.enable_profiling = True
    so.profile_file_prefix = os.path.join(tempfile.gettempdir(), "itantra_macs")
    s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    s.run(None, feeds)
    prof = s.end_profiling()
    total = 0
    for e in json.load(open(prof)):
        a = e.get("args", {})
        if e.get("cat") != "Node" or not e["name"].endswith("_kernel_time"):
            continue
        shapes = lambda key: [next(iter(x.values())) for x in a.get(key, [])]  # noqa: E731
        total += node_macs(a.get("op_name", ""), shapes("input_type_shape"), shapes("output_type_shape"))
    os.remove(prof)
    return total


def params(path: Path) -> int:
    m = onnx.load(str(path), load_external_data=False)
    n = 0
    for t in m.graph.initializer:
        n += int(np.prod(t.dims)) if t.dims else 1
    return n


def fit(xs, ys, deg):
    c = np.polyfit(np.array(xs, float), np.array(ys, float), deg)[::-1]
    return [float(v) for v in c] + [0.0] * (3 - len(c))


def f32(*shape):
    return np.random.default_rng(0).standard_normal(shape).astype(np.float32)


def main():
    rng = np.random.default_rng(0)
    res = {}
    a = M / "indic-conformer-int8/assets"

    # encoder: x = mel frames (100 per second of audio)
    xs = [200, 400, 800, 1600]
    ys = [profile_macs(a / "encoder.onnx", {"audio_signal": f32(1, 80, x), "length": np.array([x])}) for x in xs]
    res["encoder"] = {"unit": "mel frame", "coef": fit(xs, ys, 2), "params": 607_000_000}

    # CTC head: x = encoder frames
    xs = [25, 50, 100]
    ys = [profile_macs(a / "ctc_decoder.onnx", {"encoder_output": f32(1, 1024, x)}) for x in xs]
    res["ctc"] = {"unit": "encoder frame", "coef": fit(xs, ys, 1), "params": params(a / "ctc_decoder.onnx")}

    # RNN-T pieces: per call
    h = np.zeros((2, 1, 640), np.float32)
    pred = profile_macs(a / "rnnt_decoder.onnx", {"targets": np.array([[5]], np.int32),
                                                    "target_length": np.array([1], np.int32), "states.1": h,
                                                    "onnx::Slice_3": h})
    jp = profile_macs(a / "joint_pred.onnx", {"input": f32(1, 1, 640)})
    res["rnnt_pred"] = {"unit": "call", "coef": [pred + jp, 0, 0],
                        "params": params(a / "rnnt_decoder.onnx") + params(a / "joint_pred.onnx")}
    xs = [25, 50, 100]
    ys = [profile_macs(a / "joint_enc.onnx", {"input": f32(1, x, 1024)}) for x in xs]
    res["rnnt_joint_enc"] = {"unit": "encoder frame", "coef": fit(xs, ys, 1), "params": params(a / "joint_enc.onnx")}
    pre = profile_macs(a / "joint_pre_net.onnx", {"input": f32(1, 1, 640)})
    post = profile_macs(a / "joint_post_net_hi.onnx", {"input": f32(1, 1, 640)})
    res["rnnt_joint"] = {"unit": "call", "coef": [pre + post, 0, 0],
                         "params": params(a / "joint_pre_net.onnx") + params(a / "joint_post_net_hi.onnx")}

    # LID: x = mel frames
    lid = M / "lid-ambernet/ambernet.int8.onnx"
    xs = [200, 400, 800]
    ys = [profile_macs(lid, {"features": f32(1, 80, x), "length": np.array([x])}) for x in xs]
    res["lid"] = {"unit": "mel frame", "coef": fit(xs, ys, 1), "params": params(lid)}

    # VAD: per 512-sample window (+64 context)
    vad = M / "silero-vad/onnx/model.onnx"
    res["vad"] = {"unit": "window", "coef": [profile_macs(vad, {"input": f32(1, 576), "state": np.zeros((2, 1, 128), np.float32),
                                                                "sr": np.array(16000)}), 0, 0], "params": params(vad)}

    # Whisper: encoder always sees 30 s; decoder x = tokens generated so far (past length)
    w = M / "whisper-tiny-en/onnx"
    hid = f32(1, 1500, 384)
    res["whisper_enc"] = {"unit": "call", "coef": [profile_macs(w / "encoder_model_int8.onnx", {"input_features": f32(1, 80, 3000)}), 0, 0],
                          "params": params(w / "encoder_model_int8.onnx")}
    dec = ort.InferenceSession(str(w / "decoder_model_merged_int8.onnx"), providers=["CPUExecutionProvider"])
    pnames = [i.name for i in dec.get_inputs() if i.name.startswith("past_key_values")]
    xs, ys = [2, 20, 60], []
    for x in xs:
        feeds = {"input_ids": np.array([[50257]], np.int64), "encoder_hidden_states": hid,
                 "use_cache_branch": np.array([True])}
        for n in pnames:
            feeds[n] = f32(1, 6, 1500 if ".encoder." in n else x, 64)
        ys.append(profile_macs(w / "decoder_model_merged_int8.onnx", feeds))
    res["whisper_dec"] = {"unit": "past token", "coef": fit(xs, ys, 1), "params": params(w / "decoder_model_merged_int8.onnx")}

    # IndicTrans2 (each direction): encoder x = source tokens; decoder step x = source tokens (cross-attn)
    for b in ["mt-indic-en", "mt-en-indic", "mt-indic-indic"]:
        p = M / b
        xs = [10, 30, 60]
        ys = [profile_macs(p / "encoder_model.onnx", {"input_ids": rng.integers(4, 1000, (1, x)),
                                                      "attention_mask": np.ones((1, x), np.int64)}) for x in xs]
        res[f"{b}/enc"] = {"unit": "source token", "coef": fit(xs, ys, 2), "params": params(p / "encoder_model.onnx")}
        d = ort.InferenceSession(str(p / "encoder_model.onnx"), providers=["CPUExecutionProvider"]).get_outputs()[0].shape[-1]
        ys = [profile_macs(p / "decoder_model.onnx", {"input_ids": np.array([[2]]), "encoder_hidden_states": f32(1, x, d),
                                                      "encoder_attention_mask": np.ones((1, x), np.int64)}) for x in xs]
        res[f"{b}/dec"] = {"unit": "source token", "coef": fit(xs, ys, 1), "params": params(p / "decoder_model.onnx")}
    OUT.write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        print(f"{k:24s} {v['unit']:14s} " + " ".join(f"{c:.4g}" for c in v["coef"]))


if __name__ == "__main__":
    main()
