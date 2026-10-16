"""Produce quantized tiers of IndicConformer (and AmberNet LID).

  --tier int8       encoder: per-channel dynamic INT8 on every MatMul and Conv except the depthwise convs
                    (0.2M of 607M weights stay FP32: ORT's dynamic ConvInteger on depthwise conv cost +3.4 WER
                    on hi; see results/quantization.md)
  --tier int4mixed  as int8, but FFN MatMuls of layers >= 2 at block-wise INT4 (MatMulNBits, --block); first
                    2 layers and attention stay INT8 (docs/compression §2.3)
  --tier int8-pertensor / int8pc / int8b   earlier experiments, kept for reproducing results/quantization.md
  Small graphs (CTC, prediction/joint nets, per-language heads) are dynamic INT8 in every tier.
  --lid             also quantize models/lid-ambernet/ambernet.onnx → ambernet.int8.onnx

Output: models/indic-conformer-<tier>/assets/ with the same file names the runner expects.
Usage: python scripts/quantize.py --tier int8 [--block 32] [--lid]
"""

from __future__ import annotations

import argparse
import re
import shutil
import tempfile
from pathlib import Path

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "models/indic-conformer/assets"
LANGS = ["hi", "gu", "mr", "kn", "ml", "ta", "te", "or", "bn"]
SMALL = ["ctc_decoder", "rnnt_decoder", "joint_enc", "joint_pred", "joint_pre_net"] + [
    f"joint_post_net_{x}" for x in LANGS]


def mb(p: Path) -> float:
    return sum(f.stat().st_size for f in p.parent.glob(p.stem + "*") if f.is_file()) / 1e6


def int8(src: Path, dst: Path, big: bool = False, ops=("MatMul", "Gemm", "Conv", "LSTM", "Gather"),
         per_channel: bool = False, exclude: str | None = None) -> None:
    skip = []
    if exclude:
        g = onnx.load(str(src), load_external_data=False).graph
        skip = [n.name for n in g.node if re.search(exclude, n.name)]
        print(f"  keeping {len(skip)} nodes unquantized (/{exclude}/)", flush=True)
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8, op_types_to_quantize=list(ops),
                     per_channel=per_channel, use_external_data_format=big, nodes_to_exclude=skip,
                     extra_options={"MatMulConstBOnly": True})


def blockwise(dst: Path, block: int, bits: int, int4_nodes: set[str] | None = None) -> None:
    """Weight-only block-wise quantization of encoder MatMuls (MatMulNBits), then per-channel INT8 Convs.

    bits=8: every MatMul at 8 bits. With int4_nodes: those at 4 bits, all other MatMuls at 8 bits.
    """
    from onnxruntime.quantization.matmul_nbits_quantizer import (
        DefaultWeightOnlyQuantConfig,
        MatMulNBitsQuantizer,
    )

    model = onnx.load(str(SRC / "encoder.onnx"), load_external_data=True)
    all_mm = [n.name for n in model.graph.node if n.op_type == "MatMul"]
    passes = [(bits, set(all_mm) - (int4_nodes or set()))]
    if int4_nodes:
        passes.append((4, int4_nodes))
    for nbits, targets in passes:
        cfg = DefaultWeightOnlyQuantConfig(block_size=block, is_symmetric=True, bits=nbits)
        q = MatMulNBitsQuantizer(model, block_size=block, is_symmetric=True, algo_config=cfg,
                                 nodes_to_exclude=[n for n in all_mm if n not in targets])
        q.process()
        model = q.model.model
    with tempfile.TemporaryDirectory(dir=ROOT / "models") as tmp:
        mid = Path(tmp) / "enc_nbits.onnx"
        onnx.save(model, str(mid), save_as_external_data=True, location="enc_nbits.onnx.data")
        int8(mid, dst, big=True, ops=("Conv",), per_channel=True)


def encoder_int4_nodes(model: onnx.ModelProto, keep_int8_layers: int = 2) -> list[str]:
    """Encoder MatMul nodes to push to INT4: FFN weights in layers >= keep_int8_layers."""
    out = []
    for n in model.graph.node:
        if n.op_type != "MatMul":
            continue
        m = re.search(r"layers\.(\d+)[./]", n.name)
        if not m or int(m.group(1)) < keep_int8_layers:
            continue
        if "feed_forward" in n.name or any("feed_forward" in i for i in n.input):
            out.append(n.name)
    return out


DEPTHWISE = "depthwise_conv"


def int4mixed(dst: Path, block: int, exclude: str) -> None:
    """FFN MatMuls (layers >= 2) → MatMulNBits INT4; remaining MatMuls/Convs → per-channel dynamic INT8."""
    from onnxruntime.quantization.matmul_nbits_quantizer import (
        DefaultWeightOnlyQuantConfig,
        MatMulNBitsQuantizer,
    )

    model = onnx.load(str(SRC / "encoder.onnx"), load_external_data=True)
    targets = set(encoder_int4_nodes(model))
    print(f"  int4 MatMul nodes: {len(targets)}", flush=True)
    all_mm = [n.name for n in model.graph.node if n.op_type == "MatMul"]
    cfg = DefaultWeightOnlyQuantConfig(block_size=block, is_symmetric=True, bits=4)
    q = MatMulNBitsQuantizer(model, block_size=block, is_symmetric=True, algo_config=cfg,
                             nodes_to_exclude=[n for n in all_mm if n not in targets])
    q.process()
    with tempfile.TemporaryDirectory(dir=ROOT / "models") as tmp:
        mid = Path(tmp) / "enc_int4.onnx"
        onnx.save(q.model.model, str(mid), save_as_external_data=True, location="enc_int4.onnx.data")
        int8(mid, dst, big=True, ops=("MatMul", "Gemm", "Conv"), per_channel=True, exclude=exclude)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["int8", "int4mixed", "int8-pertensor", "int8pc", "int8b"])
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--lid", action="store_true")
    ap.add_argument("--exclude", help="regex of encoder node names to leave unquantized (int8/int8pc)")
    ap.add_argument("--name", help="output dir suffix (default: tier)")
    args = ap.parse_args()

    if args.tier:
        out = ROOT / f"models/indic-conformer-{args.name or args.tier}/assets"
        out.mkdir(parents=True, exist_ok=True)
        for f in ["vocab.json", "language_masks.json"]:
            shutil.copy(SRC / f, out / f)
        print("encoder …", flush=True)
        if args.tier == "int8":
            int8(SRC / "encoder.onnx", out / "encoder.onnx", big=True, ops=("MatMul", "Gemm", "Conv"),
                 per_channel=True, exclude=args.exclude or DEPTHWISE)
        elif args.tier == "int8-pertensor":
            int8(SRC / "encoder.onnx", out / "encoder.onnx", big=True, ops=("MatMul", "Gemm", "Conv"))
        elif args.tier == "int8pc":
            int8(SRC / "encoder.onnx", out / "encoder.onnx", big=True, ops=("MatMul", "Gemm", "Conv"),
                 per_channel=True, exclude=args.exclude)
        elif args.tier == "int8b":
            blockwise(out / "encoder.onnx", args.block, bits=8)
        else:
            int4mixed(out / "encoder.onnx", args.block, args.exclude or DEPTHWISE)
        print(f"  encoder: {mb(out / 'encoder.onnx'):.0f} MB", flush=True)
        for name in SMALL:
            int8(SRC / f"{name}.onnx", out / f"{name}.onnx", per_channel=args.tier != "int8-pertensor")
        total = sum(f.stat().st_size for f in out.iterdir()) / 1e6
        print(f"{args.name or args.tier}: {total:.0f} MB in {out}")

    if args.lid:
        lid_static_int8()


def lid_static_int8() -> None:
    """AmberNet is conv-only: static QDQ INT8 (calibrated activations) so ORT runs fast QLinearConv kernels.

    Dynamic INT8 measured 3x slower than FP32 on CPU (ConvInteger). Calibration: 3 smoke clips per language.
    """
    import json

    import numpy as np
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, quantize_static

    from indicstt.audio import load_audio
    from indicstt.features import log_mel

    d = ROOT / "models/lid-ambernet"

    class Reader(CalibrationDataReader):
        def __init__(self):
            feeds = []
            for lang in json.loads((d / "labels.json").read_text()):
                for p in sorted((ROOT / "data/smoke" / lang).glob("*.json"))[:3]:
                    x = log_mel(load_audio(ROOT / json.loads(p.read_text())["audio"]))
                    feeds.append({"features": x[None], "length": np.array([x.shape[1]], dtype=np.int64)})
            self.it = iter(feeds)

        def get_next(self):
            return next(self.it, None)

    quantize_static(str(d / "ambernet.onnx"), str(d / "ambernet.int8.onnx"), Reader(),
                    quant_format=QuantFormat.QDQ, per_channel=True, weight_type=QuantType.QInt8,
                    activation_type=QuantType.QUInt8, op_types_to_quantize=["Conv", "MatMul", "Gemm"])
    print(f"lid: {(d / 'ambernet.onnx').stat().st_size / 1e6:.0f} → "
          f"{(d / 'ambernet.int8.onnx').stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
