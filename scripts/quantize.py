"""Produce quantized tiers of IndicConformer (and AmberNet LID).

  --tier int8       every graph dynamic INT8 (MatMul/Gemm/Conv/LSTM/Gather weights)
  --tier int4mixed  encoder FFN MatMuls → block-wise INT4 (MatMulNBits); embeddings-side/first 2 layers and
                    attention Q/K/V stay INT8 (docs/compression §2.3); other graphs as int8
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


def int8(src: Path, dst: Path, big: bool = False, ops=("MatMul", "Gemm", "Conv", "LSTM", "Gather")) -> None:
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8, op_types_to_quantize=list(ops),
                     use_external_data_format=big, extra_options={"MatMulConstBOnly": True})


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


def quantize_encoder_int4mixed(dst: Path, block: int) -> None:
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

    model = onnx.load(str(SRC / "encoder.onnx"), load_external_data=True)
    targets = set(encoder_int4_nodes(model))
    print(f"  int4 MatMul nodes: {len(targets)}")
    if not targets:
        raise SystemExit("no FFN MatMul nodes matched; inspect node names")
    all_mm = [n.name for n in model.graph.node if n.op_type == "MatMul"]
    q = MatMulNBitsQuantizer(model, block_size=block, is_symmetric=True,
                             nodes_to_exclude=[n for n in all_mm if n not in targets])
    q.process()
    with tempfile.TemporaryDirectory(dir=ROOT / "models") as tmp:
        mid = Path(tmp) / "enc_int4.onnx"
        onnx.save(q.model.model, str(mid), save_as_external_data=True, location="enc_int4.onnx.data")
        int8(mid, dst, big=True, ops=("MatMul", "Gemm", "Conv"))  # remaining MatMuls/Convs → INT8


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["int8", "int4mixed"])
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--lid", action="store_true")
    args = ap.parse_args()

    if args.tier:
        out = ROOT / f"models/indic-conformer-{args.tier}/assets"
        out.mkdir(parents=True, exist_ok=True)
        for f in ["vocab.json", "language_masks.json"]:
            shutil.copy(SRC / f, out / f)
        print("encoder …", flush=True)
        if args.tier == "int8":
            int8(SRC / "encoder.onnx", out / "encoder.onnx", big=True, ops=("MatMul", "Gemm", "Conv"))
        else:
            quantize_encoder_int4mixed(out / "encoder.onnx", args.block)
        print(f"  encoder: {mb(out / 'encoder.onnx'):.0f} MB", flush=True)
        for name in SMALL:
            int8(SRC / f"{name}.onnx", out / f"{name}.onnx")
        total = sum(f.stat().st_size for f in out.iterdir()) / 1e6
        print(f"{args.tier}: {total:.0f} MB in {out}")

    if args.lid:
        d = ROOT / "models/lid-ambernet"
        int8(d / "ambernet.onnx", d / "ambernet.int8.onnx", ops=("MatMul", "Gemm", "Conv"))
        print(f"lid: {mb(d / 'ambernet.onnx'):.0f} → {(d / 'ambernet.int8.onnx').stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
