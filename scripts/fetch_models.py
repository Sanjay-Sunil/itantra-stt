"""Download every upstream model the engine uses into models/.

Usage: python scripts/fetch_models.py [--only NAME ...]
       python scripts/fetch_models.py --prepared OWNER/REPO   (built models, e.g. on the Hugging Face Space)
Re-running is a no-op for files already present. Gated repos need the HF account to have
accepted the access conditions (token is read from HF_TOKEN or the cached login).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import GatedRepoError

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

# name -> (repo, allow_patterns, gated)
HF_MODELS = {
    "silero-vad": ("onnx-community/silero-vad", ["onnx/model.onnx", "*.json"], False),
    "indic-conformer": ("ai4bharat/indic-conformer-600m-multilingual", None, True),
    "whisper-tiny-en": (
        "onnx-community/whisper-tiny.en",
        ["onnx/encoder_model_int8.onnx", "onnx/decoder_model_merged_int8.onnx", "*.json", "*.txt"],
        False,
    ),
    "mt-indic-indic": ("hari31416/indictrans2-indic-indic-dist-320M-ONNX-int8", None, False),
    "mt-indic-en": ("hari31416/indictrans2-indic-en-dist-200M-ONNX-int8", None, False),
    "mt-en-indic": ("hari31416/indictrans2-en-indic-dist-200M-ONNX-int8", None, False),
}

# NeMo AmberNet language-ID checkpoint (NGC, public; licence: NGC Terms of Use)
NGC_MODELS = {
    "lid-ambernet": (
        "https://api.ngc.nvidia.com/v2/models/nvidia/nemo/langid_ambernet/versions/1.12.0/files/ambernet.nemo",
        "ambernet.nemo",
    ),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_hf(name: str) -> str:
    repo, patterns, _ = HF_MODELS[name]
    dest = MODELS / name
    try:
        snapshot_download(repo, local_dir=dest, allow_patterns=patterns)
    except GatedRepoError:
        return f"BLOCKED (gated): accept access at https://huggingface.co/{repo}"
    return "ok"


def fetch_ngc(name: str) -> str:
    url, fname = NGC_MODELS[name]
    dest = MODELS / name / fname
    if dest.exists():
        return "ok (cached)"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)
    return "ok"


def write_lock() -> None:
    lock = {}
    for d in sorted(p for p in MODELS.iterdir() if p.is_dir()):
        files = [f for f in d.rglob("*") if f.is_file() and ".cache" not in f.parts]
        lock[d.name] = [
            {"path": str(f.relative_to(MODELS)), "bytes": f.stat().st_size, "sha256": sha256(f)}
            for f in sorted(files)
        ]
    (MODELS / "manifest.lock.json").write_text(json.dumps(lock, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="subset of model names")
    ap.add_argument("--no-lock", action="store_true", help="skip hashing into manifest.lock.json")
    ap.add_argument("--prepared", metavar="REPO",
                    help="download already-built (quantized/exported) models from this HF repo and exit")
    args = ap.parse_args()
    if args.prepared:
        MODELS.mkdir(exist_ok=True)
        snapshot_download(args.prepared, local_dir=MODELS)  # HF_TOKEN from the environment for private repos
        print(f"models ready from {args.prepared}")
        return 0
    names = args.only or [*HF_MODELS, *NGC_MODELS]
    MODELS.mkdir(exist_ok=True)
    failed = False
    for name in names:
        status = fetch_hf(name) if name in HF_MODELS else fetch_ngc(name)
        failed |= not status.startswith("ok")
        print(f"{name:18s} {status}")
    if not args.no_lock:
        write_lock()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
