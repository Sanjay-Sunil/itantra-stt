"""Publish the browser demo to Hugging Face: models repo made public with a licence card, page → static Space.

Dry run by default (prints what would change). Nothing is sent without --yes.

  python scripts/publish_web.py --models-repo zxnjxd/itantra-stt-models --space zxnjxd/itantra-stt-browser [--yes]

The Space is static: the visitor's browser downloads the models from the models repo and runs them locally.
Run `cd web && npm ci` first; this script runs `node build.mjs --models-url …` itself.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"

MODELS_CARD = """---
license: other
license_name: mixed
tags: [speech-recognition, translation, onnx, indic, onnxruntime-web]
---
# iTantra STT: built INT8 models

Quantized ONNX models used by the iTantra STT engine and its in-browser demo
(https://huggingface.co/spaces/{space}). Code: https://github.com/ (iTantra STT repository).

| Folder | Built from | Licence |
|---|---|---|
| indic-conformer-int8 | [ai4bharat/indic-conformer-600m-multilingual](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual), per-channel INT8, depthwise convs FP32 | MIT (© AI4Bharat) |
| lid-ambernet | [NVIDIA NeMo langid_ambernet](https://catalog.ngc.nvidia.com/orgs/nvidia/nemo/models/langid_ambernet) → ONNX, static INT8 | NVIDIA NGC Terms of Use |
| whisper-tiny-en | [onnx-community/whisper-tiny.en](https://huggingface.co/onnx-community/whisper-tiny.en) (INT8) | MIT (© OpenAI) |
| silero-vad | [onnx-community/silero-vad](https://huggingface.co/onnx-community/silero-vad) | MIT |
| mt-* | ONNX INT8 exports (hari31416) of [AI4Bharat IndicTrans2 distilled](https://huggingface.co/ai4bharat) models | MIT (© AI4Bharat) |

Each model keeps its upstream licence; using these files means accepting those terms. AI4Bharat asks users of
IndicConformer to share contact details on its upstream page: if you build on it, please also request access there.
"""

SPACE_README = """---
title: iTantra on-device STT
emoji: 🎙️
colorFrom: yellow
colorTo: red
sdk: static
pinned: false
license: apache-2.0
short_description: Indic speech-to-text + translation running in your browser
custom_headers:
  cross-origin-embedder-policy: require-corp
  cross-origin-opener-policy: same-origin
  cross-origin-resource-policy: cross-origin
---
Ten languages (Hindi, Gujarati, Marathi, Kannada, Malayalam, Tamil, Telugu, Odia, Bengali, English), transcribed and
translated **inside your browser tab** with ONNX Runtime Web. The models download once from
[{models}](https://huggingface.co/{models}) and run on your CPU; no server sees your audio.
"""

FILES = ["index.html", "app.js", "app.css", "worker.js", "compute.json", "models.json", "site.json",
         "coi-serviceworker.min.js"]
DIRS = ["lib", "ui", "vendor", "samples"]


def stage(models_repo: str, space: str) -> Path:
    url = f"https://huggingface.co/{models_repo}/resolve/main/"
    subprocess.run(["node", "build.mjs", "--models-url", url], cwd=WEB, check=True)
    st = Path(tempfile.mkdtemp(prefix="itantra-space-"))
    for f in FILES:
        shutil.copy2(WEB / f, st / f)
    for d in DIRS:
        shutil.copytree(WEB / d, st / d)
    (st / "README.md").write_text(SPACE_README.format(models=models_repo))
    # restore the local build (site.json removed) so `npm run serve` keeps using ../models
    subprocess.run(["node", "build.mjs"], cwd=WEB, check=True, capture_output=True)
    return st


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-repo", required=True)
    ap.add_argument("--space", required=True)
    ap.add_argument("--yes", action="store_true", help="actually change visibility and upload")
    args = ap.parse_args()
    st = stage(args.models_repo, args.space)
    files = sorted(p for p in st.rglob("*") if p.is_file())
    print(f"models repo {args.models_repo}: set public, replace README.md (licence table)")
    print(f"space {args.space} (static): {len(files)} files, {sum(p.stat().st_size for p in files) / 1e6:.1f} MB from {st}")
    if not args.yes:
        print("dry run: add --yes to publish")
        return
    from huggingface_hub import HfApi

    api = HfApi()
    api.upload_file(path_or_fileobj=MODELS_CARD.format(space=args.space).encode(), path_in_repo="README.md",
                    repo_id=args.models_repo, commit_message="Model card: licences and attribution")
    api.update_repo_settings(args.models_repo, private=False)
    api.create_repo(args.space, repo_type="space", space_sdk="static", private=False, exist_ok=True)
    api.upload_folder(repo_id=args.space, repo_type="space", folder_path=st, commit_message="Deploy browser demo")
    print(f"https://huggingface.co/spaces/{args.space}")


if __name__ == "__main__":
    main()
