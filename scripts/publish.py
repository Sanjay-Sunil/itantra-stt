"""Publish the dashboard to Hugging Face: built models → private model repo, app → Docker Space.

Dry run by default (lists what would be uploaded). Nothing is sent without --yes.

  python scripts/publish.py models --repo zxnjxd/itantra-stt-models [--yes]
  python scripts/publish.py space  --repo zxnjxd/itantra-stt --models-repo zxnjxd/itantra-stt-models [--yes]
  python scripts/publish.py cloudrun   (stages build/cloudrun for scripts/deploy_cloudrun.sh)

The Space downloads the models at startup with an HF_TOKEN secret (read access to the models repo),
which the owner adds in the Space settings.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
M = ROOT / "models"

# what the INT8 dashboard needs; paths relative to models/
MODEL_FILES = {
    "silero-vad": ["onnx/model.onnx"],
    "lid-ambernet": ["ambernet.int8.onnx", "labels.json"],
    "indic-conformer-int8": ["assets/*"],
    "whisper-tiny-en": ["onnx/encoder_model_int8.onnx", "onnx/decoder_model_merged_int8.onnx", "*.json", "*.txt"],
    **{d: ["*.onnx", "*.onnx.data", "*.json", "model.SRC", "model.TGT"]
       for d in ("mt-indic-indic", "mt-indic-en", "mt-en-indic")},
}

MODELS_CARD = """---
license: other
private: true
tags: [speech-recognition, translation, onnx, indic]
---
# iTantra STT — built models (private)

Quantized ONNX models used by the iTantra STT dashboard. Built from:

| Folder | Source | Licence |
|---|---|---|
| indic-conformer-int8 | ai4bharat/indic-conformer-600m-multilingual (per-channel INT8, depthwise convs FP32) | MIT (gated upstream) |
| lid-ambernet | NVIDIA NeMo langid_ambernet → ONNX, static INT8 | NVIDIA NGC Terms of Use |
| whisper-tiny-en | onnx-community/whisper-tiny.en (INT8) | MIT |
| silero-vad | onnx-community/silero-vad | MIT |
| mt-* | hari31416 ONNX INT8 exports of ai4bharat IndicTrans2 distilled models | MIT |

Kept private: upstream access conditions (IndicConformer) and NGC terms (AmberNet) apply.
"""


def model_files() -> list[Path]:
    out = []
    for d, pats in MODEL_FILES.items():
        for p in pats:
            out += [f for f in (M / d).glob(p) if f.is_file()]
    return sorted(set(out))


def stage_space(models_repo: str) -> Path:
    """Assemble the Space repo in a temp dir: code, dashboard, demo audio, Dockerfile."""
    st = Path(tempfile.mkdtemp(prefix="itantra-space-"))
    for d in ("src", "app", ".streamlit"):
        shutil.copytree(ROOT / d, st / d, ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
    (st / "scripts").mkdir()
    shutil.copy(ROOT / "scripts/fetch_models.py", st / "scripts/fetch_models.py")
    (st / "results").mkdir()
    shutil.copy(ROOT / "results/stage-profile-int8.json", st / "results/stage-profile-int8.json")
    shutil.copytree(ROOT / "data/demo", st / "data/demo")
    for lang_dir in sorted((ROOT / "data/smoke").iterdir()):  # one example clip per language
        if lang_dir.is_dir():
            first = next(iter(sorted(lang_dir.glob("*.wav"))), None)
            if first:
                (st / "data/smoke" / lang_dir.name).mkdir(parents=True)
                shutil.copy(first, st / "data/smoke" / lang_dir.name / first.name)
                shutil.copy(first.with_suffix(".json"), st / "data/smoke" / lang_dir.name / first.with_suffix(".json").name)
    for f in ("pyproject.toml", "LICENSE"):
        if (ROOT / f).exists():
            shutil.copy(ROOT / f, st / f)
    shutil.copy(ROOT / "deploy/Dockerfile", st / "Dockerfile")
    shutil.copy(ROOT / "deploy/requirements.txt", st / "requirements.txt")
    (st / "README.md").write_text((ROOT / "deploy/SPACE_README.md").read_text())
    (st / "start.sh").write_text((ROOT / "deploy/start.sh").read_text().replace("__MODELS_REPO__", models_repo))
    (st / "start.sh").chmod(0o755)
    return st


def stage_cloudrun() -> Path:
    """Build context for Cloud Run in build/cloudrun: code + demo audio + models (hard links, no copy)."""
    import os

    st = ROOT / "build/cloudrun"
    if st.exists():
        shutil.rmtree(st)
    space = stage_space("unused")
    shutil.copytree(space, st)
    shutil.rmtree(space)
    for f in ("Dockerfile", "start.sh"):
        shutil.copy(ROOT / "deploy/cloudrun" / f, st / f)
    (st / "start.sh").chmod(0o755)
    for f in model_files():
        dst = st / "models" / f.relative_to(M)
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.link(f, dst)
    return st


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["models", "space", "cloudrun"])
    ap.add_argument("--repo", help="target repo (models / space)")
    ap.add_argument("--models-repo", help="for 'space': the private models repo the Space downloads from")
    ap.add_argument("--yes", action="store_true", help="actually create the repo and upload")
    args = ap.parse_args()

    if args.what == "cloudrun":
        st = stage_cloudrun()
        files = [f for f in st.rglob("*") if f.is_file()]
        print(f"staged Cloud Run build context: {len(files)} files, "
              f"{sum(f.stat().st_size for f in files) / 1e9:.2f} GB in {st}")
        print("deploy with: scripts/deploy_cloudrun.sh")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    if args.what == "models":
        files = model_files()
        total = sum(f.stat().st_size for f in files)
        print(f"{len(files)} files, {total / 1e9:.2f} GB → {args.repo} (private model repo)")
        for f in files[:8]:
            print("  ", f.relative_to(M))
        if not args.yes:
            print("dry run: add --yes to upload")
            return
        api.create_repo(args.repo, repo_type="model", private=True, exist_ok=True)
        api.upload_file(path_or_fileobj=MODELS_CARD.encode(), path_in_repo="README.md", repo_id=args.repo)
        api.upload_folder(repo_id=args.repo, folder_path=M,
                          allow_patterns=[str(f.relative_to(M)) for f in files],
                          commit_message="Upload built INT8 models")
        print("uploaded")
    else:
        if not args.models_repo:
            raise SystemExit("--models-repo is required for 'space'")
        st = stage_space(args.models_repo)
        files = [f for f in st.rglob("*") if f.is_file()]
        print(f"staged {len(files)} files, {sum(f.stat().st_size for f in files) / 1e6:.1f} MB in {st}")
        if not args.yes:
            print("dry run: add --yes to create the Space and upload")
            return
        api.create_repo(args.repo, repo_type="space", space_sdk="docker", private=False, exist_ok=True)
        api.upload_folder(repo_id=args.repo, repo_type="space", folder_path=st, commit_message="Deploy dashboard")
        print(f"uploaded → https://huggingface.co/spaces/{args.repo}")


if __name__ == "__main__":
    main()
