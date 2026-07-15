#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_MODELS = (
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
)
ROOT = Path(__file__).resolve().parents[1]


def run(*command: str, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=ROOT, env=env)


def copy_sources(output: Path) -> None:
    source = output / "source"
    source.mkdir(parents=True, exist_ok=True)
    for name in (
        "src", "configs", "training_configs", "scripts", "pyproject.toml", "README.md",
        "LICENSE", "THIRD_PARTY_NOTICES.md",
    ):
        origin = ROOT / name
        target = source / name
        if origin.is_dir():
            shutil.copytree(origin, target, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copy2(origin, target)
    examples = source / "datasets"
    examples.mkdir(parents=True, exist_ok=True)
    for name in ("texts.example.csv", "metadata.example.csv"):
        shutil.copy2(ROOT / "datasets" / name, examples / name)


def download_wheels(output: Path) -> None:
    wheelhouse = output / "wheelhouse"
    wheelhouse.mkdir(parents=True, exist_ok=True)
    # Download all transitive dependencies plus build tools so installation can
    # run with --no-index on an empty environment.
    run(
        sys.executable, "-m", "pip", "download", "--dest", str(wheelhouse),
        "pip", "setuptools>=68", "wheel", "build", "huggingface_hub[cli]>=0.34,<2",
        "onnx>=1.16", "qwen-tts==0.1.1", str(ROOT),
    )


def ensure_huggingface_client(output: Path) -> Path:
    bootstrap = output / ".builder"
    if not (bootstrap / "huggingface_hub").exists():
        run(sys.executable, "-m", "pip", "install", "--target", str(bootstrap), "huggingface_hub>=0.34,<2")
    return bootstrap


def download_models(output: Path, model_ids: tuple[str, ...]) -> None:
    bootstrap = ensure_huggingface_client(output)
    code = (
        "from huggingface_hub import snapshot_download; "
        "import sys; snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2])"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(bootstrap) + os.pathsep + env.get("PYTHONPATH", "")
    for model_id in model_ids:
        target = output / "models" / model_id.rsplit("/", 1)[-1]
        target.mkdir(parents=True, exist_ok=True)
        run(sys.executable, "-c", code, model_id, str(target), env=env)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_support_files(output: Path, model_ids: tuple[str, ...]) -> None:
    install = """#!/bin/sh
set -eu
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
VENV=${1:-"$HERE/.venv"}
python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install --no-index --find-links "$HERE/wheelhouse" --upgrade pip setuptools wheel
"$VENV/bin/python" -m pip install --no-index --find-links "$HERE/wheelhouse" qwen-tts==0.1.1 tts-trainer onnx
printf 'Installed offline environment at %s\n' "$VENV"
"""
    (output / "install_offline.sh").write_text(install, encoding="utf-8")
    (output / "install_offline.sh").chmod(0o755)
    verify = '''from __future__ import annotations
import hashlib, json, platform, sys
from pathlib import Path
root = Path(__file__).resolve().parent
manifest = json.loads((root / "bundle-manifest.json").read_text())
errors = []
for item in manifest["files"]:
    path = root / item["path"]
    if not path.is_file(): errors.append(f"missing: {item['path']}"); continue
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != item["sha256"]: errors.append(f"checksum mismatch: {item['path']}")
if errors: raise SystemExit("Bundle verification failed:\\n" + "\\n".join(errors))
print(f"Bundle OK: {len(manifest['files'])} files, Python {sys.version.split()[0]}, {platform.platform()}")
'''
    (output / "verify_bundle.py").write_text(verify, encoding="utf-8")
    (output / "verify_bundle.py").chmod(0o755)
    readme = (
        "# TTS Trainer offline bundle\n\n"
        f"Models: {', '.join(model_ids)}\n\n"
        "Install: `./install_offline.sh /path/to/venv`\n\n"
        "Model paths are under `models/`. The wheelhouse is platform-specific; see bundle-manifest.json.\n"
    )
    (output / "README.md").write_text(readme, encoding="utf-8")


def write_manifest(output: Path, model_ids: tuple[str, ...]) -> None:
    excluded = {"bundle-manifest.json"}
    files = []
    for path in sorted(p for p in output.rglob("*") if p.is_file() and ".builder" not in p.parts):
        relative = path.relative_to(output).as_posix()
        if relative in excluded:
            continue
        files.append({"path": relative, "size": path.stat().st_size, "sha256": sha256(path)})
    manifest = {
        "format": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "models": list(model_ids),
        "total_bytes": sum(item["size"] for item in files),
        "files": files,
    }
    (output / "bundle-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def create_archive(output: Path) -> Path:
    archive = output.with_suffix(".tar")
    with tarfile.open(archive, "w") as tar:
        tar.add(output, arcname=output.name)
    return archive


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a relocatable offline TTS bundle")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--download-models", action="store_true", help="opt in to downloading model weights")
    parser.add_argument("--skip-wheels", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--archive", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    models = tuple(args.models or DEFAULT_MODELS)
    copy_sources(output)
    if not args.skip_download:
        if not args.skip_wheels: download_wheels(output)
        if args.download_models: download_models(output, models)
    write_support_files(output, models)
    write_manifest(output, models)
    if args.archive: print(f"archive: {create_archive(output)}")
    print(f"bundle: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
