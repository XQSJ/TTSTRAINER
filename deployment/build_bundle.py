#!/usr/bin/env python3
"""构建可迁移的离线 TTS 部署资源包。 / Build a relocatable offline TTS bundle.

将源码、平台专属 wheelhouse、模型权重与前端资源打包成单个目录（可选 tar 归档），
供无网、无编译器的目标机器离线安装。
Bundles source code, a platform-specific wheelhouse, model weights, and frontend
resources into one directory (optionally a tar archive) for offline install on
air-gapped, compiler-free target machines.

用法：python deployment/build_bundle.py --output dist/bundle [--download-models] [--archive]
Usage: python deployment/build_bundle.py --output dist/bundle [--download-models] [--archive]
"""
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

# 默认随包分发的 Qwen3-TTS 基座与 VoiceDesign 两个模型。 / Default Qwen3-TTS Base and VoiceDesign models shipped in the bundle.
DEFAULT_MODELS = (
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
)
# 仓库根目录，所有子进程的默认工作目录。 / Repository root; also the cwd for all subprocesses.
ROOT = Path(__file__).resolve().parents[1]


def run(*command: str, env: dict[str, str] | None = None) -> None:
    """在仓库根目录执行外部命令并在失败时抛错。 / Run an external command at repo root, raising on failure."""
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=ROOT, env=env)


def copy_sources(output: Path) -> None:
    """把仓库运行所需的源码与配置拷入 bundle 的 source/ 目录。 / Copy the sources and configs the bundle needs into its source/ directory."""
    source = output / "source"
    source.mkdir(parents=True, exist_ok=True)
    # 白名单式拷贝：只带部署所需的目录/文件，排除其余仓库内容。 / Whitelist copy: ship only what deployment needs.
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
    # 数据集只带示例文件，真实语料不随 bundle 分发。 / Ship only dataset examples; real corpora stay out of the bundle.
    examples = source / "datasets"
    examples.mkdir(parents=True, exist_ok=True)
    for name in ("texts.example.csv", "metadata.example.csv"):
        shutil.copy2(ROOT / "datasets" / name, examples / name)


def download_wheels(output: Path, *, include_quality: bool = False) -> None:
    """解析全部传递依赖并构建平台专属 wheelhouse。 / Resolve all transitive deps and build the platform-specific wheelhouse."""
    wheelhouse = output / "wheelhouse"
    wheelhouse.mkdir(parents=True, exist_ok=True)
    # Resolve all transitive dependencies and build source-only packages (notably
    # pyopenjtalk) into platform-specific wheels so the target stays compiler-free.
    requirements = [
        "pip", "setuptools>=68", "setuptools_scm>=8", "wheel", "build",
        "cmake", "ninja", "cython",
        "huggingface_hub[cli]>=0.34,<2", "onnx>=1.16", "onnxruntime>=1.18",
        "qwen-tts==0.1.1", "pyopenjtalk>=0.4.1,<0.5",
        "piper-plus-g2p[zh,ko]==0.2.0", "python-mecab-ko>=1.3,<2",
    ]
    # 质检依赖（ASR + 说话人嵌入）为可选项，按需入包。 / QC deps (ASR + speaker embedding) are opt-in.
    if include_quality:
        requirements.extend(("faster-whisper>=1.2,<2", "speechbrain>=1.1,<2"))
    # 把本项目自身也构建成 wheel，目标机免源码安装。 / Build this project itself as a wheel so targets need no source install.
    requirements.append(str(ROOT))
    run(
        sys.executable, "-m", "pip", "wheel", "--wheel-dir", str(wheelhouse),
        *requirements,
    )


def ensure_huggingface_client(output: Path) -> Path:
    """在 bundle 内自举一份 huggingface_hub，避免依赖全局环境。 / Bootstrap huggingface_hub inside the bundle instead of relying on the global env."""
    bootstrap = output / ".builder"
    if not (bootstrap / "huggingface_hub").exists():
        run(sys.executable, "-m", "pip", "install", "--target", str(bootstrap), "huggingface_hub>=0.34,<2")
    return bootstrap


def download_models(output: Path, model_ids: tuple[str, ...]) -> None:
    """按模型 ID 下载快照到 bundle 内的项目本地路径。 / Download model snapshots into the bundle's project-local layout."""
    bootstrap = ensure_huggingface_client(output)
    code = (
        "from huggingface_hub import snapshot_download; "
        "import sys; snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2])"
    )
    env = os.environ.copy()
    # 让子进程优先加载 bundle 内自举的 huggingface_hub。 / Make the subprocess prefer the bootstrapped huggingface_hub.
    env["PYTHONPATH"] = str(bootstrap) + os.pathsep + env.get("PYTHONPATH", "")
    for model_id in model_ids:
        # Keep the same project-local layout used by model_registry.py inside
        # the copied source tree. The bundle must not rely on a global HF cache.
        target = output / "source" / "models" / "qwen" / model_id.rsplit("/", 1)[-1]
        target.mkdir(parents=True, exist_ok=True)
        run(sys.executable, "-c", code, model_id, str(target), env=env)


def download_frontend_resources(output: Path) -> None:
    """下载 OpenJTalk 词典与韩文 CMU dict 等前端资源。 / Download frontend resources such as the OpenJTalk dictionary and Korean CMU dict."""
    # 临时把 src/ 加入 sys.path 以复用项目内的资源下载逻辑。 / Temporarily prepend src/ to sys.path to reuse the project's resource fetchers.
    sys.path.insert(0, str(ROOT / "src"))
    try:
        from tts_trainer.frontend.resources import (ensure_korean_cmudict,
                                                     ensure_openjtalk_dictionary)
        ensure_openjtalk_dictionary(output / "source" / "models" / "frontends")
        ensure_korean_cmudict(output / "source" / "models" / "frontends")
    finally:
        sys.path.pop(0)


def sha256(path: Path) -> str:
    """流式计算文件 SHA-256，供清单校验使用。 / Stream-compute a file's SHA-256 for manifest verification."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_support_files(output: Path, model_ids: tuple[str, ...], *, include_quality: bool = False) -> None:
    """生成安装脚本、校验脚本与 README 等随包文件。 / Emit the install script, verification script, and README shipped with the bundle."""
    # 质检包名按需追加到离线安装命令中。 / QC package names appended to the offline install command when enabled.
    quality_packages = " faster-whisper speechbrain" if include_quality else ""
    install = f"""#!/bin/sh
set -eu
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
VENV=${{1:-"$HERE/.venv"}}
python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install --no-index --find-links "$HERE/wheelhouse" --upgrade pip setuptools wheel
"$VENV/bin/python" -m pip install --no-index --find-links "$HERE/wheelhouse" qwen-tts==0.1.1 pyopenjtalk piper-plus-g2p python-mecab-ko tts-trainer onnx onnxruntime{quality_packages}
printf 'Installed offline environment at %s\n' "$VENV"
"""
    # 安装脚本：建 venv 后完全从本地 wheelhouse 离线安装（--no-index）。 / Install script: create a venv then install fully offline from the local wheelhouse (--no-index).
    (output / "install_offline.sh").write_text(install, encoding="utf-8")
    (output / "install_offline.sh").chmod(0o755)
    # 校验脚本：目标机逐文件比对清单中的 SHA-256，缺失或损坏即失败。 / Verify script: on the target, re-check every file's SHA-256 against the manifest and fail on missing/corrupt entries.
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
        "Model paths are under `source/models/qwen/`. Frontend resources are under "
        "`source/models/frontends/`. The wheelhouse is platform-specific; see bundle-manifest.json.\n"
    )
    (output / "README.md").write_text(readme, encoding="utf-8")


def write_manifest(output: Path, model_ids: tuple[str, ...]) -> None:
    """遍历 bundle 全部文件，写入带 SHA-256 的完整性清单。 / Walk every bundle file and write an integrity manifest with SHA-256 checksums."""
    # 清单自身与自举目录不能进入清单，否则鸡生蛋问题。 / The manifest itself and the bootstrap dir are excluded to avoid self-reference.
    excluded = {"bundle-manifest.json"}
    files = []
    for path in sorted(p for p in output.rglob("*") if p.is_file() and ".builder" not in p.parts):
        relative = path.relative_to(output).as_posix()
        if relative in excluded:
            continue
        files.append({"path": relative, "size": path.stat().st_size, "sha256": sha256(path)})
    # platform/machine/python 字段用于提醒 wheelhouse 与构建机绑定。 / platform/machine/python fields flag that the wheelhouse is tied to the build machine.
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
    """把 bundle 目录压成同名 tar 归档便于传输。 / Tar up the bundle directory for transfer."""
    archive = output.with_suffix(".tar")
    with tarfile.open(archive, "w") as tar:
        tar.add(output, arcname=output.name)
    return archive


def main() -> int:
    """解析命令行参数并按序执行各打包步骤。 / Parse CLI args and run the bundling steps in order."""
    parser = argparse.ArgumentParser(description="Build a relocatable offline TTS bundle")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--download-models", action="store_true", help="opt in to downloading model weights")
    parser.add_argument("--skip-wheels", action="store_true")
    parser.add_argument("--skip-frontend-resources", action="store_true")
    parser.add_argument("--include-quality", action="store_true",
                        help="include optional ASR/speaker QC Python dependencies, not weights")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--archive", action="store_true")
    args = parser.parse_args()
    # 打包顺序：源码 → 下载类步骤（可分别跳过）→ 随包文件 → 清单 → 归档。 / Order: sources → download steps (individually skippable) → support files → manifest → archive.
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    models = tuple(args.models or DEFAULT_MODELS)
    copy_sources(output)
    if not args.skip_download:
        if not args.skip_wheels: download_wheels(output, include_quality=args.include_quality)
        if not args.skip_frontend_resources: download_frontend_resources(output)
        # 模型权重体积大，默认不下载，需显式 opt-in。 / Model weights are large; download only on explicit opt-in.
        if args.download_models: download_models(output, models)
    write_support_files(output, models, include_quality=args.include_quality)
    # 清单必须最后写，才能覆盖所有随包文件。 / The manifest must be written last to cover every shipped file.
    write_manifest(output, models)
    if args.archive: print(f"archive: {create_archive(output)}")
    print(f"bundle: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
