# 离线部署包

默认教师模型：

- `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`
- `Qwen/Qwen3-TTS-12Hz-1.7B-Base`

模型文件与平台无关；`wheelhouse` 只适用于构建时记录的平台和 Python 版本。
构建器会把官方 `qwen-tts==0.1.1` wheel 及其依赖放进 `wheelhouse`，不会复制
Qwen 源码仓库，默认也不下载模型。运行项目时通过
`tts-trainer models ensure ...` 按需下载到项目的 `models/qwen/`，项目代码此后
只使用这个本地路径。

## 构建

```bash
.venv/bin/python deployment/build_bundle.py \
  --output dist/tts-trainer-offline-macos-arm64-py312
```

只有明确需要把权重也装入离线包时才增加 `--download-models`。

下载完成后可选生成单个未压缩归档（神经网络权重通常已经很难再次压缩）：

```bash
.venv/bin/python deployment/build_bundle.py \
  --output dist/tts-trainer-offline-macos-arm64-py312 \
  --skip-download --archive
```

## 离线安装

把整个目录或 `.tar` 文件复制到目标机器，然后：

```bash
python3 verify_bundle.py
./install_offline.sh /path/to/new/venv
```

离线安装完成后可检查 Qwen Python 运行时：

```bash
PYTHONPATH=source /path/to/new/venv/bin/python -m tts_trainer qwen-runtime
```

进入包内 `source/`，修改 `training_configs/train1.json` 后可执行：

```bash
PYTHONPATH=src /path/to/new/venv/bin/python -m tts_trainer run-pipeline \
  --config training_configs/train1.json
```

若目标是 Linux/CUDA 训练服务器，不能复用 macOS wheelhouse。请在相同 Linux、
Python 和 CUDA 环境中重新运行构建脚本；模型目录可以直接复用，不必重新下载。
