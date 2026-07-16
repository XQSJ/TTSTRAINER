# 离线部署包

默认教师模型：

- `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`
- `Qwen/Qwen3-TTS-12Hz-1.7B-Base`

模型文件与平台无关；`wheelhouse` 只适用于构建时记录的平台和 Python 版本。
构建器会把官方 `qwen-tts==0.1.1`、日语前端 `pyopenjtalk`、中韩
`piper-plus-g2p`/MeCab 及其 Python 构建依赖，以及 ONNX/ONNX Runtime 放进
`wheelhouse`，不会复制 Qwen 源码仓库，
默认也不下载模型。运行项目时通过
`tts-trainer models ensure ...` 按需下载到项目的 `models/qwen/`，项目代码此后
只使用这个本地路径。

Open JTalk 日语词典和韩语 CMU 词典属于运行依赖而不是 Qwen 模型，构建离线包时
默认下载并校验到 `source/models/frontends/`。如果离线包不需要这些前端，可增加
`--skip-frontend-resources`。

## 构建

```bash
.venv/bin/python deployment/build_bundle.py \
  --output dist/tts-trainer-offline-macos-arm64-py312
```

只有明确需要把权重也装入离线包时才增加 `--download-models`。
此时权重会保存到包内 `source/models/qwen/`，与运行时检测路径一致。

如需 ASR/声纹质检的 Python 依赖，增加 `--include-quality`。它仍不会下载质检
模型权重；联网目标机用 `quality-models ensure ...` 显式下载到
`source/models/quality/`，或在构建机准备后将该目录一并复制。

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

导出后建议在目标机器再验证一次前端版本、音素和 token ID：

```bash
PYTHONPATH=src /path/to/new/venv/bin/python -m tts_trainer verify-frontend \
  --model-dir artifacts/model_1
```

若目标是 Linux/CUDA 训练服务器，不能复用 macOS wheelhouse。请在相同 Linux、
Python 和 CUDA 环境中重新运行构建脚本；模型目录可以直接复用，不必重新下载。
pyopenjtalk 在没有上游 wheel 的平台会在“构建离线包的机器”上编译成 wheel，因此
构建机需要系统 C/C++ 编译器；目标离线机器不需要编译器。
构建器会显式准备 `setuptools_scm`；不要对 pyopenjtalk 的源码构建使用一个缺少
`setuptools_scm` 的 `--no-build-isolation` 环境，否则其 0.4.1 元数据会退化为
0.0.0 并被 pip 拒绝。
