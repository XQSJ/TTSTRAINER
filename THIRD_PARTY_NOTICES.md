# Third-party notices

TTSTRAINER 自有源码使用 Apache License 2.0。第三方软件和模型继续遵循各自的
许可证，本项目许可证不会替代它们。

## Qwen3-TTS

- 项目不复制或修改 Qwen3-TTS 源码。
- 样本生成通过官方 PyPI 包 `qwen-tts==0.1.1` 提供，该包标记为 Apache-2.0。
- 官方源码与许可证：https://github.com/QwenLM/Qwen3-TTS
- 官方 Python 包：https://pypi.org/project/qwen-tts/
- Qwen3-TTS 模型权重不会提交到本仓库，而是在用户明确运行生成或下载命令时，
  从对应 Hugging Face 模型仓库下载到 `models/qwen/`。
- 每个模型权重、模型卡和相关文件以其模型仓库当时提供的许可证为准。用户在
  下载、训练、再分发或商业使用前应自行确认对应版本的模型许可证。

## Python dependencies

安装过程中由 PyPI、系统包管理器或离线 wheelhouse 获取的依赖仍受各自许可证
约束。发布离线依赖包时，应保留 wheel 中附带的许可证、元数据和版权声明。

## eSpeak-ng

- TTSTRAINER 通过用户另外安装的 `espeak-ng` 命令行程序生成音素，不复制或打包
  eSpeak-ng 源码、二进制文件或语言数据。
- 官方项目：https://github.com/espeak-ng/espeak-ng
- eSpeak-ng 标记为 GPL-3.0-or-later。将其二进制或 native library 随移动 App
  一起分发时，发布者需要单独评估并履行对应许可证义务。

## pyopenjtalk / Open JTalk

- 日语训练前端通过可选依赖 `pyopenjtalk` 调用 Open JTalk；源码和词典不提交到
  本仓库，在线或离线安装时由 Python 依赖系统获取。
- pyopenjtalk：https://github.com/r9y9/pyopenjtalk （MIT）
- Open JTalk：https://open-jtalk.sourceforge.net/ （Modified BSD）
- pyopenjtalk 还包含或使用 Open JTalk、HTS Engine 与日语词典相关资源；发布者
  应检查实际安装版本附带的 LICENSE，并在移动端分发原生库或词典时保留对应声明。

## Piper Plus Mandarin/Korean G2P

- 中文与韩语训练前端通过可选依赖 `piper-plus-g2p==0.2.0`，该包标记为 MIT：
  https://github.com/ayutaz/piper-plus
- 中文路径使用 `pypinyin`（MIT）。韩语路径使用 `g2pk2`（Apache-2.0）、
  `python-mecab-ko`（BSD-3-Clause）及其词典（Apache-2.0）。
- 韩语还使用 NLTK 发布的 CMU Pronouncing Dictionary 资源。项目只在用户运行
  `frontends ensure korean` 或实际需要韩语时下载到 `models/frontends/`；发布者应
  保留下载资源内的 README/许可证信息并复核实际版本。

## 可选自动质检

- ASR 回识别使用 `faster-whisper`（MIT）和用户明确下载的
  `Systran/faster-whisper-small` 权重：https://github.com/SYSTRAN/faster-whisper
- 声纹相似度使用 SpeechBrain（Apache-2.0）和
  `speechbrain/spkrec-ecapa-voxceleb` 权重：https://github.com/speechbrain/speechbrain
- 这些依赖和权重默认不安装、不下载，也不会提交 Git。模型卡、训练数据及权重的
  许可应在实际下载和分发前再次确认。
