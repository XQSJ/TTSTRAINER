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
