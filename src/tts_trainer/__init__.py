"""紧凑多语言 TTS 模型的训练工具包。

架构总览 /
--------
Qwen3-TTS 教师蒸馏 → 文本前端（G2P 音素化）→ VITS GAN 训练 →
Piper 形状 ONNX 导出 → Android SDK（ONNX Runtime 1.22）部署。

数据流（标注对应模块）：

    text_generation.py        sample_generation.py       frontend/
    LLM 生成语料   ──►  Qwen3-TTS 蒸馏语音样本  ──►  音素化 + 契约冻结
                                                             │
       Android SDK 部署  ◄──  Piper 形状 ONNX 导出  ◄──  VITS GAN 训练
       (ORT 1.22)              vits/exporter.py          vits/trainer.py

新人入门路径 /
1. 先读 pipeline.py：理解 run_pipeline 的阶段编排
   （preflight → generate_texts → generate_samples → phonemize →
   validate → train → export）。
2. 再看 cli.py：约 20 个子命令如何映射到各模块入口。
3. 深入训练看 vits/trainer.py 的 train_vits，导出看
   vits/exporter.py 的 export_vits_onnx。

一条命令跑完全流程（entry point 定义见 pyproject.toml）：

    tts-trainer run-pipeline training_configs/xxx.json

English: Training toolkit for a compact multilingual TTS model —
Qwen3-TTS teacher distillation, G2P frontend, VITS GAN training,
Piper-shaped ONNX export, and Android deployment via ONNX Runtime 1.22.
"""

__version__ = "0.1.0"
