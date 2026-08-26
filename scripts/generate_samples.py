"""用 Qwen3-TTS 生成 VITS 训练数据集的命令行入口。 / CLI entry point that generates a VITS training dataset with Qwen3-TTS.

用法：python scripts/generate_samples.py --config training_configs/train1.json
Usage: python scripts/generate_samples.py --config training_configs/train1.json
"""
from __future__ import annotations

import argparse

from tts_trainer.sample_generation import generate_samples
from tts_trainer.logging_utils import configure_logging


def main() -> int:
    """解析配置路径并触发样本生成。 / Parse the config path and kick off sample generation."""
    configure_logging()
    parser = argparse.ArgumentParser(description="Generate a VITS training dataset with Qwen3-TTS")
    parser.add_argument("--config", default="training_configs/train1.json")
    args = parser.parse_args()
    print(generate_samples(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
