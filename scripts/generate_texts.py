"""生成多语言 TTS 训练文本的命令行入口。 / CLI entry point that generates multilingual TTS training text.

用法：python scripts/generate_texts.py --config training_configs/auto-text.example.json
Usage: python scripts/generate_texts.py --config training_configs/auto-text.example.json
"""
from __future__ import annotations

import argparse

from tts_trainer.logging_utils import configure_logging
from tts_trainer.text_generation import generate_texts


def main() -> int:
    """解析配置路径并触发生成流程。 / Parse the config path and kick off text generation."""
    configure_logging()
    parser = argparse.ArgumentParser(description="Generate multilingual TTS training text")
    parser.add_argument("--config", default="training_configs/auto-text.example.json")
    args = parser.parse_args()
    print(generate_texts(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
