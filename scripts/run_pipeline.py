"""运行配置化 TTS 训练管线的命令行入口。 / CLI entry point that runs the configured TTS training pipeline.

用法：python scripts/run_pipeline.py --config training_configs/train1.json [--max-steps N]
Usage: python scripts/run_pipeline.py --config training_configs/train1.json [--max-steps N]
"""
from __future__ import annotations

import argparse

from tts_trainer.pipeline import run_pipeline
from tts_trainer.logging_utils import configure_logging


def main() -> int:
    """解析参数并启动完整训练管线。 / Parse arguments and start the full training pipeline."""
    configure_logging()
    parser = argparse.ArgumentParser(description="Run the configured TTS training pipeline")
    parser.add_argument("--config", default="training_configs/train1.json")
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    print(run_pipeline(args.config, max_steps=args.max_steps))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
