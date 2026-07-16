from __future__ import annotations

import argparse

from tts_trainer.pipeline import run_pipeline
from tts_trainer.logging_utils import configure_logging


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description="Run the configured TTS training pipeline")
    parser.add_argument("--config", default="training_configs/train1.json")
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    print(run_pipeline(args.config, max_steps=args.max_steps))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
