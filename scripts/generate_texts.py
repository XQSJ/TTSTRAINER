from __future__ import annotations

import argparse

from tts_trainer.logging_utils import configure_logging
from tts_trainer.text_generation import generate_texts


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description="Generate multilingual TTS training text")
    parser.add_argument("--config", default="training_configs/auto-text.example.json")
    args = parser.parse_args()
    print(generate_texts(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
