from __future__ import annotations

import argparse

from tts_trainer.sample_generation import generate_samples


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a VITS training dataset with Qwen3-TTS")
    parser.add_argument("--config", default="training_configs/train1.json")
    args = parser.parse_args()
    print(generate_samples(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
