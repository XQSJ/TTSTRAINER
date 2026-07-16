from __future__ import annotations

import logging
import os


def configure_logging(level: str = "INFO") -> None:
    level = os.environ.get("TTS_TRAINER_LOG_LEVEL", level)
    numeric = getattr(logging, str(level).upper(), None)
    if not isinstance(numeric, int):
        raise ValueError(f"invalid log level: {level!r}")
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
