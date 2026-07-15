from __future__ import annotations

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .experiments import resolve_experiment


def train_many(config_paths: list[str], *, max_parallel: int = 1,
               max_steps: int | None = None) -> list[str]:
    if max_parallel < 1:
        raise ValueError("max_parallel must be at least 1")
    names = [resolve_experiment(path)[1].name for path in config_paths]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"duplicate model names in batch: {', '.join(duplicates)}")

    def run(config_path: str) -> str:
        command = [sys.executable, "-m", "tts_trainer", "train-vits", "--config", config_path]
        if max_steps is not None:
            command.extend(("--max-steps", str(max_steps)))
        subprocess.run(command, check=True, env=os.environ.copy())
        return str(Path(config_path))

    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        return list(executor.map(run, config_paths))
