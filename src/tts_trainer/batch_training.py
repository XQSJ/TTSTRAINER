"""批量并行跑多个实验训练任务。 / Run multiple experiment training jobs in parallel."""
from __future__ import annotations

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .experiments import resolve_experiment


def train_many(config_paths: list[str], *, max_parallel: int = 1,
               max_steps: int | None = None) -> list[str]:
    """并行训练多份配置对应的模型。 / Train models for multiple configs in parallel."""
    if max_parallel < 1:
        raise ValueError("max_parallel must be at least 1")
    names = [resolve_experiment(path)[1].name for path in config_paths]
    # 模型重名会在输出目录互相覆盖，必须在启动前拒绝。 / Duplicate names would overwrite each other's outputs.
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"duplicate model names in batch: {', '.join(duplicates)}")

    def run(config_path: str) -> str:
        """以子进程运行单个 train-vits 任务。 / Run one train-vits job in a subprocess."""
        command = [sys.executable, "-m", "tts_trainer", "train-vits", "--config", config_path]
        if max_steps is not None:
            command.extend(("--max-steps", str(max_steps)))
        subprocess.run(command, check=True, env=os.environ.copy())
        return str(Path(config_path))

    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        return list(executor.map(run, config_paths))
