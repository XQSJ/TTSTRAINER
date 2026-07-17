from __future__ import annotations

import os
import signal
import subprocess
import sys
from collections.abc import Sequence


SUPERVISED_ENV = "TTS_TRAINER_SUPERVISED"
GRACE_SECONDS_ENV = "TTS_TRAINER_INTERRUPT_GRACE_SECONDS"
LONG_RUNNING_COMMANDS = {
    "generate-samples",
    "generate-texts",
    "run-pipeline",
    "train",
    "train-many",
    "train-vits",
}


def should_supervise(argv: Sequence[str]) -> bool:
    """Keep a small parent process responsive while native ML code is running."""
    return (
        bool(argv)
        and argv[0] in LONG_RUNNING_COMMANDS
        and os.environ.get(SUPERVISED_ENV) != "1"
    )


def _send_signal(process: subprocess.Popen, selected: signal.Signals) -> None:
    if process.poll() is not None:
        return
    if os.name == "posix":
        os.killpg(process.pid, selected)
    else:  # pragma: no cover - exercised on Windows runners only
        process.send_signal(selected)


def _exit_code(returncode: int) -> int:
    return 128 + abs(returncode) if returncode < 0 else returncode


def _force_stop(process: subprocess.Popen) -> int:
    if process.poll() is not None:
        return _exit_code(int(process.returncode))
    print(
        "\nINTERRUPT | forcing worker shutdown now",
        file=sys.stderr,
        flush=True,
    )
    _send_signal(process, signal.SIGTERM)
    try:
        process.wait(timeout=3)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        if process.poll() is None:
            _send_signal(process, signal.SIGKILL)
        process.wait()
    return 130


def run_supervised(argv: Sequence[str]) -> int:
    """Run a long command in a child so Ctrl+C works during CUDA/native calls."""
    environment = os.environ.copy()
    environment[SUPERVISED_ENV] = "1"
    process = subprocess.Popen(
        [sys.executable, "-m", "tts_trainer", *argv],
        env=environment,
        start_new_session=True,
    )
    try:
        return _exit_code(process.wait())
    except KeyboardInterrupt:
        grace_seconds = max(0.0, float(os.environ.get(GRACE_SECONDS_ENV, "10")))
        print(
            "\nINTERRUPT | Ctrl+C received | stopping safely; completed text/WAV/cache files are kept",
            file=sys.stderr,
            flush=True,
        )
        _send_signal(process, signal.SIGINT)
        try:
            process.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            print(
                f"INTERRUPT | worker did not stop within {grace_seconds:g}s; terminating stuck native work",
                file=sys.stderr,
                flush=True,
            )
            return _force_stop(process)
        except KeyboardInterrupt:
            return _force_stop(process)
        return 130
