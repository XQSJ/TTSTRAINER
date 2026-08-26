"""长任务的守护进程与 Ctrl+C 优雅停机。 / Supervisor for long tasks and graceful Ctrl+C shutdown."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
from collections.abc import Sequence


SUPERVISED_ENV = "TTS_TRAINER_SUPERVISED"
GRACE_SECONDS_ENV = "TTS_TRAINER_INTERRUPT_GRACE_SECONDS"
# 需要守护的长命令集合。 / Long commands that run under supervision.
LONG_RUNNING_COMMANDS = {
    "generate-samples",
    "generate-texts",
    "run-pipeline",
    "train",
    "train-many",
    "train-vits",
}


def should_supervise(argv: Sequence[str]) -> bool:
    """让轻量父进程保持可响应，底层原生 ML 代码不被信号阻塞。 / Keep a small parent process responsive while native ML code is running."""
    return (
        bool(argv)
        and argv[0] in LONG_RUNNING_COMMANDS
        and os.environ.get(SUPERVISED_ENV) != "1"
    )


def _send_signal(process: subprocess.Popen, selected: signal.Signals) -> None:
    """向子进程整个进程组发信号。 / Signal the child's whole process group."""
    if process.poll() is not None:
        return
    # 子进程独立会话，需按进程组杀才能覆盖其派生的原生线程。 / The child owns a session; kill by group to reach its native threads.
    if os.name == "posix":
        os.killpg(process.pid, selected)
    else:  # pragma: no cover - exercised on Windows runners only
        process.send_signal(selected)


def _exit_code(returncode: int) -> int:
    return 128 + abs(returncode) if returncode < 0 else returncode


def _force_stop(process: subprocess.Popen) -> int:
    """强制终止卡死的子进程并返回中断码。 / Force-kill a stuck child and return the interrupt code."""
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
    """在子进程中运行长命令，使 Ctrl+C 在 CUDA/原生调用期间仍然生效。 / Run a long command in a child so Ctrl+C works during CUDA/native calls."""
    environment = os.environ.copy()
    # 标记子进程，防止它再次自我守护。 / Mark the child so it does not re-supervise itself.
    environment[SUPERVISED_ENV] = "1"
    process = subprocess.Popen(
        [sys.executable, "-m", "tts_trainer", *argv],
        env=environment,
        # 独立会话让父进程可以按进程组整组发信号。 / Own session lets the parent signal the whole group.
        start_new_session=True,
    )
    try:
        return _exit_code(process.wait())
    except KeyboardInterrupt:
        # 先给宽限期让 worker 落盘成果，超时才强杀。 / Grant a grace period to flush results before forcing.
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
