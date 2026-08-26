"""控制台日志格式化与终端进度条工具。 / Console log formatting and terminal progress utilities."""
from __future__ import annotations

import logging
import os
import sys


# ANSI 控制码与各级别颜色。 / ANSI control codes and per-level colors.
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
COLORS = {
    "DEBUG": "\033[36m",
    "INFO": "\033[32m",
    "WARNING": "\033[33m",
    "ERROR": "\033[31m",
    "CRITICAL": "\033[1;31m",
}
SECTION_WIDTH = 78


def progress_bar(current: int, total: int, *, width: int = 24) -> str:
    """渲染固定宽度的文本进度条，日志与终端均可用。 / Render a fixed-width progress bar suitable for logs and terminals."""
    ratio = min(max(current / max(total, 1), 0.0), 1.0)
    completed = min(width, int(ratio * width))
    return "[" + "█" * completed + "░" * (width - completed) + "]"


class TerminalProgress:
    """单行实时进度，日志输出前自动清除以免串行。 / One-line live progress that disappears before normal log records."""

    def __init__(self, label: str, total: int, *, enabled: bool | None = None,
                 stream=None, width: int = 24):
        self.label = label
        self.total = max(int(total), 1)
        self.stream = stream or sys.stderr
        # auto 模式仅在 tty 上启用，重定向时不产生控制字符。 / auto enables only on a tty; redirected streams stay clean.
        configured = os.environ.get("TTS_TRAINER_LIVE_PROGRESS", "auto").lower()
        if enabled is None:
            enabled = configured not in {"0", "false", "no", "never", "off"}
            if configured == "auto":
                enabled = bool(getattr(self.stream, "isatty", lambda: False)())
        self.enabled = bool(enabled)
        self.width = width
        self.rendered_width = 0

    def update(self, current: int, detail: str = "") -> None:
        """原地刷新单行进度。 / Refresh the single progress line in place."""
        if not self.enabled:
            return
        current = min(max(int(current), 0), self.total)
        percent = 100.0 * current / self.total
        text = (
            f"{self.label} {progress_bar(current, self.total, width=self.width)} "
            f"{percent:6.2f}% {current}/{self.total}"
        )
        if detail:
            text += f" | {detail}"
        # 补空格抹掉上一帧残余字符。 / Pad to erase leftovers from the previous frame.
        padding = " " * max(0, self.rendered_width - len(text))
        self.stream.write("\r" + text + padding)
        self.stream.flush()
        self.rendered_width = len(text)

    def clear(self) -> None:
        """整行擦除当前进度。 / Erase the current progress line."""
        if not self.enabled or not self.rendered_width:
            return
        self.stream.write("\r" + " " * self.rendered_width + "\r")
        self.stream.flush()
        self.rendered_width = 0

    def close(self) -> None:
        """结束时清理进度行。 / Clean up the progress line on close."""
        self.clear()


def _numeric_level(value: str, field: str) -> int:
    """把级别名称解析为数字值。 / Parse a level name into its numeric value."""
    numeric = getattr(logging, str(value).upper(), None)
    if not isinstance(numeric, int):
        raise ValueError(f"invalid {field}: {value!r}")
    return numeric


def _color_enabled(value: str | bool | None) -> bool:
    """综合 NO_COLOR、环境变量与 tty 判定是否启用颜色。 / Decide color use from NO_COLOR, env vars, and tty."""
    # 遵循 no-color.org 约定，NO_COLOR 一票否决。 / Honor the no-color.org convention unconditionally.
    if "NO_COLOR" in os.environ:
        return False
    selected = os.environ.get("TTS_TRAINER_LOG_COLOR", str(value or "auto")).lower()
    if selected in {"1", "true", "yes", "always", "on"}:
        return True
    if selected in {"0", "false", "no", "never", "off"}:
        return False
    if selected != "auto":
        raise ValueError("logging.color must be auto, always or never")
    return bool(getattr(sys.stderr, "isatty", lambda: False)()) \
        and os.environ.get("TERM", "") != "dumb"


def format_duration(seconds: float) -> str:
    """把秒数格式化为 1h 02m 03s 样式。 / Format seconds as 1h 02m 03s style."""
    total = max(0, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


class ConsoleFormatter(logging.Formatter):
    """可读的终端格式化器，重定向输出时自动去 ANSI。 / Readable terminal formatter with ANSI disabled for redirected output."""

    def __init__(self, use_color: bool):
        super().__init__()
        self.use_color = use_color

    def _paint(self, text: str, color: str, *, bold: bool = False) -> str:
        if not self.use_color:
            return text
        weight = BOLD if bold else ""
        return f"{weight}{color}{text}{RESET}"

    @staticmethod
    def _logger_name(name: str) -> str:
        """压缩内部 logger 名以对齐列宽。 / Shorten internal logger names for alignment."""
        if name.startswith("tts_trainer."):
            return name.removeprefix("tts_trainer.")
        if name.startswith("qwen_tts."):
            return "qwen_tts"
        return name

    def format(self, record: logging.LogRecord) -> str:
        """按 tts_style 渲染分区/普通两类日志。 / Render section blocks or normal lines per tts_style."""
        message = record.getMessage()
        style = getattr(record, "tts_style", "")
        if style in {"section", "success_section"}:
            divider = "━" * SECTION_WIDTH
            block = f"\n{divider}\n{message}\n{divider}"
            color = "\033[32m" if style == "success_section" else "\033[36m"
            return self._paint(block, color, bold=True)

        timestamp = self.formatTime(record, "%H:%M:%S")
        level = f"{record.levelname:<8}"
        name = self._logger_name(record.name)
        if self.use_color:
            timestamp = f"{DIM}{timestamp}{RESET}"
            level = self._paint(level, COLORS.get(record.levelname, ""), bold=True)
            name = self._paint(name, "\033[36m")
            if style == "success":
                message = self._paint(message, "\033[32m", bold=True)
            elif style == "progress":
                message = self._paint(message, "\033[36m", bold=True)
        rendered = f"{timestamp} │ {level} │ {name} │ {message}"
        if record.exc_info:
            rendered += "\n" + self.formatException(record.exc_info)
        if record.stack_info:
            rendered += "\n" + self.formatStack(record.stack_info)
        return rendered


def log_section(logger: logging.Logger, title: str, detail: str | None = None,
                *, success: bool = False) -> None:
    """输出带分隔线的分区标题。 / Emit a divider-framed section title."""
    message = title if not detail else f"{title}\n{detail}"
    logger.info(
        message,
        extra={"tts_style": "success_section" if success else "section"},
    )


def configure_logging(level: str = "INFO", *, color: str | bool | None = "auto",
                      third_party_level: str = "WARNING") -> None:
    """初始化根日志器并压低第三方库噪声。 / Set up root logging and quiet third-party libraries."""
    # 环境变量优先于参数，便于 CI 覆盖。 / Env vars override arguments for CI use.
    selected_level = os.environ.get("TTS_TRAINER_LOG_LEVEL", level)
    numeric = _numeric_level(selected_level, "log level")
    selected_third_party = os.environ.get(
        "TTS_TRAINER_THIRD_PARTY_LOG_LEVEL", third_party_level,
    )
    third_party_numeric = _numeric_level(selected_third_party, "third-party log level")
    handler = logging.StreamHandler()
    handler.setFormatter(ConsoleFormatter(_color_enabled(color)))
    logging.basicConfig(level=numeric, handlers=[handler], force=True)
    for name in ("qwen_tts", "transformers", "huggingface_hub", "urllib3"):
        logging.getLogger(name).setLevel(third_party_numeric)


def configure_logging_from_config(config: dict) -> None:
    """从配置字典的 logging 段初始化日志。 / Configure logging from a config dict's logging section."""
    settings = config.get("logging", {})
    configure_logging(
        settings.get("level", "INFO"),
        color=settings.get("color", "auto"),
        third_party_level=settings.get("third_party_level", "WARNING"),
    )
