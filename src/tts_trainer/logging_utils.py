from __future__ import annotations

import logging
import os
import sys


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
    """Render a fixed-width progress bar suitable for logs and terminals."""
    ratio = min(max(current / max(total, 1), 0.0), 1.0)
    completed = min(width, int(ratio * width))
    return "[" + "█" * completed + "░" * (width - completed) + "]"


class TerminalProgress:
    """One-line live progress that disappears before normal log records."""

    def __init__(self, label: str, total: int, *, enabled: bool | None = None,
                 stream=None, width: int = 24):
        self.label = label
        self.total = max(int(total), 1)
        self.stream = stream or sys.stderr
        configured = os.environ.get("TTS_TRAINER_LIVE_PROGRESS", "auto").lower()
        if enabled is None:
            enabled = configured not in {"0", "false", "no", "never", "off"}
            if configured == "auto":
                enabled = bool(getattr(self.stream, "isatty", lambda: False)())
        self.enabled = bool(enabled)
        self.width = width
        self.rendered_width = 0

    def update(self, current: int, detail: str = "") -> None:
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
        padding = " " * max(0, self.rendered_width - len(text))
        self.stream.write("\r" + text + padding)
        self.stream.flush()
        self.rendered_width = len(text)

    def clear(self) -> None:
        if not self.enabled or not self.rendered_width:
            return
        self.stream.write("\r" + " " * self.rendered_width + "\r")
        self.stream.flush()
        self.rendered_width = 0

    def close(self) -> None:
        self.clear()


def _numeric_level(value: str, field: str) -> int:
    numeric = getattr(logging, str(value).upper(), None)
    if not isinstance(numeric, int):
        raise ValueError(f"invalid {field}: {value!r}")
    return numeric


def _color_enabled(value: str | bool | None) -> bool:
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
    total = max(0, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


class ConsoleFormatter(logging.Formatter):
    """Readable terminal formatter with ANSI disabled for redirected output."""

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
        if name.startswith("tts_trainer."):
            return name.removeprefix("tts_trainer.")
        if name.startswith("qwen_tts."):
            return "qwen_tts"
        return name

    def format(self, record: logging.LogRecord) -> str:
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
    message = title if not detail else f"{title}\n{detail}"
    logger.info(
        message,
        extra={"tts_style": "success_section" if success else "section"},
    )


def configure_logging(level: str = "INFO", *, color: str | bool | None = "auto",
                      third_party_level: str = "WARNING") -> None:
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
    settings = config.get("logging", {})
    configure_logging(
        settings.get("level", "INFO"),
        color=settings.get("color", "auto"),
        third_party_level=settings.get("third_party_level", "WARNING"),
    )
