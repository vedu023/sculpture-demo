import logging
import os
import sys
from datetime import datetime
import warnings


class _PrettyFormatter(logging.Formatter):
    _DATEFMT = "%H:%M:%S"
    _NAME_WIDTH = 15
    _LEVEL_BADGES = {
        logging.DEBUG: "DEBUG",
        logging.INFO: "INFO",
        logging.WARNING: "WARN",
        logging.ERROR: "ERROR",
        logging.CRITICAL: "CRIT",
    }
    _COLORS = {
        logging.DEBUG: "\033[36m",  # cyan
        logging.INFO: "\033[92m",  # green
        logging.WARNING: "\033[93m",  # yellow
        logging.ERROR: "\033[91m",  # red
        logging.CRITICAL: "\033[91m\033[1m",  # bold red
    }
    _RESET = "\033[0m"

    def __init__(self, debug: bool = False):
        self._debug = debug
        self._enable_color = self._supports_color()
        super().__init__()

    def _supports_color(self) -> bool:
        stream = getattr(sys.stdout, "isatty", lambda: False)()
        if not stream:
            return False
        if os.environ.get("NO_COLOR"):
            return False
        return True

    def _short_name(self, name: str) -> str:
        if not name:
            return "root"
        return name.rsplit(".", 1)[-1]

    def _format_level(self, level: int) -> str:
        badge = self._LEVEL_BADGES.get(level, str(level))
        if self._enable_color:
            color = self._COLORS.get(level, "")
            return f"{color}{badge:<5}{self._RESET}"
        return f"{badge:<5}"

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created).strftime(self._DATEFMT)
        level = self._format_level(record.levelno)
        logger_name = f"{self._short_name(record.name):<{self._NAME_WIDTH}}"
        message = record.getMessage()

        if self._debug:
            prefix = f"{timestamp} | {level} | {logger_name}"
            return f"{prefix} | {message}"
        return f"{timestamp} {level} {logger_name}: {message}"


def setup_logging(debug: bool = False):
    """Configure root logger once for concise demo output.

    Safe to call multiple times.
    """
    root = logging.getLogger()
    level = logging.DEBUG if debug else logging.INFO
    root.setLevel(level)
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    for handler in list(root.handlers):
        if getattr(handler, "_voicebot_handler", False):
            root.removeHandler(handler)

    formatter = _PrettyFormatter(debug=debug)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler._voicebot_handler = True  # type: ignore[attr-defined]
    root.addHandler(handler)

    quiet_level = logging.DEBUG if debug else logging.WARNING
    for logger_name in (
        "faster_whisper",
        "ollama",
        "sounddevice",
        "TTS",
        "transformers",
        "huggingface_hub",
        "httpx",
        "httpcore",
        "urllib3",
    ):
        logging.getLogger(logger_name).setLevel(quiet_level)

    warnings.filterwarnings(
        "ignore",
        message="pkg_resources is deprecated as an API.*",
        category=UserWarning,
    )
