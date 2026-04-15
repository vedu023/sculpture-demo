import logging
import os
import sys
import warnings


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

    if debug:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    else:
        formatter = logging.Formatter("%(message)s")

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
