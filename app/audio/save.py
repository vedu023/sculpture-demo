import logging
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

from app.types import CapturedUtterance

logger = logging.getLogger(__name__)


def save_utterance(
    utterance: CapturedUtterance | np.ndarray,
    sample_rate: int | None = None,
    output_dir: str | Path = "logs",
) -> Path:
    """Save int16 PCM audio to a timestamped WAV file. Returns the file path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"utterance_{timestamp}.wav"
    filepath = output_dir / filename

    if isinstance(utterance, CapturedUtterance):
        audio = utterance.samples
        active_sample_rate = utterance.sample_rate
    else:
        audio = utterance
        if sample_rate is None:
            raise ValueError("sample_rate is required when saving a raw numpy audio array")
        active_sample_rate = sample_rate

    with wave.open(str(filepath), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(active_sample_rate)
        wf.writeframes(audio.tobytes())

    duration = len(audio) / active_sample_rate
    logger.info("Saved %.1fs utterance to %s", duration, filepath)
    return filepath
