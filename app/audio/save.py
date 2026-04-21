import logging
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

from app.types import CapturedUtterance

logger = logging.getLogger(__name__)


def _to_pcm16(audio: np.ndarray) -> np.ndarray:
    """Convert float/int audio into int16 PCM suitable for 16-bit WAV."""
    if np.issubdtype(audio.dtype, np.integer):
        return np.asarray(audio, dtype=np.int16)

    if not np.issubdtype(audio.dtype, np.floating):
        audio = np.asarray(audio, dtype=np.float32)
    else:
        audio = np.asarray(audio, dtype=np.float32)

    if not np.isfinite(audio).all():
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

    max_abs = np.max(np.abs(audio))
    if max_abs > 1.0:
        audio = audio / max_abs
    return (np.clip(audio, -1.0, 1.0) * np.float32(32767.0)).astype(np.int16)


def save_utterance(
    utterance: CapturedUtterance | np.ndarray,
    sample_rate: int | None = None,
    output_dir: str | Path = "logs",
) -> Path:
    """Save captured utterance audio as 16-bit WAV. Returns the file path."""
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

    pcm = _to_pcm16(audio)
    with wave.open(str(filepath), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(active_sample_rate)
        wf.writeframes(pcm.tobytes())

    duration = len(audio) / active_sample_rate
    logger.info("Saved %.1fs utterance to %s", duration, filepath)
    return filepath
