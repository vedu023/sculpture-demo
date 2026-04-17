import logging

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


class AudioOutput:
    def __init__(
        self,
        default_sample_rate: int = 16000,
        device: int | None = None,
        volume: float = 1.0,
    ):
        self.default_sample_rate = default_sample_rate
        self.device = device
        self.volume = max(0.0, min(1.0, volume))

    def set_volume(self, volume: float):
        self.volume = max(0.0, min(1.0, volume))

    def get_volume_percent(self) -> int:
        return int(round(self.volume * 100))

    def _apply_volume(self, audio: np.ndarray) -> np.ndarray:
        if self.volume >= 0.999:
            return audio

        frames = np.asarray(audio)
        if np.issubdtype(frames.dtype, np.integer):
            max_value = float(np.iinfo(frames.dtype).max)
            scaled = (frames.astype(np.float32) / max_value) * self.volume
            return np.clip(scaled, -1.0, 1.0)

        scaled = frames.astype(np.float32, copy=False) * self.volume
        return np.clip(scaled, -1.0, 1.0)

    def play(self, audio: np.ndarray, sample_rate: int | None = None, block: bool = True):
        """Play audio array through speakers.

        Set `block=False` to return immediately while playback continues in the
        background.
        """
        active_sample_rate = sample_rate or self.default_sample_rate
        playback_audio = self._apply_volume(audio)
        logger.debug(
            "Playing %.2fs audio on device=%s at %s Hz volume=%d%%",
            len(playback_audio) / active_sample_rate,
            self.device,
            active_sample_rate,
            self.get_volume_percent(),
        )
        sd.play(playback_audio, samplerate=active_sample_rate, device=self.device)
        if block:
            sd.wait()
