import logging

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


class AudioOutput:
    def __init__(self, default_sample_rate: int = 16000, device: int | None = None):
        self.default_sample_rate = default_sample_rate
        self.device = device

    def play(self, audio: np.ndarray, sample_rate: int | None = None):
        """Play audio array through speakers. Blocks until complete."""
        active_sample_rate = sample_rate or self.default_sample_rate
        logger.debug(
            "Playing %.2fs audio on device=%s at %s Hz",
            len(audio) / active_sample_rate,
            self.device,
            active_sample_rate,
        )
        sd.play(audio, samplerate=active_sample_rate, device=self.device)
        sd.wait()
