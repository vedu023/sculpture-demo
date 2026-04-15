import logging

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


class AudioInput:
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        block_size: int = 512,
        device: int | None = None,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.block_size = block_size
        self.device = device

    def record(self, duration: float) -> np.ndarray:
        """Record audio for a fixed duration. Returns int16 numpy array."""
        logger.debug("Recording %.1fs from device=%s", duration, self.device)
        audio = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            device=self.device,
        )
        sd.wait()
        return audio.flatten()

    def create_stream(self, callback) -> sd.InputStream:
        """Create a streaming input for continuous capture."""
        return sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            blocksize=self.block_size,
            device=self.device,
            callback=callback,
        )
