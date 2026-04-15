import logging

import numpy as np
from silero_vad_lite import SileroVAD

logger = logging.getLogger(__name__)


class VAD:
    """Voice Activity Detection using Silero VAD (ONNX-lite)."""

    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        min_speech_ms: int = 250,
        min_silence_ms: int = 500,
    ):
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_speech_ms = min_speech_ms
        self.min_silence_ms = min_silence_ms
        self._model_path = SileroVAD._get_model_path()
        self._model = SileroVAD(sample_rate)
        logger.debug("Silero VAD loaded (sample_rate=%d, threshold=%.2f)", sample_rate, threshold)

    def get_speech_prob(self, audio_chunk: np.ndarray) -> float:
        """Return speech probability (0.0–1.0) for a 512-sample int16 chunk."""
        float_audio = (audio_chunk.astype(np.float32) / 32768.0).tobytes()
        return self._model.process(float_audio)

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """Return True if the chunk is classified as speech."""
        return self.get_speech_prob(audio_chunk) >= self.threshold

    def reset(self):
        """Reset internal recurrent state between utterances."""
        self._model._lib.SileroVAD_delete(self._model._obj)
        self._model._obj = self._model._lib.SileroVAD_new(
            self._model_path.encode("utf-8"),
            self.sample_rate,
        )
