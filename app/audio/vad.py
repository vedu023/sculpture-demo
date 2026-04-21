import logging

import numpy as np
from silero_vad_lite import SileroVAD

logger = logging.getLogger(__name__)


class VAD:
    """Voice Activity Detection using Silero VAD (ONNX-lite)."""
    _window_size_samples = 512

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
        self.window_size_samples = self._window_size_samples
        logger.debug(
            "Silero VAD loaded (sample_rate=%d, threshold=%.2f, window=%d)",
            sample_rate,
            threshold,
            self.window_size_samples,
        )

    def get_speech_prob(self, audio_chunk: np.ndarray) -> float:
        """Return speech probability (0.0–1.0) for an audio chunk."""
        if np.issubdtype(audio_chunk.dtype, np.floating):
            float_audio = np.asarray(audio_chunk, dtype=np.float32)
        else:
            float_audio = audio_chunk.astype(np.float32) / 32768.0
        float_audio = np.ascontiguousarray(float_audio)
        num_samples = float_audio.size
        if num_samples == self.window_size_samples:
            return self._model.process(float_audio.tobytes())
        if num_samples == 0:
            return 0.0

        if num_samples < self.window_size_samples:
            padded = np.zeros(self.window_size_samples, dtype=np.float32)
            padded[:num_samples] = float_audio[: self.window_size_samples]
            return self._model.process(padded.tobytes())

        probs = []
        chunk_count = num_samples // self.window_size_samples
        for index in range(chunk_count):
            offset = index * self.window_size_samples
            segment = float_audio[offset : offset + self.window_size_samples]
            probs.append(self._model.process(segment.tobytes()))

        remainder = num_samples % self.window_size_samples
        if remainder:
            padded = np.zeros(self.window_size_samples, dtype=np.float32)
            padded[:remainder] = float_audio[-remainder:]
            probs.append(self._model.process(padded.tobytes()))

        return float(np.max(probs)) if probs else 0.0

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
