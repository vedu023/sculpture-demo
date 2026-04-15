import logging

import numpy as np
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


class ASREngine:
    """ASR using faster-whisper (CTranslate2 backend)."""

    def __init__(
        self,
        model_name: str = "small.en",
        device: str = "cpu",
        compute_type: str = "int8",
        beam_size: int = 1,
        language: str = "en",
        condition_on_previous_text: bool = False,
    ):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.language = language
        self.condition_on_previous_text = condition_on_previous_text
        self._model: WhisperModel | None = None

    def warmup(self):
        """Load the model and run a dummy transcription to warm up."""
        if self._model is not None:
            return
        logger.info(
            "Loading ASR model: %s (device=%s, compute=%s)",
            self.model_name,
            self.device,
            self.compute_type,
        )
        self._model = WhisperModel(
            self.model_name,
            device=self.device,
            compute_type=self.compute_type,
        )
        # Warm up with silence
        silence = np.zeros(8000, dtype=np.float32)
        segments, _ = self._model.transcribe(
            silence,
            beam_size=self.beam_size,
            language=self.language,
            condition_on_previous_text=self.condition_on_previous_text,
            vad_filter=False,
        )
        list(segments)
        logger.info("ASR model ready")

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe int16 audio to text. Returns empty string on failure."""
        self.warmup()

        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0

        segments, info = self._model.transcribe(
            audio,
            beam_size=self.beam_size,
            language=self.language,
            condition_on_previous_text=self.condition_on_previous_text,
            vad_filter=False,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()

        logger.debug("ASR [%.0f%% %s]: '%s'", info.language_probability * 100, info.language, text)
        return text
