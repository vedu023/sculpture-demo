from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from app.tts.voice_ref import should_use_cached_voice_state, voice_state_is_stale
from app.types import SynthesizedAudio

logger = logging.getLogger(__name__)


class PocketTTSBackend:
    backend_name = "pocket_tts"

    def __init__(
        self,
        voice: str = "alba",
        speed: float = 1.0,
        audio_prompt_path: Path | None = None,
        voice_state_path: Path | None = None,
    ):
        self.voice = voice
        self.speed = speed
        self._model = None
        self._voice_state = None
        self._sample_rate = 24000
        self._audio_prompt_path = audio_prompt_path
        self._voice_state_path = voice_state_path

    def warmup(self):
        if self._model is not None:
            return
        try:
            from pocket_tts import TTSModel
        except ImportError as exc:
            raise RuntimeError(
                "pocket-tts is not installed. Run: uv add pocket-tts"
            ) from exc

        logger.info("Loading Pocket TTS model")
        model = TTSModel.load_model()

        use_cached_state = should_use_cached_voice_state(
            self._audio_prompt_path,
            self._voice_state_path,
        )

        # Priority: fresh cached safetensors > updated audio prompt > default voice
        if use_cached_state and self._voice_state_path and self._voice_state_path.exists():
            logger.info("Loading cached voice state from: %s", self._voice_state_path)
            voice_state = model.get_state_for_audio_prompt(
                audio_conditioning=str(self._voice_state_path)
            )
        elif self._audio_prompt_path and self._audio_prompt_path.exists():
            if voice_state_is_stale(self._audio_prompt_path, self._voice_state_path):
                logger.info(
                    "Voice state cache is stale relative to %s; using the updated WAV until you rerun prepare-voice.",
                    self._audio_prompt_path,
                )
            logger.info("Loading custom voice from: %s", self._audio_prompt_path)
            voice_state = model.get_state_for_audio_prompt(
                audio_conditioning=str(self._audio_prompt_path)
            )
        else:
            logger.info("Using default voice: %s", self.voice)
            voice_state = model.get_state_for_audio_prompt(self.voice)

        self._model = model
        self._voice_state = voice_state
        self._sample_rate = int(getattr(model, "sample_rate", 24000))
        logger.info("Pocket TTS ready (%s)", self.describe())

    def synthesize(self, text: str) -> SynthesizedAudio:
        self.warmup()
        text = text.strip()
        if not text:
            return SynthesizedAudio(
                samples=np.zeros(0, dtype=np.float32),
                sample_rate=self._sample_rate,
                duration_ms=0.0,
            )

        try:
            audio_tensor = self._model.generate_audio(self._voice_state, text)
            samples = np.asarray(audio_tensor, dtype=np.float32).reshape(-1)
        except Exception as exc:
            logger.error("Pocket TTS synthesis failed: %s", exc)
            return SynthesizedAudio(
                samples=np.zeros(0, dtype=np.float32),
                sample_rate=self._sample_rate,
                duration_ms=0.0,
            )

        if self.speed > 0 and self.speed != 1.0:
            logger.debug("Ignoring unsupported Pocket TTS speed=%.2f", self.speed)

        duration_ms = (samples.size / self._sample_rate) * 1000.0
        return SynthesizedAudio(
            samples=samples,
            sample_rate=self._sample_rate,
            duration_ms=duration_ms,
        )

    def describe(self) -> str:
        if should_use_cached_voice_state(self._audio_prompt_path, self._voice_state_path):
            return f"{self.backend_name}/{self._voice_state_path.stem}"
        if self._audio_prompt_path and self._audio_prompt_path.exists():
            return f"{self.backend_name}/{self._audio_prompt_path.stem}"
        return f"{self.backend_name}/{self.voice}"
