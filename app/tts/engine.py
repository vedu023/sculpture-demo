from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from app.types import SynthesizedAudio

logger = logging.getLogger(__name__)


class _BaseTTSBackend:
    backend_name = "unknown"

    def warmup(self):
        raise NotImplementedError

    def synthesize(self, text: str) -> SynthesizedAudio:
        raise NotImplementedError

    def describe(self) -> str:
        return self.backend_name


class TTSEngine:
    """TTS engine using Pocket-TTS as the backend."""

    def __init__(
        self,
        backend: str = "pocket_tts",
        voice: str = "alba",
        speed: float = 1.0,
        audio_prompt_path: Path | None = None,
        voice_state_path: Path | None = None,
    ):
        if backend == "pocket_tts":
            from app.tts.pocket_tts import PocketTTSBackend
            self._backend = PocketTTSBackend(
                voice=voice, speed=speed,
                audio_prompt_path=audio_prompt_path,
                voice_state_path=voice_state_path,
            )
        else:
            raise RuntimeError(f"Unsupported TTS backend: {backend}")

    @property
    def backend_name(self) -> str:
        return self._backend.backend_name

    def describe(self) -> str:
        return self._backend.describe()

    def warmup(self):
        self._backend.warmup()

    def synthesize(self, text: str) -> SynthesizedAudio:
        return self._backend.synthesize(text)
