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
    """TTS engine with Pocket-TTS and Spark Somya backends."""

    def __init__(
        self,
        backend: str = "auto",
        voice: str = "alba",
        speed: float = 1.0,
        device: str = "cpu",
        audio_prompt_path: Path | None = None,
        voice_state_path: Path | None = None,
        spark_model_dir: Path | None = None,
        spark_repo_id: str = "somyalab/Spark_somya_TTS",
        spark_temperature: float = 0.7,
        spark_top_k: int = 50,
        spark_top_p: float = 0.95,
    ):
        if backend not in {"auto", "pocket_tts", "spark_somya_tts"}:
            raise RuntimeError(f"Unsupported TTS backend: {backend}")
        self.backend = backend
        self._voice = voice
        self._speed = speed
        self._device = device
        self._audio_prompt_path = audio_prompt_path
        self._voice_state_path = voice_state_path
        self._spark_model_dir = spark_model_dir
        self._spark_repo_id = spark_repo_id
        self._spark_temperature = spark_temperature
        self._spark_top_k = spark_top_k
        self._spark_top_p = spark_top_p
        self._backends: dict[str, _BaseTTSBackend] = {}

    @property
    def backend_name(self) -> str:
        if self.backend == "auto":
            return "auto"
        return self._get_backend(self.backend).backend_name

    def describe(self, language: str | None = None) -> str:
        if self.backend == "auto":
            if language in {"hi", "kn"}:
                return self._get_backend("spark_somya_tts").describe()
            return f"auto({self._get_backend('pocket_tts').describe()}, {self._spark_description()})"
        return self._get_backend(self.backend).describe()

    def warmup(self):
        if self.backend == "auto":
            self._get_backend("pocket_tts").warmup()
            return
        self._get_backend(self.backend).warmup()

    def synthesize(self, text: str, language: str = "en") -> SynthesizedAudio:
        backend = self._select_backend(language)
        return backend.synthesize(text)

    def _select_backend(self, language: str) -> _BaseTTSBackend:
        if self.backend == "spark_somya_tts":
            return self._get_backend("spark_somya_tts")
        if self.backend == "pocket_tts":
            return self._get_backend("pocket_tts")
        if language in {"hi", "kn"}:
            return self._get_backend("spark_somya_tts")
        return self._get_backend("pocket_tts")

    def _get_backend(self, backend_name: str) -> _BaseTTSBackend:
        backend = self._backends.get(backend_name)
        if backend is not None:
            return backend

        if backend_name == "pocket_tts":
            from app.tts.pocket_tts import PocketTTSBackend

            backend = PocketTTSBackend(
                voice=self._voice,
                speed=self._speed,
                audio_prompt_path=self._audio_prompt_path,
                voice_state_path=self._voice_state_path,
            )
        elif backend_name == "spark_somya_tts":
            from app.tts.spark_somya import SparkSomyaTTSBackend

            if self._spark_model_dir is None:
                raise RuntimeError("Spark Somya TTS requires a model directory.")
            backend = SparkSomyaTTSBackend(
                model_dir=self._spark_model_dir,
                repo_id=self._spark_repo_id,
                device=self._device,
                audio_prompt_path=self._audio_prompt_path,
                temperature=self._spark_temperature,
                top_k=self._spark_top_k,
                top_p=self._spark_top_p,
            )
        else:
            raise RuntimeError(f"Unsupported TTS backend: {backend_name}")

        self._backends[backend_name] = backend
        return backend

    def _spark_description(self) -> str:
        if self._spark_model_dir is None:
            return "spark_somya_tts"
        return f"spark_somya_tts/{self._spark_model_dir.name}"
