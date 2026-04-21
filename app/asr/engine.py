from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from app.asr.vyasa_onnx import VyasaOnnxRuntime
import numpy as np
from faster_whisper import WhisperModel

from app.language import detect_text_language, normalize_language_mode


logger = logging.getLogger(__name__)
_INDIC_LANGUAGE_CODES = {"hi", "kn"}


@dataclass
class _BasicInfo:
    language: str | None = None
    language_probability: float = 0.0
    duration: float = 0.0


class ASREngine:
    """ASR using faster-whisper or a direct ONNX runtime bundle."""

    def __init__(
        self,
        backend: str = "auto",
        model_name: str = "small.en",
        indic_model_name: str = "somyalab/Vyasa_mini_rnnt_onnx_v2",
        language_mode: str = "auto",
        device: str = "cpu",
        compute_type: str = "float32",
        beam_size: int = 2,
        fallback_beam_size: int = 2,
        language: str = "en",
        condition_on_previous_text: bool = False,
    ):
        self.backend = backend
        self.model_name = model_name
        self.indic_model_name = indic_model_name
        self.language_mode = normalize_language_mode(language_mode)
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.fallback_beam_size = fallback_beam_size
        self.language = language
        self.condition_on_previous_text = condition_on_previous_text
        self._model: Any | None = None
        self._active_backend: str | None = None

    def warmup(self):
        """Load the model and run a dummy transcription to warm up."""
        if self._model is not None:
            return
        active_backend = self._resolve_backend()
        active_model_name = self._resolve_model_name(active_backend)
        logger.info(
            "Loading ASR model: %s (backend=%s, device=%s, compute=%s)",
            active_model_name,
            active_backend,
            self.device,
            self.compute_type,
        )
        if active_backend == "faster_whisper":
            self._model = self._load_whisper_model(active_model_name)
        elif active_backend == "onnx_runtime":
            runtime = VyasaOnnxRuntime(
                repo_id=active_model_name,
                device=self.device,
            )
            runtime.warmup()
            self._model = runtime
        elif active_backend == "hybrid_auto":
            whisper_model = self._load_whisper_model(self.model_name)
            runtime = VyasaOnnxRuntime(
                repo_id=self.indic_model_name,
                device=self.device,
            )
            runtime.warmup()
            self._model = {
                "faster_whisper": whisper_model,
                "onnx_runtime": runtime,
            }
        else:
            raise ValueError(f"Unsupported ASR backend: {active_backend}")
        self._active_backend = active_backend
        logger.info("ASR model ready")

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe waveform audio to text. Returns empty string on failure."""
        self.warmup()
        audio = self._prepare_audio(audio)
        if audio.size == 0:
            logger.debug("ASR received empty audio buffer")
            return ""

        text, segments_list, info = self._run_transcribe(audio)
        info_language = getattr(info, "language", None)
        info_language_probability = getattr(info, "language_probability", 0.0)
        segment_count = len(segments_list)
        logger.debug(
            "ASR result: lang=%s (%.2f%%) segments=%d duration=%.2fs text=%r",
            info_language,
            info_language_probability * 100.0,
            segment_count,
            getattr(info, "duration", 0.0),
            text,
        )
        if segments_list:
            first_segment = segments_list[0]
            last_segment = segments_list[-1]
            logger.debug(
                "ASR timing: start=%.2fs end=%.2fs first=%r last=%r",
                float(first_segment.start),
                float(last_segment.end),
                first_segment.text.strip(),
                last_segment.text.strip(),
            )

        return text

    def _prepare_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio into a float32 [-1, 1] range for ASR backends."""
        if not np.issubdtype(audio.dtype, np.floating):
            float_audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype != np.float32:
            float_audio = audio.astype(np.float32, copy=False)
        else:
            float_audio = np.asarray(audio, dtype=np.float32)

        if not np.isfinite(float_audio).all():
            logger.debug("ASR input contained invalid values; replacing NaN/inf with 0")
            float_audio = np.nan_to_num(float_audio, nan=0.0, posinf=0.0, neginf=0.0)

        max_abs = float(np.max(np.abs(float_audio))) if float_audio.size else 0.0
        if max_abs > 1.0:
            logger.debug("ASR input clipped at %.5f; normalizing", max_abs)
            float_audio = float_audio / max_abs

        if float_audio.size:
            logger.debug(
                "ASR audio stats: dtype=%s n=%d peak=%.6f rms=%.6f",
                float_audio.dtype,
                float_audio.size,
                float(np.max(np.abs(float_audio))),
                float(np.sqrt(np.mean(float_audio * float_audio))),
            )

        return float_audio

    def _decode(
        self,
        audio: np.ndarray,
        beam_size: int,
        model: Any | None = None,
    ):
        model = model or self._model
        segments, info = model.transcribe(
            audio,
            beam_size=beam_size,
            language=self._resolve_whisper_language(),
            condition_on_previous_text=self.condition_on_previous_text,
            vad_filter=False,
        )
        return segments, info

    def _resolve_backend(self) -> str:
        backend = (self.backend or "auto").strip().lower()
        if backend == "auto":
            if self.language_mode == "english":
                return "faster_whisper"
            if self.language_mode == "indic":
                if self.indic_model_name:
                    return "onnx_runtime"
                return "faster_whisper"
            language = self._normalized_language()
            if language is None and self.indic_model_name:
                return "hybrid_auto"
            if language in _INDIC_LANGUAGE_CODES and self.indic_model_name:
                return "onnx_runtime"
            return "faster_whisper"
        if backend == "onnx_asr":
            return "onnx_runtime"
        if backend not in {"faster_whisper", "onnx_runtime"}:
            raise ValueError(f"Unsupported ASR backend: {backend}")
        return backend

    def _resolve_model_name(self, backend: str) -> str:
        if backend == "hybrid_auto":
            return f"{self.indic_model_name} + {self.model_name}"
        if backend == "onnx_runtime":
            return self.indic_model_name or self.model_name
        return self.model_name

    def _normalized_language(self) -> str | None:
        language = (self.language or "").strip().lower()
        if not language or language == "auto":
            return None
        return language

    def _resolve_whisper_language(self) -> str | None:
        language = self._normalized_language()
        if language is not None:
            return language
        if self.language_mode == "english":
            return "en"
        return None

    def _run_onnx_transcribe(
        self,
        audio: np.ndarray,
    ) -> tuple[str, list[Any], object]:
        text = self._model.transcribe(audio)
        info = _BasicInfo(
            language=self._normalized_language(),
            duration=float(audio.size) / 16000.0,
        )
        return text, [], info

    def _run_transcribe(
        self,
        audio: np.ndarray,
    ) -> tuple[str, list[Any], object]:
        if self._active_backend == "onnx_runtime":
            return self._run_onnx_transcribe(audio)
        if self._active_backend == "hybrid_auto":
            return self._run_hybrid_transcribe(audio)

        segments, info = self._decode(audio, beam_size=self.beam_size)
        segments_list = list(segments)
        text = " ".join(seg.text.strip() for seg in segments_list).strip()

        if not text and self.fallback_beam_size > self.beam_size:
            logger.debug(
                "ASR empty on beam=%d; retrying with beam=%d",
                self.beam_size,
                self.fallback_beam_size,
            )
            fallback_segments, info = self._decode(audio, beam_size=self.fallback_beam_size)
            segments_list = list(fallback_segments)
            text = " ".join(seg.text.strip() for seg in segments_list).strip()

            if text:
                logger.debug("ASR retry produced non-empty transcript")

        return text, segments_list, info

    def _load_whisper_model(self, model_name: str):
        model = WhisperModel(
            model_name,
            device=self.device,
            compute_type=self.compute_type,
        )
        silence = np.zeros(8000, dtype=np.float32)
        segments, _ = model.transcribe(
            silence,
            beam_size=self.beam_size,
            language=self._resolve_whisper_language(),
            condition_on_previous_text=self.condition_on_previous_text,
            vad_filter=False,
        )
        list(segments)
        return model

    def _run_hybrid_transcribe(
        self,
        audio: np.ndarray,
    ) -> tuple[str, list[Any], object]:
        if not isinstance(self._model, dict):
            raise RuntimeError("Hybrid ASR expected both Whisper and ONNX models to be loaded.")

        indic_model = self._model["onnx_runtime"]
        indic_text = indic_model.transcribe(audio).strip()
        indic_language = detect_text_language(indic_text, default="en")
        if indic_text and indic_language in _INDIC_LANGUAGE_CODES:
            info = _BasicInfo(
                language=indic_language,
                duration=float(audio.size) / 16000.0,
            )
            return indic_text, [], info

        whisper_model = self._model["faster_whisper"]
        segments, info = self._decode(audio, beam_size=self.beam_size, model=whisper_model)
        segments_list = list(segments)
        text = " ".join(seg.text.strip() for seg in segments_list).strip()
        return text, segments_list, info
