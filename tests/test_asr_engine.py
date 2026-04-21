from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from app.asr.engine import ASREngine


class FakeWhisperModel:
    instances: list["FakeWhisperModel"] = []

    def __init__(self, model_name: str, device: str, compute_type: str):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.transcribe_calls: list[dict[str, object]] = []
        self.__class__.instances.append(self)

    def transcribe(
        self,
        audio,
        *,
        beam_size,
        language,
        condition_on_previous_text,
        vad_filter,
    ):
        self.transcribe_calls.append(
            {
                "beam_size": beam_size,
                "language": language,
                "condition_on_previous_text": condition_on_previous_text,
                "vad_filter": vad_filter,
                "samples": len(audio),
            }
        )
        segments = [type("Segment", (), {"start": 0.0, "end": 0.1, "text": "hello world"})()]
        info = type("Info", (), {"language": language or "auto", "language_probability": 0.99, "duration": 0.1})()
        return segments, info


class FakeVyasaRuntime:
    instances: list["FakeVyasaRuntime"] = []
    next_text = "namaskara"

    def __init__(self, repo_id: str, *, device: str = "cpu"):
        self.repo_id = repo_id
        self.device = device
        self.warmed = False
        self.calls: list[np.ndarray] = []
        self.__class__.instances.append(self)

    def warmup(self):
        self.warmed = True

    def transcribe(self, audio: np.ndarray) -> str:
        self.calls.append(audio)
        return self.__class__.next_text


class ASREngineTests(unittest.TestCase):
    def setUp(self):
        FakeWhisperModel.instances.clear()
        FakeVyasaRuntime.instances.clear()
        FakeVyasaRuntime.next_text = "namaskara"

    def test_auto_backend_uses_whisper_for_english(self):
        with patch("app.asr.engine.WhisperModel", FakeWhisperModel):
            engine = ASREngine(backend="auto", model_name="small.en", language="en")
            transcript = engine.transcribe(np.ones(1600, dtype=np.int16))

        self.assertEqual(transcript, "hello world")
        self.assertEqual(engine._active_backend, "faster_whisper")
        self.assertEqual(FakeWhisperModel.instances[0].model_name, "small.en")
        self.assertEqual(FakeWhisperModel.instances[0].transcribe_calls[-1]["language"], "en")

    def test_auto_backend_uses_onnx_runtime_for_hindi(self):
        with patch("app.asr.engine.VyasaOnnxRuntime", FakeVyasaRuntime):
            engine = ASREngine(
                backend="auto",
                language="hi",
                indic_model_name="somyalab/Vyasa_mini_rnnt_onnx_v2",
            )
            transcript = engine.transcribe(np.ones(1600, dtype=np.int16))

        self.assertEqual(transcript, "namaskara")
        self.assertEqual(engine._active_backend, "onnx_runtime")
        self.assertEqual(FakeVyasaRuntime.instances[0].repo_id, "somyalab/Vyasa_mini_rnnt_onnx_v2")
        self.assertTrue(FakeVyasaRuntime.instances[0].warmed)

    def test_english_mode_forces_whisper_when_backend_is_auto(self):
        with patch("app.asr.engine.WhisperModel", FakeWhisperModel):
            engine = ASREngine(
                backend="auto",
                language="auto",
                language_mode="english",
                model_name="small.en",
            )
            transcript = engine.transcribe(np.ones(1600, dtype=np.int16))

        self.assertEqual(transcript, "hello world")
        self.assertEqual(engine._active_backend, "faster_whisper")
        self.assertEqual(FakeWhisperModel.instances[0].transcribe_calls[-1]["language"], "en")

    def test_indic_mode_forces_vyasa_when_backend_is_auto(self):
        with patch("app.asr.engine.VyasaOnnxRuntime", FakeVyasaRuntime):
            engine = ASREngine(
                backend="auto",
                language="auto",
                language_mode="indic",
                indic_model_name="somyalab/Vyasa_mini_rnnt_onnx_v2",
            )
            transcript = engine.transcribe(np.ones(1600, dtype=np.int16))

        self.assertEqual(transcript, "namaskara")
        self.assertEqual(engine._active_backend, "onnx_runtime")

    def test_legacy_onnx_asr_backend_alias_maps_to_onnx_runtime(self):
        with patch("app.asr.engine.VyasaOnnxRuntime", FakeVyasaRuntime):
            engine = ASREngine(
                backend="onnx_asr",
                language="kn",
                indic_model_name="somyalab/Vyasa_mini_rnnt_onnx_v2",
            )
            transcript = engine.transcribe(np.ones(1600, dtype=np.int16))

        self.assertEqual(transcript, "namaskara")
        self.assertEqual(engine._active_backend, "onnx_runtime")

    def test_auto_language_prefers_indic_transcript_when_script_matches(self):
        FakeVyasaRuntime.next_text = "नमस्ते दुनिया"
        with patch("app.asr.engine.WhisperModel", FakeWhisperModel), patch(
            "app.asr.engine.VyasaOnnxRuntime",
            FakeVyasaRuntime,
        ):
            engine = ASREngine(
                backend="auto",
                language="auto",
                model_name="small.en",
                indic_model_name="somyalab/Vyasa_mini_rnnt_onnx_v2",
            )
            transcript = engine.transcribe(np.ones(1600, dtype=np.int16))

        self.assertEqual(transcript, "नमस्ते दुनिया")
        self.assertEqual(engine._active_backend, "hybrid_auto")
        self.assertEqual(len(FakeWhisperModel.instances), 1)

    def test_auto_language_falls_back_to_whisper_when_indic_script_not_detected(self):
        FakeVyasaRuntime.next_text = "namaskara"
        with patch("app.asr.engine.WhisperModel", FakeWhisperModel), patch(
            "app.asr.engine.VyasaOnnxRuntime",
            FakeVyasaRuntime,
        ):
            engine = ASREngine(
                backend="auto",
                language="auto",
                model_name="small.en",
                indic_model_name="somyalab/Vyasa_mini_rnnt_onnx_v2",
            )
            transcript = engine.transcribe(np.ones(1600, dtype=np.int16))

        self.assertEqual(transcript, "hello world")
        self.assertEqual(engine._active_backend, "hybrid_auto")
