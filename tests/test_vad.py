from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import numpy as np

from app.audio import vad as vad_module


class DummyVAD:
    _model_path = "dummy"

    @staticmethod
    def _get_model_path():
        return DummyVAD._model_path

    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self._lib = MagicMock()
        self._obj = object()

    def process(self, _: bytes) -> float:
        return 0.42


class VADChunkingTests(unittest.TestCase):
    def test_exact_window_size_is_supported(self):
        with unittest.mock.patch("app.audio.vad.SileroVAD", DummyVAD):
            model = vad_module.VAD()
            self.assertEqual(model.get_speech_prob(np.zeros(512, dtype=np.float32)), 0.42)

    def test_longer_window_is_chunked(self):
        recorded_chunks: list[int] = []

        class RecordingVAD(DummyVAD):
            def process(self, data: bytes) -> float:
                recorded_chunks.append(len(data))
                return float(len(data))

        with unittest.mock.patch("app.audio.vad.SileroVAD", RecordingVAD):
            model = vad_module.VAD()
            score = model.get_speech_prob(np.ones(1024, dtype=np.float32))

            self.assertEqual(recorded_chunks, [2048, 2048])
            self.assertEqual(score, 2048.0)

    def test_short_window_is_padded(self):
        recorded_chunks: list[int] = []

        class RecordingVAD(DummyVAD):
            def process(self, data: bytes) -> float:
                recorded_chunks.append(len(data))
                return float(len(data))

        with unittest.mock.patch("app.audio.vad.SileroVAD", RecordingVAD):
            model = vad_module.VAD()
            score = model.get_speech_prob(np.ones(256, dtype=np.float32))

            self.assertEqual(recorded_chunks, [2048])
            self.assertEqual(score, 2048.0)
