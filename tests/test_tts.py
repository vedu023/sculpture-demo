from __future__ import annotations

import unittest
from unittest.mock import patch

from app.tts.engine import TTSEngine


class TTSTests(unittest.TestCase):
    @patch("app.tts.pocket_tts.PocketTTSBackend.warmup")
    def test_pocket_warmup_delegates_to_backend(self, warmup_mock):
        engine = TTSEngine(backend="pocket_tts")
        engine.warmup()
        warmup_mock.assert_called_once()

    @patch("app.tts.pocket_tts.PocketTTSBackend.synthesize")
    def test_pocket_synthesize_delegates_to_backend(self, synthesize_mock):
        engine = TTSEngine(backend="pocket_tts")
        text = "hello world"
        engine.synthesize(text)
        synthesize_mock.assert_called_once_with(text)

    def test_unknown_backend_fails_clearly(self):
        with self.assertRaises(RuntimeError):
            TTSEngine(backend="unknown")
