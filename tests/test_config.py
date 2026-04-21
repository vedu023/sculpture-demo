from __future__ import annotations

import unittest

from app.config import AppConfig


class ConfigTests(unittest.TestCase):
    def test_default_backend_is_auto_tts(self):
        config = AppConfig()
        self.assertEqual(config.tts.backend, "auto")

    def test_default_asr_supports_english_and_indic_paths(self):
        config = AppConfig()
        self.assertEqual(config.asr.backend, "auto")
        self.assertEqual(config.asr.model_name, "small.en")
        self.assertEqual(config.asr.indic_model_name, "somyalab/Vyasa_mini_rnnt_onnx_v2")
        self.assertEqual(config.asr.language_mode, "auto")
        self.assertEqual(config.asr.language, "auto")
