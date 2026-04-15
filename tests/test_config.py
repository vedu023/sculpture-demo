from __future__ import annotations

import unittest

from app.config import AppConfig


class ConfigTests(unittest.TestCase):
    def test_default_backend_is_pocket_tts(self):
        config = AppConfig()
        self.assertEqual(config.tts.backend, "pocket_tts")
