from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.tts.voice_ref import should_use_cached_voice_state, voice_state_is_stale


class VoiceRefTests(unittest.TestCase):
    def test_voice_state_is_stale_when_wav_is_newer(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            wav_path = tmp_path / "voice_ref_clean.wav"
            state_path = tmp_path / "voice_ref_clean.safetensors"
            state_path.write_text("old", encoding="utf-8")
            wav_path.write_text("new", encoding="utf-8")

            self.assertTrue(voice_state_is_stale(wav_path, state_path))
            self.assertFalse(should_use_cached_voice_state(wav_path, state_path))

    def test_cached_voice_state_is_used_when_it_is_fresh(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            wav_path = tmp_path / "voice_ref_clean.wav"
            state_path = tmp_path / "voice_ref_clean.safetensors"
            wav_path.write_text("old", encoding="utf-8")
            state_path.write_text("new", encoding="utf-8")
            state_path.touch()

            self.assertFalse(voice_state_is_stale(wav_path, state_path))
            self.assertTrue(should_use_cached_voice_state(wav_path, state_path))
