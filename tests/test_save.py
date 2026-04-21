import tempfile
import wave
import unittest
from pathlib import Path

import numpy as np

from app.audio.save import _to_pcm16, save_utterance


class SaveTests(unittest.TestCase):
    def test_to_pcm16_converts_float_audio(self):
        pcm = _to_pcm16(np.array([0.5, -0.5, 1.0, -1.0], dtype=np.float32))
        self.assertEqual(pcm.dtype, np.int16)
        self.assertEqual(tuple(pcm.tolist()), (16383, -16383, 32767, -32767))

    def test_save_utterance_accepts_float_input(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_utterance(
                np.array([0.0, 0.5, -0.5, 0.0], dtype=np.float32),
                sample_rate=16000,
                output_dir=Path(tmpdir),
            )

            with wave.open(str(path), "rb") as wf:
                self.assertEqual(wf.getsampwidth(), 2)
                self.assertEqual(wf.getframerate(), 16000)
                self.assertEqual(wf.getnchannels(), 1)
