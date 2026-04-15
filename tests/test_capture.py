from __future__ import annotations

import unittest

import numpy as np

from app.audio.capture import AudioChunk, UtteranceSegmenter


class FakeVAD:
    def __init__(self, probabilities, threshold=0.5, min_speech_ms=64, min_silence_ms=64):
        self._probabilities = list(probabilities)
        self.threshold = threshold
        self.min_speech_ms = min_speech_ms
        self.min_silence_ms = min_silence_ms
        self.reset_count = 0

    def get_speech_prob(self, audio_chunk):
        return self._probabilities.pop(0)

    def reset(self):
        self.reset_count += 1


def make_chunk(index: int, sample_rate: int = 16000, block_size: int = 512) -> AudioChunk:
    chunk_seconds = block_size / sample_rate
    return AudioChunk(
        samples=np.full(block_size, index, dtype=np.int16),
        started_at=index * chunk_seconds,
        ended_at=(index + 1) * chunk_seconds,
        chunk_index=index,
    )


class SegmenterTests(unittest.TestCase):
    def test_segmenter_preserves_preroll_and_trims_trailing_silence(self):
        vad = FakeVAD([0.0, 0.0, 0.8, 0.9, 0.9, 0.1, 0.1], min_speech_ms=64, min_silence_ms=64)
        segmenter = UtteranceSegmenter(
            vad=vad,
            sample_rate=16000,
            block_size=512,
            preroll_ms=96,
            trim_trailing_silence=True,
        )

        utterance = None
        for index in range(7):
            utterance = segmenter.feed(make_chunk(index))
            if utterance is not None:
                break

        self.assertIsNotNone(utterance)
        self.assertEqual(utterance.samples.size, 4 * 512)
        self.assertAlmostEqual(utterance.duration_ms, 128.0, delta=0.5)
        self.assertAlmostEqual(utterance.speech_ms, 96.0, delta=0.5)
        self.assertAlmostEqual(utterance.trailing_silence_ms, 64.0, delta=0.5)
        self.assertAlmostEqual(utterance.started_at, 0.032, delta=0.001)
        self.assertAlmostEqual(utterance.ended_at, 0.160, delta=0.001)

    def test_segmenter_respects_max_utterance_cutoff(self):
        vad = FakeVAD([0.9, 0.9, 0.9, 0.9], min_speech_ms=32, min_silence_ms=320)
        segmenter = UtteranceSegmenter(
            vad=vad,
            sample_rate=16000,
            block_size=512,
            preroll_ms=32,
            max_utterance_ms=96,
            trim_trailing_silence=True,
        )

        utterance = None
        for index in range(4):
            utterance = segmenter.feed(make_chunk(index))
            if utterance is not None:
                break

        self.assertIsNotNone(utterance)
        self.assertAlmostEqual(utterance.duration_ms, 96.0, delta=0.5)
        self.assertAlmostEqual(utterance.trailing_silence_ms, 0.0, delta=0.1)
        self.assertEqual(vad.reset_count, 1)
