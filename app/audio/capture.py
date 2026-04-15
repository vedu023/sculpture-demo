from __future__ import annotations

import collections
import logging
import queue
import time
from dataclasses import dataclass

import numpy as np

from app.audio.input import AudioInput
from app.audio.vad import VAD
from app.config import AppConfig
from app.types import CapturedUtterance

logger = logging.getLogger(__name__)


@dataclass
class AudioChunk:
    samples: np.ndarray
    started_at: float
    ended_at: float
    chunk_index: int


class AudioCaptureLoop:
    """Streams mic audio into a queue for the main thread to consume."""

    def __init__(self, audio_input: AudioInput, queue_maxsize: int = 128):
        self._input = audio_input
        self._queue: queue.Queue[AudioChunk] = queue.Queue(maxsize=queue_maxsize)
        self._stream = None
        self._dropped_chunks = 0
        self._chunk_index = 0
        self._capture_enabled = False

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            logger.debug("Audio callback status: %s", status)
        if not self._capture_enabled:
            return

        samples = indata.copy().reshape(-1)
        ended_at = time.time()
        started_at = ended_at - (len(samples) / self._input.sample_rate)
        chunk = AudioChunk(
            samples=samples,
            started_at=started_at,
            ended_at=ended_at,
            chunk_index=self._chunk_index,
        )
        self._chunk_index += 1

        try:
            self._queue.put_nowait(chunk)
        except queue.Full:
            self._dropped_chunks += 1
            logger.debug("Audio queue full; dropped chunks=%d", self._dropped_chunks)

    def start(self):
        if self._stream is not None:
            return
        self._stream = self._input.create_stream(self._audio_callback)
        self._stream.start()
        logger.debug("Audio capture started")

    def stop(self):
        self._capture_enabled = False
        self.drain()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            logger.debug("Audio capture stopped")

    def set_capture_enabled(self, enabled: bool):
        self._capture_enabled = enabled
        if not enabled:
            self.drain()

    def get_chunk(self, timeout: float = 1.0) -> AudioChunk | None:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain(self) -> int:
        drained = 0
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return drained
            drained += 1

    @property
    def dropped_chunks(self) -> int:
        return self._dropped_chunks


class UtteranceSegmenter:
    """Consumes audio chunks and emits complete utterances based on VAD."""

    def __init__(
        self,
        vad: VAD,
        sample_rate: int = 16000,
        block_size: int = 512,
        preroll_ms: int = 300,
        max_utterance_ms: int | None = None,
        trim_trailing_silence: bool = True,
    ):
        self._vad = vad
        self._sample_rate = sample_rate
        self._block_size = block_size
        self._chunk_ms = block_size / sample_rate * 1000
        self._speech_onset_chunks = max(1, int(round(vad.min_speech_ms / self._chunk_ms)))
        self._silence_end_chunks = max(1, int(round(vad.min_silence_ms / self._chunk_ms)))
        self._max_utterance_ms = max_utterance_ms
        self._trim_trailing_silence = trim_trailing_silence

        preroll_chunks = max(1, int(round(preroll_ms / self._chunk_ms)))
        self._pre_buffer: collections.deque[AudioChunk] = collections.deque(maxlen=preroll_chunks)
        self._utterance_chunks: list[AudioChunk] = []
        self._speech_count = 0
        self._silence_count = 0
        self._speaking = False
        self._speech_started_at: float | None = None

    def reset(self):
        self._pre_buffer.clear()
        self._utterance_chunks.clear()
        self._speech_count = 0
        self._silence_count = 0
        self._speaking = False
        self._speech_started_at = None
        self._vad.reset()

    def feed(self, chunk: AudioChunk) -> CapturedUtterance | None:
        """Feed one audio chunk. Returns an utterance when complete, else None."""
        speech_prob = self._vad.get_speech_prob(chunk.samples)
        is_speech = speech_prob >= self._vad.threshold
        logger.debug(
            "chunk=%d speech_prob=%.3f speaking=%s",
            chunk.chunk_index,
            speech_prob,
            self._speaking,
        )

        if not self._speaking:
            self._pre_buffer.append(chunk)
            self._speech_count = self._speech_count + 1 if is_speech else 0
            if self._speech_count >= self._speech_onset_chunks:
                self._speaking = True
                self._utterance_chunks = list(self._pre_buffer)
                self._speech_started_at = self._utterance_chunks[-self._speech_onset_chunks].started_at
                self._pre_buffer.clear()
                self._silence_count = 0
                logger.debug("Speech started")
            return None

        self._utterance_chunks.append(chunk)
        self._silence_count = self._silence_count + 1 if not is_speech else 0

        utterance_duration_ms = (
            (self._utterance_chunks[-1].ended_at - self._utterance_chunks[0].started_at) * 1000
        )
        forced_stop = (
            self._max_utterance_ms is not None and utterance_duration_ms >= self._max_utterance_ms
        )
        if not forced_stop and self._silence_count < self._silence_end_chunks:
            return None

        return self._build_utterance()

    def _build_utterance(self) -> CapturedUtterance:
        trailing_silence_chunks = min(self._silence_count, len(self._utterance_chunks))
        trailing_silence_ms = trailing_silence_chunks * self._chunk_ms

        trimmed_chunks = self._utterance_chunks
        if self._trim_trailing_silence and trailing_silence_chunks:
            trimmed_chunks = self._utterance_chunks[:-trailing_silence_chunks]

        if not trimmed_chunks:
            trimmed_chunks = self._utterance_chunks[:1]

        speech_end_chunk = trimmed_chunks[-1]
        samples = np.concatenate([chunk.samples for chunk in trimmed_chunks]).astype(np.int16)
        duration_ms = len(samples) / self._sample_rate * 1000
        speech_ms = max(0.0, (speech_end_chunk.ended_at - (self._speech_started_at or trimmed_chunks[0].started_at)) * 1000)
        utterance = CapturedUtterance(
            samples=samples,
            sample_rate=self._sample_rate,
            started_at=trimmed_chunks[0].started_at,
            ended_at=speech_end_chunk.ended_at,
            duration_ms=duration_ms,
            speech_ms=speech_ms,
            trailing_silence_ms=trailing_silence_ms,
            chunk_count=len(trimmed_chunks),
        )
        logger.debug(
            "Speech ended duration=%.1fms speech=%.1fms trailing=%.1fms",
            utterance.duration_ms,
            utterance.speech_ms,
            utterance.trailing_silence_ms,
        )
        self.reset()
        return utterance


class CaptureSession:
    """Long-lived microphone capture session for half-duplex interaction."""

    def __init__(self, config: AppConfig):
        self._config = config
        self._input = AudioInput(
            sample_rate=config.audio.sample_rate,
            channels=config.audio.channels,
            block_size=config.audio.block_size,
            device=config.audio.input_device,
        )
        self._vad = VAD(
            sample_rate=config.audio.sample_rate,
            threshold=config.vad.threshold,
            min_speech_ms=config.vad.min_speech_ms,
            min_silence_ms=config.vad.min_silence_ms,
        )
        self._segmenter = UtteranceSegmenter(
            vad=self._vad,
            sample_rate=config.audio.sample_rate,
            block_size=config.audio.block_size,
            preroll_ms=config.vad.preroll_ms,
            max_utterance_ms=config.audio.max_utterance_ms,
            trim_trailing_silence=config.vad.trim_trailing_silence,
        )
        self._capture = AudioCaptureLoop(
            self._input,
            queue_maxsize=config.audio.queue_maxsize,
        )
        self._started = False

    def start(self):
        if self._started:
            return
        self._capture.start()
        self._vad.get_speech_prob(np.zeros(self._config.audio.block_size, dtype=np.int16))
        self._vad.reset()
        self._started = True

    def stop(self):
        if not self._started:
            return
        self._capture.stop()
        self._started = False

    def next_utterance(self, timeout: float = 1.0) -> CapturedUtterance:
        self.start()
        self._segmenter.reset()
        self._capture.drain()
        overflow_start = self._capture.dropped_chunks
        self._capture.set_capture_enabled(True)
        try:
            while True:
                chunk = self._capture.get_chunk(timeout=timeout)
                if chunk is None:
                    continue
                utterance = self._segmenter.feed(chunk)
                if utterance is not None:
                    utterance.queue_overflow_count = self._capture.dropped_chunks - overflow_start
                    return utterance
        finally:
            self._capture.set_capture_enabled(False)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
