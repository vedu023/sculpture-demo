from __future__ import annotations

import unittest

import numpy as np

from app.config import AppConfig
from app.orchestration.controller import Controller
from app.types import CapturedUtterance, SynthesizedAudio


class FakeCaptureSession:
    def __init__(self, utterance: CapturedUtterance):
        self.utterance = utterance
        self.started = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def next_utterance(self, timeout: float = 1.0):
        return self.utterance


class FakeASR:
    def __init__(self, text: str):
        self.text = text
        self.warmed = False

    def warmup(self):
        self.warmed = True

    def transcribe(self, audio):
        return self.text


class FakeLLM:
    def __init__(self, reply: str):
        self.reply = reply
        self.warmed = False

    def warmup(self):
        self.warmed = True

    def generate(self, user_text: str):
        return self.reply


class FakeTTS:
    def __init__(self, audio: SynthesizedAudio):
        self.audio = audio
        self.warmed = False

    def warmup(self):
        self.warmed = True

    def synthesize(self, text: str):
        return self.audio

    def describe(self) -> str:
        return "pocket_tts/alba"


class FakeSpeaker:
    def __init__(self):
        self.calls = []

    def play(self, audio, sample_rate=None):
        self.calls.append((audio, sample_rate))


class FakeSessionLogger:
    def __init__(self):
        self.events = []

    def log_event(self, event_type, payload):
        self.events.append((event_type, payload))


def make_utterance() -> CapturedUtterance:
    return CapturedUtterance(
        samples=np.ones(1600, dtype=np.int16),
        sample_rate=16000,
        started_at=1.0,
        ended_at=1.1,
        duration_ms=100.0,
        speech_ms=90.0,
        trailing_silence_ms=10.0,
        chunk_count=4,
    )


class ControllerTests(unittest.TestCase):
    def test_run_turn_happy_path(self):
        logger = FakeSessionLogger()
        speaker = FakeSpeaker()
        controller = Controller(
            AppConfig(),
            capture_session=FakeCaptureSession(make_utterance()),
            asr=FakeASR("hello there"),
            llm=FakeLLM("short reply"),
            tts=FakeTTS(
                SynthesizedAudio(
                    samples=np.array([0.0, 0.1, -0.1], dtype=np.float32),
                    sample_rate=24000,
                    duration_ms=10.0,
                )
            ),
            speaker=speaker,
            session_logger=logger,
        )

        with controller:
            result = controller.run_turn(play_audio=True)

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.transcript, "hello there")
        self.assertEqual(result.reply, "short reply")
        self.assertEqual(
            result.state_path,
            (
                "LISTENING",
                "PROCESSING_ASR",
                "PROCESSING_LLM",
                "PROCESSING_TTS",
                "SPEAKING",
                "LISTENING",
            ),
        )
        self.assertEqual(len(speaker.calls), 1)
        self.assertEqual(speaker.calls[0][1], 24000)
        self.assertEqual(logger.events[-1][0], "turn")
        self.assertEqual(result.assistant_name, "Smruti")
        self.assertEqual(result.tts_backend, "pocket_tts/alba")

    def test_run_turn_recovers_on_empty_transcript(self):
        logger = FakeSessionLogger()
        speaker = FakeSpeaker()
        controller = Controller(
            AppConfig(),
            capture_session=FakeCaptureSession(make_utterance()),
            asr=FakeASR(""),
            llm=FakeLLM("unused"),
            tts=FakeTTS(
                SynthesizedAudio(samples=np.zeros(0, dtype=np.float32), sample_rate=24000, duration_ms=0.0)
            ),
            speaker=speaker,
            session_logger=logger,
        )

        with controller:
            result = controller.run_turn(play_audio=True)

        self.assertEqual(result.status, "empty_transcript")
        self.assertEqual(result.state_path, ("LISTENING", "PROCESSING_ASR", "LISTENING"))
        self.assertEqual(speaker.calls, [])
        self.assertEqual(logger.events[-1][0], "turn")

    def test_warmup_logs_smruti_and_backend_identity(self):
        logger = FakeSessionLogger()
        controller = Controller(
            AppConfig(),
            capture_session=FakeCaptureSession(make_utterance()),
            asr=FakeASR("hello there"),
            llm=FakeLLM("short reply"),
            tts=FakeTTS(
                SynthesizedAudio(
                    samples=np.array([0.0, 0.1], dtype=np.float32),
                    sample_rate=24000,
                    duration_ms=5.0,
                )
            ),
            speaker=FakeSpeaker(),
            session_logger=logger,
        )

        with self.assertLogs("app.orchestration.controller", level="INFO") as captured:
            controller.warmup()

        combined = "\n".join(captured.output)
        self.assertIn("Smruti", combined)
        self.assertIn("pocket_tts/alba", combined)
