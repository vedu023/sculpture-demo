from __future__ import annotations

import unittest

import numpy as np

from app.config import AppConfig
from app.orchestration.controller import Controller
from app.tools.executor import ToolExecutor
from app.tools.registry import ToolRegistry
from app.tools.types import ToolDecision, ToolDefinition, ToolResult
from app.types import CapturedUtterance, SynthesizedAudio


class FakeCaptureSession:
    def __init__(self, utterances: CapturedUtterance | list[CapturedUtterance]):
        if isinstance(utterances, list):
            self.utterances = list(utterances)
        else:
            self.utterances = [utterances]
        self.started = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def next_utterance(self, timeout: float = 1.0):
        if not self.utterances:
            raise RuntimeError("No fake utterances left")
        return self.utterances.pop(0)


class FakeASR:
    def __init__(self, texts: str | list[str]):
        if isinstance(texts, list):
            self.texts = list(texts)
        else:
            self.texts = [texts]
        self.warmed = False

    def warmup(self):
        self.warmed = True

    def transcribe(self, audio):
        if not self.texts:
            raise RuntimeError("No fake ASR transcript left")
        return self.texts.pop(0)


class FakeLLM:
    def __init__(
        self,
        decisions: list[ToolDecision] | None = None,
        generated_reply: str = "fallback reply",
    ):
        self.decisions = list(decisions or [])
        self.generated_reply = generated_reply
        self.warmed = False
        self.plan_calls: list[tuple[str, str]] = []
        self.generated_calls: list[tuple[str, bool]] = []
        self.remembered: list[tuple[str, str]] = []

    def warmup(self):
        self.warmed = True

    def plan_turn(self, user_text: str, planner_prompt: str):
        self.plan_calls.append((user_text, planner_prompt))
        if self.decisions:
            return self.decisions.pop(0)
        return ToolDecision(decision="speak", spoken_response="")

    def generate(self, user_text: str, remember: bool = True):
        self.generated_calls.append((user_text, remember))
        return self.generated_reply

    def remember_turn(self, user_text: str, assistant_text: str):
        self.remembered.append((user_text, assistant_text))


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

    def play(self, audio, sample_rate=None, block: bool = True):
        self.calls.append((audio, sample_rate, block))


class FakeSessionLogger:
    def __init__(self):
        self.events = []

    def log_event(self, event_type, payload):
        self.events.append((event_type, payload))


class RecordingToolHandler:
    def __init__(self, result_factory):
        self.result_factory = result_factory
        self.calls: list[dict[str, object]] = []

    def __call__(self, arguments):
        snapshot = dict(arguments)
        self.calls.append(snapshot)
        return self.result_factory(snapshot)


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


def make_tts() -> FakeTTS:
    return FakeTTS(
        SynthesizedAudio(
            samples=np.array([0.0, 0.1, -0.1], dtype=np.float32),
            sample_rate=24000,
            duration_ms=10.0,
        )
    )


def make_tooling(name: str, side_effect: bool, handler: RecordingToolHandler):
    registry = ToolRegistry(
        [
            ToolDefinition(
                name=name,
                description=f"Tool for {name}",
                parameters={"volume_percent": "integer"} if side_effect else {},
                side_effect=side_effect,
                handler=handler,
            )
        ]
    )
    return registry, ToolExecutor(registry)


class ControllerTests(unittest.TestCase):
    def test_run_turn_happy_path_uses_planner_spoken_reply(self):
        logger = FakeSessionLogger()
        speaker = FakeSpeaker()
        llm = FakeLLM(
            decisions=[ToolDecision(decision="speak", spoken_response="short reply")]
        )
        controller = Controller(
            AppConfig(),
            capture_session=FakeCaptureSession(make_utterance()),
            asr=FakeASR("hello there"),
            llm=llm,
            tts=make_tts(),
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
                "PLANNING",
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
        self.assertEqual(llm.generated_calls, [])
        self.assertEqual(llm.remembered, [("hello there", "short reply")])

    def test_run_turn_executes_read_only_tool(self):
        logger = FakeSessionLogger()
        speaker = FakeSpeaker()
        llm = FakeLLM(
            decisions=[ToolDecision(decision="tool_call", tool_name="get_time")]
        )
        handler = RecordingToolHandler(
            lambda arguments: ToolResult(
                tool_name="get_time",
                ok=True,
                data={"time": "9:41 AM"},
                spoken_response="It is 9:41 AM.",
            )
        )
        tool_registry, tool_executor = make_tooling("get_time", False, handler)
        controller = Controller(
            AppConfig(),
            capture_session=FakeCaptureSession(make_utterance()),
            asr=FakeASR("what time is it"),
            llm=llm,
            tts=make_tts(),
            speaker=speaker,
            session_logger=logger,
            tool_registry=tool_registry,
            tool_executor=tool_executor,
        )

        with controller:
            result = controller.run_turn(play_audio=True)

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.tool_name, "get_time")
        self.assertEqual(result.tool_status, "ok")
        self.assertEqual(result.tool_result, {"time": "9:41 AM"})
        self.assertEqual(result.reply, "It is 9:41 AM.")
        self.assertEqual(handler.calls, [{}])
        self.assertEqual(
            result.state_path,
            (
                "LISTENING",
                "PROCESSING_ASR",
                "PLANNING",
                "EXECUTING_TOOL",
                "PROCESSING_TTS",
                "SPEAKING",
                "LISTENING",
            ),
        )

    def test_side_effect_tool_requires_confirmation_then_executes_on_yes(self):
        logger = FakeSessionLogger()
        speaker = FakeSpeaker()
        llm = FakeLLM(
            decisions=[
                ToolDecision(
                    decision="tool_call",
                    tool_name="set_output_volume",
                    arguments={"volume_percent": 40},
                    requires_confirmation=True,
                )
            ]
        )
        handler = RecordingToolHandler(
            lambda arguments: ToolResult(
                tool_name="set_output_volume",
                ok=True,
                data={"volume_percent": arguments["volume_percent"]},
                spoken_response=f"Okay, I set the output volume to {arguments['volume_percent']} percent.",
            )
        )
        tool_registry, tool_executor = make_tooling("set_output_volume", True, handler)
        controller = Controller(
            AppConfig(),
            capture_session=FakeCaptureSession([make_utterance(), make_utterance()]),
            asr=FakeASR(["set the volume to forty percent", "yes"]),
            llm=llm,
            tts=make_tts(),
            speaker=speaker,
            session_logger=logger,
            tool_registry=tool_registry,
            tool_executor=tool_executor,
        )

        with controller:
            first = controller.run_turn(play_audio=True)
            self.assertEqual(handler.calls, [])
            second = controller.run_turn(play_audio=True)

        self.assertEqual(first.status, "confirmation_required")
        self.assertEqual(first.tool_name, "set_output_volume")
        self.assertTrue(first.confirmation_required)
        self.assertIn("40 percent", first.reply)
        self.assertEqual(
            first.state_path,
            (
                "LISTENING",
                "PROCESSING_ASR",
                "PLANNING",
                "CONFIRMING_ACTION",
                "PROCESSING_TTS",
                "SPEAKING",
                "LISTENING",
            ),
        )

        self.assertEqual(second.status, "ok")
        self.assertEqual(second.tool_name, "set_output_volume")
        self.assertEqual(second.tool_status, "ok")
        self.assertEqual(second.tool_result, {"volume_percent": 40})
        self.assertEqual(second.reply, "Okay, I set the output volume to 40 percent.")
        self.assertEqual(handler.calls, [{"volume_percent": 40}])
        self.assertEqual(len(llm.plan_calls), 1)
        self.assertEqual(
            second.state_path,
            (
                "LISTENING",
                "PROCESSING_ASR",
                "EXECUTING_TOOL",
                "PROCESSING_TTS",
                "SPEAKING",
                "LISTENING",
            ),
        )

    def test_run_turn_recovers_on_empty_transcript(self):
        logger = FakeSessionLogger()
        speaker = FakeSpeaker()
        controller = Controller(
            AppConfig(),
            capture_session=FakeCaptureSession(make_utterance()),
            asr=FakeASR(""),
            llm=FakeLLM(),
            tts=make_tts(),
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
            llm=FakeLLM(
                decisions=[ToolDecision(decision="speak", spoken_response="short reply")]
            ),
            tts=make_tts(),
            speaker=FakeSpeaker(),
            session_logger=logger,
        )

        with self.assertLogs("app.orchestration.controller", level="INFO") as captured:
            controller.warmup()

        combined = "\n".join(captured.output)
        self.assertIn("Smruti", combined)
        self.assertIn("pocket_tts/alba", combined)
