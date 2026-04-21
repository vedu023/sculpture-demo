from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from app.llm.engine import LLMEngine


def make_response(content: str = "", tool_calls=None):
    message = SimpleNamespace(
        content=content,
        tool_calls=tool_calls or [],
    )
    return SimpleNamespace(message=message)


class LLMEngineTests(unittest.TestCase):
    def test_normalize_spoken_reply_rejects_reasoning_style_output(self):
        engine = LLMEngine()

        normalized = engine._normalize_spoken_reply(
            "Okay, the user is asking if I emit reasoning tokens. Hmm, I need to think about how to respond here."
        )

        self.assertEqual(normalized, "")

    def test_plan_turn_ignores_reasoning_like_non_json_content(self):
        engine = LLMEngine()

        with patch.object(
            engine,
            "_chat_response",
            return_value=make_response(
                'Okay, the user is asking for the time, so I should think about which tool to call.'
            ),
        ):
            decision = engine.plan_turn(
                "what time is it",
                "planner prompt",
                tool_schemas=[],
            )

        self.assertEqual(decision.decision, "speak")
        self.assertEqual(decision.spoken_response, "")
        self.assertEqual(decision.tool_name, "")

    def test_generate_with_system_prompt_uses_fallback_reply(self):
        engine = LLMEngine(max_sentences=2)

        with patch.object(engine, "_generate_structured_reply", side_effect=["", ""]):
            reply = engine.generate_with_system_prompt(
                system_prompt="Say hello.",
                user_text="Greet the audience now.",
                fallback_reply="Hey, good to see you.",
            )

        self.assertEqual(reply, "Hey, good to see you.")
