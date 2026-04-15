from __future__ import annotations

import unittest

from app.config import LLMConfig
from app.persona import build_system_prompt, build_greeting_prompt


class PersonaTests(unittest.TestCase):
    def test_system_prompt_includes_identity_and_constraints(self):
        prompt = build_system_prompt("Smruti", "smruti", 2)
        self.assertIn("Smruti", prompt)
        self.assertIn("confident", prompt)
        self.assertIn("at most 2", prompt)

    def test_llm_config_builds_persona_prompt(self):
        config = LLMConfig()
        self.assertIn("Smruti", config.system_prompt)
        self.assertIn("warm", config.system_prompt)
        self.assertEqual(config.max_tokens, 48)

    def test_prompt_prefers_short_spoken_sentences(self):
        config = LLMConfig()
        self.assertIn("one sentence", config.system_prompt)

    def test_greeting_prompt_includes_name(self):
        prompt = build_greeting_prompt("Smruti", "smruti")
        self.assertIn("Smruti", prompt)
        self.assertIn("one short sentence", prompt)
