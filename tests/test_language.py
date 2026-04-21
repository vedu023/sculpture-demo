from __future__ import annotations

import unittest

from app.language import (
    default_reply_language,
    detect_text_language,
    is_mixed_indic_script,
    localize_tool_result,
)
from app.tools.types import ToolResult


class LanguageTests(unittest.TestCase):
    def test_detects_hindi_script(self):
        self.assertEqual(detect_text_language("नमस्ते"), "hi")

    def test_detects_kannada_script(self):
        self.assertEqual(detect_text_language("ನಮಸ್ಕಾರ"), "kn")

    def test_flags_mixed_hindi_kannada_script(self):
        self.assertTrue(is_mixed_indic_script("ऐसेತಿ"))

    def test_indic_mode_defaults_to_hindi_before_script_is_known(self):
        self.assertEqual(default_reply_language("auto", "indic"), "hi")

    def test_localizes_runtime_tool_reply_to_kannada(self):
        result = ToolResult(
            tool_name="set_output_volume",
            ok=True,
            data={"volume_percent": 35},
            spoken_response="Okay, I set the output volume to 35 percent.",
        )
        reply = localize_tool_result(result, "kn")
        self.assertIn("35", reply)
        self.assertIn("ವಾಲ್ಯೂಮ್", reply)
