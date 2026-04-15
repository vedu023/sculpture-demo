from __future__ import annotations

import unittest

from app.utils.text import sanitize_spoken_response


class SpokenResponseTests(unittest.TestCase):
    def test_sanitizes_markdown_and_caps_sentences(self):
        raw = """
        - **Hello there.**
        - This should stay.
        - This third sentence should be dropped.
        """
        cleaned = sanitize_spoken_response(raw, max_sentences=2)
        self.assertEqual(cleaned, "Hello there. This should stay.")

    def test_returns_empty_for_whitespace(self):
        self.assertEqual(sanitize_spoken_response("   \n  ", max_sentences=2), "")
