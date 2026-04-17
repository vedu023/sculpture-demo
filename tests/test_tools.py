from __future__ import annotations

from datetime import datetime, timezone
import unittest

from app.audio.output import AudioOutput
from app.config import AppConfig
from app.tools.builtin import build_builtin_tool_registry


class BuiltinToolTests(unittest.TestCase):
    def test_runtime_status_reports_models_and_volume(self):
        config = AppConfig()
        speaker = AudioOutput(volume=0.35)
        registry = build_builtin_tool_registry(
            config,
            speaker,
            device_provider=lambda: [],
            now_provider=lambda: datetime(2026, 4, 17, 9, 30, tzinfo=timezone.utc),
        )

        result = registry.get("get_runtime_status").handler({})

        self.assertTrue(result.ok)
        self.assertEqual(result.data["llm_model"], config.llm.model_name)
        self.assertEqual(result.data["asr_model"], config.asr.model_name)
        self.assertEqual(result.data["output_volume_percent"], 35)
        self.assertIn("35 percent", result.spoken_response)

    def test_set_output_volume_updates_speaker(self):
        config = AppConfig()
        speaker = AudioOutput(volume=1.0)
        registry = build_builtin_tool_registry(
            config,
            speaker,
            device_provider=lambda: [],
            now_provider=lambda: datetime(2026, 4, 17, 9, 30, tzinfo=timezone.utc),
        )

        result = registry.get("set_output_volume").handler({"volume_percent": "27"})

        self.assertTrue(result.ok)
        self.assertEqual(speaker.get_volume_percent(), 27)
        self.assertEqual(result.data["volume_percent"], 27)

    def test_list_audio_devices_summarizes_first_devices(self):
        config = AppConfig()
        speaker = AudioOutput()
        registry = build_builtin_tool_registry(
            config,
            speaker,
            device_provider=lambda: [
                {"name": "Built-in Mic", "max_input_channels": 2, "max_output_channels": 0},
                {"name": "USB Headset", "max_input_channels": 1, "max_output_channels": 2},
                {"name": "HDMI Out", "max_input_channels": 0, "max_output_channels": 2},
            ],
            now_provider=lambda: datetime(2026, 4, 17, 9, 30, tzinfo=timezone.utc),
        )

        result = registry.get("list_audio_devices").handler({})

        self.assertTrue(result.ok)
        self.assertEqual(result.data["count"], 3)
        self.assertIn("Built-in Mic", result.spoken_response)
        self.assertIn("USB Headset", result.spoken_response)
