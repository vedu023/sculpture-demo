from __future__ import annotations

from datetime import datetime, timezone
import unittest

from app.audio.output import AudioOutput
from app.config import AppConfig
from app.tools.calendar import parse_calendar_output
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
        self.assertEqual(result.data["asr_backend"], "auto")
        self.assertEqual(
            result.data["asr_model"],
            f"{config.asr.indic_model_name} (hi/kn auto) with {config.asr.model_name} as English fallback",
        )
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

    def test_calendar_tool_summarizes_next_event(self):
        config = AppConfig()
        speaker = AudioOutput()
        registry = build_builtin_tool_registry(
            config,
            speaker,
            device_provider=lambda: [],
            now_provider=lambda: datetime(2026, 4, 17, 9, 30, tzinfo=timezone.utc),
            calendar_provider=lambda date_scope: [
                {
                    "calendar_name": "Work",
                    "title": "Team Sync",
                    "start_text": "2026-04-17 10:00",
                    "end_text": "2026-04-17 10:30",
                    "location": "",
                },
                {
                    "calendar_name": "Work",
                    "title": "Design Review",
                    "start_text": "2026-04-17 13:00",
                    "end_text": "2026-04-17 14:00",
                    "location": "",
                },
            ],
        )

        result = registry.get("get_calendar_events").handler({"date_scope": "today", "mode": "next"})

        self.assertTrue(result.ok)
        self.assertEqual(result.data["count"], 2)
        self.assertEqual(result.data["date_scope"], "today")
        self.assertIn("Team Sync", result.spoken_response)
        self.assertIn("10:00 AM", result.spoken_response)

    def test_parse_calendar_output_reads_tab_separated_lines(self):
        parsed = parse_calendar_output(
            "Work\tTeam Sync\t2026-04-17 10:00\t2026-04-17 10:30\tRoom A\n"
            "Personal\tLunch\t2026-04-17 12:00\t2026-04-17 13:00\t\n"
        )

        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0]["title"], "Team Sync")
        self.assertEqual(parsed[1]["calendar_name"], "Personal")
