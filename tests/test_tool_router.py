from __future__ import annotations

import unittest

from app.tools.registry import ToolRegistry
from app.tools.router import route_tool_intent
from app.tools.types import ToolDefinition, ToolResult


def make_registry(*names: str) -> ToolRegistry:
    return ToolRegistry(
        [
            ToolDefinition(
                name=name,
                description=name,
                parameters={},
                side_effect=name == "set_output_volume",
                handler=lambda arguments, tool_name=name: ToolResult(
                    tool_name=tool_name,
                    ok=True,
                    data=dict(arguments),
                ),
            )
            for name in names
        ]
    )


class ToolRouterTests(unittest.TestCase):
    def test_routes_time_requests_deterministically(self):
        tool_call = route_tool_intent(
            "what time is it right now",
            make_registry("get_time"),
            current_volume_percent=50,
        )

        self.assertIsNotNone(tool_call)
        self.assertEqual(tool_call.tool_name, "get_time")
        self.assertEqual(tool_call.arguments, {})

    def test_routes_asr_variant_for_time_prompt(self):
        tool_call = route_tool_intent(
            "Hello, what time it is?",
            make_registry("get_time"),
            current_volume_percent=50,
        )

        self.assertIsNotNone(tool_call)
        self.assertEqual(tool_call.tool_name, "get_time")

    def test_routes_volume_word_numbers(self):
        tool_call = route_tool_intent(
            "set the volume to forty percent",
            make_registry("set_output_volume"),
            current_volume_percent=50,
        )

        self.assertIsNotNone(tool_call)
        self.assertEqual(tool_call.tool_name, "set_output_volume")
        self.assertEqual(tool_call.arguments["volume_percent"], 40)

    def test_routes_relative_volume_changes(self):
        tool_call = route_tool_intent(
            "turn it up a bit",
            make_registry("set_output_volume"),
            current_volume_percent=55,
        )

        self.assertIsNotNone(tool_call)
        self.assertEqual(tool_call.arguments["volume_percent"], 65)

    def test_routes_calendar_queries(self):
        tool_call = route_tool_intent(
            "what's on my calendar tomorrow",
            make_registry("get_calendar_events"),
            current_volume_percent=50,
        )

        self.assertIsNotNone(tool_call)
        self.assertEqual(tool_call.tool_name, "get_calendar_events")
        self.assertEqual(tool_call.arguments, {"date_scope": "tomorrow", "mode": "summary"})

    def test_routes_asr_variant_for_date_prompt(self):
        tool_call = route_tool_intent(
            "What date today?",
            make_registry("get_date"),
            current_volume_percent=50,
        )

        self.assertIsNotNone(tool_call)
        self.assertEqual(tool_call.tool_name, "get_date")
