from __future__ import annotations

import logging

from app.tools.registry import ToolRegistry
from app.tools.types import ToolCall, ToolResult

logger = logging.getLogger(__name__)


class ToolExecutor:
    def __init__(self, registry: ToolRegistry):
        self._registry = registry

    def execute(self, tool_call: ToolCall) -> ToolResult:
        definition = self._registry.get(tool_call.tool_name)
        if definition is None:
            return ToolResult(
                tool_name=tool_call.tool_name,
                ok=False,
                spoken_response="I do not have a tool for that yet.",
                error=f"unknown tool: {tool_call.tool_name}",
            )

        try:
            result = definition.handler(tool_call.arguments)
        except Exception as exc:
            logger.exception("Tool execution failed for %s", tool_call.tool_name)
            return ToolResult(
                tool_name=tool_call.tool_name,
                ok=False,
                spoken_response="I tried, but that tool call failed.",
                error=str(exc),
            )

        if not result.tool_name:
            result.tool_name = tool_call.tool_name
        return result
