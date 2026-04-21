from app.tools.executor import ToolExecutor
from app.tools.registry import ToolRegistry
from app.tools.types import ToolCall, ToolDecision, ToolDefinition, ToolResult


def build_builtin_tool_registry(*args, **kwargs):
    from app.tools.builtin import build_builtin_tool_registry as _build_builtin_tool_registry

    return _build_builtin_tool_registry(*args, **kwargs)


def route_tool_intent(*args, **kwargs):
    from app.tools.router import route_tool_intent as _route_tool_intent

    return _route_tool_intent(*args, **kwargs)

__all__ = [
    "ToolCall",
    "ToolDecision",
    "ToolDefinition",
    "ToolExecutor",
    "ToolRegistry",
    "ToolResult",
    "build_builtin_tool_registry",
    "route_tool_intent",
]
