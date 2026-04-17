from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolCall:
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolDecision:
    decision: str = "speak"
    spoken_response: str = ""
    tool_name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    requires_confirmation: bool = False

    @property
    def is_tool_call(self) -> bool:
        return self.decision == "tool_call" and bool(self.tool_name)


@dataclass
class ToolResult:
    tool_name: str
    ok: bool
    data: dict[str, Any] = field(default_factory=dict)
    spoken_response: str = ""
    error: str | None = None


ToolHandler = Callable[[dict[str, Any]], ToolResult]


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, str]
    side_effect: bool
    handler: ToolHandler

    def to_prompt_spec(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": dict(self.parameters),
            "side_effect": self.side_effect,
        }
