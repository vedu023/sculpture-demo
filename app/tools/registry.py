from __future__ import annotations

from app.tools.types import ToolDefinition


class ToolRegistry:
    def __init__(self, definitions: list[ToolDefinition]):
        self._definitions = {definition.name: definition for definition in definitions}

    def get(self, name: str) -> ToolDefinition | None:
        return self._definitions.get(name)

    def specs_for_prompt(self) -> list[dict[str, object]]:
        return [
            self._definitions[name].to_prompt_spec()
            for name in sorted(self._definitions)
        ]
