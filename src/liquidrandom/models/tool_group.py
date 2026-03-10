from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolVariation:
    name: str
    description: str
    parameters: dict[str, Any]
    returns: dict[str, Any]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolVariation:
        return cls(
            name=data["name"],
            description=data["description"],
            parameters=dict(data["parameters"] or {}),
            returns=dict(data["returns"] or {}),
        )

    def __str__(self) -> str:
        params = json.dumps(self.parameters, indent=2)
        returns = json.dumps(self.returns, indent=2)
        return (
            f"{self.name}: {self.description}\n"
            f"Parameters: {params}\n"
            f"Returns: {returns}"
        )


@dataclass(frozen=True)
class ToolFunction:
    canonical_name: str
    description: str
    variations: list[ToolVariation]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolFunction:
        variations_raw = data.get("variations") or []
        variations = [ToolVariation.from_dict(v) for v in variations_raw]
        return cls(
            canonical_name=data["canonical_name"],
            description=data["description"],
            variations=variations,
        )

    def __str__(self) -> str:
        variations_str = "\n".join(
            f"  Variation {i}: {v.name}" for i, v in enumerate(self.variations)
        )
        return (
            f"Function: {self.canonical_name} - {self.description}\n"
            f"Variations ({len(self.variations)}):\n{variations_str}"
        )


@dataclass(frozen=True)
class ToolGroup:
    domain: str
    description: str
    taxonomy_path: str
    tools: list[ToolFunction]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolGroup:
        tools_raw = data.get("tools_json") or data.get("tools")
        if isinstance(tools_raw, str):
            tools_raw = json.loads(tools_raw)
        tools_list = list(tools_raw or [])
        tools = [ToolFunction.from_dict(t) for t in tools_list]
        return cls(
            domain=data["domain"],
            description=data["description"],
            taxonomy_path=data["taxonomy_path"],
            tools=tools,
        )

    def __str__(self) -> str:
        tool_names = ", ".join(t.canonical_name for t in self.tools)
        total_variations = sum(len(t.variations) for t in self.tools)
        return (
            f"Tool Group: {self.domain} ({self.taxonomy_path})\n"
            f"{self.description}\n"
            f"Tools ({len(self.tools)}): {tool_names}\n"
            f"Total variations: {total_variations}"
        )
