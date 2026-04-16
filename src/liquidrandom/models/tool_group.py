from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, ClassVar

from liquidrandom._detail import DetailLevel


@dataclass(frozen=True)
class ToolVariation:
    name: str
    description: str
    parameters: dict[str, Any]
    returns: dict[str, Any]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolVariation:
        raw_params = data["parameters"] or {}
        raw_returns = data["returns"] or {}
        return cls(
            name=data["name"],
            description=data["description"],
            parameters=raw_params if isinstance(raw_params, dict) else {"parameters": raw_params},
            returns=raw_returns if isinstance(raw_returns, dict) else {"returns": raw_returns},
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
    kind: str = "default"

    _field_groups: ClassVar[dict[str, tuple[str, ...]]] = {
        "high_level": ("domain", "description", "taxonomy_path", "kind", "tools"),
        "detailed": (),
    }

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
            kind=data.get("kind") or "default",
        )

    def to_str(self, detail: DetailLevel = DetailLevel.DETAILED) -> str:
        tool_names = ", ".join(t.canonical_name for t in self.tools)
        total_variations = sum(len(t.variations) for t in self.tools)
        return (
            f"Tool Group: {self.domain} ({self.taxonomy_path})\n"
            f"{self.description}\n"
            f"Tools ({len(self.tools)}): {tool_names}\n"
            f"Total variations: {total_variations}"
        )

    def brief(self) -> str:
        return self.to_str(DetailLevel.HIGH_LEVEL)

    def detailed(self) -> str:
        return self.to_str(DetailLevel.DETAILED)

    def __str__(self) -> str:
        return self.detailed()
