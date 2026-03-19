from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from liquidrandom._detail import DetailLevel


@dataclass(frozen=True)
class InstructionComplexity:
    name: str
    level: str
    ambiguity: str
    description: str
    example: str

    _field_groups: ClassVar[dict[str, tuple[str, ...]]] = {
        "high_level": ("name", "level", "ambiguity"),
        "detailed": ("description", "example"),
    }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InstructionComplexity:
        return cls(
            name=data.get("name", ""),
            level=data["level"],
            ambiguity=data["ambiguity"],
            description=data["description"],
            example=data["example"],
        )

    def to_str(self, detail: DetailLevel = DetailLevel.DETAILED) -> str:
        base = f"{self.name} [{self.level}, ambiguity: {self.ambiguity}]"
        if detail == DetailLevel.HIGH_LEVEL:
            return base
        return f"{base} {self.description} Example: \"{self.example}\""

    def brief(self) -> str:
        return self.to_str(DetailLevel.HIGH_LEVEL)

    def detailed(self) -> str:
        return self.to_str(DetailLevel.DETAILED)

    def __str__(self) -> str:
        return self.detailed()
