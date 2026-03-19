from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from liquidrandom._detail import DetailLevel


@dataclass(frozen=True)
class ReasoningPattern:
    name: str
    category: str
    description: str
    when_to_use: str

    _field_groups: ClassVar[dict[str, tuple[str, ...]]] = {
        "high_level": ("name", "category"),
        "detailed": ("description", "when_to_use"),
    }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReasoningPattern:
        return cls(
            name=data["name"],
            category=data["category"],
            description=data["description"],
            when_to_use=data["when_to_use"],
        )

    def to_str(self, detail: DetailLevel = DetailLevel.DETAILED) -> str:
        base = f"{self.name} ({self.category})"
        if detail == DetailLevel.HIGH_LEVEL:
            return base
        return (
            f"{base}: {self.description} When to use: {self.when_to_use}"
        )

    def brief(self) -> str:
        return self.to_str(DetailLevel.HIGH_LEVEL)

    def detailed(self) -> str:
        return self.to_str(DetailLevel.DETAILED)

    def __str__(self) -> str:
        return self.detailed()
