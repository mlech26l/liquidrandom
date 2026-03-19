from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from liquidrandom._detail import DetailLevel


@dataclass(frozen=True)
class MathCategory:
    broad_topic: str
    name: str
    field: str
    description: str
    example_problems: list[str]

    _field_groups: ClassVar[dict[str, tuple[str, ...]]] = {
        "high_level": ("broad_topic", "field"),
        "detailed": ("name", "description", "example_problems"),
    }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MathCategory:
        return cls(
            broad_topic=data.get("broad_topic", ""),
            name=data["name"],
            field=data["field"],
            description=data["description"],
            example_problems=list(data["example_problems"] or []),
        )

    def to_str(self, detail: DetailLevel = DetailLevel.DETAILED) -> str:
        base = f"{self.broad_topic} ({self.field})"
        if detail == DetailLevel.HIGH_LEVEL:
            return base
        examples = "; ".join(self.example_problems)
        return (
            f"{base} - {self.name}: {self.description} "
            f"Example problems: {examples}"
        )

    def brief(self) -> str:
        return self.to_str(DetailLevel.HIGH_LEVEL)

    def detailed(self) -> str:
        return self.to_str(DetailLevel.DETAILED)

    def __str__(self) -> str:
        return self.detailed()
