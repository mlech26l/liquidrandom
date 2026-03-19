from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from liquidrandom._detail import DetailLevel


@dataclass(frozen=True)
class WritingStyle:
    category: str
    name: str
    tone: str
    characteristics: list[str]
    description: str

    _field_groups: ClassVar[dict[str, tuple[str, ...]]] = {
        "high_level": ("category", "tone"),
        "detailed": ("name", "characteristics", "description"),
    }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WritingStyle:
        return cls(
            category=data.get("category", ""),
            name=data["name"],
            tone=data["tone"],
            characteristics=list(data["characteristics"] or []),
            description=data["description"],
        )

    def to_str(self, detail: DetailLevel = DetailLevel.DETAILED) -> str:
        base = f"{self.category} (tone: {self.tone})"
        if detail == DetailLevel.HIGH_LEVEL:
            return base
        chars = ", ".join(self.characteristics)
        return (
            f"{base} - {self.name}: {self.description} "
            f"Characteristics: {chars}"
        )

    def brief(self) -> str:
        return self.to_str(DetailLevel.HIGH_LEVEL)

    def detailed(self) -> str:
        return self.to_str(DetailLevel.DETAILED)

    def __str__(self) -> str:
        return self.detailed()
