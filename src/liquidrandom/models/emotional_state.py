from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from liquidrandom._detail import DetailLevel


@dataclass(frozen=True)
class EmotionalState:
    category: str
    name: str
    intensity: str
    valence: str
    behavioral_description: str
    example: str

    _field_groups: ClassVar[dict[str, tuple[str, ...]]] = {
        "high_level": ("category", "intensity", "valence"),
        "detailed": ("name", "behavioral_description", "example"),
    }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EmotionalState:
        return cls(
            category=data.get("category", ""),
            name=data["name"],
            intensity=data["intensity"],
            valence=data["valence"],
            behavioral_description=data["behavioral_description"],
            example=data.get("example", ""),
        )

    def to_str(self, detail: DetailLevel = DetailLevel.DETAILED) -> str:
        base = (
            f"{self.category} (intensity: {self.intensity}, "
            f"valence: {self.valence})"
        )
        if detail == DetailLevel.HIGH_LEVEL:
            return base
        return (
            f"{base} - {self.name}: {self.behavioral_description} "
            f"Example: {self.example}"
        )

    def brief(self) -> str:
        return self.to_str(DetailLevel.HIGH_LEVEL)

    def detailed(self) -> str:
        return self.to_str(DetailLevel.DETAILED)

    def __str__(self) -> str:
        return self.detailed()
