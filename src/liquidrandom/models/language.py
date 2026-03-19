from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from liquidrandom._detail import DetailLevel


@dataclass(frozen=True)
class Language:
    category: str
    name: str
    region: str
    register: str
    script: str
    cultural_notes: str

    _field_groups: ClassVar[dict[str, tuple[str, ...]]] = {
        "high_level": ("category", "register"),
        "detailed": ("name", "region", "script", "cultural_notes"),
    }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Language:
        return cls(
            category=data.get("category", ""),
            name=data["name"],
            region=data["region"],
            register=data["register"],
            script=data["script"],
            cultural_notes=data["cultural_notes"],
        )

    def to_str(self, detail: DetailLevel = DetailLevel.DETAILED) -> str:
        base = f"{self.category} ({self.register})"
        if detail == DetailLevel.HIGH_LEVEL:
            return base
        return (
            f"{base} - {self.name}, {self.region}, {self.script}: "
            f"{self.cultural_notes}"
        )

    def brief(self) -> str:
        return self.to_str(DetailLevel.HIGH_LEVEL)

    def detailed(self) -> str:
        return self.to_str(DetailLevel.DETAILED)

    def __str__(self) -> str:
        return self.detailed()
