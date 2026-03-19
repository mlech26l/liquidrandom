from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from liquidrandom._detail import DetailLevel


@dataclass(frozen=True)
class Scenario:
    broad_title: str
    theme: str
    title: str
    context: str
    setting: str
    stakes: str
    description: str

    _field_groups: ClassVar[dict[str, tuple[str, ...]]] = {
        "high_level": ("broad_title", "theme", "setting"),
        "detailed": ("title", "context", "stakes", "description"),
    }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Scenario:
        return cls(
            broad_title=data.get("broad_title", ""),
            theme=data.get("theme", ""),
            title=data["title"],
            context=data["context"],
            setting=data["setting"],
            stakes=data["stakes"],
            description=data["description"],
        )

    def to_str(self, detail: DetailLevel = DetailLevel.DETAILED) -> str:
        base = f"{self.broad_title} ({self.theme}, {self.setting})"
        if detail == DetailLevel.HIGH_LEVEL:
            return base
        return (
            f"{base} - {self.title}: {self.description} "
            f"Context: {self.context}. Stakes: {self.stakes}"
        )

    def brief(self) -> str:
        return self.to_str(DetailLevel.HIGH_LEVEL)

    def detailed(self) -> str:
        return self.to_str(DetailLevel.DETAILED)

    def __str__(self) -> str:
        return self.detailed()
