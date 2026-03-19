from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from liquidrandom._detail import DetailLevel


@dataclass(frozen=True)
class Domain:
    broad_category: str
    area: str
    name: str
    parent_field: str
    description: str
    key_concepts: list[str]

    _field_groups: ClassVar[dict[str, tuple[str, ...]]] = {
        "high_level": ("broad_category", "area"),
        "detailed": ("name", "parent_field", "description", "key_concepts"),
    }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Domain:
        return cls(
            broad_category=data.get("broad_category", ""),
            area=data.get("area", ""),
            name=data["name"],
            parent_field=data["parent_field"],
            description=data["description"],
            key_concepts=list(data["key_concepts"] or []),
        )

    def to_str(self, detail: DetailLevel = DetailLevel.DETAILED) -> str:
        base = f"{self.broad_category} > {self.area}"
        if detail == DetailLevel.HIGH_LEVEL:
            return base
        concepts = ", ".join(self.key_concepts)
        return (
            f"{base} - {self.name} ({self.parent_field}): "
            f"{self.description} Key concepts: {concepts}"
        )

    def brief(self) -> str:
        return self.to_str(DetailLevel.HIGH_LEVEL)

    def detailed(self) -> str:
        return self.to_str(DetailLevel.DETAILED)

    def __str__(self) -> str:
        return self.detailed()
