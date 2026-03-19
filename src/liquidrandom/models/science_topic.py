from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from liquidrandom._detail import DetailLevel


@dataclass(frozen=True)
class ScienceTopic:
    broad_topic: str
    name: str
    scientific_field: str
    subfield: str
    description: str

    _field_groups: ClassVar[dict[str, tuple[str, ...]]] = {
        "high_level": ("broad_topic", "scientific_field"),
        "detailed": ("name", "subfield", "description"),
    }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScienceTopic:
        return cls(
            broad_topic=data.get("broad_topic", ""),
            name=data["name"],
            scientific_field=data["scientific_field"],
            subfield=data["subfield"],
            description=data["description"],
        )

    def to_str(self, detail: DetailLevel = DetailLevel.DETAILED) -> str:
        base = f"{self.broad_topic} ({self.scientific_field})"
        if detail == DetailLevel.HIGH_LEVEL:
            return base
        return f"{base} - {self.name} > {self.subfield}: {self.description}"

    def brief(self) -> str:
        return self.to_str(DetailLevel.HIGH_LEVEL)

    def detailed(self) -> str:
        return self.to_str(DetailLevel.DETAILED)

    def __str__(self) -> str:
        return self.detailed()
