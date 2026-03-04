from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ScienceTopic:
    name: str
    scientific_field: str
    subfield: str
    description: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScienceTopic:
        return cls(
            name=data["name"],
            scientific_field=data["scientific_field"],
            subfield=data["subfield"],
            description=data["description"],
        )

    def __str__(self) -> str:
        return (
            f"{self.name} ({self.scientific_field} > {self.subfield}): "
            f"{self.description}"
        )
