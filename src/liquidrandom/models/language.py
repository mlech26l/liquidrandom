from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Language:
    name: str
    region: str
    register: str
    script: str
    cultural_notes: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Language:
        return cls(
            name=data["name"],
            region=data["region"],
            register=data["register"],
            script=data["script"],
            cultural_notes=data["cultural_notes"],
        )

    def __str__(self) -> str:
        return (
            f"{self.name} ({self.region}, {self.register}, {self.script}): "
            f"{self.cultural_notes}"
        )
