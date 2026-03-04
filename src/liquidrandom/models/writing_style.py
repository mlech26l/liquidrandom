from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class WritingStyle:
    name: str
    tone: str
    characteristics: list[str]
    description: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WritingStyle:
        return cls(
            name=data["name"],
            tone=data["tone"],
            characteristics=list(data["characteristics"] or []),
            description=data["description"],
        )

    def __str__(self) -> str:
        chars = ", ".join(self.characteristics)
        return (
            f"{self.name} (tone: {self.tone}): {self.description} "
            f"Characteristics: {chars}"
        )
