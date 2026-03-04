from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EmotionalState:
    name: str
    intensity: str
    valence: str
    behavioral_description: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EmotionalState:
        return cls(
            name=data["name"],
            intensity=data["intensity"],
            valence=data["valence"],
            behavioral_description=data["behavioral_description"],
        )

    def __str__(self) -> str:
        return (
            f"{self.name} (intensity: {self.intensity}, valence: {self.valence}): "
            f"{self.behavioral_description}"
        )
