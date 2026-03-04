from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Persona:
    name: str
    age: int
    gender: str
    occupation: str
    nationality: str
    personality_traits: list[str]
    background: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Persona:
        return cls(
            name=data["name"],
            age=int(data["age"]),
            gender=data["gender"],
            occupation=data["occupation"],
            nationality=data["nationality"],
            personality_traits=list(data["personality_traits"] or []),
            background=data["background"],
        )

    def __str__(self) -> str:
        traits = ", ".join(self.personality_traits)
        return (
            f"{self.name} is a {self.age}-year-old {self.gender} from "
            f"{self.nationality}. They work as a {self.occupation}. "
            f"Personality traits: {traits}. Background: {self.background}"
        )
