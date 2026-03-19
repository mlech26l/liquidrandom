from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from liquidrandom._detail import DetailLevel


@dataclass(frozen=True)
class Persona:
    name: str
    age: int
    gender: str
    occupation: str
    nationality: str
    personality_traits: list[str]
    background: str

    _field_groups: ClassVar[dict[str, tuple[str, ...]]] = {
        "high_level": ("name", "age", "gender", "occupation", "nationality"),
        "detailed": ("personality_traits", "background"),
    }

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

    def to_str(self, detail: DetailLevel = DetailLevel.DETAILED) -> str:
        base = (
            f"{self.name} is a {self.age}-year-old {self.gender} from "
            f"{self.nationality}. They work as a {self.occupation}."
        )
        if detail == DetailLevel.HIGH_LEVEL:
            return base
        traits = ", ".join(self.personality_traits)
        return (
            f"{base} "
            f"Personality traits: {traits}. Background: {self.background}"
        )

    def brief(self) -> str:
        return self.to_str(DetailLevel.HIGH_LEVEL)

    def detailed(self) -> str:
        return self.to_str(DetailLevel.DETAILED)

    def __str__(self) -> str:
        return self.detailed()
