from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Job:
    title: str
    industry: str
    description: str
    required_skills: list[str]
    experience_level: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Job:
        return cls(
            title=data["title"],
            industry=data["industry"],
            description=data["description"],
            required_skills=list(data["required_skills"] or []),
            experience_level=data["experience_level"],
        )

    def __str__(self) -> str:
        skills = ", ".join(self.required_skills)
        return (
            f"{self.title} ({self.industry}, {self.experience_level}): "
            f"{self.description} Required skills: {skills}"
        )
