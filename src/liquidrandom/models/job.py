from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from liquidrandom._detail import DetailLevel


@dataclass(frozen=True)
class Job:
    job_category: str
    sector: str
    title: str
    industry: str
    description: str
    required_skills: list[str]
    experience_level: str

    _field_groups: ClassVar[dict[str, tuple[str, ...]]] = {
        "high_level": ("job_category", "sector", "experience_level"),
        "detailed": ("title", "industry", "description", "required_skills"),
    }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Job:
        return cls(
            job_category=data.get("job_category", ""),
            sector=data.get("sector", ""),
            title=data["title"],
            industry=data["industry"],
            description=data["description"],
            required_skills=list(data["required_skills"] or []),
            experience_level=data["experience_level"],
        )

    def to_str(self, detail: DetailLevel = DetailLevel.DETAILED) -> str:
        base = f"{self.job_category} ({self.sector}, {self.experience_level})"
        if detail == DetailLevel.HIGH_LEVEL:
            return base
        skills = ", ".join(self.required_skills)
        return (
            f"{base} - {self.title} ({self.industry}): "
            f"{self.description} Required skills: {skills}"
        )

    def brief(self) -> str:
        return self.to_str(DetailLevel.HIGH_LEVEL)

    def detailed(self) -> str:
        return self.to_str(DetailLevel.DETAILED)

    def __str__(self) -> str:
        return self.detailed()
