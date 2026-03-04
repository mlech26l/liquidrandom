from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CodingTask:
    title: str
    language: str
    difficulty: str
    description: str
    constraints: list[str]
    expected_behavior: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CodingTask:
        return cls(
            title=data["title"],
            language=data["language"],
            difficulty=data["difficulty"],
            description=data["description"],
            constraints=list(data["constraints"] or []),
            expected_behavior=data["expected_behavior"],
        )

    def __str__(self) -> str:
        constraints = "; ".join(self.constraints)
        return (
            f"[{self.language}, {self.difficulty}] {self.title}: "
            f"{self.description} Constraints: {constraints}. "
            f"Expected behavior: {self.expected_behavior}"
        )
