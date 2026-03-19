from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from liquidrandom._detail import DetailLevel


@dataclass(frozen=True)
class CodingTask:
    title: str
    language: str
    difficulty: str
    description: str
    constraints: list[str]
    expected_behavior: str
    follow_up_task: str
    change_request: str
    edge_cases: list[str]

    _field_groups: ClassVar[dict[str, tuple[str, ...]]] = {
        "high_level": ("title", "language", "difficulty"),
        "detailed": ("description", "constraints", "expected_behavior"),
        "manual": ("follow_up_task", "change_request", "edge_cases"),
    }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CodingTask:
        return cls(
            title=data["title"],
            language=data["language"],
            difficulty=data["difficulty"],
            description=data["description"],
            constraints=list(data["constraints"] or []),
            expected_behavior=data["expected_behavior"],
            follow_up_task=data.get("follow_up_task", ""),
            change_request=data.get("change_request", ""),
            edge_cases=list(data.get("edge_cases") or []),
        )

    def to_str(self, detail: DetailLevel = DetailLevel.DETAILED) -> str:
        base = f"[{self.language}, {self.difficulty}] {self.title}"
        if detail == DetailLevel.HIGH_LEVEL:
            return base
        constraints = "; ".join(self.constraints)
        return (
            f"{base}: {self.description} Constraints: {constraints}. "
            f"Expected behavior: {self.expected_behavior}"
        )

    def brief(self) -> str:
        return self.to_str(DetailLevel.HIGH_LEVEL)

    def detailed(self) -> str:
        return self.to_str(DetailLevel.DETAILED)

    def __str__(self) -> str:
        return self.detailed()
