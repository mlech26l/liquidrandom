from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MathCategory:
    name: str
    field: str
    description: str
    example_problems: list[str]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MathCategory:
        return cls(
            name=data["name"],
            field=data["field"],
            description=data["description"],
            example_problems=list(data["example_problems"] or []),
        )

    def __str__(self) -> str:
        examples = "; ".join(self.example_problems)
        return (
            f"{self.name} ({self.field}): {self.description} "
            f"Example problems: {examples}"
        )
