from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ReasoningPattern:
    name: str
    category: str
    description: str
    when_to_use: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReasoningPattern:
        return cls(
            name=data["name"],
            category=data["category"],
            description=data["description"],
            when_to_use=data["when_to_use"],
        )

    def __str__(self) -> str:
        return (
            f"{self.name} ({self.category}): {self.description} "
            f"When to use: {self.when_to_use}"
        )
