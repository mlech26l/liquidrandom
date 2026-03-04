from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Domain:
    name: str
    parent_field: str
    description: str
    key_concepts: list[str]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Domain:
        return cls(
            name=data["name"],
            parent_field=data["parent_field"],
            description=data["description"],
            key_concepts=list(data["key_concepts"] or []),
        )

    def __str__(self) -> str:
        concepts = ", ".join(self.key_concepts)
        return (
            f"{self.name} ({self.parent_field}): {self.description} "
            f"Key concepts: {concepts}"
        )
