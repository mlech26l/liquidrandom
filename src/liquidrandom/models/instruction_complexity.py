from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class InstructionComplexity:
    level: str
    ambiguity: str
    description: str
    example: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InstructionComplexity:
        return cls(
            level=data["level"],
            ambiguity=data["ambiguity"],
            description=data["description"],
            example=data["example"],
        )

    def __str__(self) -> str:
        return (
            f"[{self.level}, ambiguity: {self.ambiguity}] {self.description} "
            f'Example: "{self.example}"'
        )
