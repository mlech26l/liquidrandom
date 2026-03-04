from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Scenario:
    title: str
    context: str
    setting: str
    stakes: str
    description: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Scenario:
        return cls(
            title=data["title"],
            context=data["context"],
            setting=data["setting"],
            stakes=data["stakes"],
            description=data["description"],
        )

    def __str__(self) -> str:
        return (
            f"{self.title}: {self.description} "
            f"Setting: {self.setting}. Context: {self.context}. "
            f"Stakes: {self.stakes}"
        )
