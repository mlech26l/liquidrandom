"""Resumability checkpoint for seed generation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CategoryState:
    name: str
    taxonomy_done: bool = False
    generation_done: bool = False
    total_samples: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "taxonomy_done": self.taxonomy_done,
            "generation_done": self.generation_done,
            "total_samples": self.total_samples,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CategoryState:
        return cls(
            name=data["name"],
            taxonomy_done=data.get("taxonomy_done", False),
            generation_done=data.get("generation_done", False),
            total_samples=data.get("total_samples", 0),
        )


@dataclass
class RunState:
    categories: dict[str, CategoryState] = field(default_factory=dict)

    def get_category(self, name: str) -> CategoryState:
        if name not in self.categories:
            self.categories[name] = CategoryState(name=name)
        return self.categories[name]

    def to_dict(self) -> dict[str, Any]:
        return {
            "categories": {k: v.to_dict() for k, v in self.categories.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunState:
        state = cls()
        for k, v in data.get("categories", {}).items():
            state.categories[k] = CategoryState.from_dict(v)
        return state

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> RunState:
        p = Path(path)
        if not p.exists():
            return cls()
        with open(p, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
