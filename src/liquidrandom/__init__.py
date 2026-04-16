"""liquidrandom: Pseudo-random seed data for ML/LLM training diversity."""

from __future__ import annotations

import random

from liquidrandom._loader import load_category, load_random, load_random_where
from liquidrandom.models import (
    Aerial,
    CodingTask,
    DetailLevel,
    Domain,
    EmotionalState,
    Indoor,
    InstructionComplexity,
    Job,
    Language,
    MathCategory,
    Navigation,
    PeopleScene,
    Persona,
    ReasoningPattern,
    Scenario,
    ScienceTopic,
    ToolFunction,
    ToolGroup,
    ToolVariation,
    Warehouse,
    WritingStyle,
)

__all__ = [
    "Aerial",
    "CodingTask",
    "DetailLevel",
    "Domain",
    "EmotionalState",
    "Indoor",
    "InstructionComplexity",
    "Job",
    "Language",
    "MathCategory",
    "Navigation",
    "PeopleScene",
    "Persona",
    "ReasoningPattern",
    "Scenario",
    "ScienceTopic",
    "ToolFunction",
    "ToolGroup",
    "ToolVariation",
    "Warehouse",
    "WritingStyle",
    "aerial",
    "indoor",
    "people_scene",
    "warehouse",
    "navigation",
    "persona",
    "job",
    "coding_task",
    "math_category",
    "writing_style",
    "scenario",
    "domain",
    "science_topic",
    "language",
    "reasoning_pattern",
    "emotional_state",
    "instruction_complexity",
    "tool_group",
    "physical_tool_group",
]


def persona() -> Persona:
    """Return a random persona."""
    return random.choice(load_category("persona"))


def job() -> Job:
    """Return a random job."""
    return random.choice(load_category("job"))


def coding_task() -> CodingTask:
    """Return a random coding task."""
    return random.choice(load_category("coding_task"))


def math_category() -> MathCategory:
    """Return a random math category."""
    return random.choice(load_category("math_category"))


def writing_style() -> WritingStyle:
    """Return a random writing style."""
    return random.choice(load_category("writing_style"))


def scenario() -> Scenario:
    """Return a random scenario."""
    return random.choice(load_category("scenario"))


def domain() -> Domain:
    """Return a random domain."""
    return random.choice(load_category("domain"))


def science_topic() -> ScienceTopic:
    """Return a random science topic."""
    return random.choice(load_category("science_topic"))


def language() -> Language:
    """Return a random language."""
    return random.choice(load_category("language"))


def reasoning_pattern() -> ReasoningPattern:
    """Return a random reasoning pattern."""
    return random.choice(load_category("reasoning_pattern"))


def emotional_state() -> EmotionalState:
    """Return a random emotional state."""
    return random.choice(load_category("emotional_state"))


def instruction_complexity() -> InstructionComplexity:
    """Return a random instruction complexity level."""
    return random.choice(load_category("instruction_complexity"))


def tool_group() -> ToolGroup:
    """Return a random tool group."""
    return load_random("tool_group")


def physical_tool_group() -> ToolGroup:
    """Return a random tool group with kind='physical' (smart home, robots, AVs, drones, etc.).

    This is a convenience filter over the unified tool_group dataset;
    equivalent to repeatedly calling tool_group() until one with
    kind == 'physical' is returned.
    """
    return load_random_where("tool_group", "kind", "physical")


def aerial() -> Aerial:
    """Return a random aerial image sample."""
    return random.choice(load_category("aerial"))


def indoor() -> Indoor:
    """Return a random indoor space image sample."""
    return random.choice(load_category("indoor"))


def people_scene() -> PeopleScene:
    """Return a random people scene image sample."""
    return random.choice(load_category("people_scene"))


def warehouse() -> Warehouse:
    """Return a random warehouse image sample."""
    return random.choice(load_category("warehouse"))


def navigation() -> Navigation:
    """Return a random navigation path image sample."""
    return random.choice(load_category("navigation"))
