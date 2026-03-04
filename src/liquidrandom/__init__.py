"""liquidrandom: Pseudo-random seed data for ML/LLM training diversity."""

from __future__ import annotations

import random

from liquidrandom._loader import load_category
from liquidrandom.models import (
    CodingTask,
    Domain,
    EmotionalState,
    InstructionComplexity,
    Job,
    Language,
    MathCategory,
    Persona,
    ReasoningPattern,
    Scenario,
    ScienceTopic,
    WritingStyle,
)

__all__ = [
    "CodingTask",
    "Domain",
    "EmotionalState",
    "InstructionComplexity",
    "Job",
    "Language",
    "MathCategory",
    "Persona",
    "ReasoningPattern",
    "Scenario",
    "ScienceTopic",
    "WritingStyle",
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
