from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from liquidrandom.models.coding_task import CodingTask
from liquidrandom.models.domain import Domain
from liquidrandom.models.emotional_state import EmotionalState
from liquidrandom.models.instruction_complexity import InstructionComplexity
from liquidrandom.models.job import Job
from liquidrandom.models.language import Language
from liquidrandom.models.math_category import MathCategory
from liquidrandom.models.persona import Persona
from liquidrandom.models.reasoning_pattern import ReasoningPattern
from liquidrandom.models.scenario import Scenario
from liquidrandom.models.science_topic import ScienceTopic
from liquidrandom.models.writing_style import WritingStyle

REPO_ID = "mlech26l/liquidrandom-data"


@dataclass(frozen=True)
class CategoryInfo:
    model_class: type[Any]
    filename: str


CATEGORIES: dict[str, CategoryInfo] = {
    "persona": CategoryInfo(Persona, "persona.parquet"),
    "job": CategoryInfo(Job, "job.parquet"),
    "coding_task": CategoryInfo(CodingTask, "coding_task.parquet"),
    "math_category": CategoryInfo(MathCategory, "math_category.parquet"),
    "writing_style": CategoryInfo(WritingStyle, "writing_style.parquet"),
    "scenario": CategoryInfo(Scenario, "scenario.parquet"),
    "domain": CategoryInfo(Domain, "domain.parquet"),
    "science_topic": CategoryInfo(ScienceTopic, "science_topic.parquet"),
    "language": CategoryInfo(Language, "language.parquet"),
    "reasoning_pattern": CategoryInfo(ReasoningPattern, "reasoning_pattern.parquet"),
    "emotional_state": CategoryInfo(EmotionalState, "emotional_state.parquet"),
    "instruction_complexity": CategoryInfo(
        InstructionComplexity, "instruction_complexity.parquet"
    ),
}
