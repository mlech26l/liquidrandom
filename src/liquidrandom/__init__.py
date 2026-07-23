"""liquidrandom: Pseudo-random seed data for ML/LLM training diversity."""

from __future__ import annotations

import random
from typing import Sequence

from liquidrandom._image_loader import (
    load_image_chain,
    load_image_random,
    load_random_chain,
)
from liquidrandom._loader import load_category, load_random, load_random_where
from liquidrandom._registry import IMAGE_CATEGORIES
from liquidrandom.models import (
    CodingTask,
    DetailLevel,
    Domain,
    EmotionalState,
    ImageSample,
    InstructionComplexity,
    Job,
    Language,
    MathCategory,
    Persona,
    ReasoningPattern,
    Scenario,
    ScienceTopic,
    ToolFunction,
    ToolGroup,
    ToolVariation,
    WritingStyle,
)

__all__ = [
    "CodingTask",
    "DetailLevel",
    "Domain",
    "EmotionalState",
    "ImageSample",
    "InstructionComplexity",
    "Job",
    "Language",
    "MathCategory",
    "Persona",
    "ReasoningPattern",
    "Scenario",
    "ScienceTopic",
    "ToolFunction",
    "ToolGroup",
    "ToolVariation",
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
    "tool_group",
    "physical_tool_group",
    "image",
    "image_chain",
    "image_chain_of",
    "indoor_scene",
    "outdoor_scene",
    "aerial_view",
    "agriculture",
    "industrial",
    "automotive",
    "ui_screenshot",
    "document",
    "chart",
    "retail_product",
    "food",
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


def image(
    category: str | None = None, tags: Sequence[str] | None = None
) -> ImageSample:
    """Return a random image sample, optionally filtered by category and tags.

    Tags use AND semantics: the sample must carry all of them. With
    category=None, a random image category is picked first (note: this
    downloads that category's Parquet file on first access).
    """
    if category is None:
        category = random.choice(sorted(IMAGE_CATEGORIES))
    return load_image_random(category, tags)


def image_chain(
    category: str | None = None,
    tags: Sequence[str] | None = None,
    min_length: int = 2,
) -> list[ImageSample]:
    """Return a random edit chain (base image + edited variants), sorted by turn.

    With tags, a chain qualifies if any of its images matches all tags.
    """
    if category is None:
        category = random.choice(sorted(IMAGE_CATEGORIES))
    return load_random_chain(category, tags, min_length=min_length)


def image_chain_of(sample: ImageSample) -> list[ImageSample]:
    """Return the full edit chain the given sample belongs to, sorted by turn."""
    return load_image_chain(sample.category, sample.chain_id)


def indoor_scene(tags: Sequence[str] | None = None) -> ImageSample:
    """Return a random indoor scene image."""
    return load_image_random("indoor_scene", tags)


def outdoor_scene(tags: Sequence[str] | None = None) -> ImageSample:
    """Return a random outdoor scene image."""
    return load_image_random("outdoor_scene", tags)


def aerial_view(tags: Sequence[str] | None = None) -> ImageSample:
    """Return a random satellite or aerial image."""
    return load_image_random("aerial_view", tags)


def agriculture(tags: Sequence[str] | None = None) -> ImageSample:
    """Return a random agricultural image (farms, crops, pests, machinery)."""
    return load_image_random("agriculture", tags)


def industrial(tags: Sequence[str] | None = None) -> ImageSample:
    """Return a random factory or industrial image."""
    return load_image_random("industrial", tags)


def automotive(tags: Sequence[str] | None = None) -> ImageSample:
    """Return a random automotive image (vehicles, traffic, in-cabin)."""
    return load_image_random("automotive", tags)


def ui_screenshot(tags: Sequence[str] | None = None) -> ImageSample:
    """Return a random UI/UX screenshot image."""
    return load_image_random("ui_screenshot", tags)


def document(tags: Sequence[str] | None = None) -> ImageSample:
    """Return a random document image (receipts, forms, handwriting)."""
    return load_image_random("document", tags)


def chart(tags: Sequence[str] | None = None) -> ImageSample:
    """Return a random chart, diagram, or infographic image."""
    return load_image_random("chart", tags)


def retail_product(tags: Sequence[str] | None = None) -> ImageSample:
    """Return a random retail or product image."""
    return load_image_random("retail_product", tags)


def food(tags: Sequence[str] | None = None) -> ImageSample:
    """Return a random food or cooking image."""
    return load_image_random("food", tags)
