"""End-to-end test that exercises the public API against real HuggingFace data.

This mirrors the usage shown in the README and verifies that each category
can be fetched, returns the correct type, and produces a non-empty string.
"""

import liquidrandom
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


def test_persona() -> None:
    persona = liquidrandom.persona()
    assert isinstance(persona, Persona)
    assert persona.name
    assert persona.age > 0
    assert len(persona.personality_traits) > 0
    assert len(str(persona)) > 0


def test_job() -> None:
    job = liquidrandom.job()
    assert isinstance(job, Job)
    assert job.title
    assert job.industry
    assert len(job.required_skills) > 0
    assert len(str(job)) > 0


def test_coding_task() -> None:
    task = liquidrandom.coding_task()
    assert isinstance(task, CodingTask)
    assert task.title
    assert task.language
    assert task.difficulty
    assert len(task.constraints) > 0
    assert len(str(task)) > 0


def test_math_category() -> None:
    mc = liquidrandom.math_category()
    assert isinstance(mc, MathCategory)
    assert mc.name
    assert mc.field
    assert len(mc.example_problems) > 0
    assert len(str(mc)) > 0


def test_writing_style() -> None:
    style = liquidrandom.writing_style()
    assert isinstance(style, WritingStyle)
    assert style.name
    assert style.tone
    assert len(style.characteristics) > 0
    assert len(str(style)) > 0


def test_scenario() -> None:
    sc = liquidrandom.scenario()
    assert isinstance(sc, Scenario)
    assert sc.title
    assert sc.context
    assert sc.stakes
    assert len(str(sc)) > 0


def test_domain() -> None:
    d = liquidrandom.domain()
    assert isinstance(d, Domain)
    assert d.name
    assert d.parent_field
    assert len(d.key_concepts) > 0
    assert len(str(d)) > 0


def test_science_topic() -> None:
    st = liquidrandom.science_topic()
    assert isinstance(st, ScienceTopic)
    assert st.name
    assert st.scientific_field
    assert st.subfield
    assert len(str(st)) > 0


def test_language() -> None:
    lang = liquidrandom.language()
    assert isinstance(lang, Language)
    assert lang.name
    assert lang.region
    assert lang.register
    assert len(str(lang)) > 0


def test_reasoning_pattern() -> None:
    rp = liquidrandom.reasoning_pattern()
    assert isinstance(rp, ReasoningPattern)
    assert rp.name
    assert rp.category
    assert len(str(rp)) > 0


def test_emotional_state() -> None:
    es = liquidrandom.emotional_state()
    assert isinstance(es, EmotionalState)
    assert es.name
    assert es.intensity
    assert es.valence
    assert len(str(es)) > 0


def test_instruction_complexity() -> None:
    ic = liquidrandom.instruction_complexity()
    assert isinstance(ic, InstructionComplexity)
    assert ic.level
    assert ic.ambiguity
    assert ic.example
    assert len(str(ic)) > 0


def test_str_usable_in_fstring() -> None:
    """Verify samples work in f-strings as shown in the README usage example."""
    persona = liquidrandom.persona()
    style = liquidrandom.writing_style()
    topic = liquidrandom.science_topic()

    prompt = f"""You are {persona}
Write in the following style: {style}
Explain the following topic: {topic}"""

    assert persona.name in prompt
    assert style.name in prompt
    assert topic.name in prompt


def test_field_access() -> None:
    """Verify individual field access as shown in the README."""
    persona = liquidrandom.persona()
    assert isinstance(persona.name, str)
    assert isinstance(persona.age, int)
    assert isinstance(persona.personality_traits, list)
