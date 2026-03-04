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


def test_persona_from_dict_and_str() -> None:
    data = {
        "name": "Alice",
        "age": 30,
        "gender": "female",
        "occupation": "engineer",
        "nationality": "Canadian",
        "personality_traits": ["curious", "patient"],
        "background": "Grew up in Toronto",
    }
    p = Persona.from_dict(data)
    assert p.name == "Alice"
    assert p.age == 30
    assert p.personality_traits == ["curious", "patient"]
    s = str(p)
    assert "Alice" in s
    assert "30" in s
    assert "curious" in s


def test_job_from_dict_and_str() -> None:
    data = {
        "title": "Data Scientist",
        "industry": "Technology",
        "description": "Builds ML models",
        "required_skills": ["Python", "Statistics"],
        "experience_level": "mid",
    }
    j = Job.from_dict(data)
    assert j.title == "Data Scientist"
    assert "Python" in str(j)


def test_coding_task_from_dict_and_str() -> None:
    data = {
        "title": "Implement a trie",
        "language": "Python",
        "difficulty": "medium",
        "description": "Build a trie data structure",
        "constraints": ["O(n) insert", "O(n) search"],
        "expected_behavior": "Returns matching prefixes",
    }
    ct = CodingTask.from_dict(data)
    assert ct.language == "Python"
    assert "trie" in str(ct)


def test_math_category_from_dict_and_str() -> None:
    data = {
        "name": "Linear Equations",
        "field": "Algebra",
        "description": "Solving systems of linear equations",
        "example_problems": ["2x + 3 = 7", "x + y = 5, x - y = 1"],
    }
    mc = MathCategory.from_dict(data)
    assert mc.field == "Algebra"
    assert "Linear Equations" in str(mc)


def test_writing_style_from_dict_and_str() -> None:
    data = {
        "name": "Sardonic Critique",
        "tone": "sarcastic",
        "characteristics": ["dry humor", "understatement"],
        "description": "Academic critique with a sardonic edge",
    }
    ws = WritingStyle.from_dict(data)
    assert ws.tone == "sarcastic"
    assert "Sardonic" in str(ws)


def test_scenario_from_dict_and_str() -> None:
    data = {
        "title": "Production Outage",
        "context": "E-commerce peak season",
        "setting": "On-call at 2am",
        "stakes": "Revenue loss",
        "description": "Debugging a critical production outage",
    }
    sc = Scenario.from_dict(data)
    assert sc.stakes == "Revenue loss"
    assert "Outage" in str(sc)


def test_domain_from_dict_and_str() -> None:
    data = {
        "name": "Supply Chain Logistics",
        "parent_field": "Operations Management",
        "description": "Managing flow of goods",
        "key_concepts": ["inventory", "forecasting"],
    }
    d = Domain.from_dict(data)
    assert d.parent_field == "Operations Management"
    assert "inventory" in str(d)


def test_science_topic_from_dict_and_str() -> None:
    data = {
        "name": "Quantum Entanglement",
        "scientific_field": "Physics",
        "subfield": "Quantum Mechanics",
        "description": "Non-local correlations between particles",
    }
    st = ScienceTopic.from_dict(data)
    assert st.scientific_field == "Physics"
    assert "Quantum" in str(st)


def test_language_from_dict_and_str() -> None:
    data = {
        "name": "Brazilian Portuguese",
        "region": "Brazil",
        "register": "informal",
        "script": "Latin",
        "cultural_notes": "Rich use of diminutives",
    }
    lang = Language.from_dict(data)
    assert lang.region == "Brazil"
    assert "Portuguese" in str(lang)


def test_reasoning_pattern_from_dict_and_str() -> None:
    data = {
        "name": "Proof by Contradiction",
        "category": "Deductive",
        "description": "Assume the opposite and derive a contradiction",
        "when_to_use": "When direct proof is difficult",
    }
    rp = ReasoningPattern.from_dict(data)
    assert rp.category == "Deductive"
    assert "Contradiction" in str(rp)


def test_emotional_state_from_dict_and_str() -> None:
    data = {
        "name": "Cautiously Optimistic",
        "intensity": "moderate",
        "valence": "positive",
        "behavioral_description": "Hopeful but guarded",
    }
    es = EmotionalState.from_dict(data)
    assert es.valence == "positive"
    assert "Cautiously" in str(es)


def test_instruction_complexity_from_dict_and_str() -> None:
    data = {
        "level": "high",
        "ambiguity": "moderate",
        "description": "Multi-step specification with constraints",
        "example": "Build a REST API with auth, rate limiting, and caching",
    }
    ic = InstructionComplexity.from_dict(data)
    assert ic.level == "high"
    assert "REST API" in str(ic)
