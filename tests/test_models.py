from liquidrandom._detail import DetailLevel
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


def test_persona_detail_levels() -> None:
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
    high = p.to_str(DetailLevel.HIGH_LEVEL)
    assert "Alice" in high
    assert "engineer" in high
    assert "curious" not in high
    assert "Toronto" not in high
    assert p.brief() == high

    detailed = p.to_str(DetailLevel.DETAILED)
    assert "Alice" in detailed
    assert "curious" in detailed
    assert "Toronto" in detailed
    assert p.detailed() == detailed
    assert str(p) == detailed


def test_job_from_dict_and_str() -> None:
    data = {
        "job_category": "Data Science",
        "sector": "Financial Services",
        "title": "Senior Quantitative Analyst",
        "industry": "Hedge Funds",
        "description": "Builds ML models for trading",
        "required_skills": ["Python", "Statistics"],
        "experience_level": "senior",
    }
    j = Job.from_dict(data)
    assert j.title == "Senior Quantitative Analyst"
    assert j.job_category == "Data Science"
    assert "Python" in str(j)


def test_job_detail_levels() -> None:
    data = {
        "job_category": "Data Science",
        "sector": "Financial Services",
        "title": "Senior Quantitative Analyst",
        "industry": "Hedge Funds",
        "description": "Builds ML models for trading",
        "required_skills": ["Python", "Statistics"],
        "experience_level": "senior",
    }
    j = Job.from_dict(data)
    high = j.to_str(DetailLevel.HIGH_LEVEL)
    assert "Data Science" in high
    assert "Financial Services" in high
    assert "senior" in high
    assert "Python" not in high
    assert "Quantitative" not in high

    detailed = j.to_str(DetailLevel.DETAILED)
    assert "Quantitative" in detailed
    assert "Python" in detailed


def test_coding_task_from_dict_and_str() -> None:
    data = {
        "title": "Implement a trie",
        "language": "Python",
        "difficulty": "medium",
        "description": "Build a trie data structure",
        "constraints": ["O(n) insert", "O(n) search"],
        "expected_behavior": "Returns matching prefixes",
        "follow_up_task": "Add autocomplete functionality",
        "change_request": "Support case-insensitive search",
        "edge_cases": ["empty string input", "unicode characters"],
    }
    ct = CodingTask.from_dict(data)
    assert ct.language == "Python"
    assert "trie" in str(ct)


def test_coding_task_detail_levels() -> None:
    data = {
        "title": "Implement a trie",
        "language": "Python",
        "difficulty": "medium",
        "description": "Build a trie data structure",
        "constraints": ["O(n) insert", "O(n) search"],
        "expected_behavior": "Returns matching prefixes",
        "follow_up_task": "Add autocomplete functionality",
        "change_request": "Support case-insensitive search",
        "edge_cases": ["empty string input", "unicode characters"],
    }
    ct = CodingTask.from_dict(data)
    high = ct.to_str(DetailLevel.HIGH_LEVEL)
    assert "Python" in high
    assert "medium" in high
    assert "trie" in high
    assert "O(n)" not in high

    detailed = ct.to_str(DetailLevel.DETAILED)
    assert "O(n)" in detailed
    assert "Returns matching" in detailed

    # Manual-only fields not in any str output
    assert "autocomplete" not in detailed
    assert "case-insensitive" not in detailed
    # But accessible as attributes
    assert ct.follow_up_task == "Add autocomplete functionality"
    assert ct.change_request == "Support case-insensitive search"
    assert ct.edge_cases == ["empty string input", "unicode characters"]


def test_math_category_from_dict_and_str() -> None:
    data = {
        "broad_topic": "Equation Solving",
        "name": "Linear Equations",
        "field": "Algebra",
        "description": "Solving systems of linear equations",
        "example_problems": ["2x + 3 = 7", "x + y = 5, x - y = 1"],
    }
    mc = MathCategory.from_dict(data)
    assert mc.field == "Algebra"
    assert "Linear Equations" in str(mc)


def test_math_category_detail_levels() -> None:
    data = {
        "broad_topic": "Equation Solving",
        "name": "Linear Equations",
        "field": "Algebra",
        "description": "Solving systems of linear equations",
        "example_problems": ["2x + 3 = 7", "x + y = 5, x - y = 1"],
    }
    mc = MathCategory.from_dict(data)
    high = mc.to_str(DetailLevel.HIGH_LEVEL)
    assert "Equation Solving" in high
    assert "Algebra" in high
    assert "Linear Equations" not in high

    detailed = mc.to_str(DetailLevel.DETAILED)
    assert "Linear Equations" in detailed
    assert "2x + 3 = 7" in detailed


def test_writing_style_from_dict_and_str() -> None:
    data = {
        "category": "Academic Writing",
        "name": "Sardonic Critique",
        "tone": "sarcastic",
        "characteristics": ["dry humor", "understatement"],
        "description": "Academic critique with a sardonic edge",
    }
    ws = WritingStyle.from_dict(data)
    assert ws.tone == "sarcastic"
    assert "Sardonic" in str(ws)


def test_writing_style_detail_levels() -> None:
    data = {
        "category": "Academic Writing",
        "name": "Sardonic Critique",
        "tone": "sarcastic",
        "characteristics": ["dry humor", "understatement"],
        "description": "Academic critique with a sardonic edge",
    }
    ws = WritingStyle.from_dict(data)
    high = ws.to_str(DetailLevel.HIGH_LEVEL)
    assert "Academic Writing" in high
    assert "sarcastic" in high
    assert "dry humor" not in high

    detailed = ws.to_str(DetailLevel.DETAILED)
    assert "Sardonic" in detailed
    assert "dry humor" in detailed


def test_scenario_from_dict_and_str() -> None:
    data = {
        "broad_title": "Production incident",
        "theme": "Technology crisis",
        "title": "Production Outage",
        "context": "E-commerce peak season",
        "setting": "On-call at 2am",
        "stakes": "Revenue loss",
        "description": "Debugging a critical production outage",
    }
    sc = Scenario.from_dict(data)
    assert sc.stakes == "Revenue loss"
    assert "Outage" in str(sc)


def test_scenario_detail_levels() -> None:
    data = {
        "broad_title": "Production incident",
        "theme": "Technology crisis",
        "title": "Production Outage",
        "context": "E-commerce peak season",
        "setting": "On-call at 2am",
        "stakes": "Revenue loss",
        "description": "Debugging a critical production outage",
    }
    sc = Scenario.from_dict(data)
    high = sc.to_str(DetailLevel.HIGH_LEVEL)
    assert "Production incident" in high
    assert "Technology crisis" in high
    assert "On-call" in high
    assert "E-commerce" not in high
    assert "Revenue" not in high

    detailed = sc.to_str(DetailLevel.DETAILED)
    assert "E-commerce" in detailed
    assert "Revenue" in detailed
    assert "Production Outage" in detailed


def test_domain_from_dict_and_str() -> None:
    data = {
        "broad_category": "Business",
        "area": "Supply Chain",
        "name": "Supply Chain Logistics",
        "parent_field": "Operations Management",
        "description": "Managing flow of goods",
        "key_concepts": ["inventory", "forecasting"],
    }
    d = Domain.from_dict(data)
    assert d.parent_field == "Operations Management"
    assert "inventory" in str(d)


def test_domain_detail_levels() -> None:
    data = {
        "broad_category": "Business",
        "area": "Supply Chain",
        "name": "Supply Chain Logistics",
        "parent_field": "Operations Management",
        "description": "Managing flow of goods",
        "key_concepts": ["inventory", "forecasting"],
    }
    d = Domain.from_dict(data)
    high = d.to_str(DetailLevel.HIGH_LEVEL)
    assert "Business" in high
    assert "Supply Chain" in high
    assert "inventory" not in high
    assert "Operations" not in high

    detailed = d.to_str(DetailLevel.DETAILED)
    assert "inventory" in detailed
    assert "Operations" in detailed


def test_science_topic_from_dict_and_str() -> None:
    data = {
        "broad_topic": "Quantum Physics",
        "name": "Quantum Entanglement",
        "scientific_field": "Physics",
        "subfield": "Quantum Mechanics",
        "description": "Non-local correlations between particles",
    }
    st = ScienceTopic.from_dict(data)
    assert st.scientific_field == "Physics"
    assert "Quantum Entanglement" in str(st)


def test_science_topic_detail_levels() -> None:
    data = {
        "broad_topic": "Quantum Physics",
        "name": "Quantum Entanglement",
        "scientific_field": "Physics",
        "subfield": "Quantum Mechanics",
        "description": "Non-local correlations between particles",
    }
    st = ScienceTopic.from_dict(data)
    high = st.to_str(DetailLevel.HIGH_LEVEL)
    assert "Quantum Physics" in high
    assert "Physics" in high
    assert "Entanglement" not in high
    assert "Non-local" not in high

    detailed = st.to_str(DetailLevel.DETAILED)
    assert "Entanglement" in detailed
    assert "Non-local" in detailed


def test_language_from_dict_and_str() -> None:
    data = {
        "category": "Romance language",
        "name": "Brazilian Portuguese",
        "region": "Brazil",
        "register": "informal",
        "script": "Latin",
        "cultural_notes": "Rich use of diminutives",
    }
    lang = Language.from_dict(data)
    assert lang.region == "Brazil"
    assert "Portuguese" in str(lang)


def test_language_detail_levels() -> None:
    data = {
        "category": "Romance language",
        "name": "Brazilian Portuguese",
        "region": "Brazil",
        "register": "informal",
        "script": "Latin",
        "cultural_notes": "Rich use of diminutives",
    }
    lang = Language.from_dict(data)
    high = lang.to_str(DetailLevel.HIGH_LEVEL)
    assert "Romance language" in high
    assert "informal" in high
    assert "Latin" not in high
    assert "diminutives" not in high

    detailed = lang.to_str(DetailLevel.DETAILED)
    assert "Portuguese" in detailed
    assert "Latin" in detailed
    assert "diminutives" in detailed


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


def test_reasoning_pattern_detail_levels() -> None:
    data = {
        "name": "Proof by Contradiction",
        "category": "Deductive",
        "description": "Assume the opposite and derive a contradiction",
        "when_to_use": "When direct proof is difficult",
    }
    rp = ReasoningPattern.from_dict(data)
    high = rp.to_str(DetailLevel.HIGH_LEVEL)
    assert "Contradiction" in high
    assert "Deductive" in high
    assert "Assume" not in high

    detailed = rp.to_str(DetailLevel.DETAILED)
    assert "Assume" in detailed
    assert "direct proof" in detailed


def test_emotional_state_from_dict_and_str() -> None:
    data = {
        "category": "Hopefulness",
        "name": "Cautiously Optimistic",
        "intensity": "moderate",
        "valence": "positive",
        "behavioral_description": "Hopeful but guarded",
        "example": "Waiting for medical test results after initial positive signs",
    }
    es = EmotionalState.from_dict(data)
    assert es.valence == "positive"
    assert "Cautiously" in str(es)


def test_emotional_state_detail_levels() -> None:
    data = {
        "category": "Hopefulness",
        "name": "Cautiously Optimistic",
        "intensity": "moderate",
        "valence": "positive",
        "behavioral_description": "Hopeful but guarded",
        "example": "Waiting for medical test results after initial positive signs",
    }
    es = EmotionalState.from_dict(data)
    high = es.to_str(DetailLevel.HIGH_LEVEL)
    assert "Hopefulness" in high
    assert "moderate" in high
    assert "positive" in high
    assert "guarded" not in high

    detailed = es.to_str(DetailLevel.DETAILED)
    assert "Cautiously" in detailed
    assert "guarded" in detailed
    assert "medical test" in detailed


def test_instruction_complexity_from_dict_and_str() -> None:
    data = {
        "name": "Multi-step API specification",
        "level": "high",
        "ambiguity": "moderate",
        "description": "Multi-step specification with constraints",
        "example": "Build a REST API with auth, rate limiting, and caching",
    }
    ic = InstructionComplexity.from_dict(data)
    assert ic.level == "high"
    assert "REST API" in str(ic)


def test_instruction_complexity_detail_levels() -> None:
    data = {
        "name": "Multi-step API specification",
        "level": "high",
        "ambiguity": "moderate",
        "description": "Multi-step specification with constraints",
        "example": "Build a REST API with auth, rate limiting, and caching",
    }
    ic = InstructionComplexity.from_dict(data)
    high = ic.to_str(DetailLevel.HIGH_LEVEL)
    assert "Multi-step API" in high
    assert "high" in high
    assert "moderate" in high
    assert "REST API" not in high

    detailed = ic.to_str(DetailLevel.DETAILED)
    assert "Multi-step specification" in detailed
    assert "REST API" in detailed
