# 🎲 liquidrandom

Pseudo-random seed data for ML/LLM training diversity.

When using LLMs to generate training data, outputs tend to be repetitive and lack variety. `liquidrandom` solves this by providing a large pool of diverse, pre-generated seed data (personas, jobs, scenarios, etc.) that you can inject into your prompts to steer generation toward more varied outputs.

## 📦 Installation

```bash
pip install liquidrandom
# or
uv add liquidrandom
```

## 🚀 Quick Start

```python
import liquidrandom

# Get a random persona to inject into your LLM prompt
persona = liquidrandom.persona()
print(persona)
# Alice is a 30-year-old female from Canada. They work as an engineer. ...

# Get a random coding task
task = liquidrandom.coding_task()
print(task)
# [Python, medium] Implement a trie: Build a trie data structure ...
```

## 📋 Available Categories

| Function | Returns | Description |
|---|---|---|
| `liquidrandom.persona()` | `Persona` | 🧑 Random personas with name, age, gender, occupation, nationality, personality traits, background |
| `liquidrandom.job()` | `Job` | 💼 Professions with title, industry, description, required skills, experience level |
| `liquidrandom.coding_task()` | `CodingTask` | 💻 Programming challenges with title, language, difficulty, description, constraints, expected behavior |
| `liquidrandom.math_category()` | `MathCategory` | 🔢 Math categories with name, field, description, example problems |
| `liquidrandom.writing_style()` | `WritingStyle` | ✍️ Writing styles with name, tone, characteristics, description |
| `liquidrandom.scenario()` | `Scenario` | 🎬 Real-world scenarios with title, context, setting, stakes, description |
| `liquidrandom.domain()` | `Domain` | 🌐 Knowledge domains with name, parent field, description, key concepts |
| `liquidrandom.science_topic()` | `ScienceTopic` | 🔬 Scientific topics with name, field, subfield, description |
| `liquidrandom.language()` | `Language` | 🌍 Languages/locales with name, region, register, script, cultural notes |
| `liquidrandom.reasoning_pattern()` | `ReasoningPattern` | 🧠 Reasoning approaches with name, category, description, when to use |
| `liquidrandom.emotional_state()` | `EmotionalState` | 💭 Emotional states with name, intensity, valence, behavioral description |
| `liquidrandom.instruction_complexity()` | `InstructionComplexity` | 📐 Instruction complexity levels with level, ambiguity, description, example |

## 🛠️ Usage Example

Use `liquidrandom` to add diversity to your LLM data generation pipeline:

```python
import liquidrandom
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="<OPENROUTER_API_KEY>",
)

persona = liquidrandom.persona()
style = liquidrandom.writing_style()
topic = liquidrandom.science_topic()

prompt = f"""You are {persona}
Write in the following style: {style}
Explain the following topic: {topic}"""

response = client.chat.completions.create(
    model="liquid/lfm-2-24b-a2b",
    messages=[{"role": "user", "content": prompt}],
)
```

Each call to a `liquidrandom` function returns a typed dataclass. You can use them directly in f-strings (via `__str__`) or access their individual fields:

```python
persona = liquidrandom.persona()
print(persona.name)               # "Alice"
print(persona.age)                 # 30
print(persona.personality_traits)  # ["curious", "patient"]
```

## ⚙️ How It Works

The dataset contains 340,000+ samples across 12 categories, generated using hierarchical taxonomy trees with LLM-based quality validation and fuzzy deduplication.

Seed data is hosted on HuggingFace ([mlech26l/liquidrandom-data](https://huggingface.co/datasets/mlech26l/liquidrandom-data)) as zstd-compressed Parquet files. On first use, only the requested category file is downloaded and cached locally. Subsequent calls use the cached data.

## 📄 License

MIT
