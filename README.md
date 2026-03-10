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
| `liquidrandom.tool_group()` | `ToolGroup` | 🔧 Groups of related LLM tools/functions in OpenAI format, each with multiple parameter variations |

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

The dataset contains 340,000+ samples across 13 categories, generated using hierarchical taxonomy trees with LLM-based quality validation and fuzzy deduplication.

Seed data is hosted on HuggingFace ([mlech26l/liquidrandom-data](https://huggingface.co/datasets/mlech26l/liquidrandom-data)) as zstd-compressed Parquet files. On first use, only the requested category file is downloaded and cached locally. Subsequent calls use the cached data.

## 🔍 Example Samples

Here are some real samples from the dataset to give you a sense of the specificity and diversity:

<details>
<summary>🧑 Persona</summary>

```
Carla Antonia Fernandez is a 45-year-old Female from Argentine-Spanish.
They work as a Lead Midwife for High-Risk Pregnancies.
Personality traits: Authoritative, Nurturing, Tired, Dedicated.
Background: With twenty years of experience, Carla mentors younger staff
in the public hospital near Retiro. She focuses on patients experiencing
domestic violence during pregnancy.
```
</details>

<details>
<summary>💼 Job</summary>

```
ABET Student Survey Data Manager (Engineering Education, mid):
Manages the collection and analysis of student survey data for ABET reviews.
Ensures anonymity and data integrity for program evaluation.
Required skills: ABET Data Requirements, Survey Management, Data Privacy,
Statistical Analysis
```
</details>

<details>
<summary>💻 Coding Task</summary>

```
[CUDA, medium] Outlier Clipping Before Quantization:
Implement a preprocessing kernel that clips extreme outlier values in
attention matrices before per-token quantization to improve quantization
fidelity. Clip based on a percentile threshold per token.
Constraints: Calculate 99th percentile per token row; Clip values exceeding
threshold before quantizing; Minimize additional kernel launch overhead.
Expected behavior: Produces quantized matrices with reduced quantization
error by mitigating the impact of activation outliers.
```
</details>

<details>
<summary>🎬 Scenario</summary>

```
Shareholder ROI Question
Context: Investors question spend on empty office amenities.
Setting: Annual Shareholder Meeting
Stakes: Capital efficiency and investor confidence.
A shareholder asks why the company funds a cafeteria used by 20% of staff.
The CEO must defend the office budget or commit to shifting funds to remote
productivity stipends.
```
</details>

<details>
<summary>🔬 Science Topic</summary>

```
Impact of statin co-administration on CCB-Digoxin metabolic pathways
(Clinical Medicine, Lipidology):
This topic examines if statin co-administration competes for metabolic
pathways with CCBs and Digoxin. It determines if triple therapy increases
AV node conduction risk.
```
</details>

<details>
<summary>✍️ Writing Style</summary>

```
Vermilion Bloodletting Aesthetic (tone: Violent, vibrant, sacrificial):
The act of painting is portrayed as a vampiric ritual where the canvas
drinks the artist's vitality to gain sentience. Descriptions focus on the
vividness of red hues, suggesting that true art requires the spillage of
moral or physical blood.
Characteristics: Red pigments equated with life force, Painting acts
described as surgical procedures, Canvas absorbing blood as nourishment,
Glorification of artistic consumption
```
</details>

<details>
<summary>🔧 Tool Group</summary>

```
Tool Group: Cloud Object Versioning and History Retrieval
  (Travel > Cloud Storage > Versioning)
Tools (4): simulate_restore_impact, calculate_storage_delta,
  authorize_rollback, execute_rollback
Each tool has 8 parameter variations, e.g. for simulate_restore_impact:
  v0: simulate_restore_impact(bucket_name, object_key, target_version_id)
  v1: evaluate_rollback_safety(storage_context, version_selector, scan_preferences)
  v2: assess_restore_feasibility(cloud_resource_arn, historical_epoch, validation_flags)
  ...
```
</details>

### 🔧 Tool Groups: Usage Notes

Tool groups provide sets of related LLM tool/function definitions in OpenAI function calling format, each with 8 parameter variations. They are useful for testing agentic workflows against diverse tool signatures.

```python
import liquidrandom

group = liquidrandom.tool_group()
print(group.domain)        # "Cloud Object Versioning and History Retrieval"
print(len(group.tools))    # 3-5 tools per group

for tool in group.tools:
    print(tool.canonical_name)        # "search_flights"
    print(len(tool.variations))       # 8 parameter variations
    for var in tool.variations:
        print(var.name)               # "find_available_flights"
        print(var.parameters)         # OpenAI JSON Schema dict
        print(var.returns)            # OpenAI JSON Schema dict
```

**Known limitations:**

- **Low tool count diversity.** ~62% of groups have 3 tools, ~37% have 4, and only ~1% have 5. If your use case is sensitive to tool count, consider sampling multiple groups and combining or dropping tools to get a wider distribution.
- **Cross-variation interface drift.** The canonical (v0) tools within a group have coherent inter-tool interfaces (e.g. tool A's return value matches tool B's input). However, variations were generated independently per tool, so variation N of tool A may not perfectly align with variation N of tool B (e.g. differently named keys or slightly different semantics). This is fine for testing individual tool signatures but may cause inconsistencies if you mix-and-match variation indices across tools within a group.

## TODO / Future Improvements

- **Tool group cross-variation alignment.** Currently, variations for each tool within a group are generated independently. This means variation N of tool A and variation N of tool B have no guaranteed interface compatibility (matching parameter/return names). A future improvement would be to generate variations at the group level rather than per-tool, ensuring that all tools in a variation set have coherent inter-tool interfaces.
- **Over-specificity in some categories.** Some seed data types have far more samples than needed for good diversity coverage. In particular, `language` (~22k samples) and `writing_style` (~25k samples) are over-specified, containing highly niche entries (e.g. obscure professional terminologies, hyper-specific literary styles) that are unlikely to be useful for most generation pipelines. These categories would benefit from being regenerated with a shallower taxonomy (1-2 levels) and a much smaller target count (~500 samples) focused on broad, practical coverage. Most other categories (e.g. `job`, `persona`, `coding_task`) benefit from their large sample counts and are fine as-is.

## 📄 License

MIT
