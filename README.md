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
# Alice is a 30-year-old female from Canada. They work as an engineer. Personality traits: curious, patient. Background: Grew up in Toronto ...

# High-level output (fewer fields, broader)
print(persona.brief())
# Alice is a 30-year-old female from Canada. They work as an engineer.

# Get a random coding task
task = liquidrandom.coding_task()
print(task)
# [Python, medium] Implement a trie: Build a trie data structure ...

# Access manual-only fields directly
print(task.follow_up_task)   # "Add autocomplete functionality"
print(task.change_request)   # "Support case-insensitive search"
print(task.edge_cases)       # ["empty string input", "unicode characters"]
```

## 📋 Available Categories

Each category supports two detail levels:
- **`str(x)`** / **`x.detailed()`**: All standard fields (default)
- **`x.brief()`**: Only broad, summary-level fields

Some models also have **manual-only fields** that are never included in string output but accessible as attributes.

> **Tip:** When combining multiple seed data types in a single prompt, prefer `.brief()` for each. Brief outputs are more compatible with each other and less likely to overwhelm the prompt.

| Function | Returns | High-Level Fields | Detailed Fields | Manual-Only Fields |
|---|---|---|---|---|
| `persona()` | `Persona` | name, age, gender, occupation, nationality | + personality_traits, background | |
| `job()` | `Job` | job_category, sector, experience_level | + title, industry, description, required_skills | |
| `coding_task()` | `CodingTask` | title, language, difficulty | + description, constraints, expected_behavior | follow_up_task, change_request, edge_cases |
| `math_category()` | `MathCategory` | broad_topic, field | + name, description, example_problems | |
| `writing_style()` | `WritingStyle` | category, tone | + name, characteristics, description | |
| `scenario()` | `Scenario` | broad_title, theme, setting | + title, context, stakes, description | |
| `domain()` | `Domain` | broad_category, area | + name, parent_field, description, key_concepts | |
| `science_topic()` | `ScienceTopic` | broad_topic, scientific_field | + name, subfield, description | |
| `language()` | `Language` | category, register | + name, region, script, cultural_notes | |
| `reasoning_pattern()` | `ReasoningPattern` | name, category | + description, when_to_use | |
| `emotional_state()` | `EmotionalState` | category, intensity, valence | + name, behavioral_description, example | |
| `instruction_complexity()` | `InstructionComplexity` | name, level, ambiguity | + description, example | |
| `tool_group()` | `ToolGroup` | domain, description, taxonomy_path, kind, tools | (same) | Unified pool (`kind`: `"default"` SaaS, `"physical"` embodied) |
| `physical_tool_group()` | `ToolGroup` | same | (same) | Convenience filter: `tool_group()` constrained to `kind == "physical"` |

### 📝 Per-Category Usage Notes

**Personas** -- Use `.brief()` as the default. The brief version provides enough context (name, age, occupation, nationality) to steer generation without overwhelming the prompt.

**Jobs** -- Use `.brief()` as the default. Similar to personas, the broad job category and sector are usually sufficient.

**Coding Tasks** -- Use `.detailed()` (the default). The brief version is not specific enough to serve as a coding specification. Use the manual-only fields (`follow_up_task`, `change_request`, `edge_cases`) for multi-turn coding scenarios.

**Math Categories** -- Be aware of the wide gap between `.brief()` and `.detailed()`. Brief gives only the broad topic and field; detailed adds the specific topic name, description, and example problems.

**Writing Styles** -- Good as-is at both levels. No special considerations.

**Scenarios** -- Use `.brief()` as the default. The brief version already includes a descriptive title, theme, and setting.

**Domains** -- Be aware of the wide gap between `.brief()` and `.detailed()`. Brief gives only the broad category and area; detailed adds the specific domain name, parent field, and key concepts.

**Science Topics** -- Be aware of the wide gap between `.brief()` and `.detailed()`. Brief gives only the broad topic and field.

**Languages** -- Use `.brief()` only. The detailed version contains overly specific dialect/variant information that is rarely useful.

**Reasoning Patterns** -- Small curated list (~250 items). Best used in conjunction with other seed data types or your own randomization, not as a standalone source. Covers general-purpose reasoning techniques (counterfactual, abductive, analogical, etc.).

**Emotional States** -- Use `.brief()` as the default. The detailed version is very specific, but the `example` field in it provides valuable situational context if needed.

**Instruction Complexity** -- Small curated list (~250 items). Best used in conjunction with other seed data types. Covers instruction styles like "explain like I'm 5", "step by step", "first principles", etc.

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

# Use brief output for concise prompts
prompt = f"""You are {persona.brief()}
Write in the following style: {style.brief()}
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
print(persona.age)                # 30
print(persona.personality_traits)  # ["curious", "patient"]

# Control detail level
print(persona.brief())     # Broad overview
print(persona.detailed())  # Full details (same as str())
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

Tool groups provide sets of related LLM tool/function definitions in OpenAI function calling format, each with multiple parameter variations. They are useful for testing agentic workflows against diverse tool signatures.

```python
import liquidrandom

group = liquidrandom.tool_group()
print(group.domain)        # "Cloud Object Versioning and History Retrieval"
print(len(group.tools))    # 3-6 tools per group

for tool in group.tools:
    print(tool.canonical_name)        # "search_flights"
    print(len(tool.variations))       # 8 parameter variations
    for var in tool.variations:
        print(var.name)               # "find_available_flights"
        print(var.parameters)         # OpenAI JSON Schema dict
        print(var.returns)            # OpenAI JSON Schema dict
```

**Important notes:**

- **Variation compatibility.** Tool variations are compatible *within* the same variation index but *not across* indices. For example, tool A variation 1 is interface-compatible with tool B variation 1, but tool A variation 1 is NOT compatible with tool B variation 2. When selecting a variation, use the same index for all tools in the group.
- **Group size is 3-6 tools.** If you need more tools (e.g., 10+), sample multiple tool groups and combine them. Tools from different groups are independent and can be freely mixed.
- Tools with an empty variation list may occur occasionally; check and skip if needed.
- **`kind` discriminator.** Each `ToolGroup` carries a `kind` field: `"default"` for SaaS/cloud-style tools, `"physical"` for embodied/physical-AI tools (smart home voice assistants, mobile/manipulator/humanoid robots, autonomous vehicles, drones, industrial automation, agricultural/medical robotics, wearable AI). `liquidrandom.tool_group()` draws from the full pool; `liquidrandom.physical_tool_group()` is a convenience filter equivalent to sampling until `kind == "physical"`.

## 📄 License

MIT
