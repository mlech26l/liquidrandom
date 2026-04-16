# Example scripts and python modules on using `liquidrandom` for a synthetic audio-to-tool-calling dataset.

Instructions: Create a python script (or module) that generates synthetic data of the from:

- system prompt with tool definitions
- user turn in the form of audio input of a using speaking a question or command
- assistant response with just text or tool call+text
- another user turn with audio input ...
- ... multiple turns per conversation, eg. 2-5 turns


We are using uv for package and dependency management.

## Seeding the sample:
- Use the liquidrandom person function to make the user request tailed to the background of the persona
- Use the liquidrandom tool_group function to generate a random tool group and inject the tool definitions, get 1-3 tool groups per sample and drop 0-3 random tools (if tool list would be empty)

## Process:

Using an LLM (via openrouter.ai) we first generate full text conversation.
- This include already the tool calls and all user and assistant turns.
- Make sure in the prompting we add that the user turn will be converted to audio, thus not very long language, no special characters, etc. and add filler words to make it more natural.
- After we generated the conversation we all the LLM again but this time to check quality. Specifically we want to check that the tool calls are correct (match the definitions and parameters of the tools we injected), that there are not hallucinated arguments in the tool call, and the user turns are appropriate for audio.

Example of hallucinated argument: 
user: Set me a reminder to call John tomorrow at 3pm
assistant: calls set_reminder with arguments { "time": "tomorrow at 3pm", "contact": "John", "timezone": "PST" }
ISSUE: the timezone is hallucinated, it was not in the user request and the tool definition does not have a timezone argument.
Make sure such information are either, 1. specified by the user, or 2. not included in the tool call, or 3. specified in the system prompt or somewhere else in the previous conversation.

## additional randomization:

Even with the process discussed above, we will most likely see some bias, eg. user requests always the very first tool in the list, or every user turn asks for a tool call.
To avoid this, we will do the following:
1. shuffle the tool list when we use it for the generation of the conversation (in the saved sample we should keep the original order).
2. we should inject random (unrelated user request -> assistant response without tool call) turns in the conversation, to make it more natural and less biased towards tool calls in every turn. eg. you may want to have a seperate generation step, where we generate Q-A pair without tool in the scope. Again, use liquidrandom to seed the Q-A pair generation. 


## Conversation to audio:

We will add the conversation to audio later on.


## Saving format:

We should save the generated data (conversation) as parquet files.
Have a "text_only" folder that stores the text only data, as well as a "audio" folder that stores conversation including the audio of the user turn.
Note: since we are implementing the audio conversion later on, the audio folder should be empty now.

## Inspection:

Include a script that allows human-in-the-loop inspection of the generated conversations.

## LLM generation

IMPORTANT: Use concurrent requetss with batch-size cli argument for the genration (pick async or Threading whatever you like)
You can assume the envvar OPENROUTER_API_KEY is set and available for the script.

```python
from openai import OpenAI

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="<OPENROUTER_API_KEY>",
)

# First API call with reasoning
response = client.chat.completions.create(
  model="qwen/qwen3.5-35b-a3b",
  messages=[
          {
            "role": "user",
            "content": "How many r's are in the word 'strawberry'?"
          }
        ],
  extra_body={"reasoning": {"enabled": True}}
)

# Extract the assistant message with reasoning_details
response = response.choices[0].message

# Preserve the assistant message with reasoning_details
messages = [
  {"role": "user", "content": "How many r's are in the word 'strawberry'?"},
  {
    "role": "assistant",
    "content": response.content,
    "reasoning_details": response.reasoning_details  # Pass back unmodified
  },
  {"role": "user", "content": "Are you sure? Think carefully."}
]

# Second API call - model continues reasoning from where it left off
response2 = client.chat.completions.create(
  model="qwen/qwen3.5-35b-a3b",
  messages=messages,
  extra_body={"reasoning": {"enabled": True}}
)
```

## liquidrandom

README
🎲 liquidrandom
Pseudo-random seed data for ML/LLM training diversity.

When using LLMs to generate training data, outputs tend to be repetitive and lack variety. liquidrandom solves this by providing a large pool of diverse, pre-generated seed data (personas, jobs, scenarios, etc.) that you can inject into your prompts to steer generation toward more varied outputs.

📦 Installation
pip install liquidrandom
# or
uv add liquidrandom
🚀 Quick Start
import liquidrandom

# Get a random persona to inject into your LLM prompt
persona = liquidrandom.persona()
print(persona)
# Alice is a 30-year-old female from Canada. They work as an engineer. ...

# Get a random coding task
task = liquidrandom.coding_task()
print(task)
# [Python, medium] Implement a trie: Build a trie data structure ...
📋 Available Categories
Function	Returns	Description
liquidrandom.persona()	Persona	🧑 Random personas with name, age, gender, occupation, nationality, personality traits, background
liquidrandom.job()	Job	💼 Professions with title, industry, description, required skills, experience level
liquidrandom.coding_task()	CodingTask	💻 Programming challenges with title, language, difficulty, description, constraints, expected behavior
liquidrandom.math_category()	MathCategory	🔢 Math categories with name, field, description, example problems
liquidrandom.writing_style()	WritingStyle	✍️ Writing styles with name, tone, characteristics, description
liquidrandom.scenario()	Scenario	🎬 Real-world scenarios with title, context, setting, stakes, description
liquidrandom.domain()	Domain	🌐 Knowledge domains with name, parent field, description, key concepts
liquidrandom.science_topic()	ScienceTopic	🔬 Scientific topics with name, field, subfield, description
liquidrandom.language()	Language	🌍 Languages/locales with name, region, register, script, cultural notes
liquidrandom.reasoning_pattern()	ReasoningPattern	🧠 Reasoning approaches with name, category, description, when to use
liquidrandom.emotional_state()	EmotionalState	💭 Emotional states with name, intensity, valence, behavioral description
liquidrandom.instruction_complexity()	InstructionComplexity	📐 Instruction complexity levels with level, ambiguity, description, example
liquidrandom.tool_group()	ToolGroup	🔧 Groups of related LLM tools/functions in OpenAI format, each with multiple parameter variations
🛠️ Usage Example
Use liquidrandom to add diversity to your LLM data generation pipeline:

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
Each call to a liquidrandom function returns a typed dataclass. You can use them directly in f-strings (via __str__) or access their individual fields:

persona = liquidrandom.persona()
print(persona.name)               # "Alice"
print(persona.age)                 # 30
print(persona.personality_traits)  # ["curious", "patient"]
⚙️ How It Works
The dataset contains 340,000+ samples across 13 categories, generated using hierarchical taxonomy trees with LLM-based quality validation and fuzzy deduplication.

Seed data is hosted on HuggingFace (mlech26l/liquidrandom-data) as zstd-compressed Parquet files. On first use, only the requested category file is downloaded and cached locally. Subsequent calls use the cached data.

🔍 Example Samples
Here are some real samples from the dataset to give you a sense of the specificity and diversity:

🧑 Persona
💼 Job
💻 Coding Task
🎬 Scenario
🔬 Science Topic
✍️ Writing Style
🔧 Tool Group
🔧 Tool Groups: Usage Notes
Tool groups provide sets of related LLM tool/function definitions in OpenAI function calling format, each with 8 parameter variations. They are useful for testing agentic workflows against diverse tool signatures.

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
Known limitations:

Low tool count diversity. ~62% of groups have 3 tools, ~37% have 4, and only ~1% have 5. If your use case is sensitive to tool count, consider sampling multiple groups and combining or dropping tools to get a wider distribution.
Cross-variation interface drift. The canonical (v0) tools within a group have coherent inter-tool interfaces (e.g. tool A's return value matches tool B's input). However, variations were generated independently per tool, so variation N of tool A may not perfectly align with variation N of tool B (e.g. differently named keys or slightly different semantics). This is fine for testing individual tool signatures but may cause inconsistencies if you mix-and-match variation indices across tools within a group.
TODO / Future Improvements
Tool group cross-variation alignment. Currently, variations for each tool within a group are generated independently. This means variation N of tool A and variation N of tool B have no guaranteed interface compatibility (matching parameter/return names). A future improvement would be to generate variations at the group level rather than per-tool, ensuring that all tools in a variation set have coherent inter-tool interfaces.
Over-specificity in some categories. Some seed data types have far more samples than needed for good diversity coverage. In particular, language (~22k samples) and writing_style (~25k samples) are over-specified, containing highly niche entries (e.g. obscure professional terminologies, hyper-specific literary styles) that are unlikely to be useful for most generation pipelines. These categories would benefit from being regenerated with a shallower taxonomy (1-2 levels) and a much smaller target count (~500 samples) focused on broad, practical coverage. Most other categories (e.g. job, persona, coding_task) benefit from their large sample counts and are fine as-is.
📄 License
MIT
