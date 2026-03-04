# Liquidrandom

This is a python package for pseudo random data generation. It is made for machine learning to seed the data generation process.
ie. a typical use-case is using an LLM to generate data for training another model, which has the problem of a lack of randomness in the generated data. This package is designed to solve that problem by providing a way to add randomness to the data generation process by for instance adding random personas or jobs into the prompt.

## Package

The package is called `liquidrandom` and can be installed via pip.
We are using typed python.
We are using ty and uv for type checking and dependency management respectively.
Create a pyproject.toml file for the package.

The package should be usable as following:

```python3
import liquidrandom

persona = liquidrandom.persona()
job = liquidrandom.job()
coding_task = liquidrandom.coding_task()
math_category = liquidrandom.math_category()
writing_style = liquidrandom.writing_style()
scenario = liquidrandom.scenario()
domain = liquidrandom.domain()
science_topic = liquidrandom.science_topic()
language = liquidrandom.language()
reasoning_pattern = liquidrandom.reasoning_pattern()
emotional_state = liquidrandom.emotional_state()
instruction_complexity = liquidrandom.instruction_complexity()
```

Use types and objects for the seed data, for instance a Persona object with name, age, etc. and a Job object with name and description.
Make sure to overwrite the __str__ method of the objects to return a string representation of the object that can be used in the prompt for the LLM.

Create a clear and concise README.md file that explains the package, how to install it, and how to use it. Include examples of how to use the package in the README.md file.

## Seed data

The seed data we want to host on huggingface. Make sure only the needed data is fetched and that the package is not too heavy. 
The seed data should be cached locally after the first fetch.

## Seed data generation

Let's create a seperate folder where we put the "run once" scripts that generate the seed data and upload it to huggingface.
We want to use openrouter.ai for the generation of the seed data.

```python
from openai import OpenAI

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="<OPENROUTER_API_KEY>",
)

# First API call with reasoning
response = client.chat.completions.create(
  model="qwen/qwen3.5-397b-a17b",
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
  model="qwen/qwen3.5-397b-a17b",
  messages=messages,
  extra_body={"reasoning": {"enabled": True}}
)
```

You can assume the envvar OPENROUTER_API_KEY is set.
For uploading to huggingface, either let me know how to do it, or you the envvar HF_TOKEN.

We want to accelerate the process, thus using either async or ThreadPoolExecutor for the generation of the seed data in parallel.
The batch size should be controlable via a cli arguemnt --batch-size

Per LLM call we want to generate at least k samples, where n is controlable via a cli argument --k. Make sure to include in the prompting and the parsing of the response that there are multiple samples generated per call. 

After each generation call, we want to call the same LLM again to check if the generated data is good enough. This process should check if there are collapsed or degenerate samples, e.g. halluications or repetitive samples. If the generated data is not good enough, we discard the current call and generate new data until we have good enough data. This is to ensure that the generated data is of high quality and not too similar to each other.

In total we want to generate at least n samples per category, where n is controlable via a cli argument --n.

## Diversity strategy: Hierarchical Taxonomy Tree

To generate 10k-100k samples without collapsing into repetitive subsets, we use a two-phase hierarchical approach:

### Phase 1: Taxonomy generation

Before generating any seed data samples, first use the LLM to generate a deep taxonomy tree for each category. The tree should be broad at the top and increasingly specific at the leaves.

Example for Science topics:
```
Science
├── Physics
│   ├── Quantum Mechanics
│   │   ├── Entanglement phenomena
│   │   │   ├── Bell state preparation in ion traps
│   │   │   ├── Quantum teleportation protocols
│   │   │   └── Loophole-free Bell tests
│   │   ├── Decoherence
│   │   │   ├── Environment-induced superselection
│   │   │   └── ...
│   ├── Thermodynamics
│   │   └── ...
├── Biology
│   └── ...
```

The taxonomy depth should auto-scale based on the target sample count:
- `required_leaf_nodes = target_samples / samples_per_leaf`
- More samples → deeper tree → more leaf nodes
- The taxonomy itself should be generated in stages: first top-level branches, then expand each branch deeper, to avoid context window limitations.

The taxonomy generation should also be controllable via a cli argument `--taxonomy-depth` to control the depth of the taxonomy tree and `--samples-per-leaf` to control how many samples to generate per leaf node.

The taxonomy should be saved to disk as JSON so it can be inspected, reused, and resumed.

### Phase 2: Round-robin sample generation

Generate samples by cycling through all leaf nodes in round-robin fashion:

```
for leaf in cycle(all_leaf_nodes):
    generate k samples for this specific leaf
    validate & dedup
    if leaf.count >= target_per_leaf:
        mark leaf as done
```

This ensures that no single subtopic gets over-represented. Each generation prompt should include the leaf node's full path in the taxonomy (e.g. "Science > Physics > Quantum Mechanics > Entanglement phenomena > Bell state preparation in ion traps") to anchor the LLM to that specific subtopic.

The previously generated samples for the current leaf should be included in the prompt to avoid repetition within the same leaf.

### Deduplication: Fuzzy string matching

Use token-level fuzzy string matching to detect and reject near-duplicate samples. No ML dependency needed.

Approach:
- Use Jaccard similarity on token sets (word-level) to compare new samples against all existing samples in the same category.
- Reject samples with a Jaccard similarity above a configurable threshold (default: 0.7).
- Additionally, normalize samples before comparison (lowercase, strip punctuation, collapse whitespace) to catch trivial reformulations.
- The dedup check should run on the `__str__` representation of each sample.
- The threshold should be controllable via a cli argument `--dedup-threshold`.

## Categories of seed data

- Persona: random personas with name, age etc.
- Professions or jobs: Random professions or jobs with a name and description of the job.
- Coding tasks: Specific programming challenges and tasks (e.g. "implement a trie-based autocomplete with fuzzy matching", "write a rate limiter using the token bucket algorithm"). Should include the programming context, constraints, and expected behavior.
- Math categories: Random math categories like algebra, geometry, etc. with a description of the category.
- Writing styles / tones: Diverse writing styles and tones (e.g. "sardonic academic critique", "enthusiastic technical blogger", "dry legal prose"). Includes the style name and a description of its characteristics.
- Scenarios / situations: Real-world scenarios or situations (e.g. "debugging a production outage at 2am", "negotiating a contract with a difficult vendor"). Includes the scenario description and relevant context.
- Domains / topics: Specific knowledge domains and subtopics (e.g. "supply chain logistics for perishable goods", "comparative analysis of NoSQL databases"). Includes the domain name and a detailed description.
- Science topics: Specific scientific topics and phenomena (e.g. "quantum entanglement in photon pairs", "CRISPR-Cas9 off-target effects in gene therapy", "tidal locking in exoplanetary systems"). Includes the topic name, scientific field, and a description.
- Languages / locales: Random languages, dialects, or cultural contexts (e.g. "Brazilian Portuguese, informal register", "Kansai Japanese dialect"). Includes the language, region, register, and cultural notes.
- Reasoning patterns: Types of reasoning or problem-solving approaches (e.g. "proof by contradiction", "cost-benefit analysis with uncertainty", "analogical reasoning from biology to engineering"). Includes the pattern name and description of how it works.
- Emotional states: Specific emotional states or moods (e.g. "frustrated but trying to stay polite", "cautiously optimistic after a setback"). Includes the emotional state and behavioral description.
- Instruction complexity: Varying levels of instruction complexity and ambiguity (e.g. "vague one-liner request", "detailed multi-step specification with constraints", "contradictory requirements"). Includes the complexity level, a description, and an example.

Make sure the seed data is very specific, ie. "algebra" is not a good seed data, but "different ways to solve linear equations" is a good seed data. This is to ensure that the generated data for training is more diverse and not too similar to each other.

For the generation process, use the rich python library to show progress as well as the expected time of arrival (ETA) for the generation process.

dependencies and README for the run once seed generation scripts should be in the same folder as the scripts and different from the main package. The main package should not have any dependencies that are not needed for the generation of the seed data.

