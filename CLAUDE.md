# Liquidrandom

Python package for pseudo-random seed data generation for ML/LLM training diversity.

## Project Structure

```
├── pyproject.toml                  # hatchling build, deps: huggingface-hub, pyarrow
├── .python-version                 # 3.12
├── README.md
├── src/liquidrandom/
│   ├── __init__.py                 # Public API: persona(), job(), etc. + model re-exports
│   ├── py.typed                    # PEP 561 marker
│   ├── _loader.py                  # HuggingFace fetch + Parquet read + in-memory cache
│   ├── _registry.py                # Category → model class + filename mapping
│   └── models/                     # 12 frozen dataclasses with from_dict() and __str__()
│       ├── __init__.py
│       ├── persona.py
│       ├── job.py
│       ├── coding_task.py
│       ├── math_category.py
│       ├── writing_style.py
│       ├── scenario.py
│       ├── domain.py
│       ├── science_topic.py
│       ├── language.py
│       ├── reasoning_pattern.py
│       ├── emotional_state.py
│       └── instruction_complexity.py
├── tests/
│   ├── test_models.py              # from_dict + __str__ for all 12 models
│   └── test_loader.py              # Parquet loading, caching, public API (mocked HF)
└── seed_generation/                # Separate project with own pyproject.toml
    ├── pyproject.toml              # deps: openai, rich, typer, huggingface-hub, pyarrow
    ├── README.md
    ├── generate.py                 # CLI: `generate` and `upload-only` commands (typer)
    ├── config.py                   # Constants and defaults
    ├── categories.py               # 12 CategoryConfig with field specs and prompt templates
    ├── taxonomy.py                 # Phase 1: BFS taxonomy tree generation
    ├── sampler.py                  # Phase 2: round-robin sample generation
    ├── validator.py                # LLM quality validation per batch
    ├── dedup.py                    # Jaccard similarity dedup on token sets
    ├── llm.py                      # AsyncOpenAI client wrapper with retries
    ├── uploader.py                 # Consolidate JSONL → Parquet + upload to HF
    └── state.py                    # Checkpoint/resume state
```

## Tooling

- **Package manager**: uv
- **Type checker**: ty (run `uv run ty check src/ tests/`)
- **Tests**: pytest (`uv run pytest tests/`)
- **Typed Python**: all code uses type annotations, `from __future__ import annotations`

## Key Design Decisions

### Data format: Parquet (not JSONL)
Seed data is stored as zstd-compressed Parquet on HuggingFace (`mlech26l/liquidrandom-data`). Parquet gives ~5-10x smaller files than JSONL, and pyarrow is already in the typical ML stack. The intermediate per-leaf files during generation remain JSONL (append-friendly), converted to Parquet at upload time.

### Data loading
- `_loader.py` uses `hf_hub_download()` to fetch a single Parquet file per category (not the whole repo)
- Disk cache handled by huggingface-hub (~/.cache/huggingface/hub/)
- In-memory cache via module-level `_cache` dict avoids re-parsing within a session

### Leaf file naming: SHA-256 hash
Per-leaf sample files use `hashlib.sha256(path)[:16]` as filename to avoid filesystem path length limits from deep taxonomy paths.

### Parallelization
Both taxonomy expansion and sample generation use `asyncio.Semaphore(batch_size)` with `AsyncOpenAI`. Progress updates per-leaf completion (not per-batch) for responsive UI.

### Generation defaults
Tuned so `k` matches `samples_per_leaf` to minimize wasted LLM output:
- `n=22000, k=100, batch_size=32, taxonomy_depth=3, samples_per_leaf=100`
- Yields ~216 leaves, ~100 samples each, ~432 LLM call pairs (generate + validate)

### List field handling in models
All `from_dict()` methods use `list(data["field"] or [])` to handle None values from Parquet columns.

## Seed Data Generation

### LLM
- Model: `qwen/qwen3.5-397b-a17b` via OpenRouter (`OPENROUTER_API_KEY` env var)
- Reasoning enabled via `extra_body={"reasoning": {"enabled": True}}`
- JSON responses extracted with fallback parsing (direct, code blocks, brace matching)

### Two-phase approach
1. **Taxonomy**: BFS expansion of category tree, branching factor auto-scaled from `(n / samples_per_leaf)^(1/depth)`. Saved as JSON in `output/taxonomies/`.
2. **Round-robin sampling**: all incomplete leaves dispatched concurrently (semaphore-limited), each does generate → validate → dedup. Progress bar updates per-leaf.

### Quality pipeline
- **Validation**: second LLM call checks for empty/placeholder content, hallucinations, repetitiveness, off-topic, insufficient specificity. >50% rejection discards entire batch (up to 3 retries).
- **Dedup**: Jaccard similarity on normalized word-level token sets (default threshold 0.7). Checks within-leaf and within-batch.
- **Stall detection**: 3 consecutive rounds with zero progress stops the category.

### Upload
`python generate.py upload-only` consolidates per-leaf JSONL into per-category zstd Parquet, generates a dataset card, uploads via `HfApi` (`HF_TOKEN` env var). Repo is auto-created.

### CLI usage
```bash
cd seed_generation
uv sync

# Generate (defaults: n=22000, k=100, depth=3, spl=100, batch=32)
python generate.py generate

# Specific categories
python generate.py generate --categories persona --categories job

# Resume interrupted run
python generate.py generate --resume

# Upload to HuggingFace
python generate.py upload-only --repo-id mlech26l/liquidrandom-data
```

## Categories (12)

| Category | Model | Key Fields |
|---|---|---|
| persona | Persona | name, age, gender, occupation, nationality, personality_traits, background |
| job | Job | title, industry, description, required_skills, experience_level |
| coding_task | CodingTask | title, language, difficulty, description, constraints, expected_behavior |
| math_category | MathCategory | name, field, description, example_problems |
| writing_style | WritingStyle | name, tone, characteristics, description |
| scenario | Scenario | title, context, setting, stakes, description |
| domain | Domain | name, parent_field, description, key_concepts |
| science_topic | ScienceTopic | name, scientific_field, subfield, description |
| language | Language | name, region, register, script, cultural_notes |
| reasoning_pattern | ReasoningPattern | name, category, description, when_to_use |
| emotional_state | EmotionalState | name, intensity, valence, behavioral_description |
| instruction_complexity | InstructionComplexity | level, ambiguity, description, example |

## Package Usage

```python
import liquidrandom

persona = liquidrandom.persona()    # Returns Persona dataclass
job = liquidrandom.job()            # Returns Job dataclass
print(persona)                      # LLM-prompt-friendly string
print(persona.name)                 # Access individual fields
```
