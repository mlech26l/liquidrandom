# Liquidrandom

Python package for pseudo-random seed data generation for ML/LLM training diversity.

## Project Structure

```
├── pyproject.toml                  # hatchling build, deps: huggingface-hub, pyarrow
├── .python-version                 # 3.12
├── README.md
├── src/liquidrandom/
│   ├── __init__.py                 # Public API: persona(), job(), image(), etc. + model re-exports
│   ├── py.typed                    # PEP 561 marker
│   ├── _loader.py                  # HuggingFace fetch + Parquet read + in-memory cache (text)
│   ├── _image_loader.py            # Row-group-lazy loader for image categories + tag/chain indices
│   ├── _registry.py                # Category → model class + filename mapping (+ IMAGE_CATEGORIES)
│   └── models/                     # Frozen dataclasses with from_dict() and __str__()
│       ├── __init__.py
│       ├── persona.py … instruction_complexity.py   # 12 text models
│       ├── tool_group.py           # ToolGroup/ToolFunction/ToolVariation
│       └── image_sample.py         # ImageSample (shared by all 11 image categories)
├── tests/
│   ├── test_models.py              # from_dict + __str__ for all text models
│   ├── test_loader.py              # Parquet loading, caching, public API (mocked HF)
│   ├── test_image_models.py        # ImageSample round-trip, to_str, save/to_pil
│   └── test_image_loader.py        # Lazy row-group access, tag filter, chains (mocked HF)
└── seed_generation/                # Separate project with own pyproject.toml
    ├── pyproject.toml              # deps: openai, rich, typer, huggingface-hub, pyarrow, google-genai, pillow
    ├── README.md
    ├── generate.py                 # CLI: generate / generate-tools / generate-images / review-images / upload-only
    ├── config.py                   # Constants and defaults (text)
    ├── categories.py               # 12 CategoryConfig with field specs and prompt templates
    ├── image_config.py             # Image constants (model ids, chain shape, preview settings)
    ├── image_categories.py         # 11 ImageCategoryConfig: taxonomy seeds, tag vocab, edit palettes
    ├── gemini_image.py             # google-genai wrapper: generate/edit/VLM-validate/recompress
    ├── image_sampler.py            # Chain planning (LLM) + Nano Banana generation/edits
    ├── image_viewer.py             # HTML review gallery (quality gate before full runs)
    ├── taxonomy.py                 # Phase 1: BFS taxonomy tree generation (shared by all modalities)
    ├── sampler.py                  # Phase 2: round-robin sample generation (text)
    ├── validator.py                # LLM quality validation per batch (text)
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
- Image categories use `_image_loader.py` instead: files are multi-GB, so it keeps an open `pq.ParquetFile` handle and reads one small row group (written with `row_group_size=64`) per sample. Tag/chain lookups read all columns except `image` into cached posting lists. The eager loaders in `_loader.py` raise `ValueError` for image categories.

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

## Image Seed Data Generation

### Model & APIs
- Image generation/editing: `gemini-3.1-flash-lite-image` (Nano Banana 2 Lite) via `google-genai` (`GEMINI_API_KEY`). 1K output only; aspect ratio set per chain via `ImageConfig`.
- Image validation: cheap Gemini VLM (`VALIDATION_MODEL_NAME` in `image_config.py`) checks each base image against its prompt and verifies/corrects tags.
- Chain planning (prompts + edit instructions): qwen3.5 via OpenRouter, same `llm.py` as text.

### Chain-based generation
Each unit of work is a **chain**: 1 base image + 3 edits (`DEFAULT_EDITS_PER_BASE`). Python pre-samples per chain — edit types (weighted, without replacement, from per-category `edit_palette`s; this is what guarantees edit diversity), parent topology (each edit targets the previous image with p=0.5, else the base), and aspect ratio — then one LLM call writes prompts/instructions for k chains. A rejected base kills the chain before edits are paid for; edits are spot-checked at 10%. Chain rows are appended to the leaf JSONL only when the whole chain is done (resume = count rows, never half-chains).

### Schema
Parquet columns per image category: `image` (binary WebP q87, 1024px), `image_format`, `width`, `height`, `aspect_ratio`, `taxonomy_path`, `caption`, `prompt`, `edit_instruction` ("" for base), `tags` (list<string>, controlled vocab like `people`/`lighting:dim`), `chain_id`, `turn_index` (0 = base), `parent_turn` (-1 for base), `chain_length`. Chain rows are contiguous, ordered by turn. JSONL intermediates carry `image_base64` instead of `image`.

### Quality gate (before full runs)
```bash
cd seed_generation

# ~20 images per category for human review (does NOT mark generation done)
python generate.py generate-images --preview --categories indoor_scene

# HTML gallery: chains as strips with edit instructions + tag chips
python generate.py review-images indoor_scene

# Full run (defaults: n=20000/category, concurrency 30), resumable
python generate.py generate-images --resume
```
A one-shot live API sanity check exists: `python gemini_image.py <out_dir>` (generate → edit → validate).

## Image Categories (11)

All use the shared `ImageSample` model. Public API: `liquidrandom.image(category, tags)`, `image_chain(category, tags, min_length)`, `image_chain_of(sample)`, plus per-category functions. Tags use AND semantics. `to_pil()` needs the `liquidrandom[image]` extra (pillow); everything else is dependency-free.

`indoor_scene, outdoor_scene, aerial_view, agriculture, industrial, automotive, ui_screenshot, document, chart, retail_product, food` — each defined in `seed_generation/image_categories.py` with its taxonomy seed, tag vocabulary (`tag_attributes`), edit palette, and aspect-ratio mix.

## Categories (12)

Each model supports `to_str(detail)` with `DetailLevel.HIGH_LEVEL` or `DetailLevel.DETAILED` (default). Fields marked **[H]** are high-level, **[D]** are detailed, **[M]** are manual-only (attribute access only, not in any `__str__`).

| Category | Model | High-Level Fields | Detailed Fields | Manual Fields |
|---|---|---|---|---|
| persona | Persona | name, age, gender, occupation, nationality | personality_traits, background | |
| job | Job | job_category, sector, experience_level | title, industry, description, required_skills | |
| coding_task | CodingTask | title, language, difficulty | description, constraints, expected_behavior | follow_up_task, change_request, edge_cases |
| math_category | MathCategory | broad_topic, field | name, description, example_problems | |
| writing_style | WritingStyle | category, tone | name, characteristics, description | |
| scenario | Scenario | broad_title, theme, setting | title, context, stakes, description | |
| domain | Domain | broad_category, area | name, parent_field, description, key_concepts | |
| science_topic | ScienceTopic | broad_topic, scientific_field | name, subfield, description | |
| language | Language | category, register | name, region, script, cultural_notes | |
| reasoning_pattern | ReasoningPattern | name, category | description, when_to_use | |
| emotional_state | EmotionalState | category, intensity, valence | name, behavioral_description, example | |
| instruction_complexity | InstructionComplexity | name, level, ambiguity | description, example | |

## Package Usage

```python
import liquidrandom
from liquidrandom import DetailLevel

persona = liquidrandom.persona()    # Returns Persona dataclass
job = liquidrandom.job()            # Returns Job dataclass
print(persona)                      # Detailed output (default)
print(persona.to_str(DetailLevel.HIGH_LEVEL))  # High-level output
print(persona.name)                 # Access individual fields
```
