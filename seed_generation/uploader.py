"""Consolidate per-leaf JSONL samples into per-category Parquet files and upload to HuggingFace."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi
from rich.console import Console

from categories import CATEGORY_CONFIGS
from image_categories import IMAGE_CATEGORY_CONFIGS
from image_config import PARQUET_ROW_GROUP_SIZE

console = Console()

# Categories that use pre-built parquet files instead of JSONL consolidation.
# Maps category name to the module attribute containing the hardcoded data list.
_HARDCODED_CATEGORIES: dict[str, tuple[str, str]] = {
    "reasoning_pattern": ("hardcoded_reasoning_patterns", "REASONING_PATTERNS"),
    "instruction_complexity": ("hardcoded_instruction_complexities", "INSTRUCTION_COMPLEXITIES"),
}


def _build_hardcoded_parquet(category_name: str, dest_dir: Path) -> int:
    """Build a parquet file from a hardcoded Python list.

    Returns the number of samples written.
    """
    module_name, attr_name = _HARDCODED_CATEGORIES[category_name]
    import importlib
    module = importlib.import_module(module_name)
    data: list[dict[str, Any]] = getattr(module, attr_name)

    if not data:
        return 0

    table = pa.Table.from_pylist(data)
    output_path = dest_dir / f"{category_name}.parquet"
    pq.write_table(table, output_path, compression="zstd")

    return len(data)


# Source folders that feed into a given output parquet, each tagged with a
# `kind` value. Absent entries fall back to a single same-named folder with
# no `kind` injection.
_CATEGORY_SOURCES: dict[str, list[tuple[str, str | None]]] = {
    "tool_group": [
        ("tool_group", "default"),
        ("physical_tool_group", "physical"),
    ],
}

# Generation-only categories that are merged into another parquet and must
# not be uploaded as separate files.
_MERGED_INTO_OTHER: set[str] = {"physical_tool_group"}

# Fields that should be stored as flat columns in Parquet.
# For tool_group, the nested "tools" list is redundant with "tools_json" (string).
_DROP_FIELDS: dict[str, set[str]] = {
    "tool_group": {"tools"},
}


def _consolidate_category(output_dir: str, category_name: str, dest_dir: Path) -> int:
    """Consolidate all per-leaf JSONL files into a single category Parquet file.

    Returns the number of samples consolidated.
    """
    # Use hardcoded data if available for this category
    if category_name in _HARDCODED_CATEGORIES:
        return _build_hardcoded_parquet(category_name, dest_dir)

    sources = _CATEGORY_SOURCES.get(category_name, [(category_name, None)])
    drop = _DROP_FIELDS.get(category_name, set())

    rows: list[dict[str, Any]] = []
    for source_folder, kind_value in sources:
        samples_dir = Path(output_dir) / "samples" / source_folder
        if not samples_dir.exists():
            continue
        for jsonl_file in sorted(samples_dir.glob("*.jsonl")):
            with open(jsonl_file, encoding="utf-8") as in_f:
                for line in in_f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if drop:
                        row = {k: v for k, v in row.items() if k not in drop}
                    if kind_value is not None:
                        row["kind"] = kind_value
                    rows.append(row)

    if not rows:
        return 0

    table = pa.Table.from_pylist(rows)
    output_path = dest_dir / f"{category_name}.parquet"
    pq.write_table(table, output_path, compression="zstd")

    return len(rows)


# Explicit schema for image category parquets. Chain rows are written
# contiguously, ordered by turn_index, and small row groups let the package
# read single random samples without materializing the multi-GB file.
_IMAGE_SCHEMA = pa.schema(
    [
        pa.field("image", pa.binary()),
        pa.field("image_format", pa.string()),
        pa.field("width", pa.int64()),
        pa.field("height", pa.int64()),
        pa.field("aspect_ratio", pa.string()),
        pa.field("taxonomy_path", pa.string()),
        pa.field("caption", pa.string()),
        pa.field("prompt", pa.string()),
        pa.field("edit_instruction", pa.string()),
        pa.field("tags", pa.list_(pa.string())),
        pa.field("chain_id", pa.string()),
        pa.field("turn_index", pa.int64()),
        pa.field("parent_turn", pa.int64()),
        pa.field("chain_length", pa.int64()),
    ]
)


def _consolidate_image_category(
    output_dir: str, category_name: str, dest_dir: Path
) -> int:
    """Consolidate image category JSONL files into Parquet with binary image column.

    Reads per-leaf JSONL files where each row has an 'image_base64' field,
    decoded into the binary 'image' column. Rows of a chain stay contiguous.

    Returns the number of samples consolidated.
    """
    import base64

    samples_dir = Path(output_dir) / "samples" / category_name
    if not samples_dir.exists():
        return 0

    rows: list[dict[str, Any]] = []
    for jsonl_file in sorted(samples_dir.glob("*.jsonl")):
        with open(jsonl_file, encoding="utf-8") as in_f:
            for line in in_f:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                row = {k: v for k, v in raw.items() if k != "image_base64"}
                row["image"] = base64.b64decode(raw.get("image_base64", ""))
                rows.append(row)

    if not rows:
        return 0

    table = pa.Table.from_pylist(rows, schema=_IMAGE_SCHEMA)
    output_path = dest_dir / f"{category_name}.parquet"
    pq.write_table(
        table,
        output_path,
        compression="zstd",
        row_group_size=PARQUET_ROW_GROUP_SIZE,
    )

    return len(rows)


def _generate_dataset_card(categories: dict[str, int]) -> str:
    """Generate a dataset card README.md."""
    total = sum(categories.values())
    text_rows = ""
    image_rows = ""
    image_total = 0
    for name, count in sorted(categories.items()):
        if name in IMAGE_CATEGORY_CONFIGS:
            display = IMAGE_CATEGORY_CONFIGS[name].display_name
            image_rows += f"| {display} | {count:,} | `{name}.parquet` |\n"
            image_total += count
        else:
            display = (
                CATEGORY_CONFIGS[name].display_name
                if name in CATEGORY_CONFIGS
                else name
            )
            text_rows += f"| {display} | {count:,} | `{name}.parquet` |\n"

    size_bucket = "100K<n<1M" if total >= 100_000 else "10K<n<100K"
    task_categories = "  - text-generation\n"
    extra_tags = ""
    image_section = ""
    image_usage = ""
    if image_rows:
        task_categories += "  - text-to-image\n  - image-classification\n"
        extra_tags = "  - image\n  - multimodal\n"
        image_section = f"""
## Image Categories

{image_total:,} images (1024px WebP) generated with Gemini Nano Banana from
taxonomy-derived prompts plus diverse image edits. Images come in edit chains:
a base image (`turn_index` 0) and edited variants linked via `chain_id` /
`parent_turn`, with the applied `edit_instruction` stored per image. Every
image carries a `tags` list (controlled vocabulary, VLM-verified) for
filtered selection.

| Category | Images | File |
|---|---|---|
{image_rows}"""
        image_usage = """
img = liquidrandom.image("indoor_scene", tags=["no_people"])
chain = liquidrandom.image_chain("ui_screenshot")  # base + edited variants
img.save("sample.webp")
"""

    return f"""---
license: mit
task_categories:
{task_categories}language:
  - en
size_categories:
  - {size_bucket}
tags:
  - synthetic
  - seed-data
  - diversity
  - llm-training
{extra_tags}---

# liquidrandom-data

Diverse seed data for ML/LLM training data generation pipelines.

Used by the [liquidrandom](https://github.com/mlech26l/liquidrandom) Python package.

## Dataset Summary

This dataset contains {total:,} seed data samples across {len(categories)} categories,
generated using a hierarchical taxonomy tree approach with LLM-based quality validation
and fuzzy deduplication. Data is stored as Parquet with zstd compression.

## Categories

| Category | Samples | File |
|---|---|---|
{text_rows}{image_section}

## Usage

```python
import liquidrandom

persona = liquidrandom.persona()
print(persona)
{image_usage}```

## Generation

Data was generated using the `liquidrandom` seed generation scripts with:
- Hierarchical taxonomy trees for diversity
- LLM-based quality validation
- Jaccard similarity deduplication
"""


def consolidate_and_upload(
    output_dir: str, repo_id: str, skip_images: bool = False
) -> None:
    """Consolidate all samples and upload to HuggingFace."""
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        console.print("[red]HF_TOKEN environment variable is not set[/red]")
        raise SystemExit(1)

    api = HfApi(token=hf_token)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        category_counts: dict[str, int] = {}

        console.print("[bold]Consolidating text samples...[/bold]")
        for cat_name in CATEGORY_CONFIGS:
            if cat_name in _MERGED_INTO_OTHER:
                continue  # merged into another category's parquet
            count = _consolidate_category(output_dir, cat_name, tmp_path)
            if count > 0:
                category_counts[cat_name] = count
                console.print(f"  {cat_name}: {count:,} samples")

        if skip_images:
            console.print("[yellow]Skipping image samples (--skip-images)[/yellow]")
        else:
            console.print("[bold]Consolidating image samples...[/bold]")
            for cat_name in IMAGE_CATEGORY_CONFIGS:
                count = _consolidate_image_category(output_dir, cat_name, tmp_path)
                if count > 0:
                    category_counts[cat_name] = count
                    console.print(f"  {cat_name}: {count:,} images")

        if not category_counts:
            console.print("[red]No samples found to upload[/red]")
            return

        # Write dataset card
        readme_content = _generate_dataset_card(category_counts)
        with open(tmp_path / "README.md", "w", encoding="utf-8") as f:
            f.write(readme_content)

        console.print(f"\n[bold]Uploading to {repo_id}...[/bold]")
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        api.upload_folder(
            folder_path=str(tmp_path),
            repo_id=repo_id,
            repo_type="dataset",
        )

        total = sum(category_counts.values())
        console.print(
            f"[green]Uploaded {total:,} samples across "
            f"{len(category_counts)} categories to {repo_id}[/green]"
        )
