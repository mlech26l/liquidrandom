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


def _consolidate_image_category(
    output_dir: str, category_name: str, dest_dir: Path
) -> int:
    """Consolidate image category JSONL files into Parquet with binary image column.

    Reads per-leaf JSONL files where each row has 'image_base64' and 'image_ext'
    fields. Converts base64 to raw bytes for the Parquet binary column.
    Drops 'image_base64', 'image_ext', and 'image_prompt' fields from the output.

    Returns the number of samples consolidated.
    """
    import base64

    samples_dir = Path(output_dir) / "samples" / category_name
    if not samples_dir.exists():
        return 0

    drop_fields = {"image_base64", "image_ext", "image_prompt"}
    rows: list[dict[str, Any]] = []

    for jsonl_file in sorted(samples_dir.glob("*.jsonl")):
        with open(jsonl_file, encoding="utf-8") as in_f:
            for line in in_f:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)

                # Convert base64 image to bytes
                img_b64 = raw.get("image_base64", "")
                if img_b64:
                    image_bytes = base64.b64decode(img_b64)
                else:
                    image_bytes = b""

                row = {k: v for k, v in raw.items() if k not in drop_fields}
                row["image"] = image_bytes
                rows.append(row)

    if not rows:
        return 0

    # Build schema: text fields + binary image column
    fields_from_first = []
    for key, val in rows[0].items():
        if key == "image":
            fields_from_first.append(pa.field("image", pa.binary()))
        elif isinstance(val, list):
            fields_from_first.append(pa.field(key, pa.list_(pa.string())))
        elif isinstance(val, int):
            fields_from_first.append(pa.field(key, pa.int64()))
        else:
            fields_from_first.append(pa.field(key, pa.string()))

    schema = pa.schema(fields_from_first)
    table = pa.Table.from_pylist(rows, schema=schema)

    output_path = dest_dir / f"{category_name}.parquet"
    pq.write_table(table, output_path, compression="zstd")

    return len(rows)


def _generate_dataset_card(categories: dict[str, int]) -> str:
    """Generate a dataset card README.md."""
    total = sum(categories.values())
    rows = ""
    for name, count in sorted(categories.items()):
        if name in CATEGORY_CONFIGS:
            display = CATEGORY_CONFIGS[name].display_name
        elif name in IMAGE_CATEGORY_CONFIGS:
            display = IMAGE_CATEGORY_CONFIGS[name].display_name
        else:
            display = name
        rows += f"| {display} | {count:,} | `{name}.parquet` |\n"

    return f"""---
license: mit
task_categories:
  - text-generation
language:
  - en
size_categories:
  - 10K<n<100K
tags:
  - synthetic
  - seed-data
  - diversity
  - llm-training
---

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
{rows}

## Usage

```python
import liquidrandom

persona = liquidrandom.persona()
print(persona)
```

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
