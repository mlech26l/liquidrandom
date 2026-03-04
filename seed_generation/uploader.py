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

console = Console()


def _consolidate_category(output_dir: str, category_name: str, dest_dir: Path) -> int:
    """Consolidate all per-leaf JSONL files into a single category Parquet file.

    Returns the number of samples consolidated.
    """
    samples_dir = Path(output_dir) / "samples" / category_name
    if not samples_dir.exists():
        return 0

    rows: list[dict[str, Any]] = []
    for jsonl_file in sorted(samples_dir.glob("*.jsonl")):
        with open(jsonl_file, encoding="utf-8") as in_f:
            for line in in_f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

    if not rows:
        return 0

    table = pa.Table.from_pylist(rows)
    output_path = dest_dir / f"{category_name}.parquet"
    pq.write_table(table, output_path, compression="zstd")

    return len(rows)


def _generate_dataset_card(categories: dict[str, int]) -> str:
    """Generate a dataset card README.md."""
    total = sum(categories.values())
    rows = ""
    for name, count in sorted(categories.items()):
        display = CATEGORY_CONFIGS[name].display_name if name in CATEGORY_CONFIGS else name
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


def consolidate_and_upload(output_dir: str, repo_id: str) -> None:
    """Consolidate all samples and upload to HuggingFace."""
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        console.print("[red]HF_TOKEN environment variable is not set[/red]")
        raise SystemExit(1)

    api = HfApi(token=hf_token)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        category_counts: dict[str, int] = {}

        console.print("[bold]Consolidating samples...[/bold]")
        for cat_name in CATEGORY_CONFIGS:
            count = _consolidate_category(output_dir, cat_name, tmp_path)
            if count > 0:
                category_counts[cat_name] = count
                console.print(f"  {cat_name}: {count:,} samples")

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
