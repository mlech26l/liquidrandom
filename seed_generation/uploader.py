"""Consolidate per-leaf JSONL samples into per-category Parquet files and upload to HuggingFace."""

from __future__ import annotations

import base64
import json
import os
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi
from rich.console import Console

from categories import CATEGORY_CONFIGS
from image_categories import IMAGE_CATEGORY_CONFIGS
from image_config import PARQUET_ROW_GROUP_SIZE
from tag_normalize import TagNormalizer

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


# Rows are buffered in memory between Parquet writes. A full image category is
# several GB, so it is written incrementally: this many rows (~120 MB of WebP)
# are held at a time regardless of how large the category is.
_IMAGE_WRITE_BATCH = 512


def _renumber_chain(chain: list[dict[str, Any]]) -> bool:
    """Make turn_index contiguous from 0 and repoint parent_turn accordingly.

    A rejected edit leaves a gap in the planned turn numbers (turns [0, 1, 3]),
    which would break the invariant that turn_index is the row's position in
    the chain. Renumbering is a pure relabeling: prompts and instructions are
    untouched and every parent still refers to the same image.

    Returns True if the chain needed renumbering.
    """
    chain.sort(key=lambda r: r["turn_index"])
    remap = {row["turn_index"]: i for i, row in enumerate(chain)}
    changed = any(old != new for old, new in remap.items())
    for row in chain:
        row["turn_index"] = remap[row["turn_index"]]
        parent = row["parent_turn"]
        row["parent_turn"] = remap.get(parent, -1) if parent >= 0 else -1
        row["chain_length"] = len(chain)
    return changed


def _iter_image_rows(
    samples_dir: Path,
    normalizer: TagNormalizer,
    dropped: Counter[str],
    stats: Counter[str],
) -> Iterator[dict[str, Any]]:
    """Yield Parquet-ready rows from a category's per-leaf JSONL files.

    Rows are emitted a chain at a time so turn numbering can be repaired; the
    sampler appends whole chains, so a chain never spans leaf files.
    """
    chain: list[dict[str, Any]] = []
    for jsonl_file in sorted(samples_dir.glob("*.jsonl")):
        with open(jsonl_file, encoding="utf-8") as in_f:
            for line in in_f:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                row = {k: v for k, v in raw.items() if k != "image_base64"}
                row["image"] = base64.b64decode(raw.get("image_base64", ""))
                tags, unmapped = normalizer.normalize(list(row.get("tags") or []))
                row["tags"] = tags
                for tag in unmapped:
                    dropped[tag] += 1

                if chain and row["chain_id"] != chain[0]["chain_id"]:
                    stats["renumbered"] += _renumber_chain(chain)
                    yield from chain
                    chain = []
                chain.append(row)
    if chain:
        stats["renumbered"] += _renumber_chain(chain)
        yield from chain


def _consolidate_image_category(
    output_dir: str, category_name: str, dest_dir: Path
) -> int:
    """Consolidate image category JSONL files into Parquet with binary image column.

    Reads per-leaf JSONL files where each row has an 'image_base64' field,
    decoded into the binary 'image' column. Rows of a chain stay contiguous.
    Written incrementally so memory stays flat on multi-GB categories.

    Returns the number of samples consolidated.
    """
    samples_dir = Path(output_dir) / "samples" / category_name
    if not samples_dir.exists():
        return 0

    normalizer = TagNormalizer(IMAGE_CATEGORY_CONFIGS[category_name])
    dropped: Counter[str] = Counter()
    stats: Counter[str] = Counter()
    output_path = dest_dir / f"{category_name}.parquet"
    partial_path = output_path.with_suffix(".parquet.partial")

    total = 0
    writer: pq.ParquetWriter | None = None
    batch: list[dict[str, Any]] = []
    try:
        for row in _iter_image_rows(samples_dir, normalizer, dropped, stats):
            batch.append(row)
            if len(batch) >= _IMAGE_WRITE_BATCH:
                if writer is None:
                    writer = pq.ParquetWriter(
                        partial_path, _IMAGE_SCHEMA, compression="zstd"
                    )
                writer.write_table(
                    pa.Table.from_pylist(batch, schema=_IMAGE_SCHEMA),
                    row_group_size=PARQUET_ROW_GROUP_SIZE,
                )
                total += len(batch)
                batch = []
        if batch:
            if writer is None:
                writer = pq.ParquetWriter(
                    partial_path, _IMAGE_SCHEMA, compression="zstd"
                )
            writer.write_table(
                pa.Table.from_pylist(batch, schema=_IMAGE_SCHEMA),
                row_group_size=PARQUET_ROW_GROUP_SIZE,
            )
            total += len(batch)
    finally:
        if writer is not None:
            writer.close()

    if total == 0:
        partial_path.unlink(missing_ok=True)
        return 0

    # Rename only on success, so an interrupted run never leaves a short
    # parquet that a resume would mistake for a finished one.
    partial_path.replace(output_path)

    if dropped:
        shown = ", ".join(f"{t} ({n})" for t, n in dropped.most_common(5))
        console.print(
            f"    [yellow]dropped {sum(dropped.values()):,} off-vocabulary tag "
            f"values: {shown}[/yellow]"
        )
    if stats["renumbered"]:
        console.print(
            f"    [dim]renumbered {stats['renumbered']:,} chains that lost an "
            "edit[/dim]"
        )

    return total


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


def _remote_row_counts(repo_id: str, filenames: list[str]) -> dict[str, int]:
    """Row counts of parquet files already in the repo, read from their footers.

    Lets the dataset card keep listing categories whose source samples are not
    present locally (e.g. uploading images from a machine that only generated
    images). Returns an empty mapping if the repo cannot be read.
    """
    counts: dict[str, int] = {}
    try:
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem()
        for filename in filenames:
            path = f"datasets/{repo_id}/{filename}"
            with fs.open(path, "rb") as handle:
                counts[filename] = pq.ParquetFile(handle).metadata.num_rows
    except Exception as exc:  # network, missing repo, unreadable file
        console.print(f"[yellow]Could not read remote row counts: {exc}[/yellow]")
    return counts


def consolidate_and_upload(
    output_dir: str,
    repo_id: str,
    skip_images: bool = False,
    work_dir: str | None = None,
    force: bool = False,
) -> None:
    """Consolidate all samples and upload to HuggingFace.

    Parquet files are staged in `work_dir` (default `<output_dir>/parquet`) on
    real disk rather than a temp directory: the image categories total tens of
    GB. Staged files and files already uploaded with a matching size are
    reused, so an interrupted upload can simply be re-run.
    """
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        console.print("[red]HF_TOKEN environment variable is not set[/red]")
        raise SystemExit(1)

    api = HfApi(token=hf_token)
    staging = Path(work_dir) if work_dir else Path(output_dir) / "parquet"
    staging.mkdir(parents=True, exist_ok=True)
    console.print(f"[dim]Staging parquet files in {staging}[/dim]")

    category_counts: dict[str, int] = {}

    def _staged(cat_name: str) -> int | None:
        """Row count of an already-staged parquet, or None if it must be built."""
        path = staging / f"{cat_name}.parquet"
        if force or not path.exists():
            return None
        try:
            return pq.read_metadata(path).num_rows
        except Exception:
            return None

    console.print("[bold]Consolidating text samples...[/bold]")
    for cat_name in CATEGORY_CONFIGS:
        if cat_name in _MERGED_INTO_OTHER:
            continue  # merged into another category's parquet
        count = _staged(cat_name)
        if count is None:
            count = _consolidate_category(output_dir, cat_name, staging)
        if count > 0:
            category_counts[cat_name] = count
            console.print(f"  {cat_name}: {count:,} samples")

    if skip_images:
        console.print("[yellow]Skipping image samples (--skip-images)[/yellow]")
    else:
        console.print("[bold]Consolidating image samples...[/bold]")
        for cat_name in IMAGE_CATEGORY_CONFIGS:
            count = _staged(cat_name)
            if count is not None:
                console.print(f"  {cat_name}: {count:,} images [dim](staged)[/dim]")
            else:
                console.print(f"  {cat_name}: consolidating...")
                count = _consolidate_image_category(output_dir, cat_name, staging)
                if count > 0:
                    console.print(f"  {cat_name}: {count:,} images")
            if count > 0:
                category_counts[cat_name] = count

    if not category_counts:
        console.print("[red]No samples found to upload[/red]")
        return

    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    remote_sizes: dict[str, int] = {}
    try:
        info = api.repo_info(repo_id=repo_id, repo_type="dataset", files_metadata=True)
        remote_sizes = {s.rfilename: s.size or 0 for s in info.siblings or []}
    except Exception as exc:
        console.print(f"[yellow]Could not list remote files: {exc}[/yellow]")

    # Categories that live only in the repo (not regenerated here) still belong
    # on the dataset card.
    absent = [
        f"{name}.parquet"
        for name in list(CATEGORY_CONFIGS) + list(IMAGE_CATEGORY_CONFIGS)
        if name not in category_counts
        and name not in _MERGED_INTO_OTHER
        and f"{name}.parquet" in remote_sizes
    ]
    if absent:
        console.print(f"[dim]Reading row counts of {len(absent)} remote-only file(s)[/dim]")
        for filename, rows in _remote_row_counts(repo_id, absent).items():
            category_counts[filename.removesuffix(".parquet")] = rows

    readme_path = staging / "README.md"
    readme_path.write_text(_generate_dataset_card(category_counts), encoding="utf-8")

    console.print(f"\n[bold]Uploading to {repo_id}...[/bold]")
    for path in sorted(staging.glob("*.parquet")) + [readme_path]:
        size = path.stat().st_size
        # The card is rewritten every run and is tiny; always push it.
        if path is not readme_path and not force and remote_sizes.get(path.name) == size:
            console.print(f"  {path.name}: [dim]already uploaded[/dim]")
            continue
        console.print(f"  {path.name}: uploading {size / 1e6:,.0f} MB...")
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path.name,
            repo_id=repo_id,
            repo_type="dataset",
        )

    total = sum(category_counts.values())
    console.print(
        f"[green]Uploaded {total:,} samples across "
        f"{len(category_counts)} categories to {repo_id}[/green]"
    )
