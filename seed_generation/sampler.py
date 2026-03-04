"""Phase 2: Round-robin sample generation across taxonomy leaf nodes."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    MofNCompleteColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from categories import CategoryConfig
from config import MAX_VALIDATION_RETRIES
from dedup import dedup_batch
from llm import llm_call
from taxonomy import TaxonomyNode, save_taxonomy
from validator import validate_batch

console = Console()


def _leaf_id(leaf_path: list[str]) -> str:
    """Stable short filename from a leaf path using a hash."""
    raw = "/".join(leaf_path)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _load_leaf_samples(output_dir: str, category_name: str, leaf_path: list[str]) -> list[dict[str, Any]]:
    """Load existing samples for a specific leaf node."""
    path = Path(output_dir) / "samples" / category_name / f"{_leaf_id(leaf_path)}.jsonl"
    if not path.exists():
        return []
    samples: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def _save_leaf_samples(
    output_dir: str,
    category_name: str,
    leaf_path: list[str],
    samples: list[dict[str, Any]],
) -> None:
    """Append samples to the leaf's JSONL file."""
    path = Path(output_dir) / "samples" / category_name / f"{_leaf_id(leaf_path)}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def _sample_to_str(sample: dict[str, Any]) -> str:
    """Convert a sample dict to a string for dedup comparison."""
    parts: list[str] = []
    for key, value in sorted(sample.items()):
        if isinstance(value, list):
            parts.append(f"{key}: {', '.join(str(v) for v in value)}")
        else:
            parts.append(f"{key}: {value}")
    return " | ".join(parts)


async def _generate_for_leaf(
    client: AsyncOpenAI,
    leaf: TaxonomyNode,
    category: CategoryConfig,
    k: int,
    dedup_threshold: float,
    output_dir: str,
) -> int:
    """Generate samples for a single leaf node. Returns number of new samples accepted."""
    leaf_path_str = " > ".join(leaf.path)
    existing = _load_leaf_samples(output_dir, category.name, leaf.path)

    # Build existing samples text for the prompt
    existing_text = "None yet." if not existing else ""
    if existing:
        for s in existing[-20:]:  # Show last 20 to avoid prompt explosion
            existing_text += _sample_to_str(s) + "\n"
        if len(existing) > 20:
            existing_text = f"[showing last 20 of {len(existing)}]\n" + existing_text

    prompt = category.generation_prompt_template.format(
        k=k,
        leaf_path=leaf_path_str,
        existing_samples=existing_text,
    )

    for attempt in range(MAX_VALIDATION_RETRIES):
        try:
            # Generate samples
            result = await llm_call(client, [{"role": "user", "content": prompt}])

            if isinstance(result, dict):
                raw_samples = result.get("samples", [])
            else:
                raw_samples = result

            if not raw_samples:
                continue

            # Validate
            accepted = await validate_batch(
                client, raw_samples, category, leaf_path_str
            )

            if not accepted:
                # >50% rejected, retry
                continue

            # Dedup against existing
            deduped = dedup_batch(
                accepted, existing, _sample_to_str, dedup_threshold
            )

            if deduped:
                _save_leaf_samples(output_dir, category.name, leaf.path, deduped)
                leaf.sample_count += len(deduped)
                return len(deduped)

        except Exception as e:
            console.print(
                f"    [red]Error generating for {leaf_path_str}: {e}[/red]"
            )

    return 0


async def generate_samples(
    client: AsyncOpenAI,
    root: TaxonomyNode,
    category: CategoryConfig,
    target_samples: int,
    k: int,
    batch_size: int,
    dedup_threshold: float,
    output_dir: str,
) -> int:
    """Generate samples by cycling through all leaf nodes in round-robin.

    Returns total number of samples generated.
    """
    leaves = root.leaf_nodes()

    # Load existing counts
    for leaf in leaves:
        existing = _load_leaf_samples(output_dir, category.name, leaf.path)
        leaf.sample_count = len(existing)

    total_existing = sum(leaf.sample_count for leaf in leaves)
    if total_existing >= target_samples:
        console.print(
            f"  [green]Already have {total_existing}/{target_samples} samples[/green]"
        )
        return total_existing

    console.print(
        f"  Starting from {total_existing}/{target_samples} samples "
        f"across {len(leaves)} leaves"
    )

    semaphore = asyncio.Semaphore(batch_size)
    total_generated = total_existing

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("ETA:"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"  {category.display_name}",
            total=target_samples,
            completed=total_existing,
        )

        # Track stalls for early exit
        stall_rounds = 0

        while total_generated < target_samples:
            # Find incomplete leaves
            incomplete = [
                leaf for leaf in leaves if leaf.sample_count < leaf.target_count
            ]
            if not incomplete:
                break

            # Fire all incomplete leaves, semaphore limits concurrency
            async def process_leaf(leaf: TaxonomyNode) -> int:
                async with semaphore:
                    count = await _generate_for_leaf(
                        client, leaf, category, k, dedup_threshold, output_dir
                    )
                    # Update progress immediately as each leaf completes
                    nonlocal total_generated
                    total_generated += count
                    progress.update(task, completed=min(total_generated, target_samples))
                    return count

            tasks = [asyncio.create_task(process_leaf(leaf)) for leaf in incomplete]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            batch_total = sum(r for r in results if isinstance(r, int))

            # Save taxonomy with updated counts
            save_taxonomy(root, output_dir, category.name)

            if batch_total == 0:
                stall_rounds += 1
                console.print(
                    "    [yellow]No samples generated in this round, "
                    "some leaves may be saturated[/yellow]"
                )
                for leaf in incomplete:
                    if leaf.sample_count == 0:
                        leaf.target_count = 0
                if stall_rounds >= 3:
                    console.print(
                        "    [red]3 consecutive stalls, stopping category[/red]"
                    )
                    break
            else:
                stall_rounds = 0

    return total_generated
