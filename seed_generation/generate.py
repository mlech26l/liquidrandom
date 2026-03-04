"""CLI entrypoint for seed data generation."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from categories import CATEGORY_CONFIGS
from config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DEDUP_THRESHOLD,
    DEFAULT_K,
    DEFAULT_N,
    DEFAULT_SAMPLES_PER_LEAF,
    DEFAULT_TAXONOMY_DEPTH,
    STATE_FILE,
)
from llm import create_client
from sampler import generate_samples
from state import RunState
from taxonomy import generate_taxonomy
from uploader import consolidate_and_upload

app = typer.Typer(help="Seed data generation for liquidrandom")
console = Console()


def _resolve_categories(categories: list[str] | None) -> list[str]:
    """Resolve category names, defaulting to all."""
    if not categories:
        return list(CATEGORY_CONFIGS.keys())
    for cat in categories:
        if cat not in CATEGORY_CONFIGS:
            console.print(f"[red]Unknown category: {cat}[/red]")
            console.print(f"Available: {', '.join(sorted(CATEGORY_CONFIGS.keys()))}")
            raise typer.Exit(1)
    return categories


async def _run_generate(
    categories: list[str],
    n: int,
    k: int,
    batch_size: int,
    taxonomy_depth: int,
    samples_per_leaf: int,
    dedup_threshold: float,
    output_dir: str,
    resume: bool,
) -> None:
    client = create_client()
    state_path = str(Path(output_dir) / "state.json")
    state = RunState.load(state_path) if resume else RunState()

    results: dict[str, dict[str, int | float]] = {}
    total_start = time.time()

    console.print(f"\n[bold]Generating seed data for {len(categories)} categories[/bold]")
    console.print(f"  Target: {n} samples per category, {k} per LLM call")
    console.print(f"  Taxonomy: depth {taxonomy_depth}, {samples_per_leaf} samples/leaf")
    console.print(f"  Dedup threshold: {dedup_threshold}")
    console.print()

    for i, cat_name in enumerate(categories, 1):
        cat_config = CATEGORY_CONFIGS[cat_name]
        cat_state = state.get_category(cat_name)
        cat_start = time.time()

        console.print(
            f"[bold cyan][{i}/{len(categories)}] {cat_config.display_name}[/bold cyan]"
        )

        # Phase 1: Taxonomy
        if not cat_state.taxonomy_done or not resume:
            try:
                root = await generate_taxonomy(
                    client,
                    cat_config,
                    target_samples=n,
                    samples_per_leaf=samples_per_leaf,
                    max_depth=taxonomy_depth,
                    batch_size=batch_size,
                    output_dir=output_dir,
                )
                cat_state.taxonomy_done = True
                state.save(state_path)
            except Exception as e:
                console.print(f"  [red]Taxonomy generation failed: {e}[/red]")
                continue
        else:
            from taxonomy import load_taxonomy
            root = load_taxonomy(output_dir, cat_name)
            if root is None:
                console.print(f"  [red]No saved taxonomy found, regenerating[/red]")
                cat_state.taxonomy_done = False
                continue

        # Phase 2: Sample generation
        if not cat_state.generation_done or not resume:
            try:
                total = await generate_samples(
                    client,
                    root,
                    cat_config,
                    target_samples=n,
                    k=k,
                    batch_size=batch_size,
                    dedup_threshold=dedup_threshold,
                    output_dir=output_dir,
                )
                cat_state.generation_done = True
                cat_state.total_samples = total
                state.save(state_path)
            except KeyboardInterrupt:
                console.print("\n  [yellow]Interrupted! Saving state...[/yellow]")
                state.save(state_path)
                console.print(
                    f"  [yellow]Resume with: python generate.py generate --resume "
                    f"--output-dir {output_dir}[/yellow]"
                )
                raise
            except Exception as e:
                console.print(f"  [red]Sample generation failed: {e}[/red]")
                state.save(state_path)
                continue
        else:
            total = cat_state.total_samples
            console.print(
                f"  [green]Already complete ({total} samples)[/green]"
            )

        elapsed = time.time() - cat_start
        results[cat_name] = {"total": total, "time": elapsed}
        console.print()

    # Summary table
    total_elapsed = time.time() - total_start
    table = Table(title="Generation Summary")
    table.add_column("Category", style="cyan")
    table.add_column("Target", justify="right")
    table.add_column("Generated", justify="right")
    table.add_column("Time", justify="right")

    for cat_name in categories:
        if cat_name in results:
            r = results[cat_name]
            table.add_row(
                CATEGORY_CONFIGS[cat_name].display_name,
                str(n),
                str(r["total"]),
                f"{r['time']:.1f}s",
            )

    console.print(table)
    console.print(f"\n[bold]Total time: {total_elapsed:.1f}s[/bold]")


@app.command()
def generate(
    n: int = typer.Option(DEFAULT_N, "--n", help="Target samples per category"),
    k: int = typer.Option(DEFAULT_K, "--k", help="Samples per LLM call"),
    batch_size: int = typer.Option(
        DEFAULT_BATCH_SIZE, "--batch-size", help="Concurrent LLM calls"
    ),
    taxonomy_depth: int = typer.Option(
        DEFAULT_TAXONOMY_DEPTH, "--taxonomy-depth", help="Max taxonomy tree depth"
    ),
    samples_per_leaf: int = typer.Option(
        DEFAULT_SAMPLES_PER_LEAF, "--samples-per-leaf", help="Target samples per leaf"
    ),
    dedup_threshold: float = typer.Option(
        DEFAULT_DEDUP_THRESHOLD, "--dedup-threshold", help="Jaccard similarity threshold"
    ),
    categories: Optional[list[str]] = typer.Option(
        None, "--categories", help="Categories to generate (default: all)"
    ),
    resume: bool = typer.Option(False, "--resume", help="Resume from checkpoint"),
    output_dir: str = typer.Option("output", "--output-dir", help="Output directory"),
) -> None:
    """Generate seed data for all or selected categories."""
    resolved = _resolve_categories(categories)
    asyncio.run(
        _run_generate(
            categories=resolved,
            n=n,
            k=k,
            batch_size=batch_size,
            taxonomy_depth=taxonomy_depth,
            samples_per_leaf=samples_per_leaf,
            dedup_threshold=dedup_threshold,
            output_dir=output_dir,
            resume=resume,
        )
    )


@app.command(name="upload-only")
def upload_only(
    output_dir: str = typer.Option("output", "--output-dir", help="Output directory"),
    repo_id: str = typer.Option(
        "mlech26l/liquidrandom-data", "--repo-id", help="HuggingFace repo ID"
    ),
) -> None:
    """Consolidate samples and upload to HuggingFace."""
    consolidate_and_upload(output_dir=output_dir, repo_id=repo_id)


if __name__ == "__main__":
    app()
