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
    DEFAULT_M,
    DEFAULT_N,
    DEFAULT_SAMPLES_PER_LEAF,
    DEFAULT_TAXONOMY_DEPTH,
    DEFAULT_TOOL_K,
    DEFAULT_TOOL_N,
    DEFAULT_TOOL_SAMPLES_PER_LEAF,
    DEFAULT_TOOL_TAXONOMY_DEPTH,
    STATE_FILE,
)
from image_categories import IMAGE_CATEGORY_CONFIGS
from image_config import (
    DEFAULT_IMAGE_BATCH_SIZE,
    DEFAULT_IMAGE_CONCURRENCY,
    DEFAULT_IMAGE_DEDUP_THRESHOLD,
    DEFAULT_IMAGE_K,
    DEFAULT_IMAGE_N,
    DEFAULT_IMAGE_SAMPLES_PER_LEAF,
    DEFAULT_IMAGE_TAXONOMY_DEPTH,
)
from llm import create_client
from sampler import generate_samples
from state import RunState
from taxonomy import generate_taxonomy
from tool_sampler import generate_tool_groups
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


async def _run_generate_tools(
    categories: list[str],
    n: int,
    k: int,
    m: int,
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

    console.print(f"\n[bold]Generating tool groups for {len(categories)} categories[/bold]")
    console.print(f"  Target: {n} groups per category, {k} per LLM call, {m} variations per tool")
    console.print(f"  Taxonomy: depth {taxonomy_depth}, {samples_per_leaf} groups/leaf")
    console.print(f"  Dedup threshold: {dedup_threshold}")
    console.print()

    for i, cat_name in enumerate(categories, 1):
        cat_config = CATEGORY_CONFIGS[cat_name]
        cat_state = state.get_category(cat_name)
        cat_start = time.time()

        console.print(
            f"[bold cyan][{i}/{len(categories)}] {cat_config.display_name}[/bold cyan]"
        )

        # Phase 1: Taxonomy (reuses existing taxonomy.py)
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

        # Phase 2: Tool group generation
        if not cat_state.generation_done or not resume:
            try:
                total = await generate_tool_groups(
                    client,
                    cat_name,
                    root,
                    target_samples=n,
                    k=k,
                    m=m,
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
                    f"  [yellow]Resume with: python generate.py generate-tools --resume "
                    f"--output-dir {output_dir}[/yellow]"
                )
                raise
            except Exception as e:
                console.print(f"  [red]Tool group generation failed: {e}[/red]")
                state.save(state_path)
                continue
        else:
            total = cat_state.total_samples
            console.print(
                f"  [green]Already complete ({total} groups)[/green]"
            )

        elapsed = time.time() - cat_start
        results[cat_name] = {"total": total, "time": elapsed}
        console.print()

    # Summary table
    total_elapsed = time.time() - total_start
    table = Table(title="Tool Group Generation Summary")
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


@app.command(name="generate-tools")
def generate_tools(
    n: int = typer.Option(DEFAULT_TOOL_N, "--n", help="Target tool groups per category"),
    k: int = typer.Option(DEFAULT_TOOL_K, "--tool-k", help="Tool groups per LLM call"),
    m: int = typer.Option(DEFAULT_M, "--m", help="Variations per tool function"),
    batch_size: int = typer.Option(
        DEFAULT_BATCH_SIZE, "--batch-size", help="Concurrent LLM calls"
    ),
    taxonomy_depth: int = typer.Option(
        DEFAULT_TOOL_TAXONOMY_DEPTH, "--taxonomy-depth", help="Max taxonomy tree depth"
    ),
    samples_per_leaf: int = typer.Option(
        DEFAULT_TOOL_SAMPLES_PER_LEAF, "--samples-per-leaf", help="Target groups per leaf"
    ),
    dedup_threshold: float = typer.Option(
        DEFAULT_DEDUP_THRESHOLD, "--dedup-threshold", help="Jaccard similarity threshold"
    ),
    categories: Optional[list[str]] = typer.Option(
        None, "--categories", help="Categories to generate (default: tool_group)"
    ),
    resume: bool = typer.Option(False, "--resume", help="Resume from checkpoint"),
    output_dir: str = typer.Option("output", "--output-dir", help="Output directory"),
) -> None:
    """Generate tool group seed data with parameter variations."""
    resolved = categories if categories else ["tool_group", "physical_tool_group"]
    for cat in resolved:
        if cat not in CATEGORY_CONFIGS:
            console.print(f"[red]Unknown category: {cat}[/red]")
            raise typer.Exit(1)
    asyncio.run(
        _run_generate_tools(
            categories=resolved,
            n=n,
            k=k,
            m=m,
            batch_size=batch_size,
            taxonomy_depth=taxonomy_depth,
            samples_per_leaf=samples_per_leaf,
            dedup_threshold=dedup_threshold,
            output_dir=output_dir,
            resume=resume,
        )
    )


async def _run_generate_images(
    categories: list[str],
    n: int,
    k: int,
    batch_size: int,
    image_concurrency: int,
    taxonomy_depth: int,
    samples_per_leaf: int,
    dedup_threshold: float,
    output_dir: str,
    resume: bool,
) -> None:
    from image_sampler import generate_image_samples

    client = create_client()
    state_path = str(Path(output_dir) / "state.json")
    # Always load existing state to preserve progress from previous runs
    state = RunState.load(state_path)

    results: dict[str, dict[str, int | float]] = {}
    total_start = time.time()

    console.print(f"\n[bold]Generating image seed data for {len(categories)} categories[/bold]")
    console.print(f"  Target: {n} images per category, {k} prompts per LLM call")
    console.print(f"  Taxonomy: depth {taxonomy_depth}, {samples_per_leaf} images/leaf")
    console.print(f"  Image concurrency: {image_concurrency}, dedup threshold: {dedup_threshold}")
    console.print()

    for i, cat_name in enumerate(categories, 1):
        cat_config = IMAGE_CATEGORY_CONFIGS[cat_name]
        cat_state = state.get_category(cat_name)
        cat_start = time.time()

        console.print(
            f"[bold cyan][{i}/{len(categories)}] {cat_config.display_name}[/bold cyan]"
        )

        # Phase 1: Taxonomy (reuses existing taxonomy.py)
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

        # Phase 2+3: Image prompt generation + Replicate image generation
        if not cat_state.generation_done or not resume:
            try:
                total = await generate_image_samples(
                    client,
                    root,
                    cat_config,
                    target_samples=n,
                    k=k,
                    batch_size=batch_size,
                    image_concurrency=image_concurrency,
                    dedup_threshold=dedup_threshold,
                    output_dir=output_dir,
                )
                # Only mark done if we reached the target
                cat_state.generation_done = total >= n
                cat_state.total_samples = total
                state.save(state_path)
            except KeyboardInterrupt:
                console.print("\n  [yellow]Interrupted! Saving state...[/yellow]")
                state.save(state_path)
                console.print(
                    f"  [yellow]Resume with: python generate.py generate-images --resume "
                    f"--output-dir {output_dir}[/yellow]"
                )
                raise
            except Exception as e:
                console.print(f"  [red]Image generation failed: {e}[/red]")
                state.save(state_path)
                continue
        else:
            total = cat_state.total_samples
            console.print(
                f"  [green]Already complete ({total} images)[/green]"
            )

        elapsed = time.time() - cat_start
        results[cat_name] = {"total": total, "time": elapsed}
        console.print()

    # Summary table
    total_elapsed = time.time() - total_start
    table = Table(title="Image Generation Summary")
    table.add_column("Category", style="cyan")
    table.add_column("Target", justify="right")
    table.add_column("Generated", justify="right")
    table.add_column("Time", justify="right")

    for cat_name in categories:
        if cat_name in results:
            r = results[cat_name]
            table.add_row(
                IMAGE_CATEGORY_CONFIGS[cat_name].display_name,
                str(n),
                str(r["total"]),
                f"{r['time']:.1f}s",
            )

    console.print(table)
    console.print(f"\n[bold]Total time: {total_elapsed:.1f}s[/bold]")


@app.command(name="generate-images")
def generate_images(
    n: int = typer.Option(DEFAULT_IMAGE_N, "--n", help="Target images per category"),
    k: int = typer.Option(DEFAULT_IMAGE_K, "--k", help="Prompts per LLM call"),
    batch_size: int = typer.Option(
        DEFAULT_IMAGE_BATCH_SIZE, "--batch-size", help="Concurrent LLM calls"
    ),
    image_concurrency: int = typer.Option(
        DEFAULT_IMAGE_CONCURRENCY, "--image-concurrency", help="Concurrent Replicate API calls"
    ),
    taxonomy_depth: int = typer.Option(
        DEFAULT_IMAGE_TAXONOMY_DEPTH, "--taxonomy-depth", help="Max taxonomy tree depth"
    ),
    samples_per_leaf: int = typer.Option(
        DEFAULT_IMAGE_SAMPLES_PER_LEAF, "--samples-per-leaf", help="Target images per leaf"
    ),
    dedup_threshold: float = typer.Option(
        DEFAULT_IMAGE_DEDUP_THRESHOLD, "--dedup-threshold", help="Jaccard similarity threshold"
    ),
    categories: Optional[list[str]] = typer.Option(
        None, "--categories", help="Image categories to generate (default: all)"
    ),
    resume: bool = typer.Option(False, "--resume", help="Resume from checkpoint"),
    output_dir: str = typer.Option("output", "--output-dir", help="Output directory"),
) -> None:
    """Generate image seed data for all or selected image categories."""
    if not categories:
        resolved = list(IMAGE_CATEGORY_CONFIGS.keys())
    else:
        for cat in categories:
            if cat not in IMAGE_CATEGORY_CONFIGS:
                console.print(f"[red]Unknown image category: {cat}[/red]")
                console.print(
                    f"Available: {', '.join(sorted(IMAGE_CATEGORY_CONFIGS.keys()))}"
                )
                raise typer.Exit(1)
        resolved = categories
    asyncio.run(
        _run_generate_images(
            categories=resolved,
            n=n,
            k=k,
            batch_size=batch_size,
            image_concurrency=image_concurrency,
            taxonomy_depth=taxonomy_depth,
            samples_per_leaf=samples_per_leaf,
            dedup_threshold=dedup_threshold,
            output_dir=output_dir,
            resume=resume,
        )
    )


@app.command(name="review-images")
def review_images(
    category: str = typer.Argument(help="Image category to review"),
    max_images: int = typer.Option(500, "--max-images", help="Max images to include in gallery"),
    output_dir: str = typer.Option("output", "--output-dir", help="Output directory"),
) -> None:
    """Generate an HTML gallery page for reviewing generated images."""
    from image_viewer import generate_review_html

    if category not in IMAGE_CATEGORY_CONFIGS:
        console.print(f"[red]Unknown image category: {category}[/red]")
        console.print(
            f"Available: {', '.join(sorted(IMAGE_CATEGORY_CONFIGS.keys()))}"
        )
        raise typer.Exit(1)
    path = generate_review_html(output_dir, category, max_images=max_images)
    console.print(f"\n[bold]Open in browser: file://{path.resolve()}[/bold]")


@app.command(name="upload-only")
def upload_only(
    output_dir: str = typer.Option("output", "--output-dir", help="Output directory"),
    repo_id: str = typer.Option(
        "mlech26l/liquidrandom-data", "--repo-id", help="HuggingFace repo ID"
    ),
    skip_images: bool = typer.Option(
        False,
        "--skip-images",
        help="Skip image category consolidation/upload (any existing remote image parquets stay untouched; README will only list text categories)",
    ),
) -> None:
    """Consolidate samples and upload to HuggingFace."""
    consolidate_and_upload(
        output_dir=output_dir, repo_id=repo_id, skip_images=skip_images
    )


if __name__ == "__main__":
    app()
