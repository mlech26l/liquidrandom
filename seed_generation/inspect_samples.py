"""Manual inspection script for seed data samples.

Displays first 5, last 5, and 10 random samples for each category
using rich formatting. Pauses between categories for human review.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from categories import CATEGORY_CONFIGS

console = Console()

# Map category names to their model fields for structured display
_MODEL_FIELDS: dict[str, list[str]] = {
    "persona": ["name", "age", "gender", "occupation", "nationality", "personality_traits", "background"],
    "job": ["title", "industry", "description", "required_skills", "experience_level"],
    "coding_task": ["title", "language", "difficulty", "description", "constraints", "expected_behavior"],
    "math_category": ["name", "field", "description", "example_problems"],
    "writing_style": ["name", "tone", "characteristics", "description"],
    "scenario": ["title", "context", "setting", "stakes", "description"],
    "domain": ["name", "parent_field", "description", "key_concepts"],
    "science_topic": ["name", "scientific_field", "subfield", "description"],
    "language": ["name", "region", "register", "script", "cultural_notes"],
    "reasoning_pattern": ["name", "category", "description", "when_to_use"],
    "emotional_state": ["name", "intensity", "valence", "behavioral_description"],
    "instruction_complexity": ["level", "ambiguity", "description", "example"],
    "tool_group": ["domain", "description", "taxonomy_path"],
}


def load_all_samples(output_dir: Path, category_name: str) -> list[dict[str, Any]]:
    """Load all JSONL samples for a category."""
    samples_dir = output_dir / "samples" / category_name
    if not samples_dir.exists():
        return []

    rows: list[dict[str, Any]] = []
    for jsonl_file in sorted(samples_dir.glob("*.jsonl")):
        with open(jsonl_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def format_value(value: Any) -> str:
    """Format a field value for display."""
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)


def _parse_tools(sample: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract the tools list from a tool_group sample."""
    raw = sample.get("tools_json") or sample.get("tools")
    if isinstance(raw, str):
        raw = json.loads(raw)
    return list(raw or [])


def render_tool_group(sample: dict[str, Any], index: int) -> Panel:
    """Render a tool_group sample with nested tool/variation detail."""
    from rich.console import Group as RichGroup

    parts: list[Any] = []

    # Header fields
    header = Table(show_header=False, box=None, padding=(0, 1), expand=True)
    header.add_column("Field", style="bold cyan", no_wrap=True, ratio=1)
    header.add_column("Value", ratio=4)
    for field in ("domain", "description", "taxonomy_path"):
        if field in sample:
            header.add_row(field, format_value(sample[field]))
    parts.append(header)

    tools = _parse_tools(sample)
    for ti, tool in enumerate(tools):
        tool_table = Table(show_header=False, box=None, padding=(0, 1), expand=True)
        tool_table.add_column("Field", style="bold green", no_wrap=True, ratio=1)
        tool_table.add_column("Value", ratio=4)
        tool_table.add_row("canonical_name", tool.get("canonical_name", ""))
        tool_table.add_row("description", tool.get("description", ""))

        variations = tool.get("variations", [])
        tool_table.add_row("variations", str(len(variations)))

        for vi, var in enumerate(variations):
            params = var.get("parameters", {})
            props = params.get("properties", {})
            required = params.get("required", [])
            param_names = [
                f"{'*' if p in required else ''}{p}:{props[p].get('type', '?')}"
                for p in props
            ]
            returns = var.get("returns", {})
            ret_props = returns.get("properties", {})
            ret_names = [f"{p}:{ret_props[p].get('type', '?')}" for p in ret_props]

            tool_table.add_row(
                f"  v{vi} {var.get('name', '')}",
                f"params({', '.join(param_names)}) -> ({', '.join(ret_names)})",
            )

        parts.append(Panel(
            tool_table,
            title=f"[bold]Tool {ti + 1}/{len(tools)}[/bold]",
            border_style="green",
        ))

    return Panel(
        RichGroup(*parts),
        title=f"[bold]#{index + 1}[/bold]",
        border_style="dim",
    )


def render_sample(sample: dict[str, Any], index: int, category_name: str) -> Panel:
    """Render a single sample as a rich Panel."""
    if category_name == "tool_group":
        return render_tool_group(sample, index)

    fields = _MODEL_FIELDS.get(category_name, list(sample.keys()))

    table = Table(show_header=False, box=None, padding=(0, 1), expand=True)
    table.add_column("Field", style="bold cyan", no_wrap=True, ratio=1)
    table.add_column("Value", ratio=4)

    for field in fields:
        if field in sample:
            table.add_row(field, format_value(sample[field]))

    return Panel(table, title=f"[bold]#{index + 1}[/bold]", border_style="dim")


def inspect_category(output_dir: Path, category_name: str) -> None:
    """Display inspection samples for a single category."""
    config = CATEGORY_CONFIGS[category_name]
    samples = load_all_samples(output_dir, category_name)

    if not samples:
        console.print(f"[red]No samples found for {config.display_name}[/red]")
        return

    # Shuffle to avoid bias from file ordering (files are sorted by leaf hash,
    # so first/last files always map to the same taxonomy subtrees).
    random.shuffle(samples)

    console.rule(f"[bold magenta]{config.display_name}[/bold magenta]", style="magenta")
    console.print(f"[dim]Total samples: {len(samples):,}[/dim]\n")

    # First 5 (from shuffled order)
    console.print("[bold green]Samples 1-5[/bold green]\n")
    for i in range(min(5, len(samples))):
        console.print(render_sample(samples[i], i, category_name))

    # Last 5 (from shuffled order)
    start_idx = max(0, len(samples) - 5)
    console.print(f"\n[bold yellow]Samples {start_idx + 1}-{len(samples)}[/bold yellow]\n")
    for i in range(start_idx, len(samples)):
        console.print(render_sample(samples[i], i, category_name))

    # 10 random (from the middle, avoiding overlap with first/last 5)
    if len(samples) > 20:
        middle = samples[5:-5]
        chosen = random.sample(middle, min(20, len(middle)))
    else:
        chosen = random.sample(samples, min(20, len(samples)))

    console.print("\n[bold blue]20 additional random samples[/bold blue]\n")
    for sample in chosen:
        idx = samples.index(sample)
        console.print(render_sample(sample, idx, category_name))


def main() -> None:
    output_dir = Path("output")
    if len(sys.argv) > 1:
        output_dir = Path(sys.argv[1])

    if not output_dir.exists():
        console.print(f"[red]Output directory not found: {output_dir}[/red]")
        raise SystemExit(1)

    categories = list(CATEGORY_CONFIGS.keys())

    console.print(
        Panel(
            Text.from_markup(
                f"[bold]Inspecting seed data across {len(categories)} categories[/bold]\n"
                f"Output dir: [cyan]{output_dir}[/cyan]\n\n"
                "For each category: first 5, last 5, and 20 random samples.\n"
                "Press Enter to advance to the next category."
            ),
            title="[bold]Seed Data Inspector[/bold]",
            border_style="bright_blue",
        )
    )
    console.print()

    for i, category_name in enumerate(categories):
        inspect_category(output_dir, category_name)

        if i < len(categories) - 1:
            next_name = CATEGORY_CONFIGS[categories[i + 1]].display_name
            console.print()
            input(
                f"--- [{i + 1}/{len(categories)}] Done with "
                f"{CATEGORY_CONFIGS[category_name].display_name}. "
                f"Press Enter for {next_name} ---"
            )
            console.print()

    console.print()
    console.rule("[bold green]Inspection complete[/bold green]", style="green")


if __name__ == "__main__":
    main()
