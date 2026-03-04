"""Phase 1: Hierarchical taxonomy tree generation."""

from __future__ import annotations

import asyncio
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn, TimeRemainingColumn

from categories import CategoryConfig
from llm import llm_call

console = Console()


@dataclass
class TaxonomyNode:
    name: str
    path: list[str]
    children: list[TaxonomyNode] = field(default_factory=list)
    sample_count: int = 0
    target_count: int = 0

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def leaf_nodes(self) -> list[TaxonomyNode]:
        if self.is_leaf:
            return [self]
        leaves: list[TaxonomyNode] = []
        for child in self.children:
            leaves.extend(child.leaf_nodes())
        return leaves

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "sample_count": self.sample_count,
            "target_count": self.target_count,
            "children": [c.to_dict() for c in self.children],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaxonomyNode:
        node = cls(
            name=data["name"],
            path=list(data["path"]),
            sample_count=data.get("sample_count", 0),
            target_count=data.get("target_count", 0),
        )
        node.children = [cls.from_dict(c) for c in data.get("children", [])]
        return node


def save_taxonomy(node: TaxonomyNode, output_dir: str, category_name: str) -> Path:
    path = Path(output_dir) / "taxonomies" / f"{category_name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(node.to_dict(), f, indent=2)
    return path


def load_taxonomy(output_dir: str, category_name: str) -> TaxonomyNode | None:
    path = Path(output_dir) / "taxonomies" / f"{category_name}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return TaxonomyNode.from_dict(json.load(f))


async def _expand_node(
    client: AsyncOpenAI,
    node: TaxonomyNode,
    category: CategoryConfig,
    branching_factor: int,
    existing_siblings: list[str],
) -> list[TaxonomyNode]:
    """Expand a single node into children using the LLM."""
    path_str = " > ".join(node.path)
    siblings_str = ", ".join(existing_siblings) if existing_siblings else "none yet"

    messages = [
        {
            "role": "user",
            "content": (
                f"You are building a taxonomy tree for generating diverse {category.display_name}.\n\n"
                f"{category.taxonomy_seed_prompt}\n\n"
                f"Current node path: {path_str}\n"
                f"Already existing sibling branches at this level: {siblings_str}\n\n"
                f"Generate exactly {branching_factor} new sub-categories for this node. "
                f"Each sub-category should be distinct from the existing siblings and from each other. "
                f"Make them specific and non-overlapping.\n\n"
                f"Return a JSON object with a single key \"branches\" containing an array of "
                f"{branching_factor} strings, each being the name of a sub-category."
            ),
        }
    ]

    result = await llm_call(client, messages)
    if isinstance(result, dict):
        branches = result.get("branches", [])
    else:
        branches = result

    children: list[TaxonomyNode] = []
    for name in branches:
        if isinstance(name, str):
            child = TaxonomyNode(
                name=name.strip(),
                path=node.path + [name.strip()],
            )
            children.append(child)

    return children


async def generate_taxonomy(
    client: AsyncOpenAI,
    category: CategoryConfig,
    target_samples: int,
    samples_per_leaf: int,
    max_depth: int,
    batch_size: int,
    output_dir: str,
) -> TaxonomyNode:
    """Generate a hierarchical taxonomy tree for a category.

    Auto-scales branching factor based on target sample count and depth.
    """
    # Check for existing taxonomy
    existing = load_taxonomy(output_dir, category.name)
    if existing:
        existing_leaves = len(existing.leaf_nodes())
        required = math.ceil(target_samples / samples_per_leaf)
        if existing_leaves >= required:
            console.print(
                f"  [green]Reusing existing taxonomy ({existing_leaves} leaves)[/green]"
            )
            return existing
        console.print(
            f"  [yellow]Existing taxonomy has {existing_leaves} leaves, "
            f"need {required}. Regenerating.[/yellow]"
        )

    required_leaf_nodes = math.ceil(target_samples / samples_per_leaf)
    branching_factor = max(3, math.ceil(required_leaf_nodes ** (1.0 / max_depth)))

    console.print(
        f"  Target: {required_leaf_nodes} leaves "
        f"(branching factor: {branching_factor}, depth: {max_depth})"
    )

    root = TaxonomyNode(name=category.display_name, path=[category.display_name])

    semaphore = asyncio.Semaphore(batch_size)

    async def expand_with_semaphore(
        node: TaxonomyNode, siblings: list[str]
    ) -> list[TaxonomyNode]:
        async with semaphore:
            return await _expand_node(
                client, node, category, branching_factor, siblings
            )

    # Expand level by level (BFS)
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
        for depth in range(max_depth):
            leaves = root.leaf_nodes()
            if len(leaves) >= required_leaf_nodes:
                break

            task = progress.add_task(
                f"  Depth {depth + 1}/{max_depth}", total=len(leaves)
            )

            # Expand all current leaves in parallel
            tasks: list[tuple[TaxonomyNode, asyncio.Task[list[TaxonomyNode]]]] = []
            for leaf in leaves:
                sibling_names = [c.name for c in leaf.children]
                coro = expand_with_semaphore(leaf, sibling_names)
                tasks.append((leaf, asyncio.create_task(coro)))

            for leaf, expansion_task in tasks:
                try:
                    children = await expansion_task
                    leaf.children = children
                except Exception as e:
                    console.print(
                        f"  [red]Failed to expand {' > '.join(leaf.path)}: {e}[/red]"
                    )
                progress.advance(task)

            # Save after each level
            save_taxonomy(root, output_dir, category.name)
            current_leaves = len(root.leaf_nodes())
            console.print(
                f"  Depth {depth + 1}: {current_leaves} leaf nodes"
            )

    # Set target counts on leaves
    leaves = root.leaf_nodes()
    per_leaf = math.ceil(target_samples / len(leaves))
    for leaf in leaves:
        leaf.target_count = per_leaf

    save_taxonomy(root, output_dir, category.name)
    console.print(
        f"  [green]Taxonomy complete: {len(leaves)} leaves, "
        f"{per_leaf} samples per leaf[/green]"
    )

    return root
