"""Tool group generation: groups of related LLM tools with parameter variations."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from config import MAX_VALIDATION_RETRIES
from dedup import dedup_batch
from llm import llm_call
from taxonomy import TaxonomyNode, save_taxonomy
from tool_validator import validate_tool_groups

console = Console()


def _leaf_id(leaf_path: list[str]) -> str:
    raw = "/".join(leaf_path)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _load_leaf_samples(
    output_dir: str, leaf_path: list[str]
) -> list[dict[str, Any]]:
    path = (
        Path(output_dir) / "samples" / "tool_group" / f"{_leaf_id(leaf_path)}.jsonl"
    )
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
    leaf_path: list[str],
    samples: list[dict[str, Any]],
) -> None:
    path = (
        Path(output_dir) / "samples" / "tool_group" / f"{_leaf_id(leaf_path)}.jsonl"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def _group_to_str(sample: dict[str, Any]) -> str:
    """Convert a tool group dict to a string for dedup comparison."""
    tools = sample.get("tools", [])
    tool_names = " ".join(t.get("canonical_name", "") for t in tools)
    return f"{sample.get('domain', '')} {sample.get('description', '')} {tool_names}"


GROUP_GENERATION_PROMPT = """\
Generate {k} groups of related LLM tools/functions for the domain: {leaf_path}

Each group represents a coherent set of 3-6 related API functions that an AI agent \
would use together. Tools within a group should have inter-dependencies \
(the return type of one tool should be an input to another).

Use OpenAI function calling format for parameters and returns:
- parameters: {{"type": "object", "properties": {{...}}, "required": [...]}}
- returns: {{"type": "object", "properties": {{...}}}}

Each tool in a group needs:
- canonical_name: snake_case function name
- description: what the function does (1 sentence)
- variations: array with exactly 1 variation (the canonical form), containing:
  - name: same as canonical_name
  - description: same as tool description
  - parameters: OpenAI JSON Schema for input
  - returns: OpenAI JSON Schema for output

Previously generated groups for this domain (DO NOT repeat these):
{existing_samples}

Return a JSON object with a single key "groups" containing an array of {k} objects. \
Each object must have:
- domain (string): specific tool domain name
- description (string): what this tool group does (1-2 sentences)
- tools (array): 3-6 tool objects each with canonical_name, description, and variations"""


VARIATION_GENERATION_PROMPT = """\
Generate a new parameter variation for the tool function "{tool_name}".

Original tool description: {tool_description}
Domain context: {domain}

The variation should differ from existing variations in:
- Parameter names (e.g. "departure_date" vs "travel_date" vs "date_of_departure")
- Parameter types (e.g. string date vs integer timestamp vs object with day/month/year)
- Granularity (e.g. single "address" string vs structured address object)
- Optional vs required fields
- Nesting structure (flat vs deeply nested)

But the variation must serve the SAME semantic purpose as the original function.

Existing variations (generate something DIFFERENT from all of these):
{existing_variations}

Return a JSON object with:
- name (string): snake_case function name (a plausible alternative name)
- description (string): what the function does (rephrased)
- parameters (object): OpenAI JSON Schema format {{"type": "object", "properties": {{...}}, "required": [...]}}
- returns (object): OpenAI JSON Schema format {{"type": "object", "properties": {{...}}}}"""


async def _generate_variations_for_tool(
    client: AsyncOpenAI,
    tool: dict[str, Any],
    domain: str,
    m: int,
) -> list[dict[str, Any]]:
    """Generate M-1 additional variations for a single tool (sequential, context-dependent)."""
    variations = list(tool.get("variations", []))
    if not variations:
        return variations

    for _ in range(m - 1):
        existing_text = json.dumps(variations, indent=2)
        prompt = VARIATION_GENERATION_PROMPT.format(
            tool_name=tool["canonical_name"],
            tool_description=tool["description"],
            domain=domain,
            existing_variations=existing_text,
        )

        try:
            result = await llm_call(client, [{"role": "user", "content": prompt}])
            if isinstance(result, dict) and "name" in result:
                variations.append(result)
        except Exception as e:
            console.print(
                f"    [red]Variation generation failed for "
                f"{tool['canonical_name']}: {e}[/red]"
            )

    return variations


async def _generate_for_leaf(
    client: AsyncOpenAI,
    leaf: TaxonomyNode,
    k: int,
    m: int,
    dedup_threshold: float,
    output_dir: str,
    batch_semaphore: asyncio.Semaphore,
) -> int:
    """Generate tool groups for a single leaf. Returns number of new groups accepted."""
    leaf_path_str = " > ".join(leaf.path)
    existing = _load_leaf_samples(output_dir, leaf.path)

    existing_text = "None yet." if not existing else ""
    if existing:
        for s in existing[-10:]:
            existing_text += _group_to_str(s) + "\n"
        if len(existing) > 10:
            existing_text = f"[showing last 10 of {len(existing)}]\n" + existing_text

    prompt = GROUP_GENERATION_PROMPT.format(
        k=k,
        leaf_path=leaf_path_str,
        existing_samples=existing_text,
    )

    for _attempt in range(MAX_VALIDATION_RETRIES):
        try:
            # Phase A: Generate groups
            async with batch_semaphore:
                result = await llm_call(
                    client, [{"role": "user", "content": prompt}]
                )

            if isinstance(result, dict):
                raw_groups = result.get("groups", [])
            else:
                raw_groups = result

            if not raw_groups:
                continue

            # Phase C: Validate groups
            async with batch_semaphore:
                accepted = await validate_tool_groups(
                    client, raw_groups, leaf_path_str
                )

            if not accepted:
                continue

            # Phase B: Generate variations for each tool in accepted groups
            for group in accepted:
                tools = group.get("tools", [])
                variation_tasks = [
                    _generate_variations_for_tool(client, tool, group["domain"], m)
                    for tool in tools
                ]
                variation_results = await asyncio.gather(
                    *variation_tasks, return_exceptions=True
                )
                for tool, var_result in zip(tools, variation_results):
                    if isinstance(var_result, list):
                        tool["variations"] = var_result

            # Convert to storage format (tools_json string)
            storage_groups: list[dict[str, Any]] = []
            for group in accepted:
                storage_groups.append(
                    {
                        "domain": group["domain"],
                        "description": group["description"],
                        "taxonomy_path": leaf_path_str,
                        "tools": group["tools"],
                        "tools_json": json.dumps(
                            group["tools"], ensure_ascii=False
                        ),
                    }
                )

            # Dedup against existing
            deduped = dedup_batch(
                storage_groups, existing, _group_to_str, dedup_threshold
            )

            if deduped:
                _save_leaf_samples(output_dir, leaf.path, deduped)
                leaf.sample_count += len(deduped)
                return len(deduped)

        except Exception as e:
            console.print(
                f"    [red]Error generating for {leaf_path_str}: {e}[/red]"
            )

    return 0


async def generate_tool_groups(
    client: AsyncOpenAI,
    root: TaxonomyNode,
    target_samples: int,
    k: int,
    m: int,
    batch_size: int,
    dedup_threshold: float,
    output_dir: str,
) -> int:
    """Generate tool groups by cycling through all leaf nodes.

    Returns total number of tool groups generated.
    """
    leaves = root.leaf_nodes()

    for leaf in leaves:
        existing = _load_leaf_samples(output_dir, leaf.path)
        leaf.sample_count = len(existing)

    total_existing = sum(leaf.sample_count for leaf in leaves)
    if total_existing >= target_samples:
        console.print(
            f"  [green]Already have {total_existing}/{target_samples} tool groups[/green]"
        )
        return total_existing

    console.print(
        f"  Starting from {total_existing}/{target_samples} tool groups "
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
            "  Tool Groups",
            total=target_samples,
            completed=total_existing,
        )

        stall_rounds = 0

        while total_generated < target_samples:
            incomplete = [
                leaf for leaf in leaves if leaf.sample_count < leaf.target_count
            ]
            if not incomplete:
                break

            async def process_leaf(leaf: TaxonomyNode) -> int:
                count = await _generate_for_leaf(
                    client,
                    leaf,
                    k,
                    m,
                    dedup_threshold,
                    output_dir,
                    semaphore,
                )
                nonlocal total_generated
                total_generated += count
                progress.update(
                    task, completed=min(total_generated, target_samples)
                )
                return count

            tasks = [
                asyncio.create_task(process_leaf(leaf)) for leaf in incomplete
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            batch_total = sum(r for r in results if isinstance(r, int))

            save_taxonomy(root, output_dir, "tool_group")

            if batch_total == 0:
                stall_rounds += 1
                console.print(
                    "    [yellow]No groups generated in this round[/yellow]"
                )
                for leaf in incomplete:
                    if leaf.sample_count == 0:
                        leaf.target_count = 0
                if stall_rounds >= 3:
                    console.print(
                        "    [red]3 consecutive stalls, stopping[/red]"
                    )
                    break
            else:
                stall_rounds = 0

    return total_generated
