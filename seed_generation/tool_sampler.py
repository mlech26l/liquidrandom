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
    output_dir: str, category_name: str, leaf_path: list[str]
) -> list[dict[str, Any]]:
    path = (
        Path(output_dir) / "samples" / category_name / f"{_leaf_id(leaf_path)}.jsonl"
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
    category_name: str,
    leaf_path: list[str],
    samples: list[dict[str, Any]],
) -> None:
    path = (
        Path(output_dir) / "samples" / category_name / f"{_leaf_id(leaf_path)}.jsonl"
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
would use together. Tools within a group MUST have inter-dependencies \
(the return type of one tool should be usable as input to another).

IMPORTANT: Aim for diverse tool counts across groups. Generate some groups with 3, \
some with 4, some with 5, and some with 6 tools. Do NOT default to always using 3-4 tools.

CRITICAL: Ensure interface compatibility across all tools in the group. When tool A returns \
a field (e.g. "flight_id" of type "string"), any tool B that needs that value as input \
MUST use the SAME field name and type. Mismatched interfaces (e.g. tool A returns \
"flight_id:uuid" but tool B expects "flight_number:int") are not acceptable.

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


GROUP_VARIATION_PROMPT = """\
Generate a complete new parameter variation for ALL tools in this group simultaneously.

Domain: {domain}
Description: {group_description}

Current tools and their canonical (v0) interfaces:
{canonical_tools}

Existing group variations (generate something DIFFERENT from all of these):
{existing_variations}

CRITICAL REQUIREMENTS:
1. Generate a variation for EVERY tool in the group at once (not independently per tool).
2. All tools in this variation MUST have compatible interfaces with each other:
   - If tool A's return includes "order_id" (string), then any tool B that needs it as input \
MUST accept "order_id" (string) with the SAME name and type.
   - All cross-tool references must match exactly.
3. Each variation should differ from existing ones in parameter/return naming, types, \
granularity, or structure, but maintain the same semantic purpose.
4. Use plausible alternative function names (snake_case).

Return a JSON object with a single key "tools" containing an array of objects, \
one per tool in the same order as above. Each object must have:
- canonical_name (string): the original canonical_name (unchanged, for matching)
- variation (object): the new variation with name, description, parameters, returns"""


async def _generate_group_variation(
    client: AsyncOpenAI,
    group: dict[str, Any],
    existing_group_variations: list[list[dict[str, Any]]],
) -> list[dict[str, Any]] | None:
    """Generate one complete group-level variation (all tools at once).

    Returns a list of variation dicts (one per tool), or None on failure.
    """
    tools = group.get("tools", [])
    if not tools:
        return None

    # Format canonical tools for the prompt
    canonical_text = ""
    for i, tool in enumerate(tools):
        canonical_text += f"\nTool {i + 1}: {tool['canonical_name']}\n"
        canonical_text += f"  Description: {tool['description']}\n"
        v0 = tool.get("variations", [{}])[0] if tool.get("variations") else {}
        if v0:
            canonical_text += f"  Parameters: {json.dumps(v0.get('parameters', {}))}\n"
            canonical_text += f"  Returns: {json.dumps(v0.get('returns', {}))}\n"

    # Format existing variations
    existing_text = "None yet." if not existing_group_variations else ""
    for vi, var_set in enumerate(existing_group_variations):
        existing_text += f"\n--- Variation {vi + 1} ---\n"
        for tool_var in var_set:
            existing_text += f"  {tool_var.get('name', '?')}: params={json.dumps(tool_var.get('parameters', {}))}\n"

    prompt = GROUP_VARIATION_PROMPT.format(
        domain=group["domain"],
        group_description=group["description"],
        canonical_tools=canonical_text,
        existing_variations=existing_text,
    )

    try:
        result = await llm_call(client, [{"role": "user", "content": prompt}])
        if not isinstance(result, dict):
            return None
        tool_variations = result.get("tools", [])
        if len(tool_variations) != len(tools):
            return None

        # Extract the variation for each tool
        variations: list[dict[str, Any]] = []
        for tv in tool_variations:
            var = tv.get("variation", tv)
            if not isinstance(var, dict) or "name" not in var:
                return None
            variations.append(var)

        return variations
    except Exception as e:
        console.print(f"    [red]Group variation generation failed: {e}[/red]")
        return None


async def _generate_variations_for_group(
    client: AsyncOpenAI,
    group: dict[str, Any],
    m: int,
) -> None:
    """Generate M-1 additional group-level variations (sequential, context-dependent).

    Modifies group["tools"] in place, adding variations to each tool.
    """
    tools = group.get("tools", [])
    if not tools:
        return

    # Track all generated group variations for context
    existing_group_variations: list[list[dict[str, Any]]] = []

    for _ in range(m - 1):
        variation_set = await _generate_group_variation(
            client, group, existing_group_variations
        )

        if variation_set and len(variation_set) == len(tools):
            existing_group_variations.append(variation_set)
            # Append each tool's variation to its variations list
            for tool, var in zip(tools, variation_set):
                if "variations" not in tool:
                    tool["variations"] = []
                tool["variations"].append(var)


async def _generate_for_leaf(
    client: AsyncOpenAI,
    category_name: str,
    leaf: TaxonomyNode,
    k: int,
    m: int,
    dedup_threshold: float,
    output_dir: str,
    batch_semaphore: asyncio.Semaphore,
) -> int:
    """Generate tool groups for a single leaf. Returns number of new groups accepted."""
    leaf_path_str = " > ".join(leaf.path)
    existing = _load_leaf_samples(output_dir, category_name, leaf.path)

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

            # Phase C: Validate groups (before variations, to avoid wasting LLM calls)
            async with batch_semaphore:
                accepted = await validate_tool_groups(
                    client, raw_groups, leaf_path_str
                )

            if not accepted:
                continue

            # Phase B: Generate group-level variations (sequential per group)
            for group in accepted:
                await _generate_variations_for_group(client, group, m)

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
                _save_leaf_samples(output_dir, category_name, leaf.path, deduped)
                leaf.sample_count += len(deduped)
                return len(deduped)

        except Exception as e:
            console.print(
                f"    [red]Error generating for {leaf_path_str}: {e}[/red]"
            )

    return 0


async def generate_tool_groups(
    client: AsyncOpenAI,
    category_name: str,
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
        existing = _load_leaf_samples(output_dir, category_name, leaf.path)
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
                    category_name,
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

            save_taxonomy(root, output_dir, category_name)

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
