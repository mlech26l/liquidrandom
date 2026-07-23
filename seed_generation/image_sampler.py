"""Image sample generation: chain planning (LLM) + Nano Banana generation/edits.

Per leaf round: Python pre-samples each chain's edit types (weighted, without
replacement — this is what guarantees edit diversity across the dataset),
parent topology, and aspect ratio; one LLM call writes concrete prompts and
edit instructions for k chains; each chain then generates a base image,
validates it with a VLM, and applies the edits. A chain's rows are appended
to the leaf JSONL only when the whole chain is done, so resume semantics stay
"count rows per leaf" with no half-chains.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

from google import genai
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

from dedup import dedup_batch
from gemini_image import (
    create_image_client,
    edit_image,
    generate_image,
    recompress,
    vlm_validate,
)
from image_categories import EditType, ImageCategoryConfig
from image_config import (
    DEFAULT_EDITS_PER_BASE,
    EDIT_SEQUENTIAL_P,
    EDIT_SPOT_CHECK_RATE,
    MAX_IMAGE_RETRIES,
)
from llm import llm_call
from sampler import _leaf_id
from taxonomy import TaxonomyNode, save_taxonomy

console = Console()

CHAIN_SIZE = 1 + DEFAULT_EDITS_PER_BASE


def _load_leaf_rows(
    output_dir: str, category_name: str, leaf_path: list[str]
) -> list[dict[str, Any]]:
    path = (
        Path(output_dir)
        / "samples"
        / category_name
        / f"{_leaf_id(leaf_path)}.jsonl"
    )
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _count_leaf_rows(output_dir: str, category_name: str, leaf_path: list[str]) -> int:
    path = (
        Path(output_dir)
        / "samples"
        / category_name
        / f"{_leaf_id(leaf_path)}.jsonl"
    )
    if not path.exists():
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _append_leaf_rows(
    output_dir: str,
    category_name: str,
    leaf_path: list[str],
    rows: list[dict[str, Any]],
) -> None:
    path = (
        Path(output_dir)
        / "samples"
        / category_name
        / f"{_leaf_id(leaf_path)}.jsonl"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sample_edit_types(palette: tuple[EditType, ...], count: int) -> list[EditType]:
    """Weighted sample without replacement (Efraimidis-Spirakis)."""
    keyed = sorted(
        palette,
        key=lambda e: random.random() ** (1.0 / e.weight),
        reverse=True,
    )
    return keyed[: min(count, len(keyed))]


def _sample_aspect_ratio(ratios: dict[str, float]) -> str:
    choices = list(ratios)
    weights = [ratios[r] for r in choices]
    return random.choices(choices, weights=weights)[0]


def _sample_parent_turns(n_edits: int) -> list[int]:
    """Parent turn for each edit: first edits the base; later edits target the
    previous image with probability EDIT_SEQUENTIAL_P, else the base."""
    parents: list[int] = []
    for j in range(n_edits):
        if j == 0 or random.random() >= EDIT_SEQUENTIAL_P:
            parents.append(0)
        else:
            parents.append(j)  # previous image has turn_index j
    return parents


def _build_plan_prompt(
    category: ImageCategoryConfig,
    leaf_path_str: str,
    existing_prompts: list[str],
    chain_specs: list[list[EditType]],
) -> str:
    existing_text = "None yet."
    if existing_prompts:
        shown = existing_prompts[-20:]
        existing_text = "\n".join(f"- {p}" for p in shown)
        if len(existing_prompts) > 20:
            existing_text = (
                f"[showing last 20 of {len(existing_prompts)}]\n" + existing_text
            )

    vocab_lines = "\n".join(
        f"- {attr}: exactly one of {list(values)}"
        for attr, values in category.tag_attributes.items()
    )

    chain_lines: list[str] = []
    for i, edits in enumerate(chain_specs, 1):
        edit_desc = "; ".join(f"{e.name} ({e.guidance})" for e in edits)
        chain_lines.append(f"Chain {i} edit types, in order: {edit_desc}")

    return (
        f"You are writing prompts for an image generation model to build a "
        f"diverse {category.display_name} dataset.\n\n"
        f"Topic: {leaf_path_str}\n\n"
        f"{category.prompt_guidance}\n\n"
        f"Previously used prompts for this topic (make new ones visually and "
        f"semantically DIFFERENT):\n{existing_text}\n\n"
        f"Tag vocabulary (every image gets one tag per attribute):\n"
        f"{vocab_lines}\n\n"
        f"Create {len(chain_specs)} image chains. Each chain has a base image "
        f"plus edits of the assigned edit types:\n"
        + "\n".join(chain_lines)
        + "\n\n"
        "For each chain return:\n"
        '- "prompt": the base image generation prompt (2-4 sentences, concrete, '
        "specific to the topic, distinct from the other chains)\n"
        '- "caption": one sentence describing the base image\n'
        '- "tags": array with exactly one vocabulary value per attribute, '
        "consistent with the prompt\n"
        '- "edits": one object per assigned edit type, in order, each with:\n'
        '  - "instruction": a concrete edit instruction (imperative, 1-2 '
        "sentences) implementing that edit type for this specific image. "
        "Edits may be applied cumulatively, so each instruction must make "
        "sense on its own regardless of the other edits.\n"
        '  - "tag_updates": object mapping any attribute whose value the edit '
        "changes to its new vocabulary value (empty object if none)\n\n"
        'Return a JSON object: {"chains": [{"prompt": ..., "caption": ..., '
        '"tags": [...], "edits": [{"instruction": ..., "tag_updates": {...}}]}]}'
    )


def _apply_tag_updates(
    tags: list[str],
    updates: dict[str, str],
    tag_attributes: dict[str, tuple[str, ...]],
) -> list[str]:
    """Replace the tag of each updated attribute, keeping unknown updates out."""
    result = list(tags)
    for attr, new_value in updates.items():
        values = tag_attributes.get(attr)
        if not values or new_value not in values:
            continue
        result = [t for t in result if t not in values]
        result.append(new_value)
    return result


def _valid_plan(plan: Any, n_edits: int) -> bool:
    if not isinstance(plan, dict):
        return False
    if not plan.get("prompt") or not plan.get("caption"):
        return False
    edits = plan.get("edits")
    if not isinstance(edits, list) or len(edits) < n_edits:
        return False
    return all(isinstance(e, dict) and e.get("instruction") for e in edits)


async def _run_chain(
    image_client: genai.Client,
    image_sem: asyncio.Semaphore,
    category: ImageCategoryConfig,
    leaf_path_str: str,
    plan: dict[str, Any],
    edit_types: list[EditType],
    parent_turns: list[int],
    aspect_ratio: str,
) -> list[dict[str, Any]]:
    """Generate one full chain. Returns its JSONL rows ([] if rejected)."""
    prompt = str(plan["prompt"])
    caption = str(plan["caption"])
    candidate_tags = [t for t in plan.get("tags", []) if isinstance(t, str)]
    chain_id = hashlib.sha256(
        f"{leaf_path_str}|{prompt}".encode()
    ).hexdigest()[:16]

    async with image_sem:
        raw = await generate_image(image_client, prompt, aspect_ratio)
        webp, width, height = recompress(raw)
        accepted, base_tags = await vlm_validate(
            image_client,
            webp,
            "image/webp",
            prompt,
            candidate_tags,
            category.tag_attributes,
            category.specificity_guidance,
        )
    if not accepted:
        return []

    n_edits = len(edit_types)
    chain_length = 1 + n_edits

    def _row(
        image_webp: bytes,
        w: int,
        h: int,
        turn: int,
        parent: int,
        instruction: str,
        tags: list[str],
    ) -> dict[str, Any]:
        return {
            "image_base64": base64.b64encode(image_webp).decode("ascii"),
            "image_format": "webp",
            "width": w,
            "height": h,
            "aspect_ratio": aspect_ratio,
            "taxonomy_path": leaf_path_str,
            "caption": caption,
            "prompt": prompt,
            "edit_instruction": instruction,
            "tags": tags,
            "chain_id": chain_id,
            "turn_index": turn,
            "parent_turn": parent,
            "chain_length": chain_length,
        }

    rows = [_row(webp, width, height, 0, -1, "", base_tags)]
    # Image bytes and tags per turn, for edits that build on earlier turns.
    turn_images: dict[int, bytes] = {0: webp}
    turn_tags: dict[int, list[str]] = {0: base_tags}

    for j, edit_type in enumerate(edit_types):
        edit_plan = plan["edits"][j]
        instruction = str(edit_plan["instruction"])
        tag_updates = edit_plan.get("tag_updates") or {}
        parent = parent_turns[j]
        if parent not in turn_images:
            parent = 0
        turn = j + 1

        try:
            async with image_sem:
                edited_raw = await edit_image(
                    image_client, turn_images[parent], "image/webp", instruction
                )
                edited_webp, w, h = recompress(edited_raw)
                if random.random() < EDIT_SPOT_CHECK_RATE:
                    ok, _ = await vlm_validate(
                        image_client,
                        edited_webp,
                        "image/webp",
                        f"{prompt}\nThen edited: {instruction}",
                        turn_tags[parent],
                        category.tag_attributes,
                        category.specificity_guidance,
                    )
                    if not ok:
                        continue
        except RuntimeError:
            continue  # drop this edit, keep the rest of the chain

        tags = _apply_tag_updates(
            turn_tags[parent],
            tag_updates if isinstance(tag_updates, dict) else {},
            category.tag_attributes,
        )
        rows.append(_row(edited_webp, w, h, turn, parent, instruction, tags))
        turn_images[turn] = edited_webp
        turn_tags[turn] = tags

    # Dropped edits shrink the chain; record the actual length.
    if len(rows) != chain_length:
        for row in rows:
            row["chain_length"] = len(rows)
    return rows


async def _generate_for_leaf(
    llm_client: AsyncOpenAI,
    image_client: genai.Client,
    llm_sem: asyncio.Semaphore,
    image_sem: asyncio.Semaphore,
    leaf: TaxonomyNode,
    category: ImageCategoryConfig,
    k: int,
    dedup_threshold: float,
    output_dir: str,
) -> int:
    """One round for one leaf: plan up to k chains, generate them.

    Returns the number of new images accepted.
    """
    leaf_path_str = " > ".join(leaf.path)
    existing = _load_leaf_rows(output_dir, category.name, leaf.path)
    existing_prompts = sorted({r["prompt"] for r in existing if r.get("prompt")})

    remaining = leaf.target_count - leaf.sample_count
    n_chains = min(k, max(1, math.ceil(remaining / CHAIN_SIZE)))

    # Pre-sample per-chain structure in Python: this, not the LLM, is what
    # guarantees edit-type diversity across the dataset.
    chain_specs = [
        _sample_edit_types(category.edit_palette, DEFAULT_EDITS_PER_BASE)
        for _ in range(n_chains)
    ]

    plan_prompt = _build_plan_prompt(
        category, leaf_path_str, existing_prompts, chain_specs
    )

    for _ in range(MAX_IMAGE_RETRIES):
        try:
            async with llm_sem:
                result = await llm_call(
                    llm_client, [{"role": "user", "content": plan_prompt}]
                )
        except RuntimeError as e:
            console.print(f"    [red]Plan failed for {leaf_path_str}: {e}[/red]")
            return 0

        plans = result.get("chains", []) if isinstance(result, dict) else result
        if not isinstance(plans, list):
            continue
        indexed = [
            (i, p)
            for i, p in enumerate(plans[: len(chain_specs)])
            if _valid_plan(p, len(chain_specs[i]) if i < len(chain_specs) else 0)
        ]
        if not indexed:
            continue

        # Dedup base prompts against existing and within the batch.
        deduped = dedup_batch(
            indexed,
            existing_prompts,
            lambda item: item[1]["prompt"]
            if isinstance(item, tuple)
            else str(item),
            dedup_threshold,
        )
        if not deduped:
            continue

        chain_tasks = [
            _run_chain(
                image_client,
                image_sem,
                category,
                leaf_path_str,
                plan,
                chain_specs[i],
                _sample_parent_turns(len(chain_specs[i])),
                _sample_aspect_ratio(category.aspect_ratios),
            )
            for i, plan in deduped
        ]
        results = await asyncio.gather(*chain_tasks, return_exceptions=True)

        new_rows: list[dict[str, Any]] = []
        for res in results:
            if isinstance(res, BaseException):
                console.print(
                    f"    [red]Chain failed for {leaf_path_str}: {res}[/red]"
                )
            elif res:
                new_rows.extend(res)

        if new_rows:
            _append_leaf_rows(output_dir, category.name, leaf.path, new_rows)
            leaf.sample_count += len(new_rows)
            return len(new_rows)

    return 0


async def generate_image_samples(
    client: AsyncOpenAI,
    root: TaxonomyNode,
    category: ImageCategoryConfig,
    *,
    target_samples: int,
    k: int,
    batch_size: int,
    image_concurrency: int,
    dedup_threshold: float,
    output_dir: str,
) -> int:
    """Generate image samples by cycling through leaves in round-robin.

    `client` is the OpenRouter LLM client for chain planning; the Gemini
    image client is created here. Returns total images on disk.
    """
    image_client = create_image_client()
    leaves = root.leaf_nodes()

    for leaf in leaves:
        leaf.sample_count = _count_leaf_rows(output_dir, category.name, leaf.path)

    total_existing = sum(leaf.sample_count for leaf in leaves)
    if total_existing >= target_samples:
        console.print(
            f"  [green]Already have {total_existing}/{target_samples} images[/green]"
        )
        return total_existing

    console.print(
        f"  Starting from {total_existing}/{target_samples} images "
        f"across {len(leaves)} leaves"
    )

    llm_sem = asyncio.Semaphore(batch_size)
    image_sem = asyncio.Semaphore(image_concurrency)
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
                    image_client,
                    llm_sem,
                    image_sem,
                    leaf,
                    category,
                    k,
                    dedup_threshold,
                    output_dir,
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

            save_taxonomy(root, output_dir, category.name)

            if batch_total == 0:
                stall_rounds += 1
                console.print(
                    "    [yellow]No images generated this round, some leaves "
                    "may be saturated[/yellow]"
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
