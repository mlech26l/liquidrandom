"""LLM-based quality validation for generated samples."""

from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI

from categories import CategoryConfig
from llm import llm_call


async def validate_batch(
    client: AsyncOpenAI,
    samples: list[dict[str, Any]],
    category: CategoryConfig,
    leaf_path: str,
    model: str | None = None,
) -> list[dict[str, Any]]:
    """Validate a batch of samples using the LLM.

    Returns only accepted samples. If >50% are rejected, returns empty list
    to signal the caller to retry.
    """
    if not samples:
        return []

    samples_text = ""
    for i, sample in enumerate(samples):
        samples_text += f"\n--- Sample {i} ---\n"
        for key, value in sample.items():
            samples_text += f"  {key}: {value}\n"

    messages = [
        {
            "role": "user",
            "content": (
                f"You are a quality validator for {category.display_name} seed data.\n\n"
                f"These samples were generated for the taxonomy path: {leaf_path}\n\n"
                f"Review each sample and check for:\n"
                f"1. Empty or placeholder content (e.g. 'TBD', 'example', lorem ipsum)\n"
                f"2. Hallucination or factual impossibility\n"
                f"3. Repetitiveness (samples that are too similar to each other)\n"
                f"4. Off-topic (doesn't match the taxonomy path)\n"
                f"5. Insufficient specificity ({category.specificity_guidance})\n\n"
                f"Samples to validate:\n{samples_text}\n\n"
                f"Return a JSON object with a single key \"verdicts\" containing an array of objects, "
                f"one per sample, each with:\n"
                f"- index (integer): sample index (0-based)\n"
                f"- verdict (string): \"accept\" or \"reject\"\n"
                f"- reason (string): brief explanation"
            ),
        }
    ]

    result = await llm_call(client, messages, model=model)

    if isinstance(result, dict):
        verdicts = result.get("verdicts", [])
    else:
        verdicts = result

    accepted_indices: set[int] = set()
    rejected_count = 0

    for v in verdicts:
        if isinstance(v, dict):
            idx = v.get("index", -1)
            verdict = v.get("verdict", "reject")
            if verdict == "accept" and 0 <= idx < len(samples):
                accepted_indices.add(idx)
            else:
                rejected_count += 1

    # If >50% rejected, signal caller to retry by returning empty
    if rejected_count > len(samples) * 0.5:
        return []

    return [samples[i] for i in sorted(accepted_indices)]
