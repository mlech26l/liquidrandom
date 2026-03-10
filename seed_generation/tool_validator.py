"""Tool-specific validation for generated tool groups."""

from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI

from llm import llm_call


async def validate_tool_groups(
    client: AsyncOpenAI,
    groups: list[dict[str, Any]],
    leaf_path: str,
) -> list[dict[str, Any]]:
    """Validate generated tool groups using the LLM.

    Checks:
    1. OpenAI function calling schema compliance
    2. Inter-tool dependency coherence (return of A matches input of B)
    3. No placeholder content
    4. Tools are relevant to the leaf domain

    Returns accepted groups. Empty list if >50% rejected.
    """
    if not groups:
        return []

    groups_text = ""
    for i, group in enumerate(groups):
        groups_text += f"\n--- Group {i} ---\n"
        groups_text += f"  domain: {group.get('domain', '')}\n"
        groups_text += f"  description: {group.get('description', '')}\n"
        tools = group.get("tools", [])
        for j, tool in enumerate(tools):
            groups_text += f"  Tool {j}: {tool.get('canonical_name', '')} - {tool.get('description', '')}\n"
            variation = tool.get("variations", [{}])[0] if tool.get("variations") else {}
            if variation:
                groups_text += f"    Parameters: {variation.get('parameters', {})}\n"
                groups_text += f"    Returns: {variation.get('returns', {})}\n"

    messages = [
        {
            "role": "user",
            "content": (
                "You are a quality validator for LLM tool/function definitions.\n\n"
                f"These tool groups were generated for the taxonomy path: {leaf_path}\n\n"
                "Review each group and check for:\n"
                "1. Valid OpenAI function calling schema (parameters must have 'type': 'object', "
                "'properties', and 'required' keys)\n"
                "2. Inter-tool dependency coherence (the output of one tool should logically "
                "connect to the input of related tools in the group)\n"
                "3. No placeholder content (e.g. 'TBD', 'example', empty descriptions)\n"
                "4. Tools are relevant to the stated domain and taxonomy path\n"
                "5. Each group has 3-6 related, non-redundant tools\n\n"
                f"Groups to validate:\n{groups_text}\n\n"
                "Return a JSON object with a single key \"verdicts\" containing an array of objects, "
                "one per group, each with:\n"
                "- index (integer): group index (0-based)\n"
                "- verdict (string): \"accept\" or \"reject\"\n"
                "- reason (string): brief explanation"
            ),
        }
    ]

    result = await llm_call(client, messages)

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
            if verdict == "accept" and 0 <= idx < len(groups):
                accepted_indices.add(idx)
            else:
                rejected_count += 1

    if rejected_count > len(groups) * 0.5:
        return []

    return [groups[i] for i in sorted(accepted_indices)]
