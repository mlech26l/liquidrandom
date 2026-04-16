"""AsyncOpenAI client wrapper for OpenRouter with retries."""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any

from openai import AsyncOpenAI

from config import MODEL_NAME, OPENROUTER_BASE_URL


def create_client() -> AsyncOpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable is not set")
    return AsyncOpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)


def _extract_json(text: str) -> Any:
    """Extract JSON from text, handling markdown code blocks."""
    # Try direct parse first
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try finding first { or [ and matching to end
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        if start == -1:
            continue
        end = text.rfind(end_char)
        if end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue

    raise ValueError(f"Could not extract JSON from response: {text[:200]}")


async def llm_call(
    client: AsyncOpenAI,
    messages: list[dict[str, str]],
    *,
    use_reasoning: bool = True,
    max_retries: int = 3,
    model: str | None = None,
) -> dict[str, Any] | list[Any]:
    """Make an LLM call with retries and JSON parsing.

    Returns parsed JSON (dict or list).
    """
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            extra_body: dict[str, Any] = {}
            if use_reasoning:
                extra_body["reasoning"] = {"enabled": True}

            response = await client.chat.completions.create(
                model=model or MODEL_NAME,
                messages=messages,
                extra_body=extra_body if extra_body else None,
            )

            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response from LLM")

            return _extract_json(content)

        except Exception as e:
            last_error = e
            # Don't retry on 400 errors (bad request, content moderation, etc.)
            if hasattr(e, "status_code") and e.status_code == 400:
                break
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                await asyncio.sleep(wait)

    raise RuntimeError(
        f"LLM call failed after {max_retries} retries: {last_error}"
    )
