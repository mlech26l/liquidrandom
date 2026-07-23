"""Async Gemini (Nano Banana) client wrapper for image generation and editing."""

from __future__ import annotations

import asyncio
import io
import json
import os
from typing import Any

from google import genai
from google.genai import types
from PIL import Image

from image_config import (
    IMAGE_MODEL_NAME,
    MAX_IMAGE_RETRIES,
    VALIDATION_MODEL_NAME,
    WEBP_QUALITY,
)
from llm import _extract_json


def create_image_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set")
    return genai.Client(api_key=api_key)


def _extract_image_bytes(response: Any) -> bytes | None:
    """Pull the first inline image out of a generate_content response."""
    for candidate in response.candidates or []:
        if candidate.content is None:
            continue
        for part in candidate.content.parts or []:
            if part.inline_data is not None and part.inline_data.data:
                return part.inline_data.data
    return None


async def _image_call(
    client: genai.Client,
    contents: list[Any],
    aspect_ratio: str | None,
) -> bytes:
    """generate_content call returning image bytes, with retries."""
    config = types.GenerateContentConfig(
        response_modalities=["TEXT", "IMAGE"],
        image_config=types.ImageConfig(aspect_ratio=aspect_ratio)
        if aspect_ratio
        else None,
    )
    last_error: Exception | None = None
    for attempt in range(MAX_IMAGE_RETRIES):
        try:
            response = await client.aio.models.generate_content(
                model=IMAGE_MODEL_NAME, contents=contents, config=config
            )
            image = _extract_image_bytes(response)
            if image is None:
                raise ValueError("No image part in response")
            return image
        except Exception as e:
            last_error = e
            status = getattr(e, "code", None) or getattr(e, "status_code", None)
            if status == 400:
                break  # bad request / safety block, retrying won't help
            if attempt < MAX_IMAGE_RETRIES - 1:
                await asyncio.sleep(2 ** (attempt + 1))
    raise RuntimeError(
        f"Image call failed after {MAX_IMAGE_RETRIES} retries: {last_error}"
    )


async def generate_image(
    client: genai.Client, prompt: str, aspect_ratio: str
) -> bytes:
    """Generate an image from a text prompt. Returns raw image bytes."""
    return await _image_call(client, [prompt], aspect_ratio)


async def edit_image(
    client: genai.Client,
    image_bytes: bytes,
    mime_type: str,
    instruction: str,
) -> bytes:
    """Apply an edit instruction to an existing image. Returns raw image bytes."""
    part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
    # Aspect ratio is inherited from the input image; don't pass one.
    return await _image_call(client, [part, instruction], None)


def recompress(image_bytes: bytes) -> tuple[bytes, int, int]:
    """Re-encode model output (usually PNG) to WebP. Returns (bytes, w, h)."""
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=WEBP_QUALITY)
    return buf.getvalue(), img.width, img.height


async def vlm_validate(
    client: genai.Client,
    image_bytes: bytes,
    mime_type: str,
    prompt: str,
    tags: list[str],
    tag_attributes: dict[str, tuple[str, ...]],
    specificity_guidance: str,
    *,
    max_retries: int = 3,
) -> tuple[bool, list[str]]:
    """Check an image against its prompt and verify/correct its tags.

    Returns (accepted, corrected_tags).
    """
    vocab_lines = "\n".join(
        f"- {attr}: choose exactly one of {list(values)}"
        for attr, values in tag_attributes.items()
    )
    text = (
        "You are validating a generated image for a training dataset.\n\n"
        f"The image was generated from this prompt:\n{prompt}\n\n"
        f"Requirement: {specificity_guidance}\n\n"
        f"Candidate tags: {tags}\n"
        f"Tag vocabulary:\n{vocab_lines}\n\n"
        "Assess:\n"
        "1. matches_prompt: does the image actually show what the prompt "
        "describes?\n"
        "2. quality_ok: is it free of major artifacts (garbled text where text "
        "matters, deformed subjects, empty/blank output)?\n"
        "3. tags: the corrected tag list — keep candidate tags that are "
        "accurate, fix any that contradict the image, using only values from "
        "the vocabulary.\n\n"
        'Return JSON: {"matches_prompt": bool, "quality_ok": bool, '
        '"tags": [".."]}'
    )
    part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = await client.aio.models.generate_content(
                model=VALIDATION_MODEL_NAME, contents=[part, text]
            )
            result = _extract_json(response.text or "")
            if not isinstance(result, dict):
                raise ValueError(f"Expected JSON object, got: {type(result)}")
            accepted = bool(result.get("matches_prompt")) and bool(
                result.get("quality_ok")
            )
            corrected = [t for t in result.get("tags", tags) if isinstance(t, str)]
            return accepted, corrected or list(tags)
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** (attempt + 1))
    raise RuntimeError(
        f"Validation call failed after {max_retries} retries: {last_error}"
    )


if __name__ == "__main__":
    # One-shot live sanity check of generate -> edit -> validate.
    # Usage: python gemini_image.py [output_dir]
    import sys
    from pathlib import Path

    _out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else ".")

    async def _sanity() -> None:
        client = create_image_client()
        print("generating...")
        raw = await generate_image(
            client, "A photorealistic photo of a red bicycle leaning "
            "against a brick wall", "4:3"
        )
        webp, w, h = recompress(raw)
        print(f"base: {len(raw)} raw bytes -> {len(webp)} webp bytes, {w}x{h}")
        print("editing...")
        edited = await edit_image(
            client, webp, "image/webp", "Make it night time with a street lamp"
        )
        ew, eww, ehh = recompress(edited)
        print(f"edit: {len(ew)} webp bytes, {eww}x{ehh}")
        print("validating...")
        ok, tags = await vlm_validate(
            client,
            ew,
            "image/webp",
            "A red bicycle against a brick wall at night",
            ["time:night", "no_people"],
            {"time": ("time:day", "time:night"), "people": ("people", "no_people")},
            "The image must show the described scene.",
        )
        print("validation:", ok, tags)
        (_out_dir / "sanity_base.webp").write_bytes(webp)
        (_out_dir / "sanity_edit.webp").write_bytes(ew)
        print(json.dumps({"ok": True}))

    asyncio.run(_sanity())
