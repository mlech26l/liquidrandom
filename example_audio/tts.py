"""Text-to-speech using Replicate's chatterbox-turbo API via HTTP."""

from __future__ import annotations

import os
import random
import time

import httpx

TTS_MODEL = "resemble-ai/chatterbox-turbo"
REPLICATE_API_URL = f"https://api.replicate.com/v1/models/{TTS_MODEL}/predictions"

VOICES = [
    "Aaron", "Abigail", "Anaya", "Andy", "Archer",
    "Brian", "Chloe", "Dylan", "Emmanuel", "Ethan",
    "Evelyn", "Gavin", "Gordon", "Ivan", "Laura",
    "Lucy", "Madison", "Marisol", "Meera", "Walter",
]


def _get_api_token() -> str:
    token = os.environ.get("REPLICATE_API_KEY")
    if not token:
        raise RuntimeError("REPLICATE_API_KEY environment variable is not set")
    return token


def pick_voice() -> str:
    return random.choice(VOICES)


def synthesize_speech(
    text: str,
    voice: str,
    *,
    max_retries: int = 6,
) -> bytes:
    """Synthesize speech from text, returning WAV bytes.

    Uses exponential backoff on rate limit / server errors.
    """
    token = _get_api_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Prefer": "wait",
    }

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            # Create prediction (synchronous with Prefer: wait)
            with httpx.Client(timeout=120) as client:
                resp = client.post(
                    REPLICATE_API_URL,
                    headers=headers,
                    json={
                        "input": {
                            "text": text,
                            "voice": voice,
                        },
                    },
                )

                if resp.status_code == 429 or resp.status_code >= 500:
                    raise _RetryableError(f"HTTP {resp.status_code}: {resp.text[:200]}")

                resp.raise_for_status()
                data = resp.json()

                status = data.get("status")
                if status == "failed":
                    error = data.get("error", "unknown error")
                    raise RuntimeError(f"Prediction failed: {error}")

                # Poll if not yet completed
                if status != "succeeded":
                    data = _poll_prediction(client, headers, data["urls"]["get"])

                output_url = data.get("output")
                if not output_url:
                    raise RuntimeError(f"No output URL in response: {data}")

                # Download the audio file
                audio_resp = client.get(output_url)
                audio_resp.raise_for_status()
                return audio_resp.content

        except _RetryableError as e:
            last_error = e
            wait = min(2 ** (attempt + 1) + random.uniform(0, 1), 120)
            time.sleep(wait)

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                last_error = e
                wait = min(2 ** (attempt + 1) + random.uniform(0, 1), 120)
                time.sleep(wait)
            else:
                raise

    raise RuntimeError(
        f"TTS call failed after {max_retries} retries: {last_error}"
    )


def _poll_prediction(
    client: httpx.Client,
    headers: dict[str, str],
    url: str,
    poll_interval: float = 1.0,
    max_wait: float = 120.0,
) -> dict:
    """Poll a prediction until it completes."""
    elapsed = 0.0
    while elapsed < max_wait:
        time.sleep(poll_interval)
        elapsed += poll_interval
        resp = client.get(url, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status")
        if status == "succeeded":
            return data
        if status in ("failed", "canceled"):
            raise RuntimeError(f"Prediction {status}: {data.get('error', 'unknown')}")
    raise RuntimeError(f"Prediction timed out after {max_wait}s")


class _RetryableError(Exception):
    pass
