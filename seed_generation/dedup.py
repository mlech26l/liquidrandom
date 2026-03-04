"""Fuzzy string matching deduplication using Jaccard similarity on token sets."""

from __future__ import annotations

import re
import string
from typing import Any, Callable


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> set[str]:
    """Word-level token set from normalized text."""
    return set(normalize(text).split())


def jaccard_similarity(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two token sets."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union


def is_duplicate(
    new_text: str, existing_text: str, threshold: float = 0.7
) -> bool:
    """Check if new_text is a near-duplicate of existing_text."""
    return jaccard_similarity(tokenize(new_text), tokenize(existing_text)) >= threshold


def dedup_batch(
    new_samples: list[Any],
    existing_samples: list[Any],
    to_str: Callable[[Any], str],
    threshold: float = 0.7,
) -> list[Any]:
    """Filter out near-duplicates from new_samples against existing_samples.

    Also deduplicates within the new batch itself.
    Returns the non-duplicate samples.
    """
    existing_tokens = [tokenize(to_str(s)) for s in existing_samples]
    accepted: list[Any] = []
    accepted_tokens: list[set[str]] = []

    for sample in new_samples:
        sample_text = to_str(sample)
        sample_tokens = tokenize(sample_text)

        # Check against existing samples
        is_dup = False
        for existing_tok in existing_tokens:
            if jaccard_similarity(sample_tokens, existing_tok) >= threshold:
                is_dup = True
                break

        if is_dup:
            continue

        # Check against already-accepted samples in this batch
        for accepted_tok in accepted_tokens:
            if jaccard_similarity(sample_tokens, accepted_tok) >= threshold:
                is_dup = True
                break

        if not is_dup:
            accepted.append(sample)
            accepted_tokens.append(sample_tokens)

    return accepted
