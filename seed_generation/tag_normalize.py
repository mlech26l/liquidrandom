"""Repair tags that drifted from a category's controlled vocabulary.

The VLM tag-verification step occasionally re-applies an attribute prefix to a
tag that already carries one (`setting:setting:institutional`) or prefixes a
bare tag (`people:no_people`). Those values are unmatchable by the package's
exact-match tag filter, so they are repaired at consolidation time — before the
data is frozen into Parquet.
"""

from __future__ import annotations

from image_categories import ImageCategoryConfig


class TagNormalizer:
    """Maps observed tags back onto one category's controlled vocabulary."""

    def __init__(self, config: ImageCategoryConfig) -> None:
        self.vocab = {t for values in config.tag_attributes.values() for t in values}
        # Bare values are only a safe fallback when they identify one tag.
        suffixes: dict[str, list[str]] = {}
        for tag in self.vocab:
            suffixes.setdefault(tag.rsplit(":", 1)[-1], []).append(tag)
        self.by_suffix = {s: t[0] for s, t in suffixes.items() if len(t) == 1}

    def normalize_tag(self, tag: str) -> str | None:
        """Return the vocabulary tag for `tag`, or None if it maps to nothing."""
        tag = tag.strip()
        if tag in self.vocab:
            return tag
        # Peel spurious leading attribute prefixes: "a:a:b" -> "a:b" -> "b".
        # Segments are stripped because the VLM sometimes writes "people: people".
        rest = tag
        while ":" in rest:
            rest = rest.split(":", 1)[1].strip()
            if rest in self.vocab:
                return rest
        return self.by_suffix.get(rest)

    def normalize(self, tags: list[str]) -> tuple[list[str], list[str]]:
        """Return (normalized tags, tags that could not be mapped).

        Order is preserved and duplicates introduced by normalization removed.
        """
        out: list[str] = []
        dropped: list[str] = []
        for tag in tags:
            fixed = self.normalize_tag(tag)
            if fixed is None:
                dropped.append(tag)
            elif fixed not in out:
                out.append(fixed)
        return out, dropped
