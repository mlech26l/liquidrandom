from __future__ import annotations

from enum import Enum


class DetailLevel(Enum):
    """Controls which fields are included when rendering a seed data model as a string.

    HIGH_LEVEL: Only broad, summary-level fields.
    DETAILED: High-level fields plus specific, descriptive fields.
    """

    HIGH_LEVEL = "high_level"
    DETAILED = "detailed"
