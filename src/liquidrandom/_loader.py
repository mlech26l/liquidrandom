from __future__ import annotations

import json
from typing import Any

import logging

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import disable_progress_bars

disable_progress_bars()
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

from liquidrandom._registry import CATEGORIES, REPO_ID

_cache: dict[str, list[Any]] = {}


def load_category(name: str) -> list[Any]:
    """Load all samples for a category, fetching from HuggingFace if needed."""
    if name in _cache:
        return _cache[name]

    if name not in CATEGORIES:
        raise ValueError(
            f"Unknown category: {name!r}. "
            f"Available: {', '.join(sorted(CATEGORIES))}"
        )

    info = CATEGORIES[name]
    path = hf_hub_download(repo_id=REPO_ID, filename=info.filename, repo_type="dataset")

    table = pq.read_table(path)
    columns = table.column_names
    col_data = {col: table.column(col).to_pylist() for col in columns}

    samples: list[Any] = []
    for i in range(table.num_rows):
        row_dict: dict[str, Any] = {}
        for col in columns:
            value = col_data[col][i]
            # Parquet stores list fields as native lists, no decoding needed
            row_dict[col] = value
        samples.append(info.model_class.from_dict(row_dict))

    _cache[name] = samples
    return samples
