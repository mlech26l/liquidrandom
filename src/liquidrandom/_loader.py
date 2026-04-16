from __future__ import annotations

import random
from typing import Any

import logging

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import disable_progress_bars

disable_progress_bars()
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

from liquidrandom._registry import CATEGORIES, REPO_ID

_cache: dict[str, list[Any]] = {}

# Categories where converting all rows to Python objects is too expensive
# (e.g. tool_group: ~6700 rows × nested JSON → ~700 MB of Python dicts).
# These keep the Arrow table cached (~24 MB) and only parse one row per call.
_LAZY_CATEGORIES: set[str] = {"tool_group"}

_lazy_tables: dict[str, pq.lib.Table] = {}

# Per-category, per-(column,value) index of row indices — populated on first
# filtered-sample call. Shape: {category: {(column, value): [row_idx, ...]}}
_lazy_filter_index: dict[str, dict[tuple[str, Any], list[int]]] = {}


def _ensure_downloaded(name: str) -> str:
    """Download the Parquet file if needed and return the local path."""
    info = CATEGORIES[name]
    return hf_hub_download(repo_id=REPO_ID, filename=info.filename, repo_type="dataset")


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
    path = _ensure_downloaded(name)

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


def load_random(name: str) -> Any:
    """Load a single random sample from a category.

    For lazy categories (tool_group), keeps the Arrow table in memory and
    only parses the selected row into a Python object. Other categories
    use the full-cache path.
    """
    if name not in CATEGORIES:
        raise ValueError(
            f"Unknown category: {name!r}. "
            f"Available: {', '.join(sorted(CATEGORIES))}"
        )

    if name not in _LAZY_CATEGORIES:
        return random.choice(load_category(name))

    if name not in _lazy_tables:
        path = _ensure_downloaded(name)
        _lazy_tables[name] = pq.read_table(path)

    table = _lazy_tables[name]
    row_idx = random.randint(0, table.num_rows - 1)
    return _parse_lazy_row(name, table, row_idx)


def _parse_lazy_row(name: str, table: pq.lib.Table, row_idx: int) -> Any:
    info = CATEGORIES[name]
    row_dict: dict[str, Any] = {}
    for col in table.column_names:
        row_dict[col] = table.column(col)[row_idx].as_py()
    return info.model_class.from_dict(row_dict)


def load_random_where(name: str, column: str, value: Any) -> Any:
    """Load a single random sample filtered by exact column value.

    Uses the lazy path for categories in _LAZY_CATEGORIES (builds and caches
    a row-index list on first call for each (column, value) pair). For eager
    categories, filters the already-materialized list.
    """
    if name not in CATEGORIES:
        raise ValueError(
            f"Unknown category: {name!r}. "
            f"Available: {', '.join(sorted(CATEGORIES))}"
        )

    if name not in _LAZY_CATEGORIES:
        matches = [s for s in load_category(name) if getattr(s, column, None) == value]
        if not matches:
            raise ValueError(
                f"No samples in {name!r} where {column}={value!r}"
            )
        return random.choice(matches)

    if name not in _lazy_tables:
        path = _ensure_downloaded(name)
        _lazy_tables[name] = pq.read_table(path)
    table = _lazy_tables[name]

    cat_index = _lazy_filter_index.setdefault(name, {})
    key = (column, value)
    if key not in cat_index:
        if column not in table.column_names:
            raise ValueError(
                f"Column {column!r} not found in {name!r} "
                f"(available: {', '.join(table.column_names)})"
            )
        col_values = table.column(column).to_pylist()
        cat_index[key] = [i for i, v in enumerate(col_values) if v == value]
    indices = cat_index[key]
    if not indices:
        raise ValueError(
            f"No samples in {name!r} where {column}={value!r}"
        )

    row_idx = random.choice(indices)
    return _parse_lazy_row(name, table, row_idx)
