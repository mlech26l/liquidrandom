"""Row-group-lazy loader for image categories.

Image parquets are multi-GB (20k images x ~200 KB WebP), so they are never
materialized whole. Instead we keep an open ParquetFile handle per category
and read a single small row group (written with row_group_size=64 at
consolidation time) per sample. Tag and chain lookups use a metadata-only
read (every column except ``image``), which is a few MB even for a 5 GB file.
"""

from __future__ import annotations

import bisect
import random
from typing import Any, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from liquidrandom._registry import CATEGORIES, IMAGE_CATEGORIES, REPO_ID
from liquidrandom.models.image_sample import ImageSample

_parquet_files: dict[str, pq.ParquetFile] = {}
# Cumulative starting row index of each row group, for bisecting row -> group.
_rg_offsets: dict[str, list[int]] = {}
# All columns except `image`, for building tag/chain indices.
_meta_tables: dict[str, pa.Table] = {}
# Posting lists: tag -> sorted row indices / chain_id -> sorted row indices.
_tag_index: dict[str, dict[str, list[int]]] = {}
_chain_index: dict[str, dict[str, list[int]]] = {}
# Single-entry row-group cache per category; chains are written contiguously,
# so loading a chain usually touches one or two groups.
_rg_cache: dict[str, tuple[int, pa.Table]] = {}


def _ensure_open(name: str) -> pq.ParquetFile:
    if name not in IMAGE_CATEGORIES:
        raise ValueError(
            f"Not an image category: {name!r}. "
            f"Available: {', '.join(sorted(IMAGE_CATEGORIES))}"
        )
    if name not in _parquet_files:
        info = CATEGORIES[name]
        path = hf_hub_download(
            repo_id=REPO_ID, filename=info.filename, repo_type="dataset"
        )
        pf = pq.ParquetFile(path)
        offsets: list[int] = []
        total = 0
        for g in range(pf.metadata.num_row_groups):
            offsets.append(total)
            total += pf.metadata.row_group(g).num_rows
        _parquet_files[name] = pf
        _rg_offsets[name] = offsets
    return _parquet_files[name]


def _read_row(name: str, row_idx: int) -> dict[str, Any]:
    pf = _ensure_open(name)
    offsets = _rg_offsets[name]
    group = bisect.bisect_right(offsets, row_idx) - 1
    cached = _rg_cache.get(name)
    if cached is not None and cached[0] == group:
        table = cached[1]
    else:
        table = pf.read_row_group(group)
        _rg_cache[name] = (group, table)
    local = row_idx - offsets[group]
    row: dict[str, Any] = {
        col: table.column(col)[local].as_py() for col in table.column_names
    }
    row["category"] = name
    return row


def _load_row(name: str, row_idx: int) -> ImageSample:
    return ImageSample.from_dict(_read_row(name, row_idx))


def _num_rows(name: str) -> int:
    return _ensure_open(name).metadata.num_rows


def _meta_table(name: str) -> pa.Table:
    if name not in _meta_tables:
        pf = _ensure_open(name)
        columns = [c for c in pf.schema_arrow.names if c != "image"]
        _meta_tables[name] = pf.read(columns=columns)
    return _meta_tables[name]


def _get_tag_index(name: str) -> dict[str, list[int]]:
    if name not in _tag_index:
        index: dict[str, list[int]] = {}
        for i, tags in enumerate(_meta_table(name).column("tags").to_pylist()):
            for tag in tags or []:
                index.setdefault(tag, []).append(i)
        _tag_index[name] = index
    return _tag_index[name]


def _get_chain_index(name: str) -> dict[str, list[int]]:
    if name not in _chain_index:
        index: dict[str, list[int]] = {}
        for i, cid in enumerate(_meta_table(name).column("chain_id").to_pylist()):
            index.setdefault(cid, []).append(i)
        _chain_index[name] = index
    return _chain_index[name]


def _rows_matching(name: str, tags: Sequence[str]) -> list[int]:
    """Row indices whose tags contain ALL of the given tags."""
    index = _get_tag_index(name)
    postings: list[list[int]] = []
    for tag in tags:
        rows = index.get(tag)
        if not rows:
            known = ", ".join(sorted(index))
            raise ValueError(
                f"No samples in {name!r} with tag {tag!r} (known tags: {known})"
            )
        postings.append(rows)
    postings.sort(key=len)
    result = set(postings[0])
    for rows in postings[1:]:
        result &= set(rows)
    if not result:
        raise ValueError(f"No samples in {name!r} matching all tags: {list(tags)}")
    return sorted(result)


def load_image_random(name: str, tags: Sequence[str] | None = None) -> ImageSample:
    """Load a single random image sample, optionally filtered by tags (AND)."""
    if tags:
        row_idx = random.choice(_rows_matching(name, tags))
    else:
        row_idx = random.randint(0, _num_rows(name) - 1)
    return _load_row(name, row_idx)


def load_image_chain(name: str, chain_id: str) -> list[ImageSample]:
    """Load all images of a chain, sorted by turn_index."""
    rows = _get_chain_index(name).get(chain_id)
    if not rows:
        raise ValueError(f"No chain {chain_id!r} in {name!r}")
    samples = [_load_row(name, i) for i in rows]
    samples.sort(key=lambda s: s.turn_index)
    return samples


def load_random_chain(
    name: str, tags: Sequence[str] | None = None, min_length: int = 2
) -> list[ImageSample]:
    """Load a random full edit chain, optionally filtered by tags.

    With tags, a chain qualifies if any of its images matches all tags.
    """
    chain_index = _get_chain_index(name)
    if tags:
        chain_ids_col = _meta_table(name).column("chain_id")
        candidates = {chain_ids_col[i].as_py() for i in _rows_matching(name, tags)}
    else:
        candidates = set(chain_index)
    eligible = [c for c in candidates if len(chain_index[c]) >= min_length]
    if not eligible:
        raise ValueError(
            f"No chains in {name!r} with >= {min_length} images"
            + (f" matching tags {list(tags)}" if tags else "")
        )
    return load_image_chain(name, random.choice(eligible))
