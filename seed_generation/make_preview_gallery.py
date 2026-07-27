"""Build the sample gallery under `preview/` from the generated Parquet files.

Picks a handful of representative images per category — one full edit chain
plus a few base images from different taxonomy branches — downscales them to
small JPEGs and writes a single Markdown page. Selection is seeded per
category, so re-running reproduces the same gallery.

Usage:
    python make_preview_gallery.py [--parquet-dir DIR] [--repo-root DIR]
"""

from __future__ import annotations

import argparse
import io
import random
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from PIL import Image

from image_categories import IMAGE_CATEGORY_CONFIGS

# Kept deliberately small: these images live in git history forever.
THUMB_MAX_EDGE = 384
JPEG_QUALITY = 72
BASES_PER_CATEGORY = 4

META_COLUMNS = [
    "taxonomy_path",
    "caption",
    "tags",
    "chain_id",
    "turn_index",
    "parent_turn",
    "chain_length",
    "edit_instruction",
]


def _row_group_of(offsets: list[int], row: int) -> tuple[int, int]:
    """Return (row group index, offset within it) for a global row index."""
    import bisect

    g = bisect.bisect_right(offsets, row) - 1
    return g, row - offsets[g]


class CategoryReader:
    """Reads single rows out of one category Parquet without loading it."""

    def __init__(self, path: Path) -> None:
        self.pf = pq.ParquetFile(path)
        self.offsets = [0]
        for i in range(self.pf.metadata.num_row_groups):
            self.offsets.append(
                self.offsets[-1] + self.pf.metadata.row_group(i).num_rows
            )
        self.meta = self.pf.read(columns=META_COLUMNS)

    def image_bytes(self, row: int) -> bytes:
        g, offset = _row_group_of(self.offsets, row)
        return self.pf.read_row_group(g, columns=["image"]).column("image")[
            offset
        ].as_py()

    def meta_row(self, row: int) -> dict[str, Any]:
        return {c: self.meta.column(c)[row].as_py() for c in META_COLUMNS}


def _pick_chain(reader: CategoryReader, rng: random.Random) -> list[int]:
    """A full-length chain whose edits are all distinct."""
    lengths = reader.meta.column("chain_length").to_pylist()
    turns = reader.meta.column("turn_index").to_pylist()
    starts = [i for i, (t, n) in enumerate(zip(turns, lengths)) if t == 0 and n == 4]
    rng.shuffle(starts)
    for start in starts:
        rows = list(range(start, start + 4))
        edits = [reader.meta.column("edit_instruction")[r].as_py() for r in rows]
        if len(set(edits[1:])) == 3:
            return rows
    return list(range(starts[0], starts[0] + 4))


def _pick_bases(
    reader: CategoryReader, rng: random.Random, exclude: set[int], count: int
) -> list[int]:
    """Base images from distinct taxonomy branches with distinct tag sets."""
    turns = reader.meta.column("turn_index").to_pylist()
    paths = reader.meta.column("taxonomy_path").to_pylist()
    candidates = [i for i, t in enumerate(turns) if t == 0 and i not in exclude]
    rng.shuffle(candidates)

    picked: list[int] = []
    seen_branch: set[str] = set()
    seen_tags: set[tuple[str, ...]] = set()
    for i in candidates:
        branch = paths[i].split(" > ")[1] if " > " in paths[i] else paths[i]
        tags = tuple(sorted(reader.meta.column("tags")[i].as_py() or []))
        if branch in seen_branch or tags in seen_tags:
            continue
        seen_branch.add(branch)
        seen_tags.add(tags)
        picked.append(i)
        if len(picked) == count:
            break
    return picked


def _write_thumb(data: bytes, dest: Path) -> int:
    img = Image.open(io.BytesIO(data))
    img.load()
    img = img.convert("RGB")
    img.thumbnail((THUMB_MAX_EDGE, THUMB_MAX_EDGE), Image.LANCZOS)
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest, "JPEG", quality=JPEG_QUALITY, optimize=True, progressive=True)
    return dest.stat().st_size


def _tag_code(tags: list[str]) -> str:
    return " ".join(f"`{t}`" for t in tags) if tags else "—"


def build(parquet_dir: Path, repo_root: Path) -> None:
    out_dir = repo_root / "preview"
    images_dir = out_dir / "images"
    total_bytes = 0
    sections: list[str] = []
    grand_total_rows = 0

    for cat, cfg in IMAGE_CATEGORY_CONFIGS.items():
        path = parquet_dir / f"{cat}.parquet"
        if not path.exists():
            print(f"skipping {cat}: {path} not found")
            continue
        reader = CategoryReader(path)
        n_rows = reader.pf.metadata.num_rows
        grand_total_rows += n_rows
        rng = random.Random(cat)

        chain_rows = _pick_chain(reader, rng)
        base_rows = _pick_bases(reader, rng, set(chain_rows), BASES_PER_CATEGORY)

        base_cells: list[str] = []
        base_notes: list[str] = []
        for row in base_rows:
            meta = reader.meta_row(row)
            name = f"{cat}/base_{meta['chain_id']}.jpg"
            total_bytes += _write_thumb(reader.image_bytes(row), images_dir / name)
            base_cells.append(f'<img src="images/{name}" width="200">')
            base_notes.append(f"{meta['caption']}<br>{_tag_code(meta['tags'])}")

        chain_cells: list[str] = []
        chain_notes: list[str] = []
        for row in chain_rows:
            meta = reader.meta_row(row)
            name = f"{cat}/chain_{meta['chain_id']}_t{meta['turn_index']}.jpg"
            total_bytes += _write_thumb(reader.image_bytes(row), images_dir / name)
            chain_cells.append(f'<img src="images/{name}" width="200">')
            label = (
                "**base image**"
                if meta["turn_index"] == 0
                else f"*edit {meta['turn_index']} of turn {meta['parent_turn']}*<br>"
                f"{meta['edit_instruction']}"
            )
            chain_notes.append(label)

        vocab = ", ".join(f"`{v}`" for vs in cfg.tag_attributes.values() for v in vs)
        sections.append(
            f"""## {cfg.display_name}

`liquidrandom.{cat}()` — {n_rows:,} images.
Tags: {vocab}

| | | | |
|---|---|---|---|
| {" | ".join(base_cells)} |
| {" | ".join(base_notes)} |

An edit chain — one base image and three edits, each applied to the base or to
an earlier turn:

| | | | |
|---|---|---|---|
| {" | ".join(chain_cells)} |
| {" | ".join(chain_notes)} |
"""
        )
        print(f"{cat}: {len(base_rows) + len(chain_rows)} images")

    page = f"""# Image Sample Gallery

A small, downscaled sample of the image seed data — {len(sections)} categories,
{grand_total_rows:,} images in total on
[HuggingFace](https://huggingface.co/datasets/mlech26l/liquidrandom-data).

These previews are {THUMB_MAX_EDGE}px JPEGs so the repository stays small. The
real images are ~1K WebP at their native aspect ratio; fetch them with
`liquidrandom.image(category, tags)`. Every image below is reachable from the
package — the filenames carry the `chain_id` it belongs to.

```python
import liquidrandom

img = liquidrandom.indoor_scene(tags=["no_people"])
chain = liquidrandom.image_chain_of(img)   # the base image and its edits
```

Regenerate this page with `python seed_generation/make_preview_gallery.py`.

{"".join(f"{s}\n" for s in sections)}"""

    (out_dir / "README.md").write_text(page, encoding="utf-8")
    print(f"\nwrote {out_dir / 'README.md'}")
    print(f"images: {total_bytes / 1e6:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parquet-dir",
        default="../image_output/output/parquet",
        help="Directory holding the per-category Parquet files",
    )
    parser.add_argument("--repo-root", default="..")
    args = parser.parse_args()
    build(Path(args.parquet_dir).resolve(), Path(args.repo_root).resolve())
