"""HTML review gallery for generated image seed data.

Renders chains as horizontal strips (base -> edits, with the edit instruction
under each edited image), plus caption, taxonomy path, and clickable tag chips
that filter the gallery. Used for the human quality gate before a full run.
"""

from __future__ import annotations

import base64
import html
import json
import math
import random
from pathlib import Path
from typing import Any

_CSS = """
body { font-family: system-ui, sans-serif; margin: 0; padding: 1rem 2rem;
       background: #111; color: #ddd; }
h1 { font-size: 1.3rem; }
.filters { position: sticky; top: 0; background: #111; padding: .5rem 0;
           border-bottom: 1px solid #333; z-index: 2; }
.chip { display: inline-block; margin: 2px; padding: 2px 10px; border-radius: 10px;
        background: #2a2a2a; border: 1px solid #444; cursor: pointer;
        font-size: .8rem; user-select: none; }
.chip.active { background: #2563eb; border-color: #2563eb; color: #fff; }
.chain { margin: 1.5rem 0; padding: 1rem; background: #1a1a1a; border-radius: 8px; }
.chain.hidden { display: none; }
.meta { font-size: .85rem; color: #999; margin-bottom: .5rem; }
.meta b { color: #ddd; }
.strip { display: flex; gap: 1rem; overflow-x: auto; align-items: flex-start; }
.frame { flex: 0 0 auto; width: 280px; }
.frame img { width: 100%; border-radius: 4px; display: block; }
.frame .label { font-size: .75rem; color: #aaa; margin-top: .3rem; }
.frame .edit { font-size: .75rem; color: #7dd3fc; margin-top: .2rem; }
.frame .tags { margin-top: .2rem; }
.tag { display: inline-block; margin: 1px; padding: 1px 6px; border-radius: 8px;
       background: #333; font-size: .7rem; }
"""

_JS = """
const active = new Set();
function toggle(chip) {
  const tag = chip.dataset.tag;
  if (active.has(tag)) { active.delete(tag); chip.classList.remove('active'); }
  else { active.add(tag); chip.classList.add('active'); }
  document.querySelectorAll('.chain').forEach(chain => {
    const tags = new Set(JSON.parse(chain.dataset.tags));
    const show = [...active].every(t => tags.has(t));
    chain.classList.toggle('hidden', !show);
  });
}
"""


def _iter_leaf_chains(jsonl: Path, limit: int) -> list[list[dict[str, Any]]]:
    """Read up to `limit` whole chains from one leaf file, stopping early.

    Chain rows are appended contiguously by the sampler, so a chain is complete
    as soon as a different chain_id appears.
    """
    chains: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    with open(jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if current and row["chain_id"] != current[0]["chain_id"]:
                chains.append(current)
                if len(chains) >= limit:
                    return chains
                current = []
            current.append(row)
    if current:
        chains.append(current)
    return chains[:limit]


def _load_chains(
    output_dir: str, category: str, max_images: int
) -> list[list[dict[str, Any]]]:
    """Load whole chains (sorted by turn) up to a total of max_images images.

    Chains are drawn from leaf files spread across the taxonomy rather than
    from whichever leaves sort first, so the gallery is representative. Files
    are read lazily and only as far as needed — a finished category holds
    gigabytes of base64 per leaf directory.
    """
    samples_dir = Path(output_dir) / "samples" / category
    files = sorted(samples_dir.glob("*.jsonl"))
    if not files:
        return []
    random.Random(0).shuffle(files)  # fixed seed: same gallery on re-render

    # Assume 4-image chains when deciding how many to pull from each leaf.
    per_leaf = max(1, math.ceil(max_images / 4 / len(files)))

    result: list[list[dict[str, Any]]] = []
    total = 0
    for jsonl in files:
        for rows in _iter_leaf_chains(jsonl, per_leaf):
            rows.sort(key=lambda r: r["turn_index"])
            if total + len(rows) > max_images and result:
                return result
            result.append(rows)
            total += len(rows)
    return result


def generate_review_html(
    output_dir: str, category: str, max_images: int = 500
) -> Path:
    chains = _load_chains(output_dir, category, max_images)
    review_dir = Path(output_dir) / "review" / category
    images_dir = review_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    all_tags: set[str] = set()
    for rows in chains:
        for row in rows:
            all_tags.update(row.get("tags") or [])

    parts: list[str] = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        f"<title>{html.escape(category)} review</title>",
        f"<style>{_CSS}</style></head><body>",
        f"<h1>{html.escape(category)} — {sum(len(c) for c in chains)} images "
        f"in {len(chains)} chains</h1>",
        "<div class='filters'>Filter by tag: ",
    ]
    for tag in sorted(all_tags):
        parts.append(
            f"<span class='chip' data-tag='{html.escape(tag)}' "
            f"onclick='toggle(this)'>{html.escape(tag)}</span>"
        )
    parts.append("</div>")

    for rows in chains:
        base = rows[0]
        chain_tags = sorted({t for r in rows for t in (r.get("tags") or [])})
        tags_json = html.escape(json.dumps(chain_tags), quote=True)
        parts.append(f"<div class='chain' data-tags='{tags_json}'>")
        parts.append(
            "<div class='meta'>"
            f"<b>{html.escape(base.get('caption', ''))}</b><br>"
            f"{html.escape(base.get('taxonomy_path', ''))} · "
            f"chain {html.escape(base.get('chain_id', ''))}</div>"
        )
        parts.append("<div class='strip'>")
        for row in rows:
            filename = f"{row['chain_id']}_{row['turn_index']}.webp"
            (images_dir / filename).write_bytes(
                base64.b64decode(row["image_base64"])
            )
            turn = row["turn_index"]
            label = (
                "base"
                if turn == 0
                else f"edit {turn} (from turn {row['parent_turn']})"
            )
            tag_chips = "".join(
                f"<span class='tag'>{html.escape(t)}</span>"
                for t in (row.get("tags") or [])
            )
            edit_line = (
                f"<div class='edit'>{html.escape(row.get('edit_instruction', ''))}"
                "</div>"
                if turn != 0
                else ""
            )
            parts.append(
                "<div class='frame'>"
                f"<img loading='lazy' src='images/{filename}'>"
                f"<div class='label'>{label} · {row['width']}x{row['height']} "
                f"({html.escape(row.get('aspect_ratio', ''))})</div>"
                f"{edit_line}"
                f"<div class='tags'>{tag_chips}</div>"
                "</div>"
            )
        parts.append("</div></div>")

    parts.append(f"<script>{_JS}</script></body></html>")

    index = review_dir / "index.html"
    index.write_text("\n".join(parts), encoding="utf-8")
    return index
