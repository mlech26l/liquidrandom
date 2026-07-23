from pathlib import Path
from typing import Any
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import liquidrandom
from liquidrandom import _image_loader
from liquidrandom.models import ImageSample

_SCHEMA = pa.schema(
    [
        pa.field("image", pa.binary()),
        pa.field("image_format", pa.string()),
        pa.field("width", pa.int64()),
        pa.field("height", pa.int64()),
        pa.field("aspect_ratio", pa.string()),
        pa.field("taxonomy_path", pa.string()),
        pa.field("caption", pa.string()),
        pa.field("prompt", pa.string()),
        pa.field("edit_instruction", pa.string()),
        pa.field("tags", pa.list_(pa.string())),
        pa.field("chain_id", pa.string()),
        pa.field("turn_index", pa.int64()),
        pa.field("parent_turn", pa.int64()),
        pa.field("chain_length", pa.int64()),
    ]
)


def _row(
    chain_id: str, turn: int, parent: int, length: int, tags: list[str]
) -> dict[str, Any]:
    return {
        "image": f"{chain_id}-{turn}".encode(),
        "image_format": "webp",
        "width": 1024,
        "height": 1024,
        "aspect_ratio": "1:1",
        "taxonomy_path": "Indoor Scenes > Kitchens",
        "caption": f"Image {turn} of chain {chain_id}.",
        "prompt": f"Prompt for chain {chain_id}",
        "edit_instruction": "" if turn == 0 else f"edit {turn}",
        "tags": tags,
        "chain_id": chain_id,
        "turn_index": turn,
        "parent_turn": parent,
        "chain_length": length,
    }


# Chain A: 3 images tagged people; chain B: 2 images tagged no_people;
# chain C: single base image. Written with row_group_size=2 so access
# spans multiple row groups.
_ROWS = [
    _row("chain_a", 0, -1, 3, ["people", "lighting:natural"]),
    _row("chain_a", 1, 0, 3, ["people", "lighting:dim"]),
    _row("chain_a", 2, 1, 3, ["people", "lighting:dim"]),
    _row("chain_b", 0, -1, 2, ["no_people", "lighting:natural"]),
    _row("chain_b", 1, 0, 2, ["no_people", "lighting:dim"]),
    _row("chain_c", 0, -1, 1, ["no_people", "lighting:natural"]),
]


@pytest.fixture()
def image_parquet(tmp_path: Path):
    parquet_path = tmp_path / "indoor_scene.parquet"
    table = pa.Table.from_pylist(_ROWS, schema=_SCHEMA)
    pq.write_table(table, parquet_path, row_group_size=2)

    _clear_caches()
    with patch(
        "liquidrandom._image_loader.hf_hub_download",
        return_value=str(parquet_path),
    ) as mock_dl:
        yield mock_dl
    _clear_caches()


def _clear_caches() -> None:
    _image_loader._parquet_files.clear()
    _image_loader._rg_offsets.clear()
    _image_loader._meta_tables.clear()
    _image_loader._tag_index.clear()
    _image_loader._chain_index.clear()
    _image_loader._rg_cache.clear()


def test_image_random(image_parquet) -> None:
    s = liquidrandom.image("indoor_scene")
    assert isinstance(s, ImageSample)
    assert s.category == "indoor_scene"
    assert s.image.startswith(b"chain_")


def test_multi_row_group_access(image_parquet) -> None:
    pf = _image_loader._ensure_open("indoor_scene")
    assert pf.metadata.num_row_groups > 1
    # Every row is reachable and matches its source data.
    for i, row in enumerate(_ROWS):
        loaded = _image_loader._load_row("indoor_scene", i)
        assert loaded.image == row["image"]
        assert loaded.turn_index == row["turn_index"]


def test_per_category_function(image_parquet) -> None:
    s = liquidrandom.indoor_scene()
    assert isinstance(s, ImageSample)


def test_tag_and_filter(image_parquet) -> None:
    for _ in range(5):
        s = liquidrandom.image("indoor_scene", tags=["people", "lighting:dim"])
        assert "people" in s.tags and "lighting:dim" in s.tags


def test_tag_no_match(image_parquet) -> None:
    with pytest.raises(ValueError, match="unknown-tag"):
        liquidrandom.image("indoor_scene", tags=["unknown-tag"])
    with pytest.raises(ValueError, match="matching all tags"):
        liquidrandom.image("indoor_scene", tags=["people", "no_people"])


def test_image_chain(image_parquet) -> None:
    chain = _image_loader.load_image_chain("indoor_scene", "chain_a")
    assert [s.turn_index for s in chain] == [0, 1, 2]
    assert [s.parent_turn for s in chain] == [-1, 0, 1]


def test_random_chain_min_length(image_parquet) -> None:
    for _ in range(5):
        chain = liquidrandom.image_chain("indoor_scene", min_length=3)
        assert len(chain) == 3
        assert chain[0].chain_id == "chain_a"


def test_random_chain_tag_filter(image_parquet) -> None:
    chain = liquidrandom.image_chain("indoor_scene", tags=["no_people"])
    assert chain[0].chain_id == "chain_b"


def test_image_chain_of(image_parquet) -> None:
    s = liquidrandom.image("indoor_scene", tags=["people"])
    chain = liquidrandom.image_chain_of(s)
    assert len(chain) == 3
    assert {c.chain_id for c in chain} == {s.chain_id}


def test_download_once(image_parquet) -> None:
    liquidrandom.image("indoor_scene")
    liquidrandom.image("indoor_scene")
    liquidrandom.image_chain("indoor_scene")
    assert image_parquet.call_count == 1


def test_eager_loader_rejects_image_category(image_parquet) -> None:
    with pytest.raises(ValueError, match="image category"):
        liquidrandom.load_category("indoor_scene")
    with pytest.raises(ValueError, match="image category"):
        liquidrandom.load_random("indoor_scene")


def test_non_image_category_rejected_by_image_loader(image_parquet) -> None:
    with pytest.raises(ValueError, match="Not an image category"):
        liquidrandom.image("persona")
