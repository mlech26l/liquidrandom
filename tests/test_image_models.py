import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from liquidrandom.models import ImageSample


SAMPLE_ROW: dict[str, Any] = {
    "image": b"\x00fakebytes",
    "image_format": "webp",
    "width": 1024,
    "height": 768,
    "aspect_ratio": "4:3",
    "category": "indoor_scene",
    "taxonomy_path": "Indoor Scenes > Kitchens",
    "caption": "A sunlit farmhouse kitchen.",
    "prompt": "A photorealistic sunlit farmhouse kitchen with copper pots",
    "edit_instruction": "",
    "tags": ["no_people", "lighting:natural"],
    "chain_id": "abc123def4567890",
    "turn_index": 0,
    "parent_turn": -1,
    "chain_length": 4,
}


def _sample(**overrides: Any) -> ImageSample:
    return ImageSample.from_dict({**SAMPLE_ROW, **overrides})


def test_from_dict_round_trip() -> None:
    s = _sample()
    assert s.image == b"\x00fakebytes"
    assert s.width == 1024
    assert s.tags == ["no_people", "lighting:natural"]
    assert s.category == "indoor_scene"
    assert s.mime_type == "image/webp"


def test_from_dict_none_lists_and_empty_edit() -> None:
    s = _sample(tags=None, edit_instruction=None)
    assert s.tags == []
    assert s.edit_instruction == ""


def test_is_base() -> None:
    assert _sample().is_base
    assert not _sample(turn_index=2, parent_turn=0).is_base


def test_to_str_contains_no_bytes() -> None:
    s = _sample()
    for text in (s.brief(), s.detailed(), str(s), repr(s)):
        assert "fakebytes" not in text
        assert len(text) < 1000
    assert "A sunlit farmhouse kitchen." in s.brief()
    assert "no_people" in s.brief()
    assert "Kitchens" in s.detailed()
    assert "base image" in s.detailed()


def test_to_str_edit_turn() -> None:
    s = _sample(turn_index=2, parent_turn=1, edit_instruction="Make it night")
    assert "edit 2 of 3" in s.detailed()
    assert "Make it night" in s.detailed()


def test_save(tmp_path: Path) -> None:
    out = _sample().save(tmp_path / "img.webp")
    assert out.read_bytes() == b"\x00fakebytes"


def test_to_pil_missing_pillow_hint() -> None:
    with patch.dict(sys.modules, {"PIL": None, "PIL.Image": None}):
        with pytest.raises(ImportError, match=r"liquidrandom\[image\]"):
            _sample().to_pil()


def test_to_pil_decodes() -> None:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 6), "red").save(buf, format="WEBP")
    s = _sample(image=buf.getvalue())
    img = s.to_pil()
    assert img.size == (8, 6)
