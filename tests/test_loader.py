import tempfile
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

import liquidrandom
from liquidrandom._loader import _cache, load_category
from liquidrandom.models import Persona


SAMPLE_PERSONA = {
    "name": "Test User",
    "age": 25,
    "gender": "non-binary",
    "occupation": "teacher",
    "nationality": "French",
    "personality_traits": ["kind"],
    "background": "Grew up in Paris",
}


def _write_parquet(path: str, rows: list[dict[str, object]]) -> None:
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def test_load_category_parses_parquet() -> None:
    _cache.clear()
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        tmp_path = f.name
    _write_parquet(tmp_path, [SAMPLE_PERSONA])

    with patch("liquidrandom._loader.hf_hub_download", return_value=tmp_path):
        result = load_category("persona")

    assert len(result) == 1
    assert isinstance(result[0], Persona)
    assert result[0].name == "Test User"
    _cache.clear()


def test_load_category_caches_in_memory() -> None:
    _cache.clear()
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        tmp_path = f.name
    _write_parquet(tmp_path, [SAMPLE_PERSONA])

    with patch("liquidrandom._loader.hf_hub_download", return_value=tmp_path) as mock_dl:
        load_category("persona")
        load_category("persona")
        assert mock_dl.call_count == 1
    _cache.clear()


def test_load_category_unknown_raises() -> None:
    try:
        load_category("nonexistent")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "nonexistent" in str(e)


def test_public_api_returns_model() -> None:
    _cache.clear()
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        tmp_path = f.name
    _write_parquet(tmp_path, [SAMPLE_PERSONA] * 5)

    with patch("liquidrandom._loader.hf_hub_download", return_value=tmp_path):
        p = liquidrandom.persona()
        assert isinstance(p, Persona)
        assert p.name == "Test User"
    _cache.clear()
