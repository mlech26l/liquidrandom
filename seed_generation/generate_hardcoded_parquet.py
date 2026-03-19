"""Generate parquet files from hardcoded seed data lists.

Usage:
    python generate_hardcoded_parquet.py [--output-dir output]

Creates reasoning_pattern.parquet and instruction_complexity.parquet from
the hardcoded Python lists in this directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from hardcoded_instruction_complexities import INSTRUCTION_COMPLEXITIES
from hardcoded_reasoning_patterns import REASONING_PATTERNS


def create_parquet(data: list[dict], output_path: Path) -> None:
    table = pa.Table.from_pylist(data)
    pq.write_table(table, str(output_path), compression="zstd")
    print(f"Wrote {len(data)} rows to {output_path}")


def main() -> None:
    output_dir = Path("output/parquet")
    if len(sys.argv) > 1:
        output_dir = Path(sys.argv[1])
    output_dir.mkdir(parents=True, exist_ok=True)

    create_parquet(REASONING_PATTERNS, output_dir / "reasoning_pattern.parquet")
    create_parquet(INSTRUCTION_COMPLEXITIES, output_dir / "instruction_complexity.parquet")

    print(f"\nDone. Files in {output_dir}/")


if __name__ == "__main__":
    main()
