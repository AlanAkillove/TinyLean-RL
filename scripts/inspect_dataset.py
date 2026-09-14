#!/usr/bin/env python3
"""Inspect a local Parquet dataset without changing its contents."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("Install data tooling with: uv sync --extra data") from exc

    path = args.parquet if args.parquet.is_absolute() else ROOT / args.parquet
    if path.is_dir():
        parquet_files = sorted(path.rglob("*.parquet"))
        if len(parquet_files) != 1:
            raise SystemExit(f"Expected exactly one parquet file under {path}, found {len(parquet_files)}")
        path = parquet_files[0]
    table = pq.read_table(path, columns=None)
    summary = {
        "artifact_type": "dataset_inspection",
        "path": str(path.resolve()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": table.num_rows,
        "columns": table.column_names,
        "schema": {field.name: str(field.type) for field in table.schema},
        "null_counts": {
            name: table[name].null_count for name in table.column_names
        },
    }
    if "statement_id" in table.column_names:
        ids = table["statement_id"].to_pylist()
        summary["unique_statement_ids"] = len(set(ids))
        summary["duplicate_statement_ids"] = len(ids) - len(set(ids))
    output = args.output or (ROOT / "experiments" / "results" / "dataset_inspection.json")
    output = output if output.is_absolute() else ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print(f"Output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
