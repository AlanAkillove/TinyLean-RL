#!/usr/bin/env python
"""Merge V2-B003 shard outputs into the main rollouts file (idempotent, strict).

Union of the main file and all shard files, keyed by (theorem_rank, replicate).
Overlapping keys MUST be byte-equal after JSON round-trip (same seed and
verified semantics); any conflict aborts without writing. Output is written
atomically, sorted by (rank, replicate), and a sha256 + count report is printed.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "experiments/results/v2_b003_rollouts.jsonl"
SHARD_DIR = ROOT / "experiments/results/shards"
SHARD_COUNT = 4
EXPECTED_TOTAL = 1536  # 192 theorems x 8 replicates
N_THEOREMS = 192
REPLICATES = 8


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    sources: list[tuple[str, list[dict]]] = [("main", read_jsonl(MAIN))]
    for index in range(SHARD_COUNT):
        path = SHARD_DIR / f"v2_b003_rollouts.shard{index}.jsonl"
        sources.append((f"shard{index}", read_jsonl(path)))

    by_key: dict[tuple[int, int], tuple[str, dict]] = {}
    conflicts = 0
    for name, records in sources:
        for record in records:
            key = (int(record["theorem_rank"]), int(record["replicate"]))
            if key in by_key:
                other_name, other = by_key[key]
                if json.dumps(other, sort_keys=True) != json.dumps(record, sort_keys=True):
                    print(f"[merge] CONFLICT key {key}: {other_name} vs {name} - aborting", file=sys.stderr)
                    conflicts += 1
                continue
            by_key[key] = (name, record)
    if conflicts:
        return 2

    merged = [by_key[key][1] for key in sorted(by_key)]
    body = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in merged)
    tmp = MAIN.with_suffix(".jsonl.tmp")
    tmp.write_text(body, encoding="utf-8")
    tmp.replace(MAIN)

    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    expected_keys = {(r, k) for r in range(1, N_THEOREMS + 1) for k in range(REPLICATES)}
    missing = sorted(expected_keys - set(by_key))
    print(
        f"[merge] {len(merged)}/{EXPECTED_TOTAL} records "
        f"({'sources: ' + ', '.join(f'{name}={len(recs)}' for name, recs in sources)}); "
        f"missing={len(missing)}; sha256={digest}"
    )
    if missing:
        print(f"[merge] missing keys (first 20): {missing[:20]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
