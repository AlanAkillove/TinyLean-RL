#!/usr/bin/env python
"""Merge V2-B002 shard outputs into the main rollouts file (idempotent, strict).

Union of the main file and all shard files, keyed by theorem_rank. Overlapping
ranks MUST be byte-equal after JSON round-trip (they were produced with the
same seed and verified semantics); any conflict aborts without writing. Output
is written atomically, sorted by rank, and a sha256 + count report is printed.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "experiments/results/v2_b002_rollouts.jsonl"
SHARD_DIR = ROOT / "experiments/results/shards"
SHARD_COUNT = 4
EXPECTED_TOTAL = 512


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    sources: list[tuple[str, list[dict]]] = [("main", read_jsonl(MAIN))]
    for index in range(SHARD_COUNT):
        path = SHARD_DIR / f"v2_b002_rollouts.shard{index}.jsonl"
        sources.append((f"shard{index}", read_jsonl(path)))

    by_rank: dict[int, tuple[str, dict]] = {}
    conflicts = 0
    for name, records in sources:
        for record in records:
            rank = int(record["theorem_rank"])
            if rank in by_rank:
                other_name, other = by_rank[rank]
                if json.dumps(other, sort_keys=True) != json.dumps(record, sort_keys=True):
                    print(f"[merge] CONFLICT rank {rank}: {other_name} vs {name} - aborting", file=sys.stderr)
                    conflicts += 1
                continue
            by_rank[rank] = (name, record)
    if conflicts:
        return 2

    merged = [by_rank[rank][1] for rank in sorted(by_rank)]
    body = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in merged)
    tmp = MAIN.with_suffix(".jsonl.tmp")
    tmp.write_text(body, encoding="utf-8")
    tmp.replace(MAIN)

    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    missing = sorted(set(range(1, EXPECTED_TOTAL + 1)) - set(by_rank))
    print(
        f"[merge] {len(merged)}/{EXPECTED_TOTAL} records "
        f"({'sources: ' + ', '.join(f'{name}={len(recs)}' for name, recs in sources)}); "
        f"missing={len(missing)}; sha256={digest}"
    )
    if missing:
        print(f"[merge] missing ranks (first 20): {missing[:20]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
