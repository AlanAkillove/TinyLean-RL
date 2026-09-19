#!/usr/bin/env python
"""Seed the V2-B002 shard files with the already-completed ranks (idempotent).

Each shard file must contain the records whose rank belongs to that shard so
that (a) each shard runner skips them via its own resume logic, and (b) the
final merge is a clean union with strict equality checks. Records are copied
byte-identically (JSON round-trip) from the main rollouts file.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "experiments/results/v2_b002_rollouts.jsonl"
SHARD_DIR = ROOT / "experiments/results/shards"
SHARD_COUNT = 4


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    main_recs = read_jsonl(MAIN)
    if not main_recs:
        print(f"[prepare] main file empty/absent: {MAIN}")
        return 0
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    for index in range(SHARD_COUNT):
        shard_path = SHARD_DIR / f"v2_b002_rollouts.shard{index}.jsonl"
        have = {int(r["theorem_rank"]) for r in read_jsonl(shard_path)}
        to_add = [
            r for r in main_recs
            if int(r["theorem_rank"]) % SHARD_COUNT == index and int(r["theorem_rank"]) not in have
        ]
        if to_add:
            with shard_path.open("a", encoding="utf-8") as handle:
                for record in to_add:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[prepare] shard{index}: +{len(to_add)} seeded (now {len(have) + len(to_add)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
