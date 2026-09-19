#!/usr/bin/env python
"""V2-B003 independent evaluation set: first 192 reserve-pool components.

B003 (multi-seed compute-response + Expected Oracle) is the gated successor of
V2-B002. Its evaluation set is frozen from the untouched reserve pool:

  - eval set = lexicographically first 192 components of
    v2_b002_reserve_pool.json (216), keeping 24 as buffer;
  - the buffer may NOT be promoted into the evaluation without a recorded
    protocol deviation;
  - nothing here may be generated, verified or trained on before the B003
    manifest is committed AND the B002 GO decision is recorded;
  - disjointness from V1 / V2-A / V2-B content holds by construction of the
    reserve pool (see v2_build_family_components.py).

Deterministic, fail-closed (reserve artifact pinned by sha256).
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RESERVE_REL = "experiments/manifests/v2/v2_b002_reserve_pool.json"
RESERVE_SHA256 = "2378688d0c1b36f1bb03ac2aded728da7e9bdf26927516c87bab067e7e07285a"
OUTPUT_REL = "experiments/manifests/v2/v2_b003_eval_set.json"
EXPECTED_RESERVE = 216
EXPECTED_EVAL = 192


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision() -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze the V2-B003 evaluation set.")
    parser.add_argument("--reserve", default=RESERVE_REL)
    parser.add_argument("--output", default=OUTPUT_REL)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    reserve_path = Path(args.reserve)
    if not reserve_path.is_absolute():
        reserve_path = ROOT / reserve_path
    actual_sha = sha256_file(reserve_path)
    if actual_sha != RESERVE_SHA256:
        print(f"[ERROR] reserve sha256 {actual_sha} != pinned {RESERVE_SHA256}", file=sys.stderr)
        return 2

    reserve = json.loads(reserve_path.read_text(encoding="utf-8"))
    components = reserve["reserve_components"]
    if len(components) != EXPECTED_RESERVE:
        print(f"[ERROR] reserve size {len(components)} != {EXPECTED_RESERVE}", file=sys.stderr)
        return 2
    if [c["component_id"] for c in components] != sorted(c["component_id"] for c in components):
        print("[ERROR] reserve is not sorted by component_id", file=sys.stderr)
        return 2

    chosen = components[:EXPECTED_EVAL]
    buffer = components[EXPECTED_EVAL:]
    if len(buffer) != EXPECTED_RESERVE - EXPECTED_EVAL:
        print("[ERROR] buffer size mismatch", file=sys.stderr)
        return 2

    def domain(component: dict) -> str:
        name = component["names_sample"][0]
        return name.split("_")[0]

    result = {
        "artifact_type": "v2_b003_eval_set",
        "experiment": "V2-B003",
        "status": "frozen",
        "purpose": (
            "family-clean independent evaluation set for the B003 multi-seed "
            "compute-response + Expected Oracle analysis"
        ),
        "rules": [
            "evaluation set = lexicographically first 192 reserve-pool components (component_id), keeping 24 as buffer",
            "the buffer may not be promoted into the evaluation without a recorded protocol deviation",
            "nothing in this set may be generated, verified or trained on before the B003 manifest is committed AND the B002 GO decision is recorded",
            "the set stays frozen through B003; any change requires an amendment before launch",
        ],
        "inputs": {
            "reserve_pool": {"path": str(reserve_path), "sha256": actual_sha},
        },
        "counts": {
            "reserve": len(components),
            "evaluation": len(chosen),
            "buffer": len(buffer),
            "statements_covered": sum(c["size"] for c in chosen),
        },
        "size_distribution": {
            str(k): v for k, v in sorted(collections.Counter(c["size"] for c in chosen).items())
        },
        "domain_distribution": dict(
            collections.Counter(domain(c) for c in chosen).most_common()
        ),
        "buffer_component_ids": [c["component_id"] for c in buffer],
        "evaluation_components": [
            {
                "component_id": c["component_id"],
                "size": c["size"],
                "representative_statement_id": c["representative_statement_id"],
                "names_sample": c["names_sample"],
            }
            for c in chosen
        ],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
        "script": "scripts/v2_b003_freeze_eval.py",
    }

    print(f"reserve {len(components)} -> eval {len(chosen)} + buffer {len(buffer)}")
    print(f"statements covered by eval: {result['counts']['statements_covered']}")
    print(f"size distribution: {result['size_distribution']}")
    print(f"domain distribution: {result['domain_distribution']}")

    if args.dry_run:
        print("[dry-run] nothing written")
        return 0
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"artifact: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
