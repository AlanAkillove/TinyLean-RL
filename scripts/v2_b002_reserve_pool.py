#!/usr/bin/env python
"""V2-B002 untouched reserve pool: the frozen source for B003 evaluation.

The family-component registry has 728 B2-eligible components; the V2-B002 pilot
uses the first 512 (lexicographic component_id; verified unbiased - the id is
content-addressed, chi2 pilot-vs-rest p=0.605). This script freezes the
REMAINING 216 components as the untouched reserve pool:

  - the 512 pilot components may be used as predictor training/development data
    after B002 completes;
  - the B003 Expected-Oracle / allocator independent evaluation must be drawn
    ONLY from this reserve pool (target 160-192 components, keeping the
    remainder as buffer);
  - nothing in this pool may be generated, verified or trained on before the
    B003 manifest freezes its selection.

Deterministic, fail-closed (pilot/reserve must partition the eligible set).
"""

from __future__ import annotations

import argparse
import collections
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REGISTRY_REL = "experiments/manifests/v2/family_component_registry.json"
PILOT_REL = "experiments/manifests/v2/v2_b002_pilot_set.json"
OUTPUT_REL = "experiments/manifests/v2/v2_b002_reserve_pool.json"
EXPECTED_ELIGIBLE = 728
EXPECTED_PILOT = 512
EXPECTED_COMPONENTS = 1706


def sha256_file(path: Path) -> str:
    import hashlib

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
    parser = argparse.ArgumentParser(description="Freeze the V2-B002 untouched reserve pool.")
    parser.add_argument("--registry", default=REGISTRY_REL)
    parser.add_argument("--pilot", default=PILOT_REL)
    parser.add_argument("--output", default=OUTPUT_REL)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    def resolve(path: str) -> Path:
        candidate = Path(path)
        return candidate if candidate.is_absolute() else ROOT / candidate

    registry_path = resolve(args.registry)
    pilot_path = resolve(args.pilot)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))

    if registry["counts"]["components"] != EXPECTED_COMPONENTS:
        print("[ERROR] registry component count drifted", file=sys.stderr)
        return 2
    eligible = sorted(
        (c for c in registry["components"] if c.get("eligible_for_b2")),
        key=lambda c: c["component_id"],
    )
    if len(eligible) != EXPECTED_ELIGIBLE:
        print(f"[ERROR] eligible components {len(eligible)}, expected {EXPECTED_ELIGIBLE}", file=sys.stderr)
        return 2

    pilot_ids = {t["component_id"] for t in pilot["theorems"]}
    if len(pilot_ids) != EXPECTED_PILOT:
        print(f"[ERROR] pilot component ids {len(pilot_ids)}, expected {EXPECTED_PILOT}", file=sys.stderr)
        return 2
    eligible_ids = {c["component_id"] for c in eligible}
    if not pilot_ids <= eligible_ids:
        print("[ERROR] pilot components are not a subset of eligible components", file=sys.stderr)
        return 2

    reserve = [c for c in eligible if c["component_id"] not in pilot_ids]
    if len(reserve) != EXPECTED_ELIGIBLE - EXPECTED_PILOT:
        print("[ERROR] reserve size mismatch", file=sys.stderr)
        return 2
    if pilot_ids & {c["component_id"] for c in reserve}:
        print("[ERROR] pilot/reserve overlap - refusing", file=sys.stderr)
        return 2

    result = {
        "artifact_type": "v2_b002_reserve_pool",
        "experiment": "V2-B002",
        "status": "frozen",
        "purpose": (
            "untouched family-clean component pool reserved for the V2-B003 "
            "Expected-Oracle / allocator independent evaluation"
        ),
        "rules": [
            "the 512 V2-B002 pilot components may become predictor training/development data after B002 completes",
            "the B003 independent evaluation must be drawn ONLY from this reserve pool (target 160-192 components, remainder kept as buffer)",
            "no component in this pool may be generated, verified or trained on before the B003 manifest freezes its selection",
            "any deviation is a protocol violation and must be recorded, not silently re-drawn",
        ],
        "inputs": {
            "registry": {"path": str(registry_path), "sha256": sha256_file(registry_path)},
            "pilot_set": {"path": str(pilot_path), "sha256": sha256_file(pilot_path)},
        },
        "counts": {
            "eligible": len(eligible),
            "pilot": len(pilot_ids),
            "reserve": len(reserve),
            "b003_target": "160-192 components",
            "b003_buffer": len(reserve) - 192,
        },
        "size_distribution": {
            str(k): v for k, v in sorted(collections.Counter(c["size"] for c in reserve).items())
        },
        "reserve_components": [
            {
                "component_id": c["component_id"],
                "size": c["size"],
                "representative_statement_id": c["representative_statement_id"],
                "names_sample": c["names"][:3],
                "sources": c["sources"],
            }
            for c in reserve
        ],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
        "script": "scripts/v2_b002_reserve_pool.py",
    }

    print(f"eligible {len(eligible)} = pilot {len(pilot_ids)} + reserve {len(reserve)}")
    print(f"reserve size distribution: {result['size_distribution']}")
    print(f"b003 capacity check: reserve {len(reserve)} >= target 192 + buffer {len(reserve)-192}")

    if args.dry_run:
        print("[dry-run] nothing written")
        return 0
    output_path = resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"artifact: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
