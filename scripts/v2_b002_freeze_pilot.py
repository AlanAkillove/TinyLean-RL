#!/usr/bin/env python
"""V2-B002 pilot freeze: family-clean compute-response pilot selection.

Selects the V2-B002 pilot theorem set from the family-component registry
(output of scripts/v2_build_family_components.py) under the FROZEN rule:

  pilot = the first 512 eligible components in lexicographic component_id
  order; one deterministic representative per component (the member with the
  smallest sha256(statement_id)).

Eligible = the component touches none of V1-used / E023 holdout / A3-primary
(A001) / A-reserve / B1-audit-reserved / C-joint-holdout AND every member sits
in B-train/B-validation/B-test (family-level no-contact guarantee).

Pure metadata; deterministic; fail-closed on any registry drift. The output
artifact fixes the pilot BEFORE the V2-B002 manifest is written and committed;
the manifest records this artifact's sha256.
"""

from __future__ import annotations

import argparse
import collections
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from pyarrow import parquet

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from p3c_build_fixed_set import DEFAULT_DATASET
from v2_family_leakage_audit import sha256_file, sha256_hex

REGISTRY_REL = "experiments/manifests/v2/family_component_registry.json"
OUTPUT_REL = "experiments/manifests/v2/v2_b002_pilot_set.json"
EXPECTED_STATEMENTS = 7620
EXPECTED_COMPONENTS = 1706
PILOT_SIZE = 512


def git_revision() -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="V2-B002 pilot freeze (pure metadata).")
    parser.add_argument("--registry", default=REGISTRY_REL)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output", default=OUTPUT_REL)
    parser.add_argument("--pilot-size", type=int, default=PILOT_SIZE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    def resolve(path: str) -> Path:
        candidate = Path(path)
        return candidate if candidate.is_absolute() else ROOT / candidate

    registry_path = resolve(args.registry)
    registry_artifact = json.loads(registry_path.read_text(encoding="utf-8"))
    counts = registry_artifact["counts"]
    if counts["statements"] != EXPECTED_STATEMENTS or counts["components"] != EXPECTED_COMPONENTS:
        print("[ERROR] component registry counts drifted - refusing", file=sys.stderr)
        return 2

    components = registry_artifact["components"]
    eligible = sorted(
        (c for c in components if c.get("eligible_for_b2")), key=lambda c: c["component_id"]
    )
    if len(eligible) < args.pilot_size:
        print(f"[ERROR] only {len(eligible)} eligible components for pilot size {args.pilot_size}", file=sys.stderr)
        return 2
    selected = eligible[: args.pilot_size]

    # fail-closed: representative integrity and pairwise uniqueness
    for comp in selected:
        if comp["representative_statement_id"] not in comp["member_statement_ids"]:
            print(f"[ERROR] representative not a member in {comp['component_id']}", file=sys.stderr)
            return 2
    representatives = [comp["representative_statement_id"] for comp in selected]
    if len(set(representatives)) != len(representatives):
        print("[ERROR] duplicate representative statement id across components", file=sys.stderr)
        return 2

    # dataset re-check: representatives must exist and match the frozen dataset
    dataset_path = resolve(args.dataset)
    rows = parquet.read_table(dataset_path).to_pylist()
    unique_rows: dict[str, dict] = {}
    for row in rows:
        unique_rows.setdefault(row["statement_id"], row)
    if len(unique_rows) != EXPECTED_STATEMENTS:
        print(f"[ERROR] dataset has {len(unique_rows)} unique ids, expected {EXPECTED_STATEMENTS}", file=sys.stderr)
        return 2
    missing = [sid for sid in representatives if sid not in unique_rows]
    if missing:
        print(f"[ERROR] {len(missing)} representative ids missing from the dataset", file=sys.stderr)
        return 2

    theorems = []
    for rank, comp in enumerate(selected, start=1):
        sid = comp["representative_statement_id"]
        row = unique_rows[sid]
        formal = row["formal_statement"]
        natural_language = row.get("natural_language") or ""
        theorems.append(
            {
                "rank": rank,
                "component_id": comp["component_id"],
                "component_size": comp["size"],
                "statement_id": sid,
                "name": row["name"],
                "formal_statement": formal,
                "statement_sha256": sha256_hex(formal),
                "natural_language": natural_language,
                "natural_language_sha256": sha256_hex(natural_language),
            }
        )

    size_dist = dict(sorted(collections.Counter(t["component_size"] for t in theorems).items()))
    result = {
        "artifact_type": "v2_b002_pilot_set",
        "experiment": "V2-B002",
        "status": "frozen_pending_preregistration",
        "selection": {
            "registry_artifact": str(registry_path),
            "registry_sha256": sha256_file(registry_path),
            "rule": (
                "eligible components sorted lexicographically by component_id, first N; "
                "one deterministic representative per component = member with min sha256(statement_id)"
            ),
            "pilot_size": len(theorems),
            "eligible_components_total": len(eligible),
            "eligible_statements_total": sum(c["size"] for c in eligible),
            "selection_seed": None,
            "note": (
                "no random seed is used: both the component order (content-addressed component_id) "
                "and the representative (min sha256(statement_id)) are deterministic functions of the "
                "frozen registry"
            ),
        },
        "safety": {
            "family_level_no_contact": (
                "every selected component is eligible_for_b2 in the registry: no member touches "
                "V1-used / E023 holdout / A3-primary (A001) / A-reserve / B1-audit-reserved / "
                "C-joint-holdout, and all members sit in B-train/B-validation/B-test"
            ),
            "representatives_unique": True,
            "dataset_match": "all representatives resolved in the frozen Promptset parquet",
        },
        "theorem_count": len(theorems),
        "component_size_distribution": size_dist,
        "theorems": theorems,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
        "script": "scripts/v2_b002_freeze_pilot.py",
    }

    print(f"eligible components {len(eligible)} ({sum(c['size'] for c in eligible)} statements)")
    print(f"selected {len(theorems)} components, {sum(t['component_size'] for t in theorems)} statements")
    print(f"component-size distribution of the pilot: {size_dist}")
    print("name-family sample:", [t["name"] for t in theorems[:5]])
    by_prefix = collections.Counter(t["name"].split("_")[0] for t in theorems)
    print("name-prefix distribution:", dict(sorted(by_prefix.items())))

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
