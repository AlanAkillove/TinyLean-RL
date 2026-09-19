#!/usr/bin/env python
"""V2 family-component registry builder (Track B, 2026-09-20).

Builds the family-component registry that V2-B002 (and the future family-granular
registry amendment for the Track C holdout) must select theorems from. A component
is a connected component of the UNION of three relations over the 7,620 unique
Promptset statements:

  L3_name_family  theorem name minus the trailing ``_v<digits>`` variant suffix
                  (source-problem family; primary key of the leakage audit)
  L2_skeleton     L1-strong normalized statement with the first declaration name
                  abstracted: ``(theorem|lemma|example) <name>`` -> ``\\1 _``
  L4_nl           L1-strong normalization of a NON-EMPTY ``natural_language``

All three relations are used together (union of edges), exactly as frozen for the
Track C holdout in docs/v2/family_leakage_audit.md section 5.3. Pure metadata: no
GPU, no Lean, no training; deterministic and fail-closed on any input drift
(recomputes the frozen registry hash, the V1-used union size 763, the E023
holdout size 128, the pool 6729 and the A001 set hash exactly like
scripts/v2_family_leakage_audit.py).

B2 eligibility is decided per COMPONENT (fail-closed): a component is eligible
for the B2 development pool only if every member sits in B-train / B-validation /
B-test and the component touches none of: V1-used, E023 holdout, A3-primary (the
A001 selection set), A-reserve, C-joint-holdout (Track C reservation) or the
B1-audit-reserved statements. This delivers the family-level no-contact
guarantee, not just statement-level isolation.

Artifact (git-tracked, metadata only):
  experiments/manifests/v2/family_component_registry.json
Each component entry carries: component_id (content-addressed), member statement
ids, names/sources, role counts, exclusion reasons, B2 eligibility and the
deterministic representative (min sha256(statement_id) over members) that the
V2-B002 pilot freeze uses.

Read-only over frozen inputs: no set, role or rule is modified.
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

from p3c_build_fixed_set import DEFAULT_DATASET, dump_formal_statements, normalize
from v2_family_leakage_audit import family_key, sha256_file, sha256_hex, skeleton_text, strong_text

ROLE_LABELS = (
    "B-train",
    "B-validation",
    "B-test",
    "A-selection",
    "C-joint-holdout",
    "B1-audit-reserved",
)
B_POOL_ROLES = ("B-train", "B-validation", "B-test")
EXPECTED = {
    "unique_ids": 7620,
    "v1_used": 763,
    "holdout": 128,
    "pool": 6729,
    "a001": 512,
    "b1_reserved": 16,
}
REGISTRY_REL = "experiments/manifests/v2/theorem_role_registry.json"
A001_SET_REL = "experiments/manifests/v2/v2_a001_selection_set.json"
B1_RESERVED_REL = "experiments/manifests/v2/b1_reserved_statement_ids.json"
OUTPUT_REL = "experiments/manifests/v2/family_component_registry.json"
GIANT_WARNING_THRESHOLD = 76  # > 1% of the 7,620 statements


def git_revision() -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()


class UnionFind:
    """Deterministic union-find (union by rank, path compression on find)."""

    def __init__(self, ids):
        self.parent = {i: i for i in ids}
        self.rank = {i: 0 for i in ids}

    def find(self, x: str) -> str:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def relation_groups(ids, key_of) -> dict[str, list[str]]:
    """Group ids by a non-empty relation key; only multi-id groups are returned."""

    groups: dict[str, list[str]] = {}
    for identifier in sorted(ids):
        key = key_of(identifier)
        if key:
            groups.setdefault(key, []).append(identifier)
    return {key: members for key, members in groups.items() if len(members) > 1}


def build_components(ids, relations):
    """Connected components over the union of (layer_name, key_of) relations.

    Relations are applied in the given order; per-layer merge counts are the
    EFFECTIVE new unions contributed at that step (order-sensitive by design).
    Returns (components sorted by member list, per-layer merge stats).
    """

    uf = UnionFind(ids)
    stats = {}
    for layer, key_of in relations:
        groups = relation_groups(ids, key_of)
        effective = 0
        for key in sorted(groups):
            members = groups[key]
            anchor = members[0]
            for other in members[1:]:
                if uf.union(anchor, other):
                    effective += 1
        stats[layer] = {
            "multi_id_groups": len(groups),
            "ids_in_multi_id_groups": sum(len(v) for v in groups.values()),
            "largest_group": max((len(v) for v in groups.values()), default=0),
            "effective_new_unions": effective,
        }
    by_root: dict[str, list[str]] = collections.defaultdict(list)
    for identifier in sorted(ids):
        by_root[uf.find(identifier)].append(identifier)
    components = sorted((sorted(v) for v in by_root.values()), key=lambda v: v[0])
    return components, stats


def classify_component(member_set, exclusion_sets, b_pool):
    """Fail-closed per-component B2 classification (pure function)."""

    contains = {name: len(member_set & ids) for name, ids in exclusion_sets.items()}
    reasons = [name for name, count in contains.items() if count]
    eligible = (not reasons) and len(member_set & b_pool) == len(member_set)
    return contains, reasons, eligible


def component_id_of(members) -> str:
    """Content-addressed, stable id for a component (member ids are canonical)."""

    return "fc-" + sha256_hex("\n".join(members))[:12]


def representative_of(members) -> str:
    """Deterministic representative: min sha256(statement_id) over members."""

    return min(members, key=lambda sid: sha256_hex(sid))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="V2 family-component registry builder (pure metadata).")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--registry", default=REGISTRY_REL)
    parser.add_argument("--a001-set", default=A001_SET_REL)
    parser.add_argument("--b1-reserved", default=B1_RESERVED_REL)
    parser.add_argument("--holdout", default="experiments/manifests/m1_final_holdout.json")
    parser.add_argument("--e013", default="experiments/results/p3_0_promptset_temp1.json")
    parser.add_argument("--e016", default="experiments/results/p3_b_n8_calibration.json")
    parser.add_argument("--p3c-fixed-set", default="experiments/manifests/p3c_fixed_set.json")
    parser.add_argument("--mechanism-set", default="experiments/manifests/igr_mechanism_set.json")
    parser.add_argument("--output", default=OUTPUT_REL)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    def resolve(path: str) -> Path:
        candidate = Path(path)
        return candidate if candidate.is_absolute() else ROOT / candidate

    # ---- dataset (unique rows); same fail-closed premise as the leakage audit
    dataset_path = resolve(args.dataset)
    rows = parquet.read_table(dataset_path).to_pylist()
    unique_rows: dict[str, dict] = {}
    for row in rows:
        unique_rows.setdefault(row["statement_id"], row)
    if len(unique_rows) != EXPECTED["unique_ids"]:
        print(f"[ERROR] dataset has {len(unique_rows)} unique ids, expected 7620", file=sys.stderr)
        return 2
    names = {sid: row["name"] for sid, row in unique_rows.items()}
    statements = {sid: row["formal_statement"] for sid, row in unique_rows.items()}
    natural = {sid: (row.get("natural_language") or "") for sid, row in unique_rows.items()}
    sources = {sid: (row.get("source") or "") for sid, row in unique_rows.items()}

    # ---- frozen registry (roles hash recomputed)
    registry_path = resolve(args.registry)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    canonical_roles = json.dumps(registry["roles"], sort_keys=True, separators=(",", ":"))
    if sha256_hex(canonical_roles) != registry["roles_sha256"]:
        print("[ERROR] registry roles_sha256 does not recompute - refusing", file=sys.stderr)
        return 2
    roles = registry["roles"]
    pool_ids = set().union(*(set(roles[role]) for role in ROLE_LABELS))
    if len(pool_ids) != EXPECTED["pool"]:
        print(f"[ERROR] pool from roles is {len(pool_ids)}, expected 6729", file=sys.stderr)
        return 2

    # ---- A001 selection set (A3-primary), hash-checked against the registry
    a001_path = resolve(args.a001_set)
    if sha256_file(a001_path) != registry["a3_set"]["sha256"]:
        print("[ERROR] A001 set sha256 does not match the registry record", file=sys.stderr)
        return 2
    a001_ids = set(roles["A3-primary"])
    a001_file_ids = {t["statement_id"] for t in json.loads(a001_path.read_text(encoding="utf-8"))["theorems"]}
    if a001_file_ids != a001_ids or len(a001_ids) != EXPECTED["a001"]:
        print("[ERROR] A001 set file and registry A3-primary disagree", file=sys.stderr)
        return 2

    # ---- B1-audit-reserved carve-out
    b1_record = json.loads(resolve(args.b1_reserved).read_text(encoding="utf-8"))
    b1_ids = set(b1_record["statement_ids"])
    if len(b1_ids) != EXPECTED["b1_reserved"]:
        print(f"[ERROR] B1 reserved set is {len(b1_ids)}, expected 16", file=sys.stderr)
        return 2

    # ---- V1 consumption sources (rebuilt exactly like v2_family_leakage_audit.py)
    v1_sources: dict[str, set[str]] = {}
    v1_sources["E013 promptset rollouts"] = set(
        json.loads(resolve(args.e013).read_text(encoding="utf-8"))["sampled_statement_ids"]
    )
    v1_sources["E016 n8 calibration"] = set(
        json.loads(resolve(args.e016).read_text(encoding="utf-8"))["sampled_statement_ids"]
    )
    v1_sources["E018 dev set (P3C)"] = {
        t["statement_id"] for t in json.loads(resolve(args.p3c_fixed_set).read_text(encoding="utf-8"))["theorems"]
    }
    v1_sources["E020-M mechanism set"] = {
        t["statement_id"] for t in json.loads(resolve(args.mechanism_set).read_text(encoding="utf-8"))["theorems"]
    }
    formal_to_ids: dict[str, list[str]] = {}
    for sid, text in statements.items():
        formal_to_ids.setdefault(normalize(text), []).append(sid)
    for dump_dir in sorted(ROOT.glob("runs/*/rollout_data")):
        if not any(dump_dir.glob("*.jsonl")):
            continue
        texts, unparsed = dump_formal_statements(dump_dir)
        if unparsed:
            print(f"[ERROR] {dump_dir}: {len(unparsed)} unparsed dump line(s)", file=sys.stderr)
            return 2
        matched: set[str] = set()
        for text in sorted(texts):
            matched.update(formal_to_ids.get(text, []))
        v1_sources[f"training dumps {dump_dir.parent.name}"] = matched
    v1_used = set().union(*v1_sources.values())
    if len(v1_used) != EXPECTED["v1_used"]:
        print(f"[ERROR] V1-used union is {len(v1_used)}, expected 763", file=sys.stderr)
        return 2

    # ---- sealed E023 holdout
    holdout_path = resolve(args.holdout)
    holdout_ids = {t["statement_id"] for t in json.loads(holdout_path.read_text(encoding="utf-8"))["theorems"]}
    if len(holdout_ids) != EXPECTED["holdout"] or holdout_ids & v1_used:
        print("[ERROR] sealed holdout drift or overlap with the V1-used union", file=sys.stderr)
        return 2

    # ---- components over the union of the three frozen relations
    all_ids = sorted(unique_rows)
    relations = [
        ("L3_name_family", lambda sid: family_key(names[sid])),
        ("L2_skeleton", lambda sid: skeleton_text(statements[sid])),
        ("L4_nl", lambda sid: strong_text(natural[sid])),
    ]
    components, merge_stats = build_components(all_ids, relations)

    # ---- per-component classification (fail-closed, per COMPONENT not per id)
    role_sets = {role: set(members) for role, members in roles.items()}
    b_pool = set().union(*(role_sets[role] for role in B_POOL_ROLES))
    exclusion_sets = {
        "v1_used": v1_used,
        "e023_holdout": holdout_ids,
        "a001_selection": a001_ids,
        "a_reserve": role_sets["A-reserve"],
        "b1_audit_reserved": b1_ids,
        "c_joint_holdout": role_sets["C-joint-holdout"],
    }
    entries = []
    for members in components:
        member_set = set(members)
        contains, reasons, eligible = classify_component(member_set, exclusion_sets, b_pool)
        entries.append(
            {
                "component_id": component_id_of(members),
                "size": len(members),
                "member_statement_ids": members,
                "names": sorted({names[m] for m in members}),
                "sources": dict(sorted(collections.Counter(sources[m] for m in members).items())),
                "roles": {role: len(member_set & role_sets[role]) for role in ROLE_LABELS},
                "contains": contains,
                "exclusion_reasons": reasons,
                "eligible_for_b2": eligible,
                "representative_statement_id": representative_of(members),
            }
        )
    if len(entries) != len(components) or len({e["component_id"] for e in entries}) != len(entries):
        print("[ERROR] component id collision - refusing", file=sys.stderr)
        return 2
    membership = sum(e["size"] for e in entries)
    if membership != EXPECTED["unique_ids"]:
        print(f"[ERROR] component membership sums to {membership}, expected 7620", file=sys.stderr)
        return 2

    # ---- statistics for the manual giant-component review
    size_dist = {str(k): v for k, v in sorted(collections.Counter(e["size"] for e in entries).items())}
    largest = sorted(entries, key=lambda e: (-e["size"], e["component_id"]))[:10]
    giant = [e for e in entries if e["size"] > GIANT_WARNING_THRESHOLD]
    eligible_entries = [e for e in entries if e["eligible_for_b2"]]
    exclusion_counts = {
        "components_total": len(entries),
        "components_excluded": len(entries) - len(eligible_entries),
        "eligible_components": len(eligible_entries),
        "eligible_statements": sum(e["size"] for e in eligible_entries),
        "by_reason_component_counts": {
            name: sum(1 for e in entries if name in e["exclusion_reasons"]) for name in exclusion_sets
        },
    }

    result = {
        "artifact_type": "v2_family_component_registry",
        "status": "complete",
        "naming": (
            "family-component registry (2026-09-20, Track B); components are connected components of the "
            "union of L3 name-family, L2 skeleton and normalized non-empty natural_language relations; "
            "read-only over frozen inputs"
        ),
        "definition": {
            "relations": {
                "L3_name_family": "theorem name minus trailing '_v<digits>' variant suffix",
                "L2_skeleton": "L1-strong (NFKC + whitespace-collapse) with first declaration name abstracted",
                "L4_nl": "L1-strong of non-empty natural_language",
            },
            "merge_order": ["L3_name_family", "L2_skeleton", "L4_nl"],
            "component": "connected component of the union of the three relations",
            "eligibility": (
                "a component is eligible for the B2 development pool only if every member is in "
                "B-train/B-validation/B-test and the component touches none of: V1-used, E023 holdout, "
                "A3-primary (A001), A-reserve, B1-audit-reserved, C-joint-holdout"
            ),
            "representative": "min sha256(statement_id) over the component's members",
        },
        "inputs": {
            "dataset": {"path": str(dataset_path), "sha256": sha256_file(dataset_path)},
            "registry": {"path": str(registry_path), "sha256": sha256_file(registry_path), "roles_sha256": registry["roles_sha256"]},
            "a001_set": {"path": str(a001_path), "sha256": registry["a3_set"]["sha256"]},
            "b1_reserved": {"path": str(resolve(args.b1_reserved)), "sha256": sha256_file(resolve(args.b1_reserved))},
            "e023_holdout": {"path": str(holdout_path), "sha256": sha256_file(holdout_path)},
            "v1_sources": {
                name: sha256_file(resolve(path))
                for name, path in (
                    ("E013", args.e013),
                    ("E016", args.e016),
                    ("P3C", args.p3c_fixed_set),
                    ("mechanism", args.mechanism_set),
                )
            },
        },
        "counts": {
            "statements": len(all_ids),
            "components": len(entries),
            "components_multi_id": sum(1 for e in entries if e["size"] > 1),
            "components_singleton": sum(1 for e in entries if e["size"] == 1),
        },
        "merge_contributions": merge_stats,
        "component_size_distribution": size_dist,
        "largest_components": [
            {
                "component_id": e["component_id"],
                "size": e["size"],
                "names_sample": e["names"][:5],
                "exclusion_reasons": e["exclusion_reasons"],
                "roles": {k: v for k, v in e["roles"].items() if v},
            }
            for e in largest
        ],
        "giant_component_warning": (
            {
                "threshold": GIANT_WARNING_THRESHOLD,
                "components_over_threshold": [
                    {"component_id": e["component_id"], "size": e["size"], "names_sample": e["names"][:5]}
                    for e in sorted(giant, key=lambda e: (-e["size"], e["component_id"]))
                ],
            }
            if giant
            else None
        ),
        "exclusion_counts": exclusion_counts,
        "components": entries,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
        "script": "scripts/v2_build_family_components.py",
    }

    print(json.dumps({k: v for k, v in result.items() if k != "components"}, indent=2, ensure_ascii=False)[:6000])
    print("--- summary ---")
    print(f"statements {result['counts']['statements']} -> components {result['counts']['components']} "
          f"(multi-id {result['counts']['components_multi_id']}, singleton {result['counts']['components_singleton']})")
    for layer, stat in merge_stats.items():
        print(f"  {layer}: multi-id groups {stat['multi_id_groups']}, "
              f"effective new unions {stat['effective_new_unions']}, largest group {stat['largest_group']}")
    print(f"eligible components {exclusion_counts['eligible_components']} "
          f"covering {exclusion_counts['eligible_statements']} statements")
    print(f"exclusions by reason (component counts): {exclusion_counts['by_reason_component_counts']}")
    if giant:
        print(f"[WARNING] {len(giant)} component(s) exceed {GIANT_WARNING_THRESHOLD} members - manual review required")
    else:
        print("no giant component above the warning threshold")

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
