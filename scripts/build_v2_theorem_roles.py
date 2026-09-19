#!/usr/bin/env python
"""Build the V2 theorem-role registry (canonical coordination, 2026-09-19).

Deterministically partitions the V2 test pool - the Promptset statements with
no V1 consumption history - into the theorem-level roles used by Track A / B / C:

    B-train 60% | B-validation 10% | B-test 10% | A-selection 10% | C-joint-holdout 10%

Design rules
------------
- The split unit is the NORMALIZED formal statement
  (``p3c_build_fixed_set.normalize``: per-line rstrip + whole-text strip), so
  near-duplicate families move together (experiment_protocol S2). All statement
  ids sharing one normalized text receive the same role.
- Role = ``sha256("<salt>|<normalized_statement>")``: first 16 hex digits ->
  mod 10000 -> cumulative bucket [0,6000) B-train / [6000,7000) B-validation /
  [7000,8000) B-test / [8000,9000) A-selection / [9000,10000) C-joint-holdout.
  The assignment is reproducible from (dataset revision, exclusion sources,
  B1-reserved list, salt, ratios, A3 split rule) alone.
- Fail-closed (the ``build_final_holdout.py`` pattern): any unparseable training
  dump, unmatched dump statement, missing exclusion source, unknown B1 id, or
  unexpected exclusion-count drift aborts the build.
- ``B1-audit-reserved``: the 16 statements registered for fly122's V2-B001
  (raw id V2-E001) budget-semantics audit are carved out of every role and
  registered under their own role (the V2-E001 ``usage_note`` requires them to
  stay out of the future B2 train/val/test pools and final evaluations too).
- The A-selection role is split deterministically by a second order hash:
  A3-primary = maximal group-aligned prefix of <= ``--a3-count`` ids;
  the remainder is A-reserve (reserved for the conditional second A-track
  decision only). A3-primary is materialized as the frozen V2-A001 selection set.

Outputs (committed):
* ``experiments/manifests/v2/theorem_role_registry.json``
* ``experiments/manifests/v2/v2_a001_selection_set.json``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from pyarrow import parquet

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from p3c_build_fixed_set import DATASET_REVISION, DEFAULT_DATASET, dump_formal_statements, normalize

ROLE_SALT = "tinylean-v2-theorem-roles-20260919"
A3_ORDER_SALT = "tinylean-v2-a3-order-20260919"
ROLE_BUCKETS = [
    ("B-train", 0, 6000),
    ("B-validation", 6000, 7000),
    ("B-test", 7000, 8000),
    ("A-selection", 8000, 9000),
    ("C-joint-holdout", 9000, 10000),
]
B1_RESERVED_ROLE = "B1-audit-reserved"


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_revision() -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()


def role_of(normalized_statement: str, salt: str) -> tuple[str, int]:
    digest = sha256_hex(f"{salt}|{normalized_statement}")
    bucket = int(digest[:16], 16) % 10000
    for role, low, high in ROLE_BUCKETS:
        if low <= bucket < high:
            return role, bucket
    raise AssertionError(f"bucket {bucket} outside every role range")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--e013", default="experiments/results/p3_0_promptset_temp1.json")
    parser.add_argument("--e016", default="experiments/results/p3_b_n8_calibration.json")
    parser.add_argument("--p3c-fixed-set", default="experiments/manifests/p3c_fixed_set.json")
    parser.add_argument("--mechanism-set", default="experiments/manifests/igr_mechanism_set.json")
    parser.add_argument("--holdout", default="experiments/manifests/m1_final_holdout.json")
    parser.add_argument("--b1-reserved", default="experiments/manifests/v2/b1_reserved_statement_ids.json")
    parser.add_argument("--salt", default=ROLE_SALT)
    parser.add_argument("--a3-order-salt", default=A3_ORDER_SALT)
    parser.add_argument("--a3-count", type=int, default=512)
    parser.add_argument("--expected-used-union", type=int, default=763)
    parser.add_argument("--expected-holdout", type=int, default=128)
    parser.add_argument("--output", default="experiments/manifests/v2/theorem_role_registry.json")
    parser.add_argument("--a3-set-output", default="experiments/manifests/v2/v2_a001_selection_set.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = ROOT / dataset_path
    rows = parquet.read_table(dataset_path).to_pylist()
    unique_rows: dict[str, dict] = {}
    for row in rows:
        unique_rows.setdefault(row["statement_id"], row)
    ids = list(unique_rows.keys())
    formal_to_ids: dict[str, list[str]] = {}
    for statement_id, row in unique_rows.items():
        formal_to_ids.setdefault(normalize(row["formal_statement"]), []).append(statement_id)

    # ---- exclusion sources (V1-used union; same sources as build_final_holdout.py)
    excluded_sources: dict[str, set[str]] = {}
    e013 = json.loads((ROOT / args.e013).read_text(encoding="utf-8"))
    excluded_sources["E013"] = set(e013["sampled_statement_ids"])
    e016 = json.loads((ROOT / args.e016).read_text(encoding="utf-8"))
    excluded_sources["E016"] = set(e016["sampled_statement_ids"])
    p3c = json.loads((ROOT / args.p3c_fixed_set).read_text(encoding="utf-8"))
    excluded_sources["P3C development set (E018)"] = {t["statement_id"] for t in p3c["theorems"]}
    mechanism = json.loads((ROOT / args.mechanism_set).read_text(encoding="utf-8"))
    excluded_sources["IGR mechanism set (E020-M)"] = {t["statement_id"] for t in mechanism["theorems"]}

    for path in sorted(ROOT.glob("runs/*/rollout_data")):
        if not any(path.glob("*.jsonl")):
            continue
        label = f"training dumps {path.parent.name}"
        texts, unparsed = dump_formal_statements(path)
        if unparsed:
            print(f"[ERROR] {label}: {len(unparsed)} unparsed line(s) - refusing to build", file=sys.stderr)
            for sample in unparsed[:5]:
                print(f"  {sample}", file=sys.stderr)
            return 2
        source_ids: set[str] = set()
        unmatched: list[str] = []
        for text in sorted(texts):
            matches = formal_to_ids.get(text)
            if not matches:
                unmatched.append(text[:120])
                continue
            source_ids.update(matches)
        if unmatched:
            print(f"[ERROR] {label}: {len(unmatched)} statements unmatched to the dataset", file=sys.stderr)
            for sample in unmatched[:5]:
                print(f"  {sample!r}", file=sys.stderr)
            return 2
        excluded_sources[label] = source_ids

    if not any(key.startswith("training dumps ") for key in excluded_sources):
        print("[ERROR] no training dump directories discovered - refusing to build", file=sys.stderr)
        return 2

    used_union: set[str] = set().union(*excluded_sources.values())
    if len(used_union) != args.expected_used_union:
        print(
            f"[ERROR] V1-used union is {len(used_union)}, expected {args.expected_used_union} "
            "(state drift: investigate before rebuilding)",
            file=sys.stderr,
        )
        return 2

    holdout = json.loads((ROOT / args.holdout).read_text(encoding="utf-8"))
    holdout_ids = {t["statement_id"] for t in holdout["theorems"]}
    if len(holdout_ids) != args.expected_holdout:
        print(
            f"[ERROR] sealed holdout has {len(holdout_ids)} ids, expected {args.expected_holdout}",
            file=sys.stderr,
        )
        return 2
    if holdout_ids & used_union:
        print("[ERROR] sealed holdout overlaps the V1-used union - investigate", file=sys.stderr)
        return 2

    combined_excluded = used_union | holdout_ids
    pool_ids = [identifier for identifier in ids if identifier not in combined_excluded]

    # ---- B1-audit-reserved ids (fly122 budget-semantics audit)
    b1 = json.loads((ROOT / args.b1_reserved).read_text(encoding="utf-8"))
    b1_ids = set(b1["statement_ids"])
    unknown = b1_ids - set(unique_rows)
    if unknown:
        print(f"[ERROR] B1 ids not found in the dataset: {sorted(unknown)}", file=sys.stderr)
        return 2

    # ---- role partition by normalized-statement groups
    pool_groups: dict[str, list[str]] = {}
    for identifier in pool_ids:
        norm = normalize(unique_rows[identifier]["formal_statement"])
        pool_groups.setdefault(norm, []).append(identifier)

    roles: dict[str, list[str]] = {role: [] for role, _, _ in ROLE_BUCKETS}
    roles[B1_RESERVED_ROLE] = []
    b1_absorbed_groups: list[dict] = []
    for norm, group_ids in sorted(pool_groups.items(), key=lambda item: item[0]):
        role, _bucket = role_of(norm, args.salt)
        reserved = [identifier for identifier in group_ids if identifier in b1_ids]
        if reserved:
            roles[B1_RESERVED_ROLE].extend(sorted(group_ids))
            b1_absorbed_groups.append(
                {
                    "normalized_sha256": sha256_hex(norm),
                    "bucket_role": role,
                    "statement_ids": sorted(group_ids),
                    "reserved_ids": sorted(reserved),
                    "group_absorbed": len(group_ids) > len(reserved),
                }
            )
        else:
            roles[role].extend(sorted(group_ids))

    b1_in_pool = sorted(b1_ids & set(pool_ids))
    b1_out_of_pool = sorted(b1_ids - set(pool_ids))

    # ---- A-selection split (A3-primary = group-aligned prefix <= a3-count)
    a_selection_ids = set(roles["A-selection"])
    a_groups = [
        (sha256_hex(f"{args.a3_order_salt}|{norm}"), norm, sorted(group_ids))
        for norm, group_ids in pool_groups.items()
        if group_ids[0] in a_selection_ids  # every id of a group shares the role
    ]
    a_groups.sort(key=lambda item: item[0])
    a3_primary: list[str] = []
    a_reserve: list[str] = []
    for _, _, group_ids in a_groups:
        if len(a3_primary) + len(group_ids) <= args.a3_count:
            a3_primary.extend(group_ids)
        else:
            a_reserve.extend(group_ids)
    roles["A3-primary"] = a3_primary
    roles["A-reserve"] = a_reserve

    for role, ids_ in roles.items():
        duplicate = len(ids_) - len(set(ids_))
        if duplicate:
            print(f"[ERROR] role {role} contains {duplicate} duplicate id(s)", file=sys.stderr)
            return 2
    all_role_ids = [identifier for ids_ in roles.values() for identifier in ids_]
    if len(all_role_ids) != len(pool_ids) + len(a3_primary) + len(a_reserve):
        # pool ids appear once in the five role buckets (+reserved) and again in
        # the A split; A3-primary/A-reserve duplicate A-selection ids by design.
        print(
            f"[ERROR] coverage mismatch: roles total {len(all_role_ids)}, expected "
            f"{len(pool_ids) + len(a3_primary) + len(a_reserve)}",
            file=sys.stderr,
        )
        return 2
    covered_base = set()
    for role, _, _ in ROLE_BUCKETS:
        covered_base |= set(roles[role])
    covered_base |= set(roles[B1_RESERVED_ROLE])
    if covered_base != set(pool_ids):
        missing = sorted(set(pool_ids) - covered_base)[:5]
        extra = sorted(covered_base - set(pool_ids))[:5]
        print(f"[ERROR] pool coverage mismatch; missing={missing} extra={extra}", file=sys.stderr)
        return 2

    # ---- A001 selection set (materialized from A3-primary, order = order hash)
    a3_order: list[str] = []
    for _, _, group_ids in a_groups:
        for identifier in group_ids:
            if identifier in set(a3_primary):
                a3_order.append(identifier)
    if len(a3_order) != len(a3_primary):
        print("[ERROR] A3 order reconstruction failed", file=sys.stderr)
        return 2

    a001_theorems = []
    for index, identifier in enumerate(a3_order):
        row = unique_rows[identifier]
        a001_theorems.append(
            {
                "theorem_index": index,
                "statement_id": identifier,
                "name": row.get("name", identifier),
                "formal_statement": row["formal_statement"],
                "natural_language": row.get("natural_language") or "",
            }
        )
    a001_artifact = {
        "artifact_type": "v2_a001_selection_set",
        "experiment": "V2-A001",
        "track": "A",
        "naming": "frozen Track-A checkpoint-selection set (V2-A001); materialized from the A-selection partition (A3-primary) of the V2 theorem-role registry",
        "selection_seed": 20260919,
        "selection_method": (
            "deterministic: A-selection role (sha256 bucket of the normalized statement) "
            f"ordered by sha256('{args.a3_order_salt}|<normalized statement>'), maximal "
            f"group-aligned prefix of <= {args.a3_count} ids"
        ),
        "dataset_path": str(dataset_path.resolve()),
        "dataset_revision": DATASET_REVISION,
        "source_registry": "experiments/manifests/v2/theorem_role_registry.json",
        "sealed": True,
        "freeze_rule": "after this commit: no re-draw, no theorem removal/addition, no reordering",
        "theorem_count": len(a001_theorems),
        "theorems": a001_theorems,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
    }

    registry = {
        "artifact_type": "v2_theorem_role_registry",
        "status": "frozen",
        "naming": "canonical V2 theorem-role registry (coordination 2026-09-19)",
        "dataset": {
            "path": str(dataset_path.resolve()),
            "revision": DATASET_REVISION,
            "unique_statement_ids": len(ids),
        },
        "exclusions": {
            "v1_used_union": len(used_union),
            "v1_used_sources": {source: len(source_ids) for source, source_ids in excluded_sources.items()},
            "sealed_e023_holdout": len(holdout_ids),
            "combined_excluded_union": len(combined_excluded),
            "rationale": (
                "V2 roles are built only from statements with no V1 consumption history "
                "(763 used) and no V1 final-evaluation history (128 sealed E023 holdout); "
                "this follows the approved Track-A A1 memo and fly122's B1 practice."
            ),
        },
        "pool": {
            "eligible_statement_ids": len(pool_ids),
            "eligible_normalized_groups": len(pool_groups),
        },
        "partition_rule": {
            "unit": "normalized formal statement (p3c_build_fixed_set.normalize: per-line rstrip + whole-text strip)",
            "hash": "sha256",
            "input": f"'{args.salt}' + '|' + normalized_statement",
            "bucket": "int(sha256_hex[:16], 16) % 10000",
            "roles": {role: [low, high] for role, low, high in ROLE_BUCKETS},
            "salt": args.salt,
        },
        "a_selection_split": {
            "order_hash": f"sha256('{args.a3_order_salt}' + '|' + normalized_statement), ascending",
            "a3_count_target": args.a3_count,
            "a3_primary_count": len(roles["A3-primary"]),
            "a_reserve_count": len(roles["A-reserve"]),
            "usage": {
                "A3-primary": "frozen as the V2-A001 selection set (experiments/manifests/v2/v2_a001_selection_set.json)",
                "A-reserve": "reserved for the conditional second A-track decision (V2-A002-class) only; no other use without a new amendment",
            },
        },
        "b1_audit_reserved": {
            "source": b1["source"],
            "n_registered": len(b1_ids),
            "in_pool": len(b1_in_pool),
            "out_of_pool": len(b1_out_of_pool),
            "out_of_pool_ids": b1_out_of_pool,
            "in_pool_ids": b1_in_pool,
            "absorbed_groups": b1_absorbed_groups,
            "rule": "every in-pool B1 id (and its whole normalized family) is relabelled to the B1-audit-reserved role; recorded bucket_role shows the would-be role",
        },
        "counts": {
            **{role: len(ids_) for role, ids_ in roles.items() if role not in {"A3-primary", "A-reserve"}},
            "A3-primary": len(roles["A3-primary"]),
            "A-reserve": len(roles["A-reserve"]),
        },
        "roles": {role: sorted(ids_) for role, ids_ in roles.items()},
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
        "script": "scripts/build_v2_theorem_roles.py",
    }

    canonical_roles = json.dumps(registry["roles"], sort_keys=True, separators=(",", ":"))
    registry["roles_sha256"] = sha256_hex(canonical_roles)

    a001_bytes = json.dumps(a001_artifact, indent=2, ensure_ascii=False) + "\n"
    registry["a3_set"] = {
        "path": "experiments/manifests/v2/v2_a001_selection_set.json",
        "theorem_count": len(a001_theorems),
        "sha256": sha256_hex(a001_bytes),
    }
    registry_bytes = json.dumps(registry, indent=2, ensure_ascii=False) + "\n"

    print(json.dumps({k: v for k, v in registry.items() if k != "roles"}, indent=2, ensure_ascii=False))
    print(f"a001 set: {len(a001_theorems)} theorems; registry roles_sha256={registry['roles_sha256']}")
    if args.dry_run:
        print("[dry-run] nothing written")
        return 0

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(registry_bytes, encoding="utf-8")
    a3_path = ROOT / args.a3_set_output
    a3_path.parent.mkdir(parents=True, exist_ok=True)
    a3_path.write_text(a001_bytes, encoding="utf-8")
    print(f"registry: {output_path}")
    print(f"a001 set: {a3_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
