#!/usr/bin/env python
"""V2 theorem-family leakage audit (pure metadata; 2026-09-20).

The canonical V2 theorem-role registry (2026-09-19) partitions the pool by the
EXACT normalized formal statement (per-line rstrip + whole-text strip). That
guarantees exact-statement isolation only: the Promptset stores one source
problem as many `_v<digits>` variants, and exact-statement hashing cannot keep
those near-duplicate families together. This audit - requested before the A4
`theta_RL*` freeze - quantifies the actual family leakage at four
canonicalization layers, WITHOUT changing the frozen V2-A001 set, its selection
rule or the sampling protocol:

  L0 exact        ``p3c_build_fixed_set.normalize`` (the current split unit)
  L1 strong       NFKC + whitespace-collapse + strip
  L2 skeleton     L1 + first declaration name abstracted (``theorem _ ...``)
  L3 name-family  theorem name with the trailing ``_v<digits>`` variant suffix
                  stripped (source-problem family)
  L4 nl           NFKC + whitespace-collapse of ``natural_language`` (non-empty)

Reports:
  1. A001 (512) vs every V1 consumption source (E013/E016 rollouts, E018 dev
     set, E020-M mechanism set, training dumps, sealed E023 holdout) per layer;
  2. cross-role family overlap of the six base roles (five buckets + the
     B1-audit-reserved carve-out) and of the A3-primary x A-reserve split;
  3. impact numbers for future amendments: A001 effective cluster count, the
     family-clean subset of C-joint-holdout, B1 family siblings inside the
     future B2 pools, and the ``source`` provenance distribution.

This artifact is evidence for POSITIONING and for future B2/C holdout
amendments only: it must never be used to re-draw or delete V2-A001 theorems
(one-shot freeze). No GPU, no Lean, no training; deterministic and fail-closed
on any input drift.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import subprocess
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from pyarrow import parquet

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from p3c_build_fixed_set import DATASET_REVISION, DEFAULT_DATASET, dump_formal_statements, normalize

ROLE_LABELS = (
    "B-train",
    "B-validation",
    "B-test",
    "A-selection",
    "C-joint-holdout",
    "B1-audit-reserved",
)
LAYER_LABELS = ("L0_exact", "L1_strong", "L2_skeleton", "L3_name_family", "L4_nl")
EXPECTED = {"unique_ids": 7620, "v1_used": 763, "holdout": 128, "pool": 6729, "a001": 512}
REGISTRY_REL = "experiments/manifests/v2/theorem_role_registry.json"
A001_SET_REL = "experiments/manifests/v2/v2_a001_selection_set.json"
VARIANT_SUFFIX_RE = re.compile(r"_v\d+$")
DECL_NAME_RE = re.compile(r"\b(theorem|lemma|example)\s+\S+")
WHITESPACE_RE = re.compile(r"\s+")


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


# --------------------------------------------------------------------------
# canonicalization layers (pure functions; unit-tested)
# --------------------------------------------------------------------------


def strong_text(text: str) -> str:
    """L1: NFKC + collapse every whitespace run + strip."""

    return WHITESPACE_RE.sub(" ", unicodedata.normalize("NFKC", text)).strip()


def skeleton_text(text: str) -> str:
    """L2: L1 with the first declaration name abstracted (``theorem _ ...``)."""

    return DECL_NAME_RE.sub(r"\1 _", strong_text(text), count=1)


def family_key(name: str) -> str:
    """L3: theorem name without the trailing ``_v<digits>`` variant suffix."""

    return VARIANT_SUFFIX_RE.sub("", name) or name


def group_map(ids, key_of) -> dict[str, list[str]]:
    """Deterministically group ids by key (sorted keys and values)."""

    groups: dict[str, list[str]] = {}
    for identifier in sorted(ids):
        groups.setdefault(key_of(identifier), []).append(identifier)
    return groups


def overlap_report(a_ids, b_ids, key_of, *, include_lists: bool, cap: int = 10) -> dict:
    """Family-level overlap between two id collections under one layer key."""

    groups_a = group_map(a_ids, key_of)
    groups_b = group_map(b_ids, key_of)
    shared = sorted(set(groups_a) & set(groups_b))
    implicated_a = sorted(i for i in a_ids if key_of(i) in set(shared))
    implicated_b = sorted(i for i in b_ids if key_of(i) in set(shared))
    report = {
        "shared_groups": len(shared),
        "a_ids_in_shared_groups": len(implicated_a),
        "b_ids_in_shared_groups": len(implicated_b),
        "examples": [
            {
                "key": key,
                "a_ids": groups_a[key][:3],
                "b_ids": groups_b[key][:3],
            }
            for key in shared[:cap]
        ],
    }
    if include_lists:
        report["a_ids_implicated"] = implicated_a
        report["b_ids_implicated"] = implicated_b
    return report


def spanning_stats(role_ids: dict[str, list[str]], key_of, cap: int = 10) -> dict:
    """Cross-role family-spanning statistics over mutually exclusive roles."""

    role_labels = sorted(role_ids)
    groups = {role: group_map(role_ids[role], key_of) for role in role_labels}
    key_to_roles: dict[str, set[str]] = {}
    for role in role_labels:
        for key in groups[role]:
            key_to_roles.setdefault(key, set()).add(role)

    pairwise = {}
    for i, role_a in enumerate(role_labels):
        for role_b in role_labels[i + 1 :]:
            shared = sorted(set(groups[role_a]) & set(groups[role_b]))
            pairwise[f"{role_a} x {role_b}"] = {
                "shared_groups": len(shared),
                "examples": shared[:3],
            }

    spanning = {key: roles for key, roles in sorted(key_to_roles.items()) if len(roles) >= 2}
    role_id_coverage = {
        role: sum(1 for identifier in role_ids[role] if len(key_to_roles[key_of(identifier)]) >= 2)
        for role in role_labels
    }
    n_roles_distribution = collections.Counter(len(roles) for roles in key_to_roles.values())
    examples = [
        {"key": key, "roles": sorted(roles)}
        for key, roles in sorted(spanning.items(), key=lambda item: (-len(item[1]), item[0]))[:cap]
    ]
    return {
        "role_group_counts": {role: len(groups[role]) for role in role_labels},
        "pairwise_shared_groups": pairwise,
        "families_spanning_2plus_roles": len(spanning),
        "total_groups": len(key_to_roles),
        "ids_in_spanning_families": sum(role_id_coverage.values()),
        "role_id_coverage": role_id_coverage,
        "n_roles_distribution": {str(k): v for k, v in sorted(n_roles_distribution.items())},
        "examples": examples,
    }


# --------------------------------------------------------------------------
# main audit
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="V2 theorem-family leakage audit (pure metadata).")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--registry", default=REGISTRY_REL)
    parser.add_argument("--a001-set", default=A001_SET_REL)
    parser.add_argument("--holdout", default="experiments/manifests/m1_final_holdout.json")
    parser.add_argument("--b1-reserved", default="experiments/manifests/v2/b1_reserved_statement_ids.json")
    parser.add_argument("--e013", default="experiments/results/p3_0_promptset_temp1.json")
    parser.add_argument("--e016", default="experiments/results/p3_b_n8_calibration.json")
    parser.add_argument("--p3c-fixed-set", default="experiments/manifests/p3c_fixed_set.json")
    parser.add_argument("--mechanism-set", default="experiments/manifests/igr_mechanism_set.json")
    parser.add_argument("--output", default="experiments/manifests/v2/family_leakage_audit.json")
    parser.add_argument("--example-cap", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    def resolve(path: str) -> Path:
        candidate = Path(path)
        return candidate if candidate.is_absolute() else ROOT / candidate

    dataset_path = resolve(args.dataset)
    registry_path = resolve(args.registry)
    a001_path = resolve(args.a001_set)

    # ---- dataset (unique rows)
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

    # ---- registry (frozen partition)
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
    a001_ids = set(roles["A3-primary"])
    a001_sha = sha256_file(a001_path)
    if a001_sha != registry["a3_set"]["sha256"]:
        print("[ERROR] A001 set sha256 does not match the registry record", file=sys.stderr)
        return 2
    a001_file_ids = {t["statement_id"] for t in json.loads(a001_path.read_text(encoding="utf-8"))["theorems"]}
    if a001_file_ids != a001_ids or len(a001_ids) != EXPECTED["a001"]:
        print("[ERROR] A001 set file and registry A3-primary disagree", file=sys.stderr)
        return 2

    # ---- V1 consumption sources (rebuilt exactly like build_v2_theorem_roles.py)
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
    holdout_ids = {
        t["statement_id"] for t in json.loads(resolve(args.holdout).read_text(encoding="utf-8"))["theorems"]
    }
    if len(holdout_ids) != EXPECTED["holdout"] or holdout_ids & v1_used:
        print("[ERROR] sealed holdout drift or overlap with the V1-used union", file=sys.stderr)
        return 2
    b1_record = json.loads(resolve(args.b1_reserved).read_text(encoding="utf-8"))
    b1_ids = set(b1_record["statement_ids"])

    # ---- layer key functions
    layer_keys = {
        "L0_exact": lambda sid: normalize(statements[sid]),
        "L1_strong": lambda sid: strong_text(statements[sid]),
        "L2_skeleton": lambda sid: skeleton_text(statements[sid]),
        "L3_name_family": lambda sid: family_key(names[sid]),
        "L4_nl": lambda sid: strong_text(natural[sid]),
    }
    layer_definitions = {
        "L0_exact": "p3c_build_fixed_set.normalize: per-line rstrip + whole-text strip (current split unit)",
        "L1_strong": "NFKC + whitespace-collapse + strip",
        "L2_skeleton": "L1 + first declaration name abstracted: (theorem|lemma|example) <name> -> '\\1 _'",
        "L3_name_family": "theorem name with the trailing '_v<digits>' variant suffix stripped",
        "L4_nl": "NFKC + whitespace-collapse of natural_language (empty texts excluded)",
    }

    # ---- layer statistics
    all_ids = sorted(unique_rows)
    layer_stats = {}
    for layer, key_of in layer_keys.items():
        entry = {}
        for scope_name, scope_ids in (("all_7620", all_ids), ("pool_6729", sorted(pool_ids)), ("a001_512", sorted(a001_ids))):
            groups = group_map(scope_ids, key_of)
            multi = {k: v for k, v in groups.items() if len(v) > 1 and k}
            entry[scope_name] = {
                "groups": len(groups),
                "multi_id_groups": len(multi),
                "ids_in_multi_id_groups": sum(len(v) for v in multi.values()),
            }
        layer_stats[layer] = entry
    if layer_stats["L0_exact"]["all_7620"]["groups"] != EXPECTED["unique_ids"]:
        print("[ERROR] L0 premise changed: normalized statements are not globally unique", file=sys.stderr)
        return 2

    # ---- A001 vs V1 targets, per layer
    v1_targets = dict(v1_sources)
    v1_targets["V1-used union"] = v1_used
    v1_targets["E023 sealed holdout"] = holdout_ids
    v1_targets["V1 union (used + holdout)"] = v1_used | holdout_ids
    a001_vs_v1 = {}
    for target_name, target_ids in v1_targets.items():
        per_layer = {}
        for layer, key_of in layer_keys.items():
            per_layer[layer] = overlap_report(
                a001_ids, target_ids, key_of, include_lists=True, cap=args.example_cap
            )
        a001_vs_v1[target_name] = per_layer
    for target_name in ("V1-used union", "E023 sealed holdout"):
        zero = all(
            a001_vs_v1[target_name][layer]["shared_groups"] == 0 for layer in ("L0_exact", "L1_strong")
        )
        if not zero:
            print(
                f"[ERROR] {target_name} shares exact/strong-normalized statements with A001 - "
                "the id-level exclusion did not hold; investigate",
                file=sys.stderr,
            )
            return 2

    # ---- cross-role family overlap (base roles; plus the A split)
    cross_role = {}
    for layer in ("L0_exact", "L1_strong", "L2_skeleton", "L3_name_family"):
        key_of = layer_keys[layer]
        spanning = spanning_stats({role: roles[role] for role in ROLE_LABELS}, key_of, cap=args.example_cap)
        spanning["a3_split"] = {
            "shared_groups": len(
                set(group_map(roles["A3-primary"], key_of)) & set(group_map(roles["A-reserve"], key_of))
            ),
            "examples": sorted(
                set(group_map(roles["A3-primary"], key_of))
                & set(group_map(roles["A-reserve"], key_of))
            )[:3],
        }
        cross_role[layer] = spanning
    if cross_role["L0_exact"]["families_spanning_2plus_roles"] != 0:
        print("[ERROR] L0 cross-role spanning families are not zero - registry drift", file=sys.stderr)
        return 2

    # ---- impact numbers
    family_of = layer_keys["L3_name_family"]
    a001_groups = group_map(a001_ids, family_of)
    a001_multi = {k: v for k, v in a001_groups.items() if len(v) > 1}
    effective_clusters = len(a001_groups)
    ci_inflation = (EXPECTED["a001"] / effective_clusters) ** 0.5

    contaminating_union = v1_used | holdout_ids
    for role in ROLE_LABELS:
        if role != "C-joint-holdout":
            contaminating_union |= set(roles[role])
    contaminating_keys = {family_of(i) for i in contaminating_union}
    c_joint = set(roles["C-joint-holdout"])
    clean_ids = sorted(i for i in c_joint if family_of(i) not in contaminating_keys)
    a_selection_keys = {family_of(i) for i in roles["A-selection"]}
    clean_vs_selection = sorted(i for i in c_joint if family_of(i) not in a_selection_keys)

    b1_in_pool = sorted(b1_ids & pool_ids)
    b1_families = {family_of(i) for i in b1_in_pool}
    b1_siblings_by_role = {
        role: sum(1 for i in roles[role] if family_of(i) in b1_families and i not in b1_ids)
        for role in ROLE_LABELS
        if role != "B1-audit-reserved"
    }

    source_distribution = {"a001_512": dict(sorted(collections.Counter(sources[i] for i in a001_ids).items()))}
    for role in ROLE_LABELS:
        source_distribution[f"role:{role}"] = dict(
            sorted(collections.Counter(sources[i] for i in roles[role]).items())
        )
    source_distribution["v1_used_union"] = dict(
        sorted(collections.Counter(sources[i] for i in v1_used).items())
    )
    source_distribution["e023_holdout"] = dict(
        sorted(collections.Counter(sources[i] for i in holdout_ids).items())
    )

    shared_union = a001_vs_v1["V1 union (used + holdout)"]["L3_name_family"]
    result = {
        "artifact_type": "v2_family_leakage_audit",
        "status": "complete",
        "naming": (
            "pure-metadata theorem-family leakage audit (2026-09-20); evidence for result positioning "
            "and future B2/C holdout amendments only - no set, rule or protocol change"
        ),
        "freeze_guard": {
            "a001_unchanged": True,
            "sets_modified": [],
            "rule": (
                "this audit must never be used to re-draw or delete V2-A001 theorems; the one-shot "
                "freeze and the preregistered selection rule stand"
            ),
        },
        "inputs": {
            "dataset": {
                "path": str(dataset_path),
                "revision": DATASET_REVISION,
                "sha256": sha256_file(dataset_path),
            },
            "registry": {"path": str(registry_path), "sha256": sha256_file(registry_path), "roles_sha256": registry["roles_sha256"]},
            "a001_set": {"path": str(a001_path), "sha256": a001_sha},
            "e023_holdout": {"path": str(resolve(args.holdout)), "sha256": sha256_file(resolve(args.holdout))},
            "b1_reserved": {"path": str(resolve(args.b1_reserved)), "sha256": sha256_file(resolve(args.b1_reserved))},
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
            "unique_ids": len(unique_rows),
            "v1_used": len(v1_used),
            "e023_holdout": len(holdout_ids),
            "pool": len(pool_ids),
            "a001": len(a001_ids),
        },
        "layer_definitions": layer_definitions,
        "layer_stats": layer_stats,
        "a001_vs_v1": a001_vs_v1,
        "cross_role": cross_role,
        "impact": {
            "a001_effective_clusters": {
                "ids": len(a001_ids),
                "families_L3": effective_clusters,
                "families_with_multiple_ids": len(a001_multi),
                "ids_in_multiple_id_families": sum(len(v) for v in a001_multi.values()),
                "note": (
                    "family clustering shrinks the effective sample; treat borderline paired CIs on the "
                    f"A001 set as ~{ci_inflation:.2f}x wider than the nominal i.i.d. width"
                ),
            },
            "track_c_clean_holdout": {
                "definition": (
                    "C-joint-holdout ids whose L3 family does not intersect the union of V1-used, E023 "
                    "holdout, B-train, B-validation, B-test, A-selection and B1-audit-reserved"
                ),
                "c_joint_total": len(c_joint),
                "clean_ids": len(clean_ids),
                "clean_families": len({family_of(i) for i in clean_ids}),
                "contaminated_ids": len(c_joint) - len(clean_ids),
                "clean_vs_a_selection_only": len(clean_vs_selection),
            },
            "b1_family_siblings": {
                "b1_in_pool": len(b1_in_pool),
                "b1_families": len(b1_families),
                "siblings_by_role": b1_siblings_by_role,
                "note": (
                    "family siblings of B1-audit-reserved statements inside the other roles; a "
                    "family-clean B1 no-contact guarantee would need to reserve these too"
                ),
            },
            "source_distribution": source_distribution,
        },
        "verdict": {
            "exact_statement_isolation": "holds by construction (L0 groups == ids everywhere)",
            "family_isolation": (
                "does NOT hold: L3 families span the V1 sets and the role partition; see "
                "a001_vs_v1 and cross_role for the measured extents"
            ),
            "a001_positions": (
                f"{shared_union['a_ids_in_shared_groups']} of {EXPECTED['a001']} A001 ids sit in L3 "
                "families that also contain V1-consumed statements; A001 therefore remains a "
                "selection device for the theta_RL* choice only"
            ),
            "confirmatory_claim": (
                "no A001 delta (even with ci_low > 0) is a confirmatory RL-vs-theta0 capability "
                "claim; that is reserved for the family-clean Track C holdout"
            ),
        },
        "recommendations": [
            "Position the A001 outcome as checkpoint selection only; report the family overlap next to it.",
            "Draft the next registry amendment with family-granular splits (L3 as the primary key, L2 as the audit layer); no re-split of the frozen current partition.",
            "Track B (fly122): derive B2 train/val/test from family-complete groups and extend the B1 carve-out to its family siblings if the no-contact guarantee must hold at family level.",
            "Track C: freeze the final holdout from the family-clean C-joint-holdout subset computed here, or re-partition under the family-granular amendment before the final evaluation.",
        ],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
        "script": "scripts/v2_family_leakage_audit.py",
    }

    output_path = resolve(args.output)
    print(json.dumps({k: v for k, v in result.items() if k != "a001_vs_v1"}, indent=2, ensure_ascii=False))
    print("--- A001 vs V1 (L3 name-family layer) ---")
    for target_name, per_layer in result["a001_vs_v1"].items():
        stat = per_layer["L3_name_family"]
        print(
            f"  {target_name}: shared families {stat['shared_groups']}, "
            f"A001 ids implicated {stat['a_ids_in_shared_groups']}"
        )
    if args.dry_run:
        print("[dry-run] nothing written")
        return 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"audit artifact: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
