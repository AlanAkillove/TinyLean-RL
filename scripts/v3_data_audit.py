#!/usr/bin/env python3
"""V3-0 data audit (READ-ONLY).

Answers the Stage V3-0 questions for the Jev-inspired RLVR controller study using
only the existing seed1/seed2/seed3 GRPO rollout dumps. Nothing is trained, no
hidden states are extracted, no rollouts are generated, no V2 artifact is written.

Parsing assumptions are copied verbatim from the repo's canonical analyzers so the
numbers are comparable with published V1/V2 artifacts:
  * group reconstruction by `input` equality (scripts/rollout_dynamics.py:33)
  * JSONL split on "\\n" only (scripts/p3c_build_fixed_set.py:56-59)
  * FORMAL_BLOCK_RE + normalize -> statement_id join (p3c_build_fixed_set.py:37-41)
  * statement_id -> component_id via the frozen V2 family-component registry

Outputs:
  docs/v3/data_audit_stats.json   (machine-readable; NEW, not a V2 frozen artifact)
  stdout                            (human-readable summary for owner review)
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "v3" / "data_audit_stats.json"

SEED_DIRS = {
    "seed1": ROOT / "runs" / "p3b_pilot" / "rollout_data",
    "seed2": ROOT / "runs" / "m1_seed2" / "rollout_data",
    "seed3": ROOT / "runs" / "m1_seed3" / "rollout_data",
}
DATASET = ROOT / "data" / "raw" / "kimina_promptset" / "data" / "train-00000-of-00001.parquet"
REGISTRY = ROOT / "experiments" / "manifests" / "v2" / "family_component_registry.json"

FORMAL_BLOCK_RE = re.compile(r"# Formal Statement:\s*\n```lean4\n(.*?)\n```", re.DOTALL)
N = 8  # rollout.n


def normalize(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def is_system_error(tool_feedback: str) -> bool:
    return isinstance(tool_feedback, str) and tool_feedback.lstrip().startswith("# System Error:")


def load_statement_map() -> dict[str, str]:
    """normalized formal statement -> statement_id (dedupe by statement_id)."""
    t = pq.read_table(DATASET, columns=["statement_id", "formal_statement"])
    sid = t.column("statement_id").to_pylist()
    fst = t.column("formal_statement").to_pylist()
    m: dict[str, str] = {}
    for statement_id, formal in zip(sid, fst):
        m.setdefault(normalize(formal), statement_id)
    return m


def load_component_map() -> dict[str, str]:
    """statement_id -> component_id (invert the frozen registry)."""
    reg = json.loads(REGISTRY.read_text())
    m: dict[str, str] = {}
    for comp in reg["components"]:
        for member in comp["member_statement_ids"]:
            m[member] = comp["component_id"]
    return m


def reconstruct_groups(seed_dir: Path):
    """Yield one dict per GRPO group, and return structural diagnostics."""
    groups = []
    diag = {"files": 0, "lines": 0, "unparsable": 0, "irregular_block": 0, "no_formal_block": 0}
    for path in sorted(seed_dir.glob("*.jsonl"), key=lambda p: int(p.stem)):
        diag["files"] += 1
        step = int(path.stem)
        # Parse whole file first (split on \n ONLY).
        records = []
        for line in path.read_text(encoding="utf-8").split("\n"):
            if not line.strip():
                continue
            diag["lines"] += 1
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                diag["unparsable"] += 1
        # Group by contiguity + identical `input`.
        i = 0
        while i < len(records):
            block_input = records[i]["input"]
            j = i
            scores, accs, syserrs, fmt_errs = [], [], [], []
            while j < len(records) and records[j]["input"] == block_input:
                scores.append(float(records[j]["score"]))
                accs.append(float(records[j]["acc"]))
                syserrs.append(is_system_error(records[j].get("tool_feedback", "")))
                fmt_errs.append(records[j].get("format_error", "No error.") != "No error.")
                j += 1
            size = j - i
            if size != N:
                diag["irregular_block"] += 1
            mm = FORMAL_BLOCK_RE.search(block_input)
            if not mm:
                diag["no_formal_block"] += 1
            groups.append(
                {
                    "step": step,
                    "size": size,
                    "statement_text": normalize(mm.group(1)) if mm else None,
                    "score_sum": int(sum(scores)),
                    "acc_sum": int(sum(accs)),
                    "n_syserr": sum(syserrs),
                    "all_syserr": bool(all(syserrs)),
                    "n_format_error": sum(fmt_errs),
                }
            )
            i = j
    return groups, diag


def classify(score_sum: int, size: int) -> str:
    if score_sum == 0:
        return "all_fail"
    if score_sum >= size:
        return "all_success"
    return "informative"


def main() -> int:
    sid_map = load_statement_map()
    comp_map = load_component_map()

    per_seed = {}
    all_groups = []  # (seed, group)
    for seed, d in SEED_DIRS.items():
        groups, diag = reconstruct_groups(d)
        for g in groups:
            g["seed"] = seed
            g["statement_id"] = sid_map.get(g["statement_text"]) if g["statement_text"] else None
            g["component_id"] = comp_map.get(g["statement_id"]) if g["statement_id"] else None
            g["label_score"] = classify(g["score_sum"], g["size"])
            g["label_acc"] = classify(g["acc_sum"], g["size"])
        per_seed[seed] = {"groups": len(groups), "diag": diag}
        all_groups.extend(groups)

    # ---- Q1/Q2/Q3/Q6: counts + base rates -------------------------------------
    def rate_table(gs):
        c = defaultdict(int)
        for g in gs:
            c[g["label_score"]] += 1
        n = len(gs)
        return {
            "n_groups": n,
            "all_fail": c["all_fail"],
            "informative": c["informative"],
            "all_success": c["all_success"],
            "informative_rate": round(c["informative"] / n, 4) if n else None,
            "all_fail_rate": round(c["all_fail"] / n, 4) if n else None,
            "all_success_rate": round(c["all_success"] / n, 4) if n else None,
        }

    for seed in SEED_DIRS:
        per_seed[seed]["base_rate"] = rate_table([g for g in all_groups if g["seed"] == seed])

    pooled_rate = rate_table(all_groups)

    # label agreement score-based vs acc-based
    disagree = sum(1 for g in all_groups if g["label_score"] != g["label_acc"])
    cand_score_acc_gap = sum(1 for g in all_groups if g["score_sum"] != g["acc_sum"])

    # ---- Q4/Q5: recurrence ------------------------------------------------------
    def within_seed(seed):
        steps_by_stmt = defaultdict(set)
        for g in all_groups:
            if g["seed"] == seed and g["statement_id"]:
                steps_by_stmt[g["statement_id"]].add(g["step"])
        distinct = len(steps_by_stmt)
        multi = sum(1 for s in steps_by_stmt.values() if len(s) > 1)
        hist = defaultdict(int)
        for s in steps_by_stmt.values():
            hist[len(s)] += 1
        return {"distinct_theorems": distinct, "theorems_in_gt1_step": multi, "step_count_hist": dict(hist)}

    within = {seed: within_seed(seed) for seed in SEED_DIRS}

    stmt_by_seed = {seed: {g["statement_id"] for g in all_groups if g["seed"] == seed and g["statement_id"]} for seed in SEED_DIRS}
    comp_by_seed = {seed: {g["component_id"] for g in all_groups if g["seed"] == seed and g["component_id"]} for seed in SEED_DIRS}
    s1, s2, s3 = (stmt_by_seed[k] for k in ("seed1", "seed2", "seed3"))
    c1, c2, c3 = (comp_by_seed[k] for k in ("seed1", "seed2", "seed3"))
    cross_seed = {
        "distinct_statement_ids_total": len(s1 | s2 | s3),
        "s1ns2": len(s1 & s2), "s1ns3": len(s1 & s3), "s2ns3": len(s2 & s3),
        "all_three_stmt": len(s1 & s2 & s3),
        "distinct_components_total": len(c1 | c2 | c3),
        "c1nc2": len(c1 & c2), "c1nc3": len(c1 & c3), "c2nc3": len(c2 & c3),
        "all_three_comp": len(c1 & c2 & c3),
        "components_in_gt1_seed": sum(1 for comp in (c1 | c2 | c3) if sum(comp in x for x in (c1, c2, c3)) > 1),
    }

    # ---- Q7/Q8: anomalies -------------------------------------------------------
    def anom(gs):
        return {
            "groups_with_syserr_candidate": sum(1 for g in gs if g["n_syserr"] > 0),
            "groups_all_syserr": sum(1 for g in gs if g["all_syserr"]),
            "candidates_syserr": sum(g["n_syserr"] for g in gs),
            "groups_size_ne_8": sum(1 for g in gs if g["size"] != N),
        }

    anomalies = {seed: anom([g for g in all_groups if g["seed"] == seed]) for seed in SEED_DIRS}
    anomalies["pooled"] = anom(all_groups)

    # censoring sensitivity: drop every group containing a system-error candidate, recompute IGR
    clean = [g for g in all_groups if g["n_syserr"] == 0]
    pooled_rate_censored = rate_table(clean)

    # ---- Q9: family mapping coverage -------------------------------------------
    n_groups_no_stmt = sum(1 for g in all_groups if not g["statement_id"])
    n_groups_no_comp = sum(1 for g in all_groups if not g["component_id"])
    coverage = {
        "groups_unmapped_to_statement_id": n_groups_no_stmt,
        "groups_unmapped_to_component_id": n_groups_no_comp,
        "distinct_statement_ids": len({g["statement_id"] for g in all_groups if g["statement_id"]}),
        "distinct_components": len({g["component_id"] for g in all_groups if g["component_id"]}),
    }

    # ---- Q10: usable family-clean supervised size + proposed split --------------
    # unit of analysis = group (one n=8 rollout of one theorem at one step).
    # split unit = component (family); all groups of all member theorems move together.
    groups_by_comp = defaultdict(list)
    for g in all_groups:
        if g["component_id"]:
            groups_by_comp[g["component_id"]].append(g)
    comp_inf = {c: sum(1 for g in gs if g["label_score"] == "informative") for c, gs in groups_by_comp.items()}
    comp_n = {c: len(gs) for c, gs in groups_by_comp.items()}
    comps = sorted(groups_by_comp)
    # deterministic component ordering (component_id string) then a 70/15/15 split
    import hashlib
    def bucket(c: str) -> float:
        return int(hashlib.sha256(c.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    shuffled = sorted(comps, key=bucket)
    n_comp = len(shuffled)
    n_tr = int(round(0.70 * n_comp)); n_dev = int(round(0.15 * n_comp))
    split = {"train": shuffled[:n_tr], "dev": shuffled[n_tr:n_tr + n_dev], "test": shuffled[n_tr + n_dev:]}
    split_report = {}
    for name, cs in split.items():
        gs = [g for c in cs for g in groups_by_comp[c]]
        split_report[name] = {"n_components": len(cs), "n_groups": len(gs), **rate_table(gs)}
    # family size distribution over the ~443 seed components
    size_hist = defaultdict(int)
    for c in comps:
        size_hist[comp_n[c]] += 1

    result = {
        "artifact": "v3_data_audit_stats",
        "unit_of_analysis": "group (one n=8 GRPO rollout of one theorem at one step)",
        "split_unit": "component_id (V2 frozen family component)",
        "n_copied_parsing_assumptions": {
            "group_regroup_by": "input equality + contiguity",
            "jsonl_split": "\\n only",
            "stmt_id_join": "FORMAL_BLOCK_RE + normalize -> statement_id -> component_id",
        },
        "per_seed": per_seed,
        "structural_diag": {seed: per_seed[seed]["diag"] for seed in SEED_DIRS},
        "pooled_base_rate": pooled_rate,
        "pooled_base_rate_syserr_censored": pooled_rate_censored,
        "label_source": {
            "definition_informative": "0 < sum(score over n=8) < 8",
            "score_vs_acc_group_label_disagree": disagree,
            "groups_where_score_sum_ne_acc_sum": cand_score_acc_gap,
        },
        "within_seed_recurrence": within,
        "cross_seed_overlap": cross_seed,
        "anomalies": anomalies,
        "family_mapping_coverage": coverage,
        "family_clean_split": {
            "total_components_in_scope": n_comp,
            "component_group_size_hist": dict(sorted(size_hist.items())),
            "largest_components": sorted(((comp_n[c], c) for c in comps), reverse=True)[:10],
            "proposed_70_15_15": split_report,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
