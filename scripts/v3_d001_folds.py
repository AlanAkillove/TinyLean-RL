#!/usr/bin/env python3
"""V3-D001 §3 / §20-step5 — freeze nested family-grouped CV folds.

Assigns every family component a deterministic outer fold (5) and, within each outer
training set, an inner fold (4), then writes a component-keyed manifest so the fly122
formal run uses the *identical* partition (no rebuild ambiguity).

The fold assignment is keyed by `component_id`, NOT by record, so any downstream run
that reconstructs the 686 valid groups maps each to the same fold, and the hard
constraint outer_train ∩ outer_test = ∅ (and inner ∩ = ∅) holds by construction.

Output: experiments/manifests/v3/v3_d001_folds.json   (+ stdout summary)
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from v3_d001_lib import build_records, stratified_group_kfold

DATASET = "data/raw/kimina_promptset/data/train-00000-of-00001.parquet"
REGISTRY = "experiments/manifests/v2/family_component_registry.json"
OUT = ROOT / "experiments" / "manifests" / "v3" / "v3_d001_folds.json"

N_OUTER = 5
N_INNER = 4
SEED = 20260923


def main() -> int:
    built = build_records(ROOT, DATASET, REGISTRY)
    records = built["valid"]
    y = np.array([r["y_score"] for r in records])
    comps = np.array([r["component_id"] for r in records])
    assert all(c is not None for c in comps), "unmapped component"

    outer = stratified_group_kfold(y, comps, N_OUTER, seed=SEED)

    # verify hard group-disjointness (outer)
    for c in np.unique(comps):
        assert len(set(outer[comps == c])) == 1, f"component {c} straddles outer folds"

    inner_by_outer = {}
    for f in range(N_OUTER):
        tr = np.where(outer != f)[0]
        iy = y[tr]
        ig = comps[tr]
        if len(np.unique(ig)) < N_INNER:
            sub = np.zeros(len(tr), dtype=int)
        else:
            sub = stratified_group_kfold(iy, ig, N_INNER, seed=SEED + 1000 + f)
        # disjointness within this outer-train partition
        for c in np.unique(ig):
            assert len(set(sub[ig == c])) == 1, f"inner leak for {c} in outer {f}"
        inner_by_outer[f] = {str(ig[i]): int(sub[i]) for i in range(len(tr))}

    # component -> outer fold map + per-record label summary
    comp_outer = {}
    comp_pos = {}
    comp_n = {}
    for r, fo in zip(records, outer):
        c = r["component_id"]
        comp_outer[c] = int(fo)
        comp_pos[c] = comp_pos.get(c, 0) + r["y_score"]
        comp_n[c] = comp_n.get(c, 0) + 1

    payload = {
        "outer": comp_outer,
        "inner_by_outer": {str(k): v for k, v in inner_by_outer.items()},
        "seed": SEED, "n_outer": N_OUTER, "n_inner": N_INNER,
    }
    folds_hash = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    per_fold = {}
    for f in range(N_OUTER):
        rows = [i for i, fo in enumerate(outer) if fo == f]
        per_fold[f] = {
            "n_components": len({comps[i] for i in rows}),
            "n_groups": len(rows),
            "n_informative": int(y[rows].sum()),
            "informative_rate": round(float(y[rows].mean()), 4),
        }

    result = {
        "artifact_type": "v3_d001_folds",
        "status": "FROZEN (committed before any test outcome)",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "group_unit": "component_id",
        "scheme": "nested StratifiedGroupKFold equivalent (quadratic-load greedy), outer 5 / inner 4",
        "seed": SEED,
        "n_outer": N_OUTER,
        "n_inner": N_INNER,
        "valid_groups": len(records),
        "n_components": len(comp_outer),
        "folds_hash": folds_hash,
        "outer_fold_profile": per_fold,
        "component_outer_fold": comp_outer,
        "component_positive_count": comp_pos,
        "component_group_count": comp_n,
        "inner_fold_by_outer": inner_by_outer,
        "constraint_check": {
            "outer_group_disjoint": True,
            "inner_group_disjoint": True,
            "note": "verified by assertion at build time + tests/test_v3_d001_folds.py",
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in
                      ["folds_hash", "valid_groups", "n_components", "outer_fold_profile"]}, indent=2))
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
