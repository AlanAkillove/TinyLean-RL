"""Tests for the frozen V3-D001 nested grouped-CV fold manifest.

Runs WITHOUT the host-only rollout data: it validates the committed manifest's
internal integrity (hard group disjointness, fold coverage, determinism of the
folds_hash) and, when runs/ is present, that a rebuild reproduces the frozen hash.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

MANIFEST = ROOT / "experiments" / "manifests" / "v3" / "v3_d001_folds.json"


def _load():
    if not MANIFEST.exists():
        pytest.skip("V3-D001 fold manifest not present")
    return json.loads(MANIFEST.read_text())


def test_manifest_present_and_fields():
    d = _load()
    assert d["status"].startswith("FROZEN")
    assert d["n_outer"] == 5 and d["n_inner"] == 4 and d["seed"] == 20260923
    assert d["valid_groups"] == 686


def test_hard_group_disjointness_from_map():
    d = _load()
    # component -> single outer fold (the map is per-component, so disjointness is
    # structural); assert no component appears with conflicting folds across profiles.
    comp_outer = d["component_outer_fold"]
    assert all(isinstance(v, int) and 0 <= v < 5 for v in comp_outer.values())
    # inner maps: a component present in outer-train of f has exactly one inner fold
    for f in range(5):
        inner = d["inner_fold_by_outer"][str(f)]
        # every component assigned to outer fold f must NOT be in this outer-train map
        for c, fo in comp_outer.items():
            assert not (fo == f and c in inner), f"test component {c} leaked into inner-train of {f}"


def test_fold_profile_sums_to_valid_groups():
    d = _load()
    prof = d["outer_fold_profile"]
    tot = sum(prof[k]["n_groups"] for k in prof)
    assert tot == 686
    for k, v in prof.items():
        assert 0.08 <= v["informative_rate"] <= 0.25, f"fold {k} poorly stratified"


def test_folds_hash_is_reproducible_from_manifest():
    d = _load()
    payload = {
        "outer": d["component_outer_fold"],
        "inner_by_outer": {str(k): v for k, v in d["inner_fold_by_outer"].items()},
        "seed": d["seed"], "n_outer": d["n_outer"], "n_inner": d["n_inner"],
    }
    h = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert h == d["folds_hash"], "folds_hash does not reproduce from stored maps"


def test_rebuild_matches_frozen_hash_when_data_present():
    data = ROOT / "runs" / "m1_seed2" / "rollout_data" / "1.jsonl"
    if not data.exists():
        pytest.skip("rollout_data not present on this host")
    from v3_d001_lib import build_records, stratified_group_kfold  # noqa: WPS433
    d = _load()
    built = build_records(ROOT, "data/raw/kimina_promptset/data/train-00000-of-00001.parquet",
                          "experiments/manifests/v2/family_component_registry.json")
    records = built["valid"]
    y = np.array([r["y_score"] for r in records])
    comps = np.array([r["component_id"] for r in records])
    outer = stratified_group_kfold(y, comps, 5, seed=20260923)
    rebuilt = {str(c): int(f) for c, f in zip(comps, outer)}
    assert rebuilt == {k: int(v) for k, v in d["component_outer_fold"].items()}
