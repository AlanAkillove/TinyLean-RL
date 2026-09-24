"""Tests for the V3-R001 family-clean pool builder (§17 step 4).

Two jobs. First, the pure helpers: component closure must add whole families rather than single
statements, unmappable ids must be REPORTED instead of dropped, and the contact scan must resolve
prompts to statement ids the way V3-D001 does. Second, a consistency check that re-derives the
headline numbers of the committed artifact, so docs/v3/V3-R001_family_clean_pool.md cannot drift
away from the data it describes. No parquet, no tokenizer, no GPU, no rollout.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "v3_r001_pool_build.py"
ARTIFACT = ROOT / "experiments" / "manifests" / "v3" / "V3-R001_family_clean_pool.json"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("v3_r001_pool_build", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


# --------------------------------------------------------------------------------------------------
# pure helpers
# --------------------------------------------------------------------------------------------------

def test_build_component_maps_is_two_way(mod):
    reg = {"components": [
        {"component_id": "fc-a", "size": 2, "member_statement_ids": ["1", "2"],
         "representative_statement_id": "1", "sources": {}, "exclusion_reasons": ["v1_used"],
         "eligible_for_b2": False},
        {"component_id": "fc-b", "size": 1, "member_statement_ids": ["3"],
         "representative_statement_id": "3", "sources": {}, "exclusion_reasons": [],
         "eligible_for_b2": True}]}
    stmt_to_comp, comps = mod.build_component_maps(reg)
    assert stmt_to_comp == {"1": "fc-a", "2": "fc-a", "3": "fc-b"}
    assert comps["fc-a"]["exclusion_reasons"] == ["v1_used"]
    assert comps["fc-b"]["eligible_for_b2"] is True


def test_resolve_components_closes_statements_up_to_whole_families(mod):
    """Naming ONE member of a family must block the family, and an unknown id must be reported."""
    reg = {"components": [
        {"component_id": "fc-a", "size": 2, "member_statement_ids": ["1", "2"],
         "representative_statement_id": "1", "sources": {}, "exclusion_reasons": [],
         "eligible_for_b2": True},
        {"component_id": "fc-b", "size": 1, "member_statement_ids": ["3"],
         "representative_statement_id": "3", "sources": {}, "exclusion_reasons": [],
         "eligible_for_b2": True}]}
    stmt_to_comp, comps = mod.build_component_maps(reg)
    cls = {"x": {"statements": ["1", "999"], "components": ["fc-ghost"], "status": "consumed"}}
    out = mod.resolve_components(cls, stmt_to_comp, comps, known=set(stmt_to_comp))["x"]
    assert out["component_ids"] == ["fc-a", "fc-ghost"]      # family closure + the named ghost
    assert out["n_components_resolved"] == 2
    assert out["unknown_component_ids"] == ["fc-ghost"]      # reported, never silently dropped
    assert out["statement_ids_not_in_registry"] == ["999"]


def test_local_contact_scan_dedupes_a_group_and_resolves_prompts(mod, tmp_path, monkeypatch):
    """8 candidates sharing one prompt are ONE contact; unresolvable prompts are counted, not guessed."""
    d = tmp_path / "rollout_data"
    d.mkdir()
    formal = "example := by\n  rfl"
    prompt = f"# Instructions\n# Formal Statement:\n```lean4\n{formal}\n```\n"
    lines = [json.dumps({"input": prompt, "score": 1.0}) for _ in range(8)]
    lines.append(json.dumps({"input": "# Formal Statement:\n```lean4\nother := by rfl\n```\n",
                             "score": 0.0}))
    (d / "1.jsonl").write_text("\n".join(lines) + "\n")
    monkeypatch.setattr(mod, "ROOT", tmp_path, raising=True)
    monkeypatch.setattr(mod, "ROLLOUT_DIRS", ["rollout_data"], raising=True)
    sid_map = {mod._normalize(formal): ("77", "synthetic")}
    out = mod.local_contact_scan(sid_map)
    assert out["statements_contacted_on_disk"] == 1
    assert out["statement_ids"] == ["77"]
    assert out["dirs_scanned"]["rollout_data"] == {"present": True, "files": 1,
                                                   "unique_statement_ids": 1,
                                                   "unmatched_prompts": 1}


def test_pool_hash_commits_the_reading_as_well_as_the_ids(mod):
    r = {"candidate_component_ids": ["fc-a"], "candidate_statement_ids": ["1"],
         "component_to_candidate": {"fc-a": "1"}}
    assert mod.pool_hash_of(r, "consumed_only") == mod.pool_hash_of(dict(r), "consumed_only")
    assert mod.pool_hash_of(r, "consumed_only") != mod.pool_hash_of(r, "owner_literal")
    assert mod.pool_hash_of({**r, "candidate_statement_ids": ["2"]}, "consumed_only") \
        != mod.pool_hash_of(r, "consumed_only")


# --------------------------------------------------------------------------------------------------
# the committed artifact vs the document that describes it
# --------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def art():
    if not ARTIFACT.exists():
        pytest.skip("pool artifact not built yet")
    return json.loads(ARTIFACT.read_text())


def test_capacity_headlines_match_the_narrative(art):
    """docs/v3/V3-R001_family_clean_pool.md §3-4 quote exactly these numbers."""
    want = {"owner_literal": 86, "owner_literal_wider_v1": 56,
            "owner_literal_all_reservations_honored": 0, "consumed_only": 221,
            "registry_labels_maximal": 0}
    assert art["capacity_verdict"]["capacity_by_reading"] == want
    assert art["primary_reading"] == "owner_literal_wider_v1"
    assert art["capacity_verdict"]["readings_that_can_support_N128"] == ["consumed_only"]
    assert art["capacity_verdict"]["primary_reading_can_support_N128_without_reusing_a_family"] \
        is False


def test_readings_are_nested_and_the_union_is_the_extraction_set(art):
    pools = {k: set(v["candidate_statement_ids"]) for k, v in art["pools_by_reading"].items()}
    lit, wid, con = pools["owner_literal"], pools["owner_literal_wider_v1"], pools["consumed_only"]
    assert wid <= lit                                   # wider V1 blocks a superset of carve-outs
    assert lit & con == wid                             # literal ^ consumed_only differ by C alone
    assert sorted(lit | wid | con) == art["extraction_union"]
    assert len(art["extraction_union"]) == 251


def test_every_reading_reports_only_pools_it_can_defend(art):
    for name, p in art["pools_by_reading"].items():
        r = art["readings"][name]["result"]
        assert len(p["candidate_statement_ids"]) == r["one_candidate_per_component"]
        assert len(p["candidate_statement_ids"]) == len(set(p["candidate_statement_ids"]))
        assert len(p["component_to_candidate"]) == r["one_candidate_per_component"]
        assert set(p["component_to_candidate"].values()) == set(p["candidate_statement_ids"])
        assert len(p["candidate_component_ids"]) == r["components_remaining_after_prompt_filter"]
        assert p["max_N_if_k_theorems_per_component"]["1"] == r["one_candidate_per_component"]
        assert p["max_N_if_k_theorems_per_component"]["2"] \
            >= p["max_N_if_k_theorems_per_component"]["1"]
        assert p["pool_hash"]


def test_no_carve_out_evaporated_on_an_unrecognized_id(art):
    for name, c in art["usage_classes"].items():
        unmapped = c["unmapped"]
        assert unmapped["statement_ids_not_in_registry"] == [], name
        assert unmapped["unknown_component_ids"] == [], name
        assert c["n_components_resolved"] > 0, name


def test_token_filter_reproduces_the_deployment_pool_audit(art):
    """The pool's eligibility filter must be the SAME computation step 3 proved for the trainer."""
    c = art["token_filter_consistency_with_pool_audit"]
    assert c["rows_match"] is True and c["statements_match"] is True
    assert c["this_run_rows_kept"] == 24246 and c["this_run_statements_kept"] == 7613


def test_contact_evidence_confirms_the_consumed_reserved_split(art):
    ev = art["local_rollout_contact_evidence"]
    assert ev["sanity_check_v1_seed123_fully_contacted"] is True
    assert ev["unique_statements_with_local_rollout_output"] == 612
    assert ev["reserved_classes_with_any_local_contact"] == []
    assert ev["per_class"]["v1_seed123_rollouts"]["with_local_rollout_output"] == 612
    # Track C is the decisive class: zero on-disk contact, and releasing it alone reaches N=128
    assert ev["per_class"]["c_joint_holdout"] == {"declared_status": "reserved-never-used",
                                                 "statements_in closure": 680,
                                                 "with_local_rollout_output": 0}
    assert art["marginal_cost_of_each_carveout_in_primary_reading"]["c_joint_holdout"][
        "candidates_returned_if_this_class_were_not_excluded"] == 221
    assert any("no longer on this host" in lim for lim in ev["limits"])


def test_the_pool_is_a_pure_subtraction_of_frozen_inputs(art):
    """No reading may keep a component the frozen registry marks as V1-touched."""
    reg = json.loads((ROOT / "experiments/manifests/v2/family_component_registry.json").read_text())
    blocked_by_registry_v1 = {str(c["component_id"])
                              for c in reg["components"]
                              if "v1_used" in (c.get("exclusion_reasons") or [])}
    pool = art["pools_by_reading"]["consumed_only"]["component_to_candidate"]
    assert set(pool) & blocked_by_registry_v1 == set()   # consumed_only still removes all v1_used
    assert art["status"].startswith("POOL FROZEN")
    assert "no rollout" in art["status"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
