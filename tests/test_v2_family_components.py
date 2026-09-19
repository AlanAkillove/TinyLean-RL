"""Tests for the V2 family-component registry builder (pure-metadata)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from v2_build_family_components import (
    UnionFind,
    build_components,
    classify_component,
    component_id_of,
    relation_groups,
    representative_of,
)
from v2_family_leakage_audit import family_key, sha256_hex, skeleton_text, strong_text


def _relations(names, statements, naturals):
    return [
        ("L3_name_family", lambda sid: family_key(names[sid])),
        ("L2_skeleton", lambda sid: skeleton_text(statements[sid])),
        ("L4_nl", lambda sid: strong_text(naturals[sid])),
    ]


class TestUnionFind:
    def test_union_returns_only_on_real_merge(self):
        uf = UnionFind(["a", "b", "c"])
        assert uf.union("a", "b") is True
        assert uf.union("b", "a") is False
        assert uf.union("a", "c") is True
        assert len({uf.find(x) for x in ("a", "b", "c")}) == 1


class TestRelationGroups:
    def test_skips_empty_keys(self):
        groups = relation_groups(["a", "b", "c"], lambda sid: {"a": "k", "b": "k", "c": ""}[sid])
        assert groups == {"k": ["a", "b"]}

    def test_values_sorted(self):
        assert relation_groups(["b", "a"], lambda sid: "k") == {"k": ["a", "b"]}


class TestBuildComponents:
    def test_l3_family_merge(self):
        ids = ["fam1_v0001", "fam1_v0002", "other_v0001"]
        names = {i: i for i in ids}
        statements = {i: f"theorem t{i} : {n} = {n} := by rfl" for n, i in enumerate(ids)}
        naturals = {i: f"nl {i}" for i in ids}
        comps, stats = build_components(ids, _relations(names, statements, naturals))
        assert comps == [["fam1_v0001", "fam1_v0002"], ["other_v0001"]]
        assert stats["L3_name_family"]["effective_new_unions"] == 1

    def test_l2_skeleton_cross_name_merge(self):
        ids = ["a", "b"]
        names = {i: i for i in ids}
        statements = {"a": "theorem a : 1 = 1 := by rfl", "b": "theorem b : 1 = 1 := by rfl"}
        naturals = {"a": "lhs", "b": "rhs"}
        comps, stats = build_components(ids, _relations(names, statements, naturals))
        assert comps == [["a", "b"]]
        assert stats["L2_skeleton"]["effective_new_unions"] == 1

    def test_nl_merge_when_other_layers_disjoint(self):
        ids = ["x_v0001", "y_v0002"]
        names = {i: i for i in ids}
        statements = {"x_v0001": "theorem x : Nat := 1", "y_v0002": "theorem y : Int := 2"}
        naturals = {"x_v0001": "Prove the claim.", "y_v0002": "Prove the claim."}
        comps, stats = build_components(ids, _relations(names, statements, naturals))
        assert comps == [["x_v0001", "y_v0002"]]
        assert stats["L4_nl"]["effective_new_unions"] == 1
        assert stats["L3_name_family"]["effective_new_unions"] == 0
        assert stats["L2_skeleton"]["effective_new_unions"] == 0

    def test_empty_nl_never_merges(self):
        ids = ["x_v0001", "y_v0002"]
        names = {i: i for i in ids}
        statements = {"x_v0001": "theorem x : Nat := 1", "y_v0002": "theorem y : Int := 2"}
        naturals = {"x_v0001": "", "y_v0002": ""}
        comps, _ = build_components(ids, _relations(names, statements, naturals))
        assert len(comps) == 2

    def test_transitivity_across_layers(self):
        ids = ["a_v0001", "a_v0002", "c_v0001"]
        names = {i: i for i in ids}
        statements = {
            "a_v0001": "theorem p : Nat := 1",
            "a_v0002": "theorem q : Bool := true",
            "c_v0001": "theorem r : Char := 'x'",
        }
        naturals = {"a_v0001": "N1", "a_v0002": "N2", "c_v0001": "N2"}
        comps, _ = build_components(ids, _relations(names, statements, naturals))
        assert comps == [["a_v0001", "a_v0002", "c_v0001"]]

    def test_deterministic_output(self):
        ids = ["b2", "b1", "a1"]
        names = {i: i for i in ids}
        statements = {i: f"theorem t{i} : True := by trivial" for i in ids}
        naturals = {i: "" for i in ids}
        first, _ = build_components(ids, _relations(names, statements, naturals))
        second, _ = build_components(list(reversed(ids)), _relations(names, statements, naturals))
        assert first == second


class TestIds:
    def test_component_id_content_addressed(self):
        assert component_id_of(["a", "b"]).startswith("fc-")
        assert component_id_of(["a", "b"]) == component_id_of(["a", "b"])
        assert component_id_of(["a", "b"]) != component_id_of(["a", "c"])

    def test_representative_is_min_hash_member(self):
        members = ["a", "b", "c"]
        assert representative_of(members) == min(members, key=sha256_hex)


class TestClassify:
    def test_b2_eligibility_rules(self):
        exclusion_sets = {"v1_used": {"x"}, "e023_holdout": set()}
        b_pool = {"a", "b", "c"}
        contains, reasons, eligible = classify_component({"a", "b"}, exclusion_sets, b_pool)
        assert eligible is True and reasons == [] and contains == {"v1_used": 0, "e023_holdout": 0}
        _, reasons, eligible = classify_component({"a", "z"}, exclusion_sets, b_pool)
        assert eligible is False and reasons == []
        _, reasons, eligible = classify_component({"a", "x"}, exclusion_sets, b_pool)
        assert eligible is False and reasons == ["v1_used"]
