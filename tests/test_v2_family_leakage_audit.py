"""Tests for the V2 theorem-family leakage audit (pure functions only).

No file IO, no dataset, no GPU: the canonicalization layers and the
overlap/spanning helpers are exercised on synthetic inputs.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import v2_family_leakage_audit as audit


def test_family_key_strips_variant_suffix():
    assert audit.family_key("number_theory_29735_v17562") == "number_theory_29735"
    assert audit.family_key("algebra_10231_v0001") == "algebra_10231"
    assert audit.family_key("algebra_10231") == "algebra_10231"
    assert audit.family_key("n7") == "n7"
    assert audit.family_key("algebra240229") == "algebra240229"
    assert audit.family_key("") == ""


def test_family_key_does_not_strip_non_terminal_or_other_tokens():
    assert audit.family_key("foo_v1_bar") == "foo_v1_bar"
    assert audit.family_key("foo_vX") == "foo_vX"
    assert audit.family_key("foo_2") == "foo_2"


def test_strong_text_normalizes_unicode_and_whitespace():
    assert audit.strong_text("  Ａ\tﬁ  x\n\n y  ") == "A fi x y"
    assert audit.strong_text("a\r\nb") == "a b"
    assert audit.strong_text(audit.strong_text(" x  y ")) == "x y"
    assert audit.strong_text("") == ""


def test_skeleton_text_abstracts_only_the_first_declaration_name():
    got = audit.skeleton_text("theorem foo (x : ℕ) : P := by\n  rfl")
    # NFKC folds mathematical alphanumerics, so ℕ becomes N (documented layer behavior).
    assert got == "theorem _ (x : N) : P := by rfl"
    assert audit.skeleton_text("lemma Bar.baz : Q") == "lemma _ : Q"
    assert audit.skeleton_text("import Mathlib\n\n#check foo") == "import Mathlib #check foo"
    once = audit.skeleton_text("theorem foo : P")
    assert audit.skeleton_text(once) == once  # idempotent


def test_group_map_sorts_keys_and_values():
    groups = audit.group_map(["b", "a", "c"], lambda s: "g1" if s in {"a", "b"} else "g2")
    assert groups == {"g1": ["a", "b"], "g2": ["c"]}


def _family_of(token: str) -> str:
    return token.split(":")[0]


def test_overlap_report_counts_and_lists():
    key = _family_of
    report = audit.overlap_report(
        ["fam1:a", "fam1:b", "fam2:c"], ["fam1:d", "fam3:e"], key, include_lists=True
    )
    assert report["shared_groups"] == 1
    assert report["a_ids_in_shared_groups"] == 2
    assert report["b_ids_in_shared_groups"] == 1
    assert report["a_ids_implicated"] == ["fam1:a", "fam1:b"]
    assert report["b_ids_implicated"] == ["fam1:d"]
    assert report["examples"] == [{"key": "fam1", "a_ids": ["fam1:a", "fam1:b"], "b_ids": ["fam1:d"]}]


def test_overlap_report_without_lists_and_empty_intersection():
    report = audit.overlap_report(["x"], ["y"], lambda token: token, include_lists=False)
    assert report["shared_groups"] == 0
    assert report["examples"] == []
    assert "a_ids_implicated" not in report


def test_spanning_stats_detects_cross_role_families():
    key = _family_of
    roles = {"R1": ["f1:a", "f2:b"], "R2": ["f1:c", "f3:d"]}
    stats = audit.spanning_stats(roles, key)
    assert stats["families_spanning_2plus_roles"] == 1
    assert stats["total_groups"] == 3
    assert stats["role_id_coverage"] == {"R1": 1, "R2": 1}
    assert stats["ids_in_spanning_families"] == 2
    assert stats["pairwise_shared_groups"]["R1 x R2"]["shared_groups"] == 1
    assert stats["n_roles_distribution"] == {"1": 2, "2": 1}
    assert stats["examples"] == [{"key": "f1", "roles": ["R1", "R2"]}]


def test_spanning_stats_disjoint_roles():
    stats = audit.spanning_stats({"R1": ["a"], "R2": ["b"]}, lambda token: token)
    assert stats["families_spanning_2plus_roles"] == 0
    assert stats["ids_in_spanning_families"] == 0
    assert stats["role_group_counts"] == {"R1": 1, "R2": 1}
