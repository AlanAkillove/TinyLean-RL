"""Fixtures for the V5-R001 H1 external-pool builder (scripts/v5_h1_build_pool.py).

These tests are pure-CPU and never touch the network, any model or any holdout artifact.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from v5_h1_build_pool import (
    STRICT_OPTION,
    build_components,
    canonicalize,
    check_text,
    family_key,
    skeleton_text,
    strong_text,
)

CLEAN = """import Mathlib

theorem algebra_4013 {a b c : ℝ} (h : a * b * c = 1) :
    a / (a * b + a + 1) = 1 := by
"""


def test_canonicalize_clean_statement() -> None:
    canonical, reason, name = canonicalize(CLEAN)
    assert reason is None
    assert canonical is not None
    assert canonical.endswith(":= by")
    assert name == "algebra_4013"
    # eligibility semantics: autoImplicit pinned false directly after the imports
    assert canonical.startswith(f"import Mathlib\n{STRICT_OPTION}")


def test_canonicalize_strips_placeholder_and_body() -> None:
    canonical, reason, _ = canonicalize(CLEAN.replace(":= by", ":= by sorry"))
    assert reason is None and canonical.endswith(":= by")

    body = CLEAN.replace(":= by", ":= by\n  norm_num")
    canonical, reason, _ = canonicalize(body)
    assert reason is None and canonical.endswith(":= by")
    assert "norm_num" not in canonical


def test_canonicalize_completes_bare_assign() -> None:
    canonical, reason, _ = canonicalize(CLEAN.replace(":= by", ":=").rstrip() + "\n")
    assert reason is None and canonical.endswith(":= by")


def test_canonicalize_drops_informal_prelude_in_block_comment() -> None:
    noisy = "import Mathlib\n\n/- Some informal text $$\n\\[ x \\] -/\n\ntheorem foo : True := by\n"
    canonical, reason, name = canonicalize(noisy)
    assert reason is None
    assert canonical.startswith(f"import Mathlib\n{STRICT_OPTION}")
    assert "informal" not in canonical
    assert name == "foo"


def test_canonicalize_rejections() -> None:
    cases = {
        "num_decl_not_1": "import Mathlib\n\ntheorem a : True := by\n\ntheorem b : True := by\n",
        "other_decl": "import Mathlib\n\ndef helper := 1\n\ntheorem a : True := by\n",
        "other_decl_noncomputable": "import Mathlib\n\nnoncomputable def helper := 1\n\ntheorem a : True := by\n",
        "hash_cmd": "import Mathlib\n\n#eval 1\n\ntheorem a : True := by\n",
        "no_import": "theorem a : True := by\n",
        "prelude_junk": "import Mathlib\n\nby_cases h : True\n\ntheorem a : True := by\n",
        "sorry_inside": "import Mathlib\n\ntheorem a : (by sorry : True) = True := by\n",
        "explicit_autoimplicit": "import Mathlib\nset_option autoImplicit false\n\ntheorem a : True := by\n",
    }
    for expected, text in cases.items():
        canonical, reason, _ = canonicalize(text)
        want = "other_decl" if expected == "other_decl_noncomputable" else expected
        assert reason == want, (expected, reason)
        assert canonical is None


def test_canonicalize_allows_open_and_variable_prelude() -> None:
    text = "import Mathlib\nopen Nat\nvariable (n : ℕ)\n\ntheorem foo (h : n = n) : n = n := by\n"
    canonical, reason, _ = canonicalize(text)
    assert reason is None and canonical.endswith(":= by")


def test_layer_helpers() -> None:
    assert strong_text("a  b\nc") == "a b c"
    # NFKC maps double-struck letters to ASCII (the frozen layer rule)
    assert skeleton_text("theorem foo_bar {x : ℕ} : True") == "theorem _ {x : N} : True"
    assert family_key("number_theory_29735_v17562") == "number_theory_29735"
    assert family_key("algebra_4013") == "algebra_4013"
    assert family_key(None) is None


def test_components_merge_by_skeleton_and_keep_distinct_rows_apart() -> None:
    rows = [
        {
            "statement_id": "b" * 64,
            "canonical": "import Mathlib\n\ntheorem foo : True := by",
            "name": "foo",
            "problem": "p1",
        },
        {
            "statement_id": "a" * 64,
            "canonical": "import Mathlib\n\ntheorem bar : True := by",
            "name": "bar",
            "problem": "p2",
        },
        {
            "statement_id": "c" * 64,
            "canonical": "import Mathlib\n\ntheorem baz : 1 = 1 := by",
            "name": "baz",
            "problem": "p3",
        },
    ]
    component_of, members = build_components(rows)
    assert component_of["a" * 64] == component_of["b" * 64]  # same skeleton (name abstracted)
    assert component_of["c" * 64] != component_of["a" * 64]
    assert len(members) == 2
    assert members[component_of["a" * 64]] == sorted(["a" * 64, "b" * 64])


def test_components_do_not_merge_rows_without_names() -> None:
    rows = [
        {"statement_id": "a" * 64, "canonical": "import Mathlib\n\ntheorem t : True ∧ True := by", "name": None, "problem": "p1"},
        {"statement_id": "b" * 64, "canonical": "import Mathlib\n\ntheorem t : True ∧ True := by", "name": None, "problem": "p2"},
    ]
    _, members = build_components(rows)
    assert len(members) == 1  # merged by skeleton (identical canonical text), not by a None family


def test_check_text_is_statement_plus_sorry() -> None:
    text = check_text(CLEAN.strip())
    assert text.endswith(":= by\n  sorry")
