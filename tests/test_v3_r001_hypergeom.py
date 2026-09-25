"""V3-R001 §20 -- the exact hypergeometric machinery, checked against exact rational arithmetic.

R001's confirmatory gate is a single discrete number: how many of the analyzed informative theorems
land in the frozen top-20% block. The whole preregistration therefore rests on three functions in
`scripts/v3_r001_gate.py` (`exact_p`, `critical_function`, `attainable_alpha`) and on the boundary
table committed in `V3-R001_gate.json`. If any of them is off by one, mis-rounded or silently
approximate, the gate's p-value means something other than what the document says.

These tests do not re-run the same formula in the same library and call that a check. The reference
here is `fractions.Fraction` over `math.comb` -- exact integer arithmetic, no floating point anywhere
in the reference path -- swept over the entire preregistered design range: the real pooled design
(N=128, m=26) and the real within-synthetic design (N=45, m=9), every possible k and x, plus a grid
of other (N, m) pairs to catch a rule that happens to work only at the design point.

What is additionally pinned: monotonicity of the tail in x, the *definitional* property of the
critical function (it rejects at c[k] and does not reject at c[k]-1), and agreement between the
frozen artifact's committed boundary table and the code that produced it.
"""

from __future__ import annotations

import itertools
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import v3_r001_gate as G

GATE_REL = "experiments/manifests/v3/V3-R001_gate.json"

# the two designs the preregistration actually runs, plus a sweep that must not be design-specific
DESIGNS = [(128, 26), (45, 9)]
SWEEP = [(N, G.block_size(N)) for N in range(20, 141, 7)] + [(60, 30), (128, 1), (128, 127)]


def ref_tail(N: int, m: int, k: int, x: int) -> Fraction:
    """Exact P(X >= x) for X ~ Hypergeometric(N, k successes in the population, m drawn)."""
    if x <= 0:
        return Fraction(1)
    lo, hi = max(0, k - (N - m)), min(k, m)
    if not lo <= x <= hi:
        return Fraction(0) if x > hi else Fraction(1)
    total = math.comb(N, m)
    return sum(Fraction(math.comb(k, j) * math.comb(N - k, m - j), total) for j in range(x, hi + 1))


@pytest.mark.parametrize("N,m", DESIGNS)
def test_exact_p_matches_exact_rational_arithmetic(N: int, m: int) -> None:
    """Every (k, x) the design can produce, against a Fraction reference. No float is trusted."""
    for k in range(N + 1):
        hi = min(k, m)
        for x in range(hi + 2):
            exact = ref_tail(N, m, k, x)
            got = G.exact_p(N, m, k, x)
            assert abs(got - float(exact)) < 1e-12, f"p({N},{m},{k},{x}): {got} != {float(exact)}"
            # a rejection decision must not sit close enough to alpha for float noise to flip it
            if 0.0 < float(exact) < 1.0:
                assert abs(float(exact) - G.ALPHA) > 1e-9, (
                    f"the exact p at ({N},{m},{k},{x}) is indistinguishable from alpha; the gate's "
                    "decision would rest on floating point rather than on the data")


@pytest.mark.parametrize("N,m", DESIGNS + SWEEP)
def test_tail_is_weakly_monotone_decreasing_in_x(N: int, m: int) -> None:
    """Requiring more block positives can never raise the p-value."""
    for k in range(0, N + 1, 3):
        ps = [G.exact_p(N, m, k, x) for x in range(min(k, m) + 2)]
        assert all(a >= b - 1e-15 for a, b in itertools.pairwise(ps)), f"non-monotone tail at N={N} k={k}"


@pytest.mark.parametrize("N,m", DESIGNS + SWEEP)
def test_critical_function_is_exactly_the_smallest_rejecting_count(N: int, m: int) -> None:
    """c[k] rejects and c[k]-1 does not. That is the whole definition; off-by-one fails here."""
    c = G.critical_function(N, m, G.ALPHA)
    assert len(c) == N + 1
    assert not math.isfinite(c[0]), "k=0 has no positives, so no rejection region can exist"
    for k in range(1, N + 1):
        cv = c[k]
        lo, hi = max(0, k - (N - m)), min(k, m)
        rejecting = [x for x in range(lo, hi + 1) if G.exact_p(N, m, k, x) <= G.ALPHA]
        if math.isfinite(cv):
            assert int(cv) == min(rejecting), f"c[{k}]={cv} is not the smallest rejecting x"
            assert G.exact_p(N, m, k, int(cv)) <= G.ALPHA
            assert G.exact_p(N, m, k, int(cv) - 1) > G.ALPHA
        else:
            assert not rejecting, f"c[{k}]=inf but x in {rejecting} would reject"


@pytest.mark.parametrize("N,m", DESIGNS)
def test_no_rejection_region_only_when_the_block_cannot_be_enriched(N: int, m: int) -> None:
    """c[k] is infinite exactly when even a fully-positive block cannot reach alpha."""
    c = G.critical_function(N, m, G.ALPHA)
    for k in range(1, N + 1):
        p_at_best = G.exact_p(N, m, k, min(k, m))
        assert math.isfinite(c[k]) == (p_at_best <= G.ALPHA), (
            f"N={N} m={m} k={k}: best attainable p={p_at_best}, c={c[k]}")


def test_attainable_alpha_is_the_frozen_rule_s_size_under_h0() -> None:
    """Recompute the unconditional size with an independent exact mixture over k ~ Bin(N, pi)."""
    for N, m, pi in ((128, 26, 0.1391), (45, 9, 0.3444)):
        c = G.critical_function(N, m, G.ALPHA)
        total = Fraction(0)
        for k in range(1, N + 1):
            if not math.isfinite(c[k]):
                continue
            pmf = Fraction(math.comb(N, k)) * Fraction(pi).limit_denominator(10**9) ** k * \
                (1 - Fraction(pi).limit_denominator(10**9)) ** (N - k)
            total += pmf * ref_tail(N, m, k, int(c[k]))
        got = G.attainable_alpha(N, m, pi)
        assert abs(got - float(total)) < 1e-6, f"attainable alpha at N={N}: {got} != {float(total)}"
        assert got <= G.ALPHA + 1e-12, "the honest size of the rule must not exceed its nominal alpha"


def test_frozen_gate_boundary_table_agrees_with_the_code() -> None:
    """The committed artifact's boundaries are the ones this code computes today."""
    gate = json.loads((ROOT / GATE_REL).read_text())
    N, m = gate["design"]["N_nominal"], gate["design"]["top_block_size"]
    assert m == G.block_size(N)
    c = G.critical_function(N, m, G.ALPHA)
    rows = gate["pooled_gate"]["P1"]["boundary_table"]
    assert len(rows) > 20, "the table is supposed to cover the plausible observed range of k"
    for row in rows:
        k = row["k_total_positives"]
        cv = row["reject_if_block_positives_at_least"]
        assert cv == (int(c[k]) if math.isfinite(c[k]) else None), f"boundary disagrees at k={k}"
        if cv is None:
            continue
        for field in ("exact_p_at_the_boundary", "conditional_level_at_this_k"):
            assert row[field] == pytest.approx(G.exact_p(N, m, k, cv), abs=1e-5), field
        assert row["implied_enrichment_at_the_boundary"] == pytest.approx((cv / m) / (k / N), rel=1e-3)
    synth = gate["within_synthetic_gate"]["S1"]
    cs = G.critical_function(synth["n"], synth["m_top20"], G.ALPHA)
    assert synth["reject_if_block_positives_at_least"] == int(cs[synth["k_at_the_design_anchor"]])
    assert synth["identifiable"] is math.isfinite(cs[synth["k_at_the_design_anchor"]])


def test_block_size_is_the_frozen_twenty_percent_rounding_rule() -> None:
    """The design's top block: 26 of 128 and 9 of 45, by the committed rounding rule."""
    assert G.block_size(128) == 26
    assert G.block_size(45) == 9
    assert G.TOP_FRACTION == 0.20
    for n in (1, 4, 5, 6, 9, 10, 128):
        assert abs(G.block_size(n) - 0.20 * n) <= 0.5
