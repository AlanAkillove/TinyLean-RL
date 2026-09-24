#!/usr/bin/env python3
"""V3-R001 §14 — EXACT power for every candidate gate, computed BEFORE the gate is frozen.

Owner §14: "exact power calculation from N and prospective positive rate FIRST, then freeze the
gate", and whether the GO also requires B2 > B1 significance is to be "decided by power, not forced".
This script produces the numbers that decision needs, and nothing else. No rollout, no GPU.

Why the primary gate can be EXACT rather than simulated. R001 draws one theorem per family
component, so the analyzed theorems are independent draws and the §9 statistic is a two-cell
comparison: how many informative groups land in the top-f block of the frozen q ranking. Under H0
(q unrelated to the outcome) that count is Hypergeometric(N, k, m) given k total positives, so the
one-sided critical value is exact - no bootstrap needed for the test itself (the family-component
bootstrap remains the reporting device for the confidence interval). Power is then a double
convolution of two Binomials, computed exactly.

Two things are deliberately NOT hidden:
  * The prospective prevalence pi is unknown - that is one of R001's objects - so every cell is
    reported for a RANGE of pi anchored on what D001 actually measured, including the source-mix
    prediction, which is the mechanism that could make the clean pool's prevalence LOWER than V1's.
  * The effect size E (true enrichment of the selected block) is a design assumption, so the
    headline output is the minimum detectable E at 80% power, not a power at an assumed E.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from functools import cache
from pathlib import Path

import numpy as np
from scipy.stats import binom, hypergeom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

D001_RESULTS = "experiments/manifests/v3/V3-D001_results.json"
POOL = "experiments/manifests/v3/V3-R001_family_clean_pool.json"
OUT_DEFAULT = "experiments/manifests/v3/V3-R001_power.json"

ALPHA = 0.05
TARGET_POWER = 0.80
Ns = [56, 86, 128, 192, 221]                 # 56/86 = literal readings, 128/192 = owner's §7
                                             # preference, 221 = consumed_only capacity
FS = [0.10, 0.20, 0.30, 0.50]                # top-f selection block (§9 reports 10/20/30)
ES = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]          # true enrichment of the selected block
CENSOR_FALLBACK = 0.04722                    # 34/720 groups infra-censored in V1 (D001 audit)


def load(rel: str):
    return json.loads((ROOT / rel).read_text())


@cache
def critical_values(N: int, m: int, alpha: float = ALPHA) -> tuple:
    """c[k] = smallest number of selected-set positives that rejects H0 at level alpha, given k
    total positives among N theorems. inf where no rejection is reachable. Depends only on (N, m,
    alpha), so it is cached and reused across the whole prevalence/effect grid."""
    c = np.full(N + 1, np.inf)
    for k in range(1, N + 1):
        lo, hi = max(0, k - (N - m)), min(k, m)
        xs = np.arange(lo, hi + 1)
        sf = hypergeom.sf(xs - 1, N, k, m)
        hit = np.flatnonzero(sf <= alpha)
        if hit.size:
            c[k] = xs[hit[0]]
    return tuple(c)


def exact_power(N: int, m: int, pi: float, e_true: float) -> dict:
    """P(reject) with k_sel ~ Bin(m, pi_sel), k_rest ~ Bin(N-m, pi_rest), independently.

    An enrichment of E at block fraction f is only a coherent hypothesis while the block can hold
    that many positives without emptying the rest of the pool: f*E*pi <= pi, i.e. E <= 1/f. Beyond
    that the pair (E, pi) is impossible rather than powerful, and saying so is the honest answer.
    """
    f = m / N
    pi_sel = min(0.99, e_true * pi)
    pi_rest_raw = (pi - f * pi_sel) / (1 - f)
    # tolerance: at E = 1 the two rates must be identical, and float rounding must not turn the
    # null hypothesis into an "impossible" effect (which would report power where there is none)
    feasible = -1e-12 <= pi_rest_raw <= pi_sel + 1e-12 and e_true > 0.0
    if not feasible:
        return {"power": None, "pi_sel": pi_sel, "pi_rest": None, "feasible": False,
                "why_infeasible": (f"enrichment {e_true} at block fraction {f:.2f} needs "
                                   f"prevalence*E*f = {f * pi_sel:.4f} > prevalence {pi}, so the "
                                   "unselected block would need a negative informative rate "
                                   "(the ceiling is E <= 1/f)"),
                "expected_positives_selected": m * pi_sel,
                "expected_positives_total": N * pi}
    return {"power": _reject_probability(N, m, pi, pi_sel, min(max(pi_rest_raw, 0.0), pi_sel)),
            "pi_sel": pi_sel, "pi_rest": pi_rest_raw, "feasible": True,
            "expected_positives_selected": m * pi_sel, "expected_positives_total": N * pi}


def _reject_probability(N: int, m: int, pi: float, pi_sel: float, pi_rest: float) -> float:
    c = np.asarray(critical_values(N, m))
    ks = np.arange(m + 1)
    kr = np.arange(N - m + 1)
    joint = binom.pmf(ks, m, pi_sel)[:, None] * binom.pmf(kr, N - m, pi_rest)[None, :]
    k_total = ks[:, None] + kr[None, :]
    reject = ks[:, None] >= c[k_total]
    return float(joint[reject].sum())


def min_detectable_e(N: int, m: int, pi: float, target: float = TARGET_POWER) -> float | str:
    """Smallest true enrichment with power >= target, searched on the exact power curve.

    Returns a string when no enrichment that is COHERENT with prevalence pi can reach the target,
    which is a statement about the design, not a numerical failure.
    """
    ceiling = 1.0 / (m / N)
    hi, found = 1.0, False
    while hi < ceiling:
        step = min(hi * 1.5, ceiling)
        p = exact_power(N, m, pi, step)["power"]
        if p is not None and p >= target:
            hi, found = step, True
            break
        if step >= ceiling:
            break
        hi = step
    if not found:
        at_ceiling = exact_power(N, m, pi, ceiling)["power"]
        return (f"unreachable: even the maximum coherent enrichment 1/f = {ceiling:.1f} "
                f"(every positive inside the block) gives power "
                f"{at_ceiling if at_ceiling is None else round(at_ceiling, 4)}")
    lo = 1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        p = exact_power(N, m, pi, mid)["power"]
        if p is not None and p >= target:
            hi = mid
        else:
            lo = mid
    return round(hi, 4)


def prevalence_anchors() -> dict:
    """Everything the power calc needs that D001 already measured, read from the frozen artifacts."""
    d001, pool = load(D001_RESULTS), load(POOL)
    src = d001["stationarity"]["source_prevalence"]
    per_source = {k: v["prevalence"] for k, v in src.items() if v["prevalence"] is not None}
    pred = {}
    for reading, p in pool["pools_by_reading"].items():
        mix = p["sources_of_one_per_component_eligible"]
        n = sum(mix.values())
        if not n:
            pred[reading] = {"n_candidates": 0}
            continue
        # V1's per-source informative rate applied to the clean pool's source mix. The rates are
        # measured on V1's own (multiplicity-weighted) realized theorems, so this is a prediction
        # with a stated bias, not an assumption-free estimate.
        est = sum(mix[s] / n * per_source[s] for s in mix if s in per_source)
        pred[reading] = {"n_candidates": n, "source_mix": mix,
                         "predicted_prevalence_from_V1_source_rates": round(est, 4)}
    return {"group_prevalence_V1": d001["dataset"]["prevalence"],
            "theorem_level_rate_note": ("0.1709 = unweighted mean over the 588 labelled statements "
                                        "of their informative-group rate (measured in the §5 pool "
                                        "build; group-level is 0.1516)"),
            "source_prevalence_V1": per_source,
            "infra_censoring_rate": CENSOR_FALLBACK,
            "d001_topk_enrichment": {"B2": {f"top{int(f*100)}": d001["B2_block18_PRIMARY"]
                                            [f"top{int(f*100)}_enrichment"] for f in (.1, .2, .3)},
                                     "B1": {f"top{int(f*100)}": d001["B1_handcrafted"]
                                            [f"top{int(f*100)}_enrichment"] for f in (.1, .2, .3)}},
            "d001_auprc": {"B2": d001["B2_block18_PRIMARY"]["auprc"],
                           "B1": d001["B1_handcrafted"]["auprc"],
                           "delta": d001["B2_block18_PRIMARY"]["delta_auprc_vs_B1"]},
            "predicted_clean_pool_prevalence": pred}


def analyze_prevalence(anch: dict) -> dict:
    pred = anch["predicted_clean_pool_prevalence"]
    vals = [v["predicted_prevalence_from_V1_source_rates"] for v in pred.values() if v.get("n_candidates")]
    return {"grid": [0.06, 0.09, 0.14, 0.17],
            "rationale": ("0.17 = D001's theorem-level informative rate, the most optimistic anchor; "
                          "0.14 = the predicted clean-pool rate under the consumed_only source mix; "
                          "0.09 = the predicted rate under the strict 56-family pool, whose mix is "
                          "half autoformalizer; 0.06 = a pessimistic rounding of that, since V1's "
                          "per-source rates come from theorems it chose to re-draw many times."),
            "predictions": pred, "predicted_range": [min(vals), max(vals)],
            "why_this_matters": (
                "Informative groups in V1 were overwhelmingly a synthetic-source phenomenon "
                "(synthetic 0.344, autoformalizer 0.030, human 0.022). The clean pool is NOT the same "
                "source mix, so the number of positives R001 can even hope to observe is set by the "
                "mix, and owner §11's source-concentration question is therefore also the power "
                "question - not only a robustness question.")}


def power_grid(anch: dict) -> dict:
    out = {}
    for N in Ns:
        per_n = {}
        for f in FS:
            m = max(1, round(f * N))
            cells = {}
            for pi in [0.06, 0.09, 0.14, 0.17]:
                mde = min_detectable_e(N, m, pi)
                powers = {}
                for e in ES:
                    r = exact_power(N, m, pi, e)
                    powers[str(e)] = (round(r["power"], 4) if r["power"] is not None
                                      else "impossible at this prevalence")
                cells[str(pi)] = {
                    "selected_block_size": m,
                    "expected_positives_in_block_at_this_prevalence": round(m * pi, 2),
                    "enrichment_ceiling_N_over_m": round(N / m, 2),
                    "realized_size_of_the_exact_test_under_H0": round(
                        exact_power(N, m, pi, 1.0)["power"], 4),
                    "min_detectable_enrichment_at_80pct_power": mde,
                    "power_by_true_enrichment": powers,
                    "expected_analyzable_N_after_censoring": round(N * (1 - CENSOR_FALLBACK), 1)}
            per_n[f"top{int(f*100)}"] = cells
        out[str(N)] = per_n
    return out


def delta_auprc_verdict(anch: dict) -> dict:
    """Can 'B2 > B1 significantly' be a GO requirement at any feasible N?

    Exact CI widths are what D001 measured for the SAME comparison at 433 family components:
    full-procedure mean +0.109, 95% CI [-0.025, +0.237] (half-width 0.131). The interval of a
    component-level bootstrap scales as 1/sqrt(n_components), so the achievable half-width at N
    one-per-family components is 0.131*sqrt(433/N). That is an approximation (it assumes the same
    between-component variance), and it is used only to answer a NO-GO-direction question: whether a
    requirement is *out of reach*, where a conservative approximation is enough.
    """
    a = anch["d001_auprc"]
    n0, half0, mean0 = 433, 0.131, 0.109
    out = {}
    for N in Ns:
        hw = half0 * (n0 / N) ** 0.5
        out[str(N)] = {"approx_ci_halfwidth_on_delta_auprc": round(hw, 4),
                       "would_exclude_zero_if_delta_equalled_D001_point_estimate":
                           bool(mean0 - hw > 0),
                       "d001_point_delta_would_need_to_be_at_least": round(hw, 4)}
    return {"d001_reference": {"n_components": n0, "mean_delta_auprc": mean0,
                               "ci95_halfwidth": half0, "auprc_B2": a["B2"], "auprc_B1": a["B1"]},
            "by_N": out,
            "conclusion": (
                "No feasible N gets the B2-vs-B1 delta anywhere near significance: D001 was already "
                "inconclusive at 433 components, and R001's whole pool is 56-221 families. So §14's "
                "optional clause resolves the documented way - the gate must NOT require B2 > B1 "
                "significance, because that requirement is unpowered by construction and would "
                "manufacture a NO-GO regardless of whether the controller works prospectively."),
            "what_B1_still_does": (
                "B1 stays a REPORTED comparator on the identical sample (§10, zero extra GPU): point "
                "estimates of AUPRC and top-20 enrichment for both arms, with the difference and its "
                "interval, and the honest reading that the interval covers zero.")}


def smallest_N_for_power(pi: float, e_true: float, f: float = 0.20,
                         target: float = TARGET_POWER) -> int | str:
    """How many family-clean theorems the gate actually needs, at this prevalence and effect."""
    for N in range(20, 601):
        m = max(1, round(f * N))
        if m >= N:
            continue
        r = exact_power(N, m, pi, e_true)
        if r["power"] is not None and r["power"] >= target:
            return N
    return f"not reached below 600 theorems at prevalence {pi}, enrichment {e_true}"


def design_verdicts(anch: dict) -> dict:
    """One verdict per POOL READING, at that reading's own predicted prevalence.

    This is where §5's capacity question and §14's gate question meet: a reading is usable only if
    the pool it leaves can detect an effect at least as large as the one D001 already measured.
    """
    e_ref = anch["d001_topk_enrichment"]["B2"]["top20"]
    out = {}
    for reading, pred in anch["predicted_clean_pool_prevalence"].items():
        n_pool = pred.get("n_candidates", 0)
        if not n_pool:
            out[reading] = {"pool_capacity": 0, "usable": False,
                            "reason": "this reading leaves no family-clean component at all"}
            continue
        pi = pred["predicted_prevalence_from_V1_source_rates"]
        m = max(1, round(0.20 * n_pool))
        p_full = exact_power(n_pool, m, pi, e_ref)
        need = smallest_N_for_power(pi, e_ref)
        out[reading] = {
            "pool_capacity": n_pool, "predicted_prevalence": pi,
            "top20_block_size_at_full_pool": m,
            "expected_positives_in_block": round(m * pi, 2),
            "min_detectable_enrichment_at_80pct_power": min_detectable_e(n_pool, m, pi),
            "power_at_D001_observed_top20_enrichment": (round(p_full["power"], 4)
                                                        if p_full["power"] is not None else None),
            "N_needed_for_80pct_power_at_that_effect": need,
            "usable": isinstance(need, int) and need <= n_pool,
            "note": ("'expected_positives_in_block' is the whole story: a block holding ~1 positive "
                      "cannot carry a test no matter how good the ranking is.")}
    return {"reference_effect_D001_top20_enrichment": e_ref, "by_reading": out}


def gate_recommendation(grid: dict, anch: dict) -> dict:
    """Which (N, block, prevalence) combinations can carry a GO gate at all."""
    rows = []
    for N, per_n in grid.items():
        for blk, cells in per_n.items():
            for pi, c in cells.items():
                mde = c["min_detectable_enrichment_at_80pct_power"]
                rows.append((int(N), blk, float(pi), mde if isinstance(mde, float) else None))
    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    d001_b2_top20 = anch["d001_topk_enrichment"]["B2"]["top20"]
    viable = [r for r in rows if r[3] is not None and r[3] <= d001_b2_top20]
    return {"d001_observed_B2_top20_enrichment": d001_b2_top20,
            "cells_where_D001_effect_size_would_be_detectable_at_80pct":
                [{"N": n, "block": b, "prevalence": p, "min_detectable_enrichment": round(e, 3)}
                 for n, b, p, e in viable],
            "binding_constraint": (
                "Power is set by the ABSOLUTE number of informative groups inside the selected block, "
                "which is N*pi*f. Nothing about the controller can fix a sample that contains only a "
                "handful of positives, which is why the pool-capacity question (owner §5 vs §7) is "
                "the real gate on the gate."),
            "discreteness_warning": (
                "The exact test's realized size at alpha=0.05 is NOT 0.05 at these sample sizes - it "
                "is the largest reachable level below it, so power is a step function of (N, m) and "
                "can be non-monotone in N. That is why every grid cell reports its own realized size, "
                "and why the gate must name N and the block fraction, not just 'p < 0.05': the "
                "preregistration has to fix the design whose actual level it is willing to defend."),
            "recommended_shape": {
                "primary": ("one-sided exact test (hypergeometric, alpha=0.05) on the top-20% block of "
                            "the frozen q ranking + reported Katz-log 95% CI on the enrichment ratio"),
                "requires": "prospective enrichment > 1 AND exact-test p < 0.05",
                "does_not_require": "B2 > B1 significance (unpowered, see delta_auprc_verdict)",
                "sample_size": ("N is only meaningful once the owner fixes the pool reading; at the "
                                "strict reading the design is a descriptive enrichment estimate, not a "
                                "powered test - and the prereg must say so rather than dress a "
                                "56-theorem sample as a hypothesis test.")}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()

    anch = prevalence_anchors()
    prev = analyze_prevalence(anch)
    grid = power_grid(anch)
    delta = delta_auprc_verdict(anch)
    designs = design_verdicts(anch)
    rec = gate_recommendation(grid, anch)

    payload = {
        "artifact_type": "v3_r001_power_analysis",
        "status": ("PRE-GATE POWER ANALYSIS - read-only, no rollout, no labels, no sample drawn; "
                   "written before the gate is frozen exactly as owner §14 requires"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorized_by": "owner directive 2026-09-24 §14 and §17 step 5",
        "host": {"hostname": os.uname().nodename,
                 "git_revision": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                                capture_output=True, text=True,
                                                check=False).stdout.strip()},
        "question": ("What sample size and what gate can this project actually power, given the pool "
                     "that survives §5?"),
        "method": {
            "test": ("H0: the selected top-f block holds no more informative theorems than the rest. "
                     "Given k informative theorems among N, the count inside the frozen block of size "
                     "m is Hypergeometric(N, k, m); rejection uses the exact one-sided critical value "
                     "at alpha=0.05."),
            "power": ("EXACT, not simulated: sum over (k_sel, k_rest) of independent Binomials "
                      "Bin(m, E*pi) and Bin(N-m, pi_rest), pi_rest chosen so the overall rate is pi."),
            "validity_condition": ("one theorem per family component makes the analyzed units "
                                   "independent, so no family-cluster correction is needed for the "
                                   "TEST; the bootstrap CI over components remains for reporting. If "
                                   "the owner instead approves >1 theorem per family, this exactness "
                                   "is lost and the CI must be component-clustered."),
            "why_not_auc_based": ("AUC/AUPRC power needs a model of the whole score distribution and "
                                  "would be false precision here; the gate is a decision about a "
                                  "selected block, so the block test is the quantity to power.")},
        "prevalence_anchors": {**anch, "prevalence_grid": prev},
        "power_grid_by_N_and_block": grid,
        "design_verdicts_by_pool_reading": designs,
        "delta_auprc_verdict": delta,
        "gate_recommendation": rec,
        "inputs": {"d001_results": D001_RESULTS, "pool": POOL},
        "prohibitions_respected": [
            "no rollout, no generation, no verifier, no GPU, no RL optimizer step",
            "no GO criterion chosen after seeing any R001 outcome - none exists yet",
            ("no new controller task, no MLP, no new layer, no mean pooling (this file touches no "
             "model)"),
            "no frozen artifact modified; D001 and V2 outputs are read-only inputs",
        ],
    }
    path = ROOT / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))

    print(json.dumps({
        "anchors": {k: anch[k] for k in ("group_prevalence_V1", "source_prevalence_V1",
                                         "d001_topk_enrichment", "d001_auprc")},
        "predicted_clean_pool_prevalence": anch["predicted_clean_pool_prevalence"],
        "min_detectable_enrichment_top20": {
            N: {pi: grid[N]["top20"][pi]["min_detectable_enrichment_at_80pct_power"]
                for pi in grid[N]["top20"]} for N in grid},
        "power_at_D001_top20_effect_3_47": {
            N: {pi: grid[N]["top20"][pi]["power_by_true_enrichment"].get("3.5")
                for pi in grid[N]["top20"]} for N in grid},
        "design_verdicts": designs["by_reading"],
        "delta_auprc": delta["by_N"], "recommended_shape": rec["recommended_shape"]}, indent=2))
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
