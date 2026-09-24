#!/usr/bin/env python3
"""V3-R001 §17 step 6 — the FROZEN gate: exact boundaries, thresholds and outcome taxonomy.

Owner decisions 2026-09-24 §7-§12 freeze the design (contamination reading `consumed_only`,
N_nominal = 128, top fraction 20%) and require that the gate be written in terms of the design's
REAL attainable level rather than a nominal 0.05 that a discrete exact test cannot reach. This
script turns that decision into a machine-readable artifact, so the preregistration document and the
post-outcome analysis read the same numbers instead of two prose versions of them.

Nothing here sees an outcome. The inputs are the frozen pool, the frozen sample and the frozen
per-source prevalence anchors from §14's power analysis. No rollout, no GPU, no verifier, no label.

Why the test is exact: R001 draws one theorem per family component, so the analyzed units are
independent and the primary statistic is a two-cell count -- how many of the k informative theorems
land in the top-m block. Under H0 that count is Hypergeometric(N, k, m) given k, so the one-sided
critical value is exact and conditional on the OBSERVED k. The critical function c[k] is a frozen
monotone computation, not a number chosen after the fact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from functools import cache
from pathlib import Path

import numpy as np
from scipy.stats import hypergeom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import v3_r001_power as PW  # the frozen exact-power machinery, imported not re-derived

POOL = "experiments/manifests/v3/V3-R001_family_clean_pool.json"
PREDICTIONS = "experiments/manifests/v3/V3-R001_predictions.json"
FORMAL = "experiments/manifests/v3/v3_r001_formal_sample.json"
RESERVE = "experiments/manifests/v3/v3_final_holdout_reserve.json"
POWER = "experiments/manifests/v3/V3-R001_power.json"
OUT_DEFAULT = "experiments/manifests/v3/V3-R001_gate.json"

# ---- owner-decided design constants (2026-09-24 decisions 3, 4, 8, 9) --------------------------
READING = "consumed_only"                         # decision 1: contamination definition
N_NOMINAL = 128                                   # decision 4: N=192 rejected
TOP_FRACTION = 0.20                               # decisions 8/9: the confirmatory block
ALPHA = 0.05                                      # the nominal level of the exact test
ENRICHMENT_FLOOR = 1.5                            # P2 / S2 point-estimate floor
CI_LOWER_BOUND = 1.0                              # P2 / S2 bootstrap CI lower bound must exceed
BOOT_N_REP = 10_000                               # D001's frozen component-bootstrap setting
BOOT_SEED = 20260924                              # the directive date, as with the draw seed
MIN_POSITIVES = 5                                 # decision 7-D: below this -> INCONCLUSIVE-BY-DATA
MIN_ANALYZED_FRACTION = 0.80                      # decision 7-D: censoring beyond 20% -> same
LAMBDA_GRID = (0.5, 0.8)                          # decision 12: diagnostics only, never selected on

OUTCOME_TAXONOMY = {
    "A_GO-SEMANTIC": ("pooled gate PASS and within-synthetic gate PASS -- the controller retains "
                      "prospective ranking signal inside the dominant source stratum, so the pooled "
                      "enrichment is not merely source routing. The only outcome under which the "
                      "owner may consider V3-R002 decision-guided RL."),
    "B_SOURCE-DRIVEN-ONLY": ("pooled gate PASS and within-synthetic gate FAIL -- prospective "
                             "enrichment exists but current evidence cannot separate it from "
                             "source-level routing. A valid result; no expensive RL intervention."),
    "C_NO-GO": ("pooled gate FAIL -- the controller did not enrich informative groups on unseen "
                "families prospectively. The Jev-inspired RL intervention line stops."),
    "D_INCONCLUSIVE-BY-DATA": ("the sample could not carry the test: analyzed positives below "
                               f"{MIN_POSITIVES}, or fewer than {MIN_ANALYZED_FRACTION:.0%} of "
                               "N_nominal survived the verifier. NOT a NO-GO, and not a licence to "
                               "switch to another candidate sample."),
}


def load(rel: str):
    return json.loads((ROOT / rel).read_text())


def sha(*parts) -> str:
    """Byte-identical to v3_r001_prescore.sha -- the hash of a frozen object must not depend on
    which script computed it (pinned by test_the_two_freeze_scripts_hash_identically)."""
    h = hashlib.sha256()
    for p in parts:
        h.update(json.dumps(p, sort_keys=True, separators=(",", ":"), default=str).encode())
    return h.hexdigest()


def block_size(n: int) -> int:
    """The frozen top-f block size for a sample of n analyzed theorems."""
    return round(TOP_FRACTION * n)


def exact_p(N: int, m: int, k: int, x: int) -> float:
    """One-sided hypergeometric upper tail: P(X >= x | N, k successes, m drawn)."""
    if x <= 0:
        return 1.0
    lo, hi = max(0, k - (N - m)), min(k, m)
    if not lo <= x <= hi:
        return 0.0 if x > hi else 1.0
    return float(hypergeom.sf(x - 1, N, k, m))


@cache
def critical_function(N: int, m: int, alpha: float = ALPHA) -> tuple:
    """c[k] = smallest x whose exact one-sided p is <= alpha, given k total positives (inf: none)."""
    return PW.critical_values(N, m, alpha)


def attainable_alpha(N: int, m: int, pi: float) -> float:
    """P(reject) of the frozen conditional rule UNDER H0 at prevalence pi, mixing over k ~ Bin(N, pi).

    This is the honest size of the test: discreteness puts it below ALPHA, and it is a step function
    of (N, m, pi). Reporting a nominal 0.05 for a rule that never exceeds 0.03 would overstate what
    the gate's p-value means.
    """
    c = critical_function(N, m)
    tot = 0.0
    for k in range(1, N + 1):
        if not math.isfinite(c[k]):
            continue
        tot += float(PW.binom.pmf(k, N, pi)) * float(hypergeom.sf(c[k] - 1, N, k, m))
    return tot


def boundary_table(N: int, m: int, k_min: int = 2, k_max: int = 40) -> list[dict]:
    """The frozen rejection boundary, one row per possible observed total k."""
    c = critical_function(N, m)
    rows = []
    for k in range(k_min, min(k_max, N) + 1):
        cv = c[k]
        rows.append({
            "k_total_positives": k,
            "reject_if_block_positives_at_least": (int(cv) if math.isfinite(cv) else None),
            "implied_enrichment_at_the_boundary": (round((cv / m) / (k / N), 3)
                                                   if math.isfinite(cv) else None),
            "exact_p_at_the_boundary": round(exact_p(N, m, k, int(cv)), 5) if math.isfinite(cv) else None,
            "conditional_level_at_this_k": (round(float(hypergeom.sf(cv - 1, N, k, m)), 5)
                                            if math.isfinite(cv) else None),
        })
    return rows


def expected_positives_by_source(component_records: list[dict], src_pi: dict) -> dict:
    """The §14 prevalence prediction, recomputed from the sample's OWN source labels."""
    n = len(component_records)
    mix = {s: sum(1 for r in component_records if r["source"] == s) for s in sorted(src_pi)}
    pi = sum(mix[s] / n * src_pi[s] for s in mix)
    return {"n": n, "source_counts": mix, "predicted_prevalence_pi": round(pi, 4),
            "expected_positives": round(pi * n, 2),
            "how_computed": ("V1's per-source informative rate applied to this sample's own source "
                             "mix; a DESIGN anchor for power, not a measurement of R001")}


def mixture_policy(q: list[float], lam: float) -> dict:
    """Owner decision 12: P_lam(i) = (1-lam)/N + lam * q_i / sum(q), diagnostics only.

    Every entry is strictly positive whenever q >= 0 and lam <= 1, which is the property that the
    old epsilon=0 sampler destroyed. It is asserted here, not assumed.
    """
    a = np.asarray(q, dtype=float)
    if a.size == 0 or not np.all(np.isfinite(a)) or np.any(a < 0):
        raise SystemExit("FATAL: mixture policy needs a non-empty finite non-negative q vector")
    tot = float(a.sum())
    if tot <= 0.0:
        raise SystemExit("FATAL: all q are zero, so the mixture policy is undefined -- refuse to "
                         "report an ESS/entropy for a policy that does not exist")
    p = (1.0 - lam) / a.size + lam * a / tot
    if np.any(p <= 0.0):
        raise SystemExit("FATAL: a mixture policy put P(i) at or below zero (the epsilon=0 failure "
                         "the owner permanently abandoned)")
    if not math.isclose(float(p.sum()), 1.0, abs_tol=1e-12):
        raise SystemExit(f"FATAL: mixture policy does not normalize (sum={float(p.sum())})")
    nz = p[p > 0]
    return {"lambda": lam, "n": int(a.size),
            "min_probability": float(p.min()), "max_probability": float(p.max()),
            "expected_IGR": None,
            "expected_IGR_note": ("= sum_i P(i) * y_i, computable only once the rollout labels "
                                  "exist; None here because no outcome exists yet"),
            "effective_sample_size": round(float(1.0 / (p ** 2).sum()), 3),
            "entropy_bits": round(float(-(nz * np.log2(nz)).sum()), 4),
            "uniform_ess": float(a.size), "policy_weights_sha256": sha(list(np.round(p, 12)))}


def pooled_gate(pooled: dict, synth: dict) -> dict:
    """Owner decisions 7-9 as an executable rule, so the taxonomy cannot be argued later."""
    g = {"P1_exact_test": pooled["block_positives"] >= pooled["reject_if_at_least"],
         "P2_enrichment_and_CI": (pooled["enrichment"] >= ENRICHMENT_FLOOR
                                  and pooled["enrichment_ci_lower"] > CI_LOWER_BOUND),
         "S1_exact_test_within_synthetic": synth["block_positives"] >= synth["reject_if_at_least"],
         "S2_enrichment_and_CI_within_synthetic": (synth["enrichment"] >= ENRICHMENT_FLOOR
                                                   and synth["enrichment_ci_lower"] > CI_LOWER_BOUND)}
    ident = (pooled["analyzed_positives"] >= MIN_POSITIVES
             and pooled["n_analyzed"] >= MIN_ANALYZED_FRACTION * pooled["n_nominal"])
    if not ident:
        return {**g, "outcome": "D_INCONCLUSIVE-BY-DATA",
                "why": (f"n_analyzed={pooled['n_analyzed']} vs N_nominal={pooled['n_nominal']}, "
                        f"analyzed positives={pooled['analyzed_positives']}"),
                        "r002_eligible": False}
    if g["P1_exact_test"] and g["P2_enrichment_and_CI"]:
        if synth["identifiable"] and g["S1_exact_test_within_synthetic"] \
                and g["S2_enrichment_and_CI_within_synthetic"]:
            out = "A_GO-SEMANTIC"
        elif not synth["identifiable"]:
            out = "D_INCONCLUSIVE-BY-DATA"
        else:
            out = "B_SOURCE-DRIVEN-ONLY"
    else:
        out = "C_NO-GO"
    return {**g, "outcome": out,
            "why": "see OUTCOME_TAXONOMY in this artifact for what each branch licenses",
            "r002_eligible": out == "A_GO-SEMANTIC"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()

    pool, pred, formal, power = load(POOL), load(PREDICTIONS), load(FORMAL), load(POWER)
    reserve = load(RESERVE)
    src_pi = {k: float(v) for k, v in
              power["prevalence_anchors"]["source_prevalence_V1"].items()}

    # the gate refuses to exist over a sample that touches the sealed reserve (owner decision 13)
    sealed = {c["component_id"] for c in reserve["components"]}
    touched = sorted({t["component_id"] for t in formal["theorems"]} & sealed)
    if touched:
        raise SystemExit(f"FATAL: the formal sample contains {len(touched)} SEALED reserve "
                         f"components ({touched[:3]}); the allocation is broken")
    if formal["sample_sha256"] != pred["prospective_samples"][formal["predictions_sample_key"]][
            "sample_sha256"]:
        raise SystemExit("FATAL: the formal sample hash does not match the hash committed with the "
                         "frozen predictions -- this is not the sample §14's power ran on")
    pool_comps = set(pool["pools_by_reading"][READING]["candidate_component_ids"])
    formal_comps = {t["component_id"] for t in formal["theorems"]}
    if formal_comps | sealed != pool_comps or len(formal_comps) + len(sealed) != len(pool_comps):
        raise SystemExit(f"FATAL: the formal sample and the sealed reserve are not an exact "
                         f"partition of the {len(pool_comps)}-component {READING} pool")

    if formal["reading"] != READING or formal["N_nominal"] != N_NOMINAL:
        raise SystemExit(f"FATAL: the formal sample manifest is {formal['reading']}|N="
                         f"{formal['N_nominal']}, but the gate is frozen for "
                         f"{READING}|N={N_NOMINAL}")
    recs = formal["theorems"]
    m = block_size(N_NOMINAL)
    if formal["top20pct_block"]["size"] != m:
        raise SystemExit(f"FATAL: formal manifest block size {formal['top20pct_block']['size']} "
                         f"!= frozen round({TOP_FRACTION}*{N_NOMINAL}) = {m}")

    exp = expected_positives_by_source(recs, src_pi)
    pi = exp["predicted_prevalence_pi"]
    k_hat = round(pi * N_NOMINAL)
    c = critical_function(N_NOMINAL, m)
    if not math.isfinite(c[k_hat]):
        raise SystemExit(f"FATAL: no rejection region at the design anchor k={k_hat}")

    synth = [r for r in recs if r["source"] == "synthetic"]
    ms = block_size(len(synth))
    pis = src_pi["synthetic"]
    ks = round(pis * len(synth))
    cs = critical_function(len(synth), ms)

    q = [r["q_B2_controller"] for r in recs]
    diag = {f"lambda={lam}": mixture_policy(q, lam) for lam in LAMBDA_GRID}

    pooled_cell = {
        "test": ("exact one-sided conditional Hypergeometric(N_analyzed, k_observed, m): reject iff "
                 "p <= 0.05, i.e. iff the block count reaches the frozen critical function"),
        "N_nominal": N_NOMINAL, "m_top20": m, "pi_design_anchor": pi,
        "k_at_the_design_anchor": k_hat,
        "reject_if_block_positives_at_least": int(c[k_hat]),
        "implied_enrichment_at_the_boundary": round((c[k_hat] / m) / (k_hat / N_NOMINAL), 3),
        "attainable_alpha_unconditional_at_pi": round(attainable_alpha(N_NOMINAL, m, pi), 4),
        "conditional_level_at_the_anchor_k": round(
            float(hypergeom.sf(c[k_hat] - 1, N_NOMINAL, k_hat, m)), 4),
        "nominal_alpha": ALPHA,
        "min_detectable_enrichment_at_80pct_power": PW.min_detectable_e(N_NOMINAL, m, pi, 0.8),
        "expected_positives_in_block_under_the_alternative": round(
            m * min(0.99, pi * float(PW.min_detectable_e(N_NOMINAL, m, pi, 0.8))), 2)
            if isinstance(PW.min_detectable_e(N_NOMINAL, m, pi, 0.8), float) else None,
        "boundary_table": boundary_table(N_NOMINAL, m),
        "expected": exp,
        "recomputation_rule": ("the test is evaluated on the ANALYZED set: N_analyzed and "
                               "m = round(0.20 * N_analyzed) enter the SAME frozen functions, so "
                               "infra-censoring shrinks the design without changing the rule"),
    }
    synth_cell = {
        "stratum": "synthetic", "n": len(synth), "m_top20": ms,
        "pi_from_V1_synthetic_rate": pis, "k_at_the_design_anchor": ks,
        "reject_if_block_positives_at_least": (int(cs[ks]) if math.isfinite(cs[ks]) else None),
        "implied_enrichment_at_the_boundary": (round((cs[ks] / ms) / (ks / len(synth)), 3)
                                               if math.isfinite(cs[ks]) else None),
        "attainable_alpha_unconditional_at_pi": round(attainable_alpha(len(synth), ms, pis), 4),
        "conditional_level_at_the_anchor_k": (round(float(hypergeom.sf(cs[ks] - 1, len(synth), ks, ms)), 4)
                                              if math.isfinite(cs[ks]) else None),
        "min_detectable_enrichment_at_80pct_power": PW.min_detectable_e(len(synth), ms, pis, 0.8),
        "identifiable": bool(math.isfinite(cs[ks])),
        "ranking_rule": ("rank WITHIN the synthetic stratum by the frozen q_B2; the source label "
                         "itself never enters the ranking (owner decision 9)"),
        "if_unidentifiable": ("INCONCLUSIVE-BY-DATA for this co-primary, never a FAIL and never a "
                              "silent fallback to the pooled result (owner decision 9)"),
        "expected": expected_positives_by_source(synth, src_pi),
        "pre_outcome_confounding_evidence": pred["pre_outcome_source_concentration"][
            "top20pct_block_source_mix_of_union"],
    }
    never = {k: {"status": "DESIGN-ONLY / NEVER ANALYZE", "sample_sha256": v["sample_sha256"],
                 "N": v["N"]}
             for k, v in sorted(pred["prospective_samples"].items())
             if k != formal["predictions_sample_key"]}

    result = {
        "artifact_type": "v3_r001_frozen_gate",
        "status": ("FROZEN_BEFORE_ANY_OUTCOME — the gate, its boundaries, its attainable level and "
                   "the outcome taxonomy are committed before a single R001 rollout exists"),
        "provenance": ("bytes are a pure function of the frozen pool, predictions, formal sample and "
                       "power artifacts; no wall-clock, host or git field, so the sha256 pinned in "
                       "registry.yaml is re-checkable on any host by re-running"),
        "authorized_by": ("owner decisions 2026-09-24 §3-§12 on top of directive §14 and §17 step 6"),
        "question_pooled": ("Does the frozen D001 controller, scored before any outcome, enrich "
                            "reward-informative groups in the top 20% of previously unseen families?"),
        "question_within_synthetic": ("Or is the pooled effect mainly source routing -- does the "
                                      "controller retain enrichment WITHIN the synthetic stratum? "
                                      "(co-primary, owner decision 6)"),
        "design": {"reading": READING, "N_nominal": N_NOMINAL, "top_fraction": TOP_FRACTION,
                   "top_block_size": m, "one_theorem_per_component": True,
                   "formal_sample_sha256": formal["sample_sha256"],
                   "formal_sample_manifest": FORMAL},
        "pooled_gate": {"P1": pooled_cell, "P2": {
            "statistic": ("enrichment = (informative among the top-20% block) / "
                          "(informative among the analyzed sample)"),
            "point_estimate_floor": ENRICHMENT_FLOOR,
            "interval": (f"{BOOT_N_REP} reps, resample family components with replacement, "
                         f"seed {BOOT_SEED}, 2.5/97.5 percentile"),
            "require_ci_lower_bound_above": CI_LOWER_BOUND,
            "pass_rule": "P1 AND P2 (both required; owner decision 8)"},
        },
        "within_synthetic_gate": {"S1": synth_cell, "S2": {
            "point_estimate_floor": ENRICHMENT_FLOOR,
            "interval": ("same component bootstrap restricted to the synthetic stratum; with one "
                         "theorem per component the component and theorem resamples coincide, which "
                         "is stated rather than left to be discovered"),
            "require_ci_lower_bound_above": CI_LOWER_BOUND,
            "pass_rule": "S1 AND S2"},
        },
        "co_primary_note": ("Outcome A requires the pooled AND the within-synthetic gate. The "
                            "stratification is not a robustness appendix: the pre-outcome "
                            "diagnostic shows both arms' top-20% block of the whole candidate union "
                            "is 50/50 synthetic, so a pooled pass alone cannot license a "
                            "beyond-source claim."),
        "not_a_gate": {"b2_vs_b1_significance": {
            "status": "NOT A GATE",
            "reported_instead": ("AUPRC_B2, AUPRC_B1, AUPRC_prevalence and the B2-B1 delta with its "
                                 "component-bootstrap CI, on the identical sample at zero extra GPU"),
            "reason": power["delta_auprc_verdict"]["conclusion"],
            "interpretation_rule": ("if B1 ~= B2 prospectively that is NOT read as a controller "
                                    "failure; the reading turns on the pooled vs within-synthetic "
                                    "contrast (owner decision 11)")}},
        "inconclusive_by_data": {"min_analyzed_positives": MIN_POSITIVES,
                                 "min_analyzed_fraction_of_N_nominal": MIN_ANALYZED_FRACTION,
                                 "censoring_rule": ("an infra-censored group stays missing/excluded "
                                                    "and is NOT counted all-fail (owner decision 5); "
                                                    "D001's measured censoring rate was "
                                                    f"{PW.CENSOR_FALLBACK:.5f}, so ~122 of 128 "
                                                    "analyze"),
                                 "never_permitted": ("switching to another candidate sample, "
                                                     "re-drawing, re-ranking, re-fitting or "
                                                     "re-defining the label rule")},
        "sampler_diagnostics": {
            "purpose": ("offline estimand diagnostics ONLY (owner decision 12): no two-arm sampling "
                        "happens in R001 and no lambda is selected on R001 results"),
            "formula": "P_lambda(i) = (1-lambda)/N + lambda * q_i / sum(q)",
            "positivity_invariant": "P_lambda(i) > 0 for every eligible theorem, asserted in code",
            "epsilon_zero": "PERMANENTLY ABANDONED (owner directive §13)",
            "metrics_reported": ["expected_IGR", "effective_sample_size", "entropy_bits",
                                 "max_probability", "min_probability"],
            "lambda_grid": list(LAMBDA_GRID), "values": diag},
        "candidate_samples_never_to_analyze": never,
        "outcome_taxonomy": OUTCOME_TAXONOMY,
        "prohibitions_respected": [
            "no rollout, no generation, no verifier call, no label read",
            "no RL optimizer step, no gradient, no training",
            "the 10 GB training smoke remains NOT AUTHORIZED and was not run",
            "no V2 frozen artifact modified; the Track C release is recorded as V3 provenance only",
            "no gate component chosen after seeing an outcome; every threshold is owner-decided",
        ],
        "inputs_read_only": [POOL, PREDICTIONS, FORMAL, POWER],
    }
    outp = ROOT / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(result, indent=2))
    print(json.dumps({"status": result["status"],
                      "pooled": {k: pooled_cell[k] for k in
                                 ("m_top20", "pi_design_anchor", "k_at_the_design_anchor",
                                  "reject_if_block_positives_at_least",
                                  "implied_enrichment_at_the_boundary",
                                  "attainable_alpha_unconditional_at_pi",
                                  "min_detectable_enrichment_at_80pct_power")},
                      "within_synthetic": {k: synth_cell[k] for k in
                                           ("n", "m_top20", "reject_if_block_positives_at_least",
                                            "implied_enrichment_at_the_boundary",
                                            "attainable_alpha_unconditional_at_pi",
                                            "min_detectable_enrichment_at_80pct_power")},
                      "mixture": {k: {kk: vv for kk, vv in v.items()
                                      if kk in ("effective_sample_size", "entropy_bits",
                                                "min_probability", "max_probability")}
                                  for k, v in diag.items()}}, indent=2))
    print("wrote", outp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
