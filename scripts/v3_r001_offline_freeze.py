#!/usr/bin/env python3
"""[ARCHIVED] V3-R001 training-draft support — OFFLINE controller freeze, sampler diagnostics, power.

PROVENANCE ONLY. This script belongs to the *archived* V3-R001 RL-training draft
(docs/v3/archive/V3-R001-training-draft_SUPERSEDED.md), which the owner did NOT approve on
2026-09-24 (§2). Its logic is kept unmodified so that
experiments/manifests/v3/archive/V3-R001-training-draft_offline_freeze.json stays re-derivable.
Two of its outputs are permanently abandoned and must not be reused:
  * the (alpha, epsilon) sampler freeze - epsilon* = 0 gives zero sampling probability to theorems
    with q = 0 (the artifact's own starved_statement_fraction = 0.02041) and leaves q undefined for
    unlabelled pool rows; owner §13 abandons it in favour of a uniform-mixture policy.
  * every ROW-LEVEL diagnostic - the pool here is the on-disk parquet (24,418 rows / 7,620
    statements), but the real sampled population is the 1024-token-filtered one (24,246 rows /
    7,613 statements; 172 rows dropped, 7 statements with every row dropped). Verified in the
    deployment-pool audit of 2026-09-24; see docs/v3/V3-R001_deployment_pool_audit.md.

Everything below is computed from V1 history and the frozen V3-D001 offline probe. It launches no
training, generates no rollouts and touches no RL code path. Its purpose is to let the V3-R001
protocol be FROZEN before any RL outcome exists, as the owner directive requires:

  1. deployment controller : the frozen B2 recipe (theta0 block-18 last-token rep + step_norm ->
                             L2 logistic) refit ONCE on all 686 V1 seed1/2/3 valid groups, C chosen
                             by the identical inner family-grouped AUPRC criterion  ->  q_i(t)
  2. OOF q surface         : honest generalisation estimates from the frozen nested CV
  3. (alpha, epsilon) freeze: grid search over the stochastic sampler w = eps + q^alpha using OFFLINE
                             criteria only (multiplicity-preserving OOF-predicted treated IGR subject to
                             anti-collapse constraints expressed RELATIVE TO THE CONTROL ARM's own
                             row-level distribution). RL outcomes are never used.
  4. power analysis        : from the three historical per-step IGR series -> MDE per horizon and a
                             preregistered horizon proposal.

Diagnostics are reported at two levels, because the prompt pool lists a statement once per row (mean
multiplicity ~3.2) and the V1 shuffled loader therefore sampled theorems PROPORTIONAL TO ROW
MULTIPLICITY:
  * theorem-level : each scored theorem counts once (what the controller sees)
  * row-level     : each theorem weighted by its pool multiplicity (what the sampler realises)
The freeze selects on the row-level objective, since that is the deployment measure.

Output: experiments/manifests/v3/archive/V3-R001-training-draft_offline_freeze.json
        (+ runs/v3_r001/oof_cache.npz) -- the committed copy was moved to archive/ on 2026-09-24,
        and OUT below points at that new location; the JSON *contents* are unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import v3_d001_lib as LIB
import v3_d001_run as R
from v3_d001_lib import (
    average_precision,
    component_bootstrap,
    fit_logistic,
    predict_proba_logistic,
    standardize_apply,
    standardize_fit,
    stratified_group_kfold,
)

FOLDS = "experiments/manifests/v3/v3_d001_folds.json"
COMMITTED = "experiments/manifests/v3/V3-D001_results.json"
OUT = "experiments/manifests/v3/archive/V3-R001-training-draft_offline_freeze.json"
RL_POOL_RAW = "data/raw/kimina_promptset/data/train-00000-of-00001.parquet"
RL_POOL_PROCESSED = "data/processed/p3_promptset/prompt_sets/AI-MO/Kimina-Prover-Promptset/train.parquet"
DYNAMICS = {"seed1": "experiments/results/e019_dynamics.json",
            "seed2": "experiments/results/e020_seed2_dynamics.json",
            "seed3": "experiments/results/e022_seed3_dynamics.json"}
REPS = "runs/v3_d001/theta0_reps.npz"

SEED = R.SEED
N_INNER = 4
B_THEOREMS_PER_STEP = 4                    # V1 data.train_batch_size (prompts per step)
N_CANDIDATES = LIB.N                       # 8, unchanged
V1_TOTAL_STEPS = LIB.STEPS                 # 60
ALPHA_GRID = [0.25, 0.5, 1.0, 2.0, 4.0]
EPS_GRID = [0.0, 0.01, 0.025, 0.05, 0.10, 0.20, 0.40]
STEP_NORM_EVAL = [round(s / V1_TOTAL_STEPS, 4) for s in (5, 10, 15, 20, 25, 30)]
UPLIFT_SCENARIOS_PP = [5.0, 10.0, 15.0, 20.0, 26.0]
HORIZONS = [15, 20, 25, 30, 40, 60]
CONSTRAINTS = {
    "min_ESS_fraction_of_pool": 0.03,      # sampler behaves like an equal-weight pool on >=3% of it
    "min_normalized_entropy": 0.85,        # H / log K
    "max_prob_over_control": 2.0,          # most-sampled theorem <= 2x its CONTROL-arm probability.
                                           # An absolute cap vs statement-uniform is the wrong yardstick:
                                           # pool multiplicity (1..54) already puts the UNIFORM control arm
                                           # at ~4.8x statement-uniform, so an absolute cap measures the
                                           # pool's copy structure, not controller-induced concentration.
    "min_family_coverage_ratio": 0.90,     # expected distinct components per batch >= 0.9 x control
    "min_trial_family_coverage_ratio": 0.80,  # distinct components EVER drawn across the whole trial
                                              # (TRIAL_DRAWS prompts) >= 0.8 x control
}
TRIAL_DRAWS = 100                          # proposed H=25 steps x 4 prompts/step; frozen with H


def sampler_diagnostics(w, group_of, multiplicity, batch: int, n_draws: int = 0) -> dict:
    """Multinomial (with-replacement) sampler diagnostics from unnormalised weights."""
    ww = np.asarray(w, dtype=float)
    if multiplicity is not None:
        ww = ww * np.asarray(multiplicity, dtype=float)
    p = ww / ww.sum()
    keep = p > 0
    K = len(p)
    ent = float(-(p[keep] * np.log(p[keep])).sum())
    uniq_g, inv = np.unique(group_of, return_inverse=True)
    pg = np.bincount(inv, weights=p, minlength=len(uniq_g))
    out = {"K_units": K, "n_groups": len(uniq_g),
           "ESS": round(1.0 / float((p ** 2).sum()), 3),
           "ESS_fraction_of_pool": round(1.0 / float((p ** 2).sum()) / K, 5),
           "entropy_nats": round(ent, 5),
           "normalized_entropy": round(ent / math.log(K), 5) if K > 1 else None,
           "max_sampling_probability": round(float(p.max()), 8),
           "max_prob_over_uniform": round(float(p.max() * K), 4),
           "top10_probability_mass": round(float(np.sort(p)[::-1][:10].sum()), 5),
           "expected_distinct_groups_per_batch": round(
               float((1.0 - np.power(1.0 - pg, batch)).sum()), 4),
           "expected_distinct_units_per_batch": round(
               float((1.0 - np.power(1.0 - p, batch)).sum()), 4)}
    if n_draws:
        # per-batch coverage barely moves at 4 draws/step; the quantity that matters for a
        # whole trial is how many family components are EVER drawn in H x batch prompts.
        out["expected_distinct_groups_over_trial"] = round(
            float((1.0 - np.power(1.0 - pg, n_draws)).sum()), 4)
        out["fraction_of_groups_never_drawn_over_trial"] = round(
            float(np.power(1.0 - pg, n_draws).mean()), 5)
        out["min_group_probability"] = round(float(pg.min()), 8)
    return out


def wmean(w, v) -> float:
    w = np.asarray(w, dtype=float)
    return float((w * np.asarray(v, dtype=float)).sum() / w.sum())


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def step_cluster_power(step_series: list, n_steps: int, uplift: float, one_sided: bool = True) -> dict:
    """Normal approximation on the per-step (cluster) IGR spread taken from history."""
    s = np.concatenate([np.asarray(x, dtype=float) for x in step_series])
    se = math.sqrt(2.0) * float(s.std(ddof=1)) / math.sqrt(n_steps)
    z_a = 1.6449 if one_sided else 1.9600
    return {"se_of_difference_in_mean_IGR": round(se, 5),
            "minimum_detectable_uplift_80pct_power": round((z_a + 0.8416) * se, 5),
            "power_at_uplift": round(float(normal_cdf(uplift / se - z_a)), 4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap-reps", type=int, default=4000)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    t0 = time.time()

    built = LIB.build_records(ROOT, R.DATASET, R.REGISTRY)
    records, prompts = built["valid"], built["prompts"]
    folds = json.loads((ROOT / FOLDS).read_text())
    committed = json.loads((ROOT / COMMITTED).read_text())
    z = np.load(ROOT / REPS, allow_pickle=True)
    lp = [int(x) for x in z["layers"]]
    reps_map = {s: (int(z["token_lens"][i]), {L: z["reps"][i, lp.index(L)] for L in R.LAYERS})
                for i, s in enumerate(z["statement_ids"])}
    y, comps, seeds, Xb1, Xb2 = R.build_designs(records, prompts, reps_map)
    L = R.PRIMARY_LAYER
    outer = np.array([folds["component_outer_fold"][c] for c in comps], dtype=int)
    oofB2, _ = R.nested_oof(Xb2[L], y, comps, outer, folds["inner_fold_by_outer"], int(folds["n_outer"]))
    oofB1, _ = R.nested_oof(Xb1, y, comps, outer, folds["inner_fold_by_outer"], int(folds["n_outer"]))

    # ---- pool multiplicity (what the V1 shuffled loader actually sampled) --------------
    counts: dict = {}
    raw = pq.read_table(ROOT / RL_POOL_RAW, columns=["statement_id"])
    for s in raw.column("statement_id").to_pylist():
        counts[s] = counts.get(s, 0) + 1
    pc: dict = {}
    proc = pq.read_table(ROOT / RL_POOL_PROCESSED, columns=["statement_id"])
    for s in proc.column("statement_id").to_pylist():
        pc[s] = pc.get(s, 0) + 1
    mult_rec = np.array([float(counts.get(r["statement_id"], 0)) for r in records])
    assert (mult_rec > 0).all(), "a scored statement is absent from the RL pool"
    stmt_ids = sorted({r["statement_id"] for r in records})
    sid_comp = {r["statement_id"]: r["component_id"] for r in records}
    stmt_comp = np.array([sid_comp[s] for s in stmt_ids])
    mult_stmt = np.array([float(counts[s]) for s in stmt_ids])
    pool = {
        "raw_pool_rows": int(raw.num_rows), "processed_pool_rows": int(proc.num_rows),
        "raw_pool_unique_statements": len(counts), "processed_pool_unique_statements": len(pc),
        "multiplicity_identical_raw_vs_processed": bool(all(counts.get(k) == pc.get(k) for k in counts)),
        "scored_statements_present_in_pool": len(stmt_ids),
        "scored_statement_share_of_pool_rows": round(float(mult_stmt.sum() / raw.num_rows), 5),
        "multiplicity_of_scored_statements_min_median_max": [int(mult_stmt.min()),
                                                             int(np.median(mult_stmt)),
                                                             int(mult_stmt.max())],
        "pool_multiplicity_min_median_max": [int(min(counts.values())),
                                             int(np.median(list(counts.values()))),
                                             int(max(counts.values()))],
    }

    # ---- 1. deployment controller ------------------------------------------------------
    inner_full = stratified_group_kfold(y, comps, N_INNER, SEED)
    C_dep, inner_ap = R.select_C(Xb2[L], y, inner_full)
    mu, sd = standardize_fit(Xb2[L])
    w_dep = fit_logistic(standardize_apply(Xb2[L], mu, sd), y.astype(float), C_dep)
    q_in = predict_proba_logistic(standardize_apply(Xb2[L], mu, sd), w_dep)

    # ---- 2. q_i(t) surface from the deployment head ------------------------------------
    Xs = np.zeros((len(stmt_ids), Xb2[L].shape[1]))
    for i, s in enumerate(stmt_ids):
        Xs[i, :-1] = reps_map[s][1][L]
    q_t = {}
    for sn in STEP_NORM_EVAL:
        Xs[:, -1] = sn
        q_t[sn] = predict_proba_logistic(standardize_apply(Xs, mu, sd), w_dep)
    q_mid = q_t[STEP_NORM_EVAL[len(STEP_NORM_EVAL) // 2]]
    step_sensitivity = {
        "mean_abs_q_shift_step_norm_0.083_to_0.50": round(
            float(np.abs(q_t[STEP_NORM_EVAL[0]] - q_t[STEP_NORM_EVAL[-1]]).mean()), 5),
        "note": "step_norm is 1 of 1025 features; the q(t) surface is nearly flat, so the frozen sampler is "
                "effectively time-stationary inside a <=30-step trial"}

    # ---- 3. (alpha, epsilon) freeze ----------------------------------------------------
    uni_stmt = sampler_diagnostics(np.ones(len(stmt_ids)), stmt_comp, None, B_THEOREMS_PER_STEP,
                                   n_draws=TRIAL_DRAWS)
    uni_row = sampler_diagnostics(np.ones(len(stmt_ids)), stmt_comp, mult_stmt, B_THEOREMS_PER_STEP,
                                  n_draws=TRIAL_DRAWS)
    p_ctrl_row = (mult_stmt / mult_stmt.sum())
    yf = y.astype(float)
    base_row = wmean(mult_rec, yf)
    rows = []
    for alpha in ALPHA_GRID:
        for eps in EPS_GRID:
            wrec = eps + np.clip(oofB2, 1e-9, 1.0) ** alpha
            wstmt = eps + np.clip(q_mid, 1e-9, 1.0) ** alpha
            theorem = sampler_diagnostics(wstmt, stmt_comp, None, B_THEOREMS_PER_STEP,
                                          n_draws=TRIAL_DRAWS)
            rowlevel = sampler_diagnostics(wstmt, stmt_comp, mult_stmt, B_THEOREMS_PER_STEP,
                                           n_draws=TRIAL_DRAWS)
            igr_row = wmean(wrec * mult_rec, yf)
            p_treat_row = (wstmt * mult_stmt) / float((wstmt * mult_stmt).sum())
            trial_cov_ratio = (rowlevel["expected_distinct_groups_over_trial"]
                               / uni_row["expected_distinct_groups_over_trial"])
            feasible = all([
                rowlevel["ESS_fraction_of_pool"] >= CONSTRAINTS["min_ESS_fraction_of_pool"],
                rowlevel["normalized_entropy"] >= CONSTRAINTS["min_normalized_entropy"],
                rowlevel["max_sampling_probability"]
                <= CONSTRAINTS["max_prob_over_control"] * uni_row["max_sampling_probability"],
                rowlevel["expected_distinct_groups_per_batch"]
                >= CONSTRAINTS["min_family_coverage_ratio"] * uni_row["expected_distinct_groups_per_batch"],
                trial_cov_ratio >= CONSTRAINTS["min_trial_family_coverage_ratio"]])
            rows.append({"alpha": alpha, "epsilon": eps, "feasible": bool(feasible),
                         "oof_treated_IGR_record_level": round(wmean(wrec, yf), 5),
                         "oof_uplift_pp_record_level": round(100 * (wmean(wrec, yf) - float(yf.mean())), 3),
                         "oof_treated_IGR_multiplicity_weighted": round(igr_row, 5),
                         "oof_uplift_pp_multiplicity_weighted": round(100 * (igr_row - base_row), 3),
                         "theorem_level": theorem, "row_level": rowlevel,
                         "max_prob_over_control": round(
                             rowlevel["max_sampling_probability"]
                             / uni_row["max_sampling_probability"], 4),
                         "entropy_ratio_vs_control": round(
                             rowlevel["entropy_nats"] / uni_row["entropy_nats"], 4),
                         "trial_coverage_ratio_vs_control": round(trial_cov_ratio, 4),
                         "starved_statement_fraction": round(
                             float((p_treat_row < 0.1 * p_ctrl_row).mean()), 5),
                         "coverage_ratio_vs_uniform": round(
                             rowlevel["expected_distinct_groups_per_batch"]
                             / uni_row["expected_distinct_groups_per_batch"], 4)})
    feas = [r for r in rows if r["feasible"]]
    if not feas:
        print("WARNING: no (alpha, epsilon) pair satisfies every anti-collapse constraint", file=sys.stderr)
        pick = max(rows, key=lambda r: r["oof_treated_IGR_multiplicity_weighted"])
    else:
        pick = max(feas, key=lambda r: (r["oof_treated_IGR_multiplicity_weighted"], r["epsilon"]))
    # documented sensitivity: what the SAME constraints would pick under the record-level (group-uniform)
    # objective, and whether the two objectives disagree. Reported, never used to select.
    alt_r = max(feas or rows, key=lambda r: (r["oof_treated_IGR_record_level"], -r["epsilon"]))
    alt = {"objective": "maximise the record-level (group-uniform) OOF-predicted treated IGR",
           "alpha": alt_r["alpha"], "epsilon": alt_r["epsilon"],
           "feasible_under_same_constraints": bool(alt_r["feasible"]),
           "oof_treated_IGR_record_level": alt_r["oof_treated_IGR_record_level"],
           "oof_uplift_pp_multiplicity_weighted": alt_r["oof_uplift_pp_multiplicity_weighted"],
           "agrees_with_frozen_pick": bool(alt_r["alpha"] == pick["alpha"]
                                           and alt_r["epsilon"] == pick["epsilon"])}
    a_star, e_star = pick["alpha"], pick["epsilon"]
    wflat = e_star + np.clip(oofB2, 1e-9, 1.0) ** a_star
    wn = wflat * mult_rec
    wn = wn / wn.sum()
    # two COHERENT contrasts: each arm's treated/control pair shares one denominator family
    lo, hi, _ = component_bootstrap(
        records, np.zeros(len(y)), lambda idx: wmean(wn[idx], yf[idx]) - wmean(mult_rec[idx], yf[idx]),
        n_rep=args.bootstrap_reps, seed=SEED)
    lo_rec, hi_rec, _ = component_bootstrap(
        records, np.zeros(len(y)), lambda idx: wmean(wflat[idx], yf[idx]) - float(y[idx].mean()),
        n_rep=args.bootstrap_reps, seed=SEED)

    # ---- 4. power analysis -------------------------------------------------------------
    series, per_run = [], {}
    for name, rel in DYNAMICS.items():
        d = json.loads((ROOT / rel).read_text())
        vals = [float(d["per_step"][k]["igr"]) for k in sorted(d["per_step"], key=lambda x: int(x))]
        series.append(vals)
        per_run[name] = {"n_steps": len(vals), "mean_IGR": round(float(np.mean(vals)), 5),
                         "sd_per_step_IGR": round(float(np.std(vals, ddof=1)), 5),
                         "igr_steps_1_30": round(float(np.mean(vals[:30])), 5),
                         "igr_steps_31_60": round(float(np.mean(vals[30:])), 5),
                         "igr_by_10step_bins": [round(float(np.mean(vals[i:i + 10])), 4)
                                                for i in range(0, len(vals), 10)]}
    pooled = np.concatenate(series)
    power = {
        "unit_of_inference": "training step (cluster of 4 groups / 32 candidates) - the historical unit",
        "historical_basis": {"runs": list(per_run), "pooled_steps": len(pooled),
                             "pooled_mean_IGR": round(float(pooled.mean()), 5),
                             "pooled_sd_of_per_step_IGR": round(float(pooled.std(ddof=1)), 5),
                             "per_run": per_run},
        "scenario_assumption": ("the offline OOF-predicted uplift is an UPPER bound for a real trial: it treats "
                                "informativeness as a fixed property of the theorem and ignores policy drift "
                                "plus use-up of mixed groups, so 5-26 pp scenarios are all priced in"),
        "table": [{f"H={h}": {**step_cluster_power(series, h, up / 100.0), "assumed_uplift_pp": up}
                   for h in HORIZONS} for up in UPLIFT_SCENARIOS_PP],
        "steps_for_80pct_power_by_uplift_pp": {
            str(up): next((h for h in range(2, 401)
                           if step_cluster_power(series, h, up / 100.0)["power_at_uplift"] >= 0.8), None)
            for up in UPLIFT_SCENARIOS_PP},
        "offline_predicted_uplift_pp": pick["oof_uplift_pp_multiplicity_weighted"],
    }

    freeze = {
        "artifact_type": "v3_r001_offline_freeze",
        "status": "PREREGISTRATION SUPPORT (offline only; no RL launched, no rollouts generated)",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorized_by": "owner directive 2026-09-24 Part B (DRAFT ONLY - do not launch)",
        "depends_on": {"V3-D001_provenance": committed["provenance"],
                       "V3-D001_gate": committed["gate"]["ranking_GO_NO_GO"]},
        "controller": {
            "recipe": "frozen B2: theta0 block-18 last-token rep (1024) + step_norm -> L2 logistic (Newton)",
            "fit_on": "ALL 686 V1 seed1/2/3 valid groups (rollout history only; no reward/verifier input)",
            "C_selection": {"grid": R.CGRID, "criterion": "mean inner family-grouped AUPRC over 4 folds",
                            "selected_C": C_dep, "mean_inner_auprc": inner_ap},
            "standardization": "mean/std of the 686-group fit set, stored with the weights",
            "online_retraining": "NONE for all of R001",
            "weights_mu_sd": {"w": [float(x) for x in w_dep], "mu": [float(x) for x in mu],
                              "sd": [float(x) for x in sd]},
            "weights_sha256": hashlib.sha256(np.ascontiguousarray(w_dep).tobytes()
                                             + np.ascontiguousarray(mu).tobytes()
                                             + np.ascontiguousarray(sd).tobytes()).hexdigest(),
            "sanity": {"OOF_auprc_of_same_recipe": round(float(average_precision(y, oofB2)), 5),
                       "committed_OOF_auprc": committed["B2_block18_PRIMARY"]["auprc"],
                       "OOF_auprc_B1_reference": round(float(average_precision(y, oofB1)), 5),
                       "in_sample_auprc_descriptive_only": round(float(average_precision(y, q_in)), 5)},
            "step_feature_semantics": "step_norm = elapsed optimizer updates / 60, identical to the D001 "
                                      "feature; a <=30-step trial stays inside the fitted support",
            "q_t_surface_by_step": {f"step_norm={sn}": {
                "mean_q": round(float(q.mean()), 5),
                "q_p05_p50_p95_max": [round(float(v), 5) for v in np.quantile(q, [.05, .5, .95])]
                                     + [round(float(q.max()), 5)]} for sn, q in q_t.items()},
            "step_sensitivity": step_sensitivity,
            "scope_note": "these weights are the deployment controller fit on V1 history; they are NOT a "
                          "substitute for the OOF predictions used to freeze (alpha, epsilon)",
        },
        "pool": pool,
        "oof_q_distribution": {"n_records": len(oofB2), "mean": round(float(oofB2.mean()), 5),
                               "quantiles_p05_p25_p50_p75_p95": [
                                   round(float(v), 5) for v in np.quantile(oofB2, [.05, .25, .5, .75, .95])],
                               "min_max": [round(float(oofB2.min()), 5), round(float(oofB2.max()), 5)],
                               "fraction_above_prevalence": round(float((oofB2 > y.mean()).mean()), 5)},
        "sampler_form": "w_i(t) = epsilon + q_i(t)^alpha ;  P(i|t) = w_i / sum_j w_j  (stochastic, NO "
                        "deterministic top-k; prompt-pool row multiplicity preserved)",
        "grid_search": {"alpha_grid": ALPHA_GRID, "epsilon_grid": EPS_GRID,
                        "objective": "maximise the multiplicity-weighted OOF-predicted treated IGR",
                        "constraints": CONSTRAINTS,
                        "trial_draws_assumed_for_coverage": TRIAL_DRAWS,
                        "constraint_rationale": (
                            "anti-collapse constraints are measured RELATIVE TO THE CONTROL ARM's own "
                            "row-level distribution, because pool row multiplicity (1..54) already makes "
                            "the uniform control arm concentrate ~4.8x relative to statement-uniform; an "
                            "absolute cap would grade the pool's copy structure instead of the controller, "
                            "and would forbid the treatment for doing what the control already does"),
                        "uniform_references": {"theorem_level": uni_stmt, "row_level_multiplicity": uni_row},
                        "sensitivity_alternative_objective": alt,
                        "forbidden": ["any RL outcome", "any Pass@k", "any reward from a candidate run",
                                      "any retune after R001 data exists"],
                        "table": rows},
        "selected": {**pick,
                     "selection_rule": "max multiplicity-weighted OOF-predicted treated IGR among feasible "
                                       "pairs; ties toward the larger epsilon (flatter, safer)",
                     "n_feasible_pairs": len(feas),
                     "uniform_realized_IGR_record_level": round(float(y.mean()), 5),
                     "uniform_realized_IGR_multiplicity_weighted": round(base_row, 5),
                     "uplift_pp_ci95_component_bootstrap_multiplicity_weighted": [round(100 * lo, 3),
                                                                                 round(100 * hi, 3)],
                     "uplift_definition_multiplicity_weighted":
                         "treated row-weighted mean minus control row-weighted mean, both over the SAME "
                         "labelled statements with the same multiplicity weights (coherent contrast)",
                     "uplift_pp_ci95_component_bootstrap_record_level": [round(100 * lo_rec, 3),
                                                                        round(100 * hi_rec, 3)],
                     "uplift_definition_record_level":
                         "treated mean under w = epsilon + q^alpha minus the equal-weight 686-group mean "
                         "(the D001 realized prevalence); no multiplicity enters either side",
                     "caveat": "an off-policy estimate on the 686 V1 groups, not a guarantee: the treated arm "
                               "also changes which groups are informative as it learns"},
        "power_analysis": power,
        "planned_arms": {"control": "uniform sampler (w identically 1) over the same filtered prompt pool",
                         "treatment": f"decision-guided sampler with alpha={a_star}, epsilon={e_star}",
                         "theorems_per_step": B_THEOREMS_PER_STEP, "candidates_per_theorem": N_CANDIDATES,
                         "identical_everywhere_else": "model/theta0, n=8, batch, optimizer, LR, GRPO/DrGRPO "
                                                      "flags, max response length, temperature, top_p, verifier"},
        "timing": {"total_s": round(time.time() - t0, 1)},
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(freeze, indent=2))
    cache = ROOT / "runs/v3_r001/oof_cache.npz"
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, y=y, comps=comps, seeds=seeds, mult_rec=mult_rec,
                        step_norm=np.array([r["step_norm"] for r in records]),
                        oofB2=oofB2, oofB1=oofB1, q_mid=q_mid, stmt_comp=stmt_comp,
                        mult_stmt=mult_stmt, stmt_ids=np.array(stmt_ids, dtype=object))

    print(json.dumps({"selected": {k: pick[k] for k in
                                   ["alpha", "epsilon", "oof_treated_IGR_record_level",
                                    "oof_treated_IGR_multiplicity_weighted",
                                    "oof_uplift_pp_multiplicity_weighted", "coverage_ratio_vs_uniform",
                                    "max_prob_over_control", "feasible"]},
                      "sensitivity_alt_objective_pick": alt,
                      "row_level_diagnostics": pick["row_level"],
                      "uniform_row_level_reference": uni_row,
                      "uplift_ci_pp": freeze["selected"][
                          "uplift_pp_ci95_component_bootstrap_multiplicity_weighted"],
                      "steps_for_80pct": power["steps_for_80pct_power_by_uplift_pp"],
                      "pool": pool,
                      "controller": {"C": C_dep, "sha": freeze["controller"]["weights_sha256"][:16],
                                     "sanity": freeze["controller"]["sanity"]},
                      "timing": freeze["timing"]}, indent=2))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
