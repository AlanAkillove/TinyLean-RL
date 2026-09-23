#!/usr/bin/env python3
"""V3-D001 §20 steps 9-11 — fit B0/B1/B2 with nested family-grouped CV, evaluate, bootstrap, gate.

Runs the frozen protocol end-to-end on the reconstructed 686 valid groups:
  * nested CV  : outer 5 / inner 4 folds read from the committed fold manifest (component-keyed)
  * B0         : per-outer-fold prevalence (calibration / Brier null)
  * B1         : handcrafted difficulty features -> L2 logistic  (owner §7)
  * B2         : frozen theta0 last-token rep (block 18 primary; 9/27 robustness) + step -> L2 logistic (§8/§9)
  * regularization C chosen ONLY on inner family-grouped CV (criterion = inner AUPRC)
  * primary evaluation surface = 686 out-of-fold predictions, each from a model that never saw its family
  * family-component bootstrap CIs (10k reps), cross-seed family-isolated transfer, stationarity diagnostics
  * GO / NO-GO exactly per owner §14 G1-G3 + §15 calibration claim

numpy only (no sklearn). Requires runs/v3_d001/theta0_reps.npz from step 8.
Writes runs/v3_d001/v3_d001_results.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Cap BLAS threads before numpy loads: this build's multithreaded LAPACK `gesv`
# oversubscribes catastrophically (a 1026-dim Newton solve takes ~6 s at default
# threads vs ~50 ms when capped). Keeps the formal CPU fit tractable on fly122.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from v3_d001_lib import (
    B1_FEATURES,
    average_precision,
    brier,
    build_records,
    component_bootstrap,
    ece_equal_mass,
    extract_b1,
    fit_logistic,
    formal_text,
    predict_proba_logistic,
    roc_auc,
    standardize_apply,
    standardize_fit,
    stratified_group_kfold,
    topk_enrichment,
)

DATASET = "data/raw/kimina_promptset/data/train-00000-of-00001.parquet"
REGISTRY = "experiments/manifests/v2/family_component_registry.json"
FOLDS = "experiments/manifests/v3/v3_d001_folds.json"

SEED = 20260923
CGRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
TOPK = [0.10, 0.20, 0.30]
PRIMARY_LAYER = 18
ROBUST_LAYERS = [9, 27]
LAYERS = [PRIMARY_LAYER, *ROBUST_LAYERS]
STEP_BINS = [(1, 10), (11, 20), (21, 30), (31, 40), (41, 50), (51, 60)]
SOURCES = ["synthetic", "autoformalizer", "human"]
SEEDS = ["seed1", "seed2", "seed3"]
REP_DIM = 1024


# --- design matrices -------------------------------------------------------------------
def build_designs(records, prompts, reps_map):
    y = np.array([r["y_score"] for r in records], dtype=int)
    comps = np.array([r["component_id"] for r in records])
    seeds = np.array([r["seed"] for r in records])
    Xb1 = np.zeros((len(records), len(B1_FEATURES)))
    Xb2 = {L: np.zeros((len(records), REP_DIM + 1)) for L in LAYERS}
    for i, r in enumerate(records):
        tl, rep = reps_map[r["statement_id"]]
        Xb1[i] = extract_b1(formal_text(prompts[r["statement_id"]]), tl, r["source"], r["step_norm"])
        for L in LAYERS:
            Xb2[L][i, :REP_DIM] = rep[L]
            Xb2[L][i, REP_DIM] = r["step_norm"]
    return y, comps, seeds, Xb1, Xb2


# --- regularization selection (inner family-grouped CV, criterion = AUPRC) -------------
def select_C(X, y, inner_labels):
    devs = sorted({int(v) for v in inner_labels})
    best_C, best_score = CGRID[0], -np.inf
    for C in CGRID:
        scores = []
        for d in devs:
            te = inner_labels == d
            tr = ~te
            if y[te].sum() == 0 or y[tr].sum() == 0:
                continue
            mu, sd = standardize_fit(X[tr])
            w = fit_logistic(standardize_apply(X[tr], mu, sd), y[tr].astype(float), C)
            p = predict_proba_logistic(standardize_apply(X[te], mu, sd), w)
            ap = average_precision(y[te], p)
            if np.isfinite(ap):
                scores.append(ap)
        if scores:
            m = float(np.mean(scores))
            if m > best_score + 1e-12:  # strict -> ties keep the smaller (more regularised) C
                best_score, best_C = m, C
    return best_C, (round(best_score, 5) if np.isfinite(best_score) else None)


# --- nested out-of-fold predictions ----------------------------------------------------
def b0_oof(y, outer, n_outer):
    oof = np.zeros(len(y))
    for f in range(n_outer):
        oof[outer == f] = y[outer != f].mean()
    return oof


def nested_oof(X, y, comps, outer, inner_by_outer, n_outer):
    oof = np.zeros(len(y))
    chosen = {}
    for f in range(n_outer):
        tr = outer != f
        te = ~tr
        inner_lbl = np.array([inner_by_outer[str(f)][c] for c in comps[tr]], dtype=int)
        C, iscore = select_C(X[tr], y[tr], inner_lbl)
        chosen[f] = {"C": C, "inner_auprc": iscore}
        mu, sd = standardize_fit(X[tr])
        w = fit_logistic(standardize_apply(X[tr], mu, sd), y[tr].astype(float), C)
        oof[te] = predict_proba_logistic(standardize_apply(X[te], mu, sd), w)
    return oof, chosen


# --- metrics ---------------------------------------------------------------------------
def metrics(y, p):
    prevalence = float(y.mean())
    order = np.argsort(-p)
    out = {"auprc": round(float(average_precision(y, p)), 5), "auroc": round(float(roc_auc(y, p)), 5),
           "brier": round(brier(y, p), 5), "ece": round(ece_equal_mass(y, p, bins=10), 5),
           "prevalence": round(prevalence, 5)}
    for frac in TOPK:
        k = max(1, round(frac * len(y)))
        igr = float(y[order[:k]].mean())
        out[f"top{int(frac * 100)}_igr"] = round(igr, 5)
        out[f"top{int(frac * 100)}_enrichment"] = round(igr / prevalence, 5) if prevalence > 0 else None
    return out


# --- cross-seed family-isolated transfer (§13) -----------------------------------------
def fit_predict(X, y, tr, te, comps):
    inner = stratified_group_kfold(y[tr], comps[tr], 4, SEED)
    C, _ = select_C(X[tr], y[tr], inner)
    mu, sd = standardize_fit(X[tr])
    w = fit_logistic(standardize_apply(X[tr], mu, sd), y[tr].astype(float), C)
    return predict_proba_logistic(standardize_apply(X[te], mu, sd), w), C


def cross_seed(seeds, comps, y, Xb1, Xb2p):
    res = {}
    for held in SEEDS:
        te = seeds == held
        test_comps = set(comps[te])
        tr = (seeds != held) & np.array([c not in test_comps for c in comps])
        p1, c1 = fit_predict(Xb1, y, tr, te, comps)
        p2, c2 = fit_predict(Xb2p, y, tr, te, comps)
        yte = y[te]
        prevalence = float(yte.mean())
        k = max(1, round(0.2 * int(te.sum())))
        top20_igr = float(yte[np.argsort(-p2)[:k]].mean())
        res[held] = {
            "n_test_groups": int(te.sum()), "n_train_groups": int(tr.sum()),
            "test_positives": int(yte.sum()), "test_prevalence": round(prevalence, 5),
            "B1_auprc": round(float(average_precision(yte, p1)), 5),
            "B2_auprc": round(float(average_precision(yte, p2)), 5),
            "B2_top20_igr": round(top20_igr, 5),
            "B2_beats_B1": bool(average_precision(yte, p2) > average_precision(yte, p1)),
            "B2_top20_above_prevalence": bool(top20_igr > prevalence),
            "C_B1": c1, "C_B2": c2,
        }
    return res


# --- stationarity diagnostics (§5, descriptive only, must not retune the model) --------
def stationarity(records, y, seeds):
    steps = np.array([r["step"] for r in records])
    srcs = np.array([r["source"] for r in records])
    igr_by_step = {}
    for lo, hi in STEP_BINS:
        m = (steps >= lo) & (steps <= hi)
        row = {"n_groups": int(m.sum())}
        for s in SEEDS + ["pooled"]:
            sel = m if s == "pooled" else (m & (seeds == s))
            row[s] = round(float(y[sel].mean()), 4) if sel.sum() else None
        igr_by_step[f"{lo}-{hi}"] = row
    by_stmt = {}
    for r, yi in zip(records, y):
        by_stmt.setdefault(r["statement_id"], []).append((r["seed"], int(yi)))
    within, cross = _empty_consistency(), _empty_consistency()
    for obs in by_stmt.values():
        if len(obs) < 2:
            continue
        ys = [o[1] for o in obs]
        bucket = within if len({o[0] for o in obs}) == 1 else cross
        bucket["n_repeated"] += 1
        if all(v == 0 for v in ys):
            bucket["always_noninf"] += 1
        elif all(v == 1 for v in ys):
            bucket["always_inf"] += 1
        else:
            bucket["flips"] += 1
    src_prev = {}
    for src in SOURCES:
        m = srcs == src
        src_prev[src] = {"n_groups": int(m.sum()), "prevalence": round(float(y[m].mean()), 4) if m.sum() else None}
    other = ~(np.isin(srcs, SOURCES))
    src_prev["other_or_unknown"] = {"n_groups": int(other.sum()),
                                    "prevalence": round(float(y[other].mean()), 4) if other.sum() else None}
    return {"igr_by_step": igr_by_step, "repeated_within_seed": within,
            "repeated_cross_seed": cross, "source_prevalence": src_prev}


def _empty_consistency():
    return {"n_repeated": 0, "always_noninf": 0, "always_inf": 0, "flips": 0}


# --- component bootstrap helpers (§12) -------------------------------------------------
def _ci(records, p, metric_fn, seed):
    lo, hi, _ = component_bootstrap(records, p, lambda idx: metric_fn(idx), seed=seed)
    return {"lo95": round(float(lo), 5), "hi95": round(float(hi), 5)}


def _paired_ci(records, y, pA, pB, diff_fn, seed):
    lo, hi, _ = component_bootstrap(records, np.zeros(len(y)), lambda idx: diff_fn(idx), seed=seed)
    return {"lo95": round(float(lo), 5), "hi95": round(float(hi), 5)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", default="runs/v3_d001/theta0_reps.npz")
    ap.add_argument("--reps-meta", default="runs/v3_d001/theta0_reps_meta.json")
    ap.add_argument("--out", default="runs/v3_d001/v3_d001_results.json")
    args = ap.parse_args()

    if not (ROOT / args.reps).exists():
        print(f"ERROR: reps artifact missing: {args.reps}. Run scripts/v3_d001_extract.py first (§20 step8).",
              file=sys.stderr)
        return 2

    built = build_records(ROOT, DATASET, REGISTRY)
    records = built["valid"]
    prompts = built["prompts"]
    folds = json.loads((ROOT / FOLDS).read_text())

    z = np.load(ROOT / args.reps, allow_pickle=True)
    stmt_ids = list(z["statement_ids"])
    layers_present = [int(x) for x in z["layers"]]
    reps_arr, token_lens = z["reps"], z["token_lens"]
    if sorted(layers_present) != sorted(LAYERS):
        print(f"ERROR: reps layers {layers_present} != expected {LAYERS}", file=sys.stderr)
        return 2
    reps_map = {}
    for i, sid in enumerate(stmt_ids):
        reps_map[sid] = (int(token_lens[i]), {L: reps_arr[i, layers_present.index(L)] for L in LAYERS})

    pre_comps = [r["component_id"] for r in records]
    miss_stmt = [r["statement_id"] for r in records if r["statement_id"] not in reps_map]
    miss_fold = [c for c in set(pre_comps) if c not in folds["component_outer_fold"]]
    assert not miss_stmt, f"statements without reps: {miss_stmt[:5]}"
    assert not miss_fold, f"components not in frozen fold manifest: {miss_fold[:5]}"
    outer = np.array([folds["component_outer_fold"][c] for c in pre_comps], dtype=int)

    t0 = time.time()
    y, comps, seeds, Xb1, Xb2 = build_designs(records, prompts, reps_map)
    assert np.array_equal(comps, np.array(pre_comps))

    inner_by_outer = folds["inner_fold_by_outer"]
    n_outer = int(folds["n_outer"])
    oofB0 = b0_oof(y, outer, n_outer)
    oofB1, cB1 = nested_oof(Xb1, y, comps, outer, inner_by_outer, n_outer)
    oofB2 = {}
    for L in LAYERS:
        oofB2[L] = nested_oof(Xb2[L], y, comps, outer, inner_by_outer, n_outer)
    cpu_fit_s = round(time.time() - t0, 2)

    p1, p2 = oofB1, oofB2[PRIMARY_LAYER][0]
    mB0 = metrics(y, oofB0)
    mB1 = metrics(y, p1)
    mB2 = {L: metrics(y, oofB2[L][0]) for L in LAYERS}

    boot = {
        "B1_auprc": _ci(records, p1, lambda idx: average_precision(y[idx], p1[idx]), SEED),
        "B2p_auprc": _ci(records, p2, lambda idx: average_precision(y[idx], p2[idx]), SEED),
        "B2p_top20_enrichment": _ci(records, p2, lambda idx: topk_enrichment(y[idx], p2[idx], 0.20), SEED),
        "B2p_ece": _ci(records, p2, lambda idx: ece_equal_mass(y[idx], p2[idx], bins=10), SEED),
        "dAUPRC_B2_minus_B1": _paired_ci(
            records, y, p1, p2,
            lambda idx: average_precision(y[idx], p2[idx]) - average_precision(y[idx], p1[idx]), SEED),
        "dBrier_B2_minus_B1": _paired_ci(
            records, y, p1, p2,
            lambda idx: brier(y[idx], p2[idx]) - brier(y[idx], p1[idx]), SEED),
    }

    dAUPRC = round(mB2[PRIMARY_LAYER]["auprc"] - mB1["auprc"], 5)
    top20_enr_pt = mB2[PRIMARY_LAYER]["top20_enrichment"]

    xs = cross_seed(seeds, comps, y, Xb1, Xb2[PRIMARY_LAYER])
    g3_folds_ok = sum(1 for h in xs.values() if h["B2_beats_B1"] and h["B2_top20_above_prevalence"])
    G3 = g3_folds_ok >= 2

    G1 = boot["dAUPRC_B2_minus_B1"]["lo95"] > 0
    G2 = (top20_enr_pt is not None and top20_enr_pt >= 1.75) and boot["B2p_top20_enrichment"]["lo95"] > 1.0
    ranking_go = bool(G1 and G2 and G3)

    calib_ok = (mB2[PRIMARY_LAYER]["brier"] <= mB1["brier"]) and (mB2[PRIMARY_LAYER]["ece"] <= 0.10)
    if ranking_go and calib_ok:
        calibration_claim = "calibrated semantic controller (ranking GO + calibration gate pass)"
    elif ranking_go:
        calibration_claim = "rank-based theorem sampler ONLY (ranking GO, calibration gate NOT met)"
    else:
        calibration_claim = "no controller claim (ranking NO-GO)"

    reps_meta = json.loads((ROOT / args.reps_meta).read_text()) if (ROOT / args.reps_meta).exists() else {}

    result = {
        "artifact_type": "v3_d001_results",
        "status": "OFFLINE PROBE (no RL training performed)",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "theta0_weights_sha256": reps_meta.get("theta0_weights_sha256_expected"),
            "reps_content_sha256": reps_meta.get("reps_content_sha256"),
            "reps_formality": reps_meta.get("formality"),
            "reps_host": reps_meta.get("host"),
            "folds_hash": folds["folds_hash"],
            "seed": SEED, "primary_layer": PRIMARY_LAYER, "robust_layers": ROBUST_LAYERS,
            "b1_features": B1_FEATURES,
        },
        "dataset": {"valid_groups": len(records), "components": len(set(comps)),
                    "informative_positives": int(y.sum()), "prevalence": round(float(y.mean()), 5)},
        "stationarity": stationarity(records, y, seeds),
        "B0_prevalence": mB0,
        "B1_handcrafted": {**mB1, "chosen_C_by_outer_fold": cB1},
        "B2_block18_PRIMARY": {**mB2[PRIMARY_LAYER], "chosen_C_by_outer_fold": oofB2[PRIMARY_LAYER][1],
                               "delta_auprc_vs_B1": dAUPRC},
        "B2_robustness": {f"block{L}": mB2[L] for L in ROBUST_LAYERS},
        "bootstrap": boot,
        "cross_seed": xs,
        "gate": {
            "G1_dAUPRC_lower_gt_0": bool(G1),
            "G1_dAUPRC_point": dAUPRC, "G1_ci": boot["dAUPRC_B2_minus_B1"],
            "G2_top20_enrich_point_ge_1.75_and_lower_gt_1": bool(G2),
            "G2_point": top20_enr_pt, "G2_ci": boot["B2p_top20_enrichment"],
            "G3_crossseed_folds_ok": f"{g3_folds_ok}/3 (need >=2)", "G3_pass": bool(G3),
            "ranking_GO_NO_GO": "GO" if ranking_go else "NO-GO",
            "calibration_claim": calibration_claim,
        },
        "compute": {"cpu_fit_time_s": cpu_fit_s,
                    "extraction_time_s": reps_meta.get("extract_time_s"),
                    "peak_vram_gb": reps_meta.get("peak_vram_gb"),
                    "model_load_time_s": reps_meta.get("load_time_s")},
    }
    outp = ROOT / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(result, indent=2))

    print(json.dumps({"gate": result["gate"],
                      "B0": {k: mB0[k] for k in ["auprc", "brier"]},
                      "B1": {k: mB1[k] for k in ["auprc", "brier", "ece", "top20_enrichment"]},
                      "B2_block18": {k: mB2[PRIMARY_LAYER][k] for k in
                                     ["auprc", "auroc", "brier", "ece", "top20_enrichment"]},
                      "compute": result["compute"]}, indent=2))
    print("wrote", outp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
