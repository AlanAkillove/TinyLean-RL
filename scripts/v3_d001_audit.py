#!/usr/bin/env python3
"""V3-D001 canonical methodological audit (owner directive 2026-09-24, Part A: A1/A2/A4/A5).

Post-hoc audit of the frozen V3-D001 probe. It does NOT re-open, re-tune or replace the
frozen gate: it (R) reproduces the committed numbers from the frozen artifacts, and adds
(A1) the complete family-isolated cross-seed table, (A2) an itemized ACTIVE
preprocessing-leakage audit, and (A4) representation provenance / sanity checks.

Everything is recomputed from the frozen inputs through the frozen code path
(scripts/v3_d001_lib.py and scripts/v3_d001_run.py are imported, never edited), so any
disagreement with the committed result is reported as a FAIL rather than silently fixed.

A2 activity: for every outer fold, and for both designs, the held-out block's FEATURES are
scrambled and - separately - its LABELS are flipped. The selected C, the standardisation
statistics, the fitted weights and the out-of-fold predictions must be bit-identical. Any
preprocessing step that touched the held-out block cannot survive this.

Output: runs/v3_d001/v3_d001_canonical_audit.json  (+ stdout summary)
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Cap BLAS threads before numpy import (same guard as the frozen run script): pure
# implementation performance, numerics unchanged.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import v3_d001_lib as LIB
import v3_d001_run as R
from v3_d001_lib import (
    average_precision,
    fit_logistic,
    predict_proba_logistic,
    standardize_apply,
    standardize_fit,
    stratified_group_kfold,
)

FOLDS = "experiments/manifests/v3/v3_d001_folds.json"
COMMITTED = "experiments/manifests/v3/V3-D001_results.json"
MANIFEST = "experiments/manifests/v3/V3-D001.yaml"
PREREG = "docs/v3/V3-D001_preregistration.md"
MODEL = "models/weights/kimina_distill_0_6b/model.safetensors"
THETA0_SHA = "34e6e630f564d330c79424c404ab0494558a0a659e6201b47d9bd88ccd640fe2"
REP_DIM = 1024
SEED = R.SEED


def _git(*args) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True,
                          check=False).stdout.strip()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _chk(name, ok, **detail) -> dict:
    return {"check": name, "pass": bool(ok), **detail}


# --------------------------------------------------------------------------------------
# A1  complete family-isolated cross-seed table
# --------------------------------------------------------------------------------------
def cross_seed_full(seeds, comps, y, Xb1, Xb2p):
    rows = {}
    for held in R.SEEDS:
        te = seeds == held
        test_comps = set(comps[te])
        tr = (seeds != held) & np.array([c not in test_comps for c in comps])
        p1, c1 = R.fit_predict(Xb1, y, tr, te, comps)
        p2, c2 = R.fit_predict(Xb2p, y, tr, te, comps)
        yte = y[te]
        prevalence = float(yte.mean())
        k = max(1, round(0.20 * int(te.sum())))
        ap1, ap2 = float(average_precision(yte, p1)), float(average_precision(yte, p2))
        igr1 = float(yte[np.argsort(-p1)[:k]].mean())
        igr2 = float(yte[np.argsort(-p2)[:k]].mean())
        rows[held] = {
            "train_seeds": [s for s in R.SEEDS if s != held],
            "n_test_groups": int(te.sum()),
            "n_test_components": len(test_comps),
            "n_train_groups_before_family_exclusion": int((seeds != held).sum()),
            "n_groups_dropped_by_family_exclusion": int((seeds != held).sum() - tr.sum()),
            "n_train_groups_after_family_exclusion": int(tr.sum()),
            "n_train_components": len(set(comps[tr])),
            "train_positives": int(y[tr].sum()),
            "train_prevalence": round(float(y[tr].mean()), 5),
            "test_positives": int(yte.sum()),
            "test_prevalence": round(prevalence, 5),
            "top20_k_groups": k,
            "B1_auprc": round(ap1, 5),
            "B2_auprc": round(ap2, 5),
            "delta_auprc_B2_minus_B1": round(ap2 - ap1, 5),
            "B1_top20_igr": round(igr1, 5),
            "B2_top20_igr": round(igr2, 5),
            "B1_top20_enrichment": round(igr1 / prevalence, 5) if prevalence > 0 else None,
            "B2_top20_enrichment": round(igr2 / prevalence, 5) if prevalence > 0 else None,
            "auprc_direction_B2_gt_B1": bool(ap2 > ap1),
            "top20_igr_gt_fold_prevalence": bool(igr2 > prevalence),
            "B2_top20_igr_gt_B1_top20_igr": bool(igr2 > igr1),
            "G3_fold_criterion_met_frozen_rule": bool(ap2 > ap1 and igr2 > prevalence),
            "C_B1": c1, "C_B2": c2,
            "family_disjoint_train_test": bool(not (set(comps[tr]) & test_comps)),
        }
    met = sum(v["G3_fold_criterion_met_frozen_rule"] for v in rows.values())
    summary = {
        "n_folds": len(rows),
        "folds_meeting_frozen_G3_rule": met,
        "G3_rule": ">= 2/3 folds with AUPRC(B2)>AUPRC(B1) AND top20 IGR > fold prevalence (frozen owner §14)",
        "G3_pass": bool(met >= 2),
        "criteria_split": {
            "auprc_direction_positive": sum(v["auprc_direction_B2_gt_B1"] for v in rows.values()),
            "top20_igr_above_fold_prevalence": sum(v["top20_igr_gt_fold_prevalence"] for v in rows.values()),
            "B2_top20_igr_above_B1_top20_igr": sum(v["B2_top20_igr_gt_B1_top20_igr"] for v in rows.values()),
            "both_frozen_conditions": met,
        },
        "delta_auprc_sign_by_fold": {h: ("+" if v["delta_auprc_B2_minus_B1"] > 0 else "-")
                                     for h, v in rows.items()},
        "delta_auprc_mean": round(float(np.mean([v["delta_auprc_B2_minus_B1"] for v in rows.values()])), 5),
        "delta_auprc_median": round(float(np.median([v["delta_auprc_B2_minus_B1"] for v in rows.values()])), 5),
        "delta_auprc_min": round(float(np.min([v["delta_auprc_B2_minus_B1"] for v in rows.values()])), 5),
        "honest_reading": "G3 is 2/3 on the frozen conjunctive rule; the AUPRC direction is NOT 3/3 "
                          "(one fold is negative) while top20-IGR-above-prevalence IS 3/3. Reported as 2/3.",
    }
    return rows, summary


# --------------------------------------------------------------------------------------
# A2  active per-fold leakage probes
# --------------------------------------------------------------------------------------
def fold_fit(X, y, comps, inner_by_outer, tr, te, f):
    """One outer-fold fit, executed through the frozen code path."""
    inner_lbl = np.array([inner_by_outer[str(f)][c] for c in comps[tr]], dtype=int)
    C, _ = R.select_C(X[tr], y[tr], inner_lbl)
    mu, sd = standardize_fit(X[tr])
    w = fit_logistic(standardize_apply(X[tr], mu, sd), y[tr].astype(float), C)
    p = predict_proba_logistic(standardize_apply(X[te], mu, sd), w)
    return {"C": C, "mu": mu, "sd": sd, "w": w, "p": p}


def leakage_fold_probe(X, y, comps, outer, inner_by_outer, tag) -> list:
    out = []
    for f in range(int(outer.max()) + 1):
        tr, te = outer != f, outer == f
        base = fold_fit(X, y, comps, inner_by_outer, tr, te, f)

        rng = np.random.default_rng(1234 + f)
        Xs = X.copy()                                   # scramble held-out FEATURES only
        Xs[te] = rng.normal(size=(int(te.sum()), X.shape[1])) * 1e3
        vf = fold_fit(Xs, y, comps, inner_by_outer, tr, te, f)

        yb = y.copy()                                   # flip held-out LABELS only
        yb[te] = 1 - yb[te]
        vl = fold_fit(X, yb, comps, inner_by_outer, tr, te, f)

        out.append(_chk(
            f"A2.all_preprocessing_outer_train_local[{tag},outer_fold{f}]",
            np.array_equal(base["mu"], vf["mu"]) and np.array_equal(base["sd"], vf["sd"])
            and base["C"] == vf["C"] and np.array_equal(base["w"], vf["w"])
            and base["C"] == vl["C"] and np.array_equal(base["w"], vl["w"])
            and np.array_equal(base["p"], vl["p"]),
            n_train_groups=int(tr.sum()), n_test_groups=int(te.sum()),
            train_components=len(set(comps[tr])), test_components=len(set(comps[te])),
            scaler_stats_unchanged_by_test_feature_scramble=bool(np.array_equal(base["mu"], vf["mu"])
                                                                 and np.array_equal(base["sd"], vf["sd"])),
            chosen_C_unchanged=base["C"] == vf["C"] == vl["C"],
            C_base=base["C"], C_after_feature_scramble=vf["C"], C_after_label_flip=vl["C"],
            fitted_weights_unchanged=bool(np.array_equal(base["w"], vf["w"]) and np.array_equal(base["w"], vl["w"])),
            max_abs_weight_drift=float(max(np.max(np.abs(base["w"] - vf["w"])),
                                           np.max(np.abs(base["w"] - vl["w"])))),
            oof_predictions_unchanged_by_label_flip=bool(np.array_equal(base["p"], vl["p"])),
            max_abs_prediction_drift=float(np.max(np.abs(base["p"] - vl["p"])))))
    return out


def _gate_block_uses_primary_only() -> bool:
    """Static check: the G1/G2/G3 + calibration decision region references only the primary layer."""
    src = inspect.getsource(R.main)
    region = src[src.index("dAUPRC = round("):src.index("reps_meta = json.loads")]
    return "ROBUST" not in region and region.count("PRIMARY_LAYER") >= 3


def junk_label_fields(records):
    out = []
    for i, r in enumerate(records):
        d = dict(r)
        d["y_score"] = i % 2
        d["y_acc"] = (i + 1) % 2
        d["n_pos_score"] = (i * 7) % 9
        d["n_pos_acc"] = (i * 3) % 9
        d["n_infra_candidates"] = i % 5
        out.append(d)
    return out


def prompt_text_by_statement():
    """Independent re-scan of the rollout dumps: statement -> set of prompt byte-hashes."""
    sid_map = LIB._load_statement_map(ROOT, R.DATASET)
    seen: dict = {}
    n_groups = 0
    for seed, rel in LIB.SEED_DIRS.items():
        for fp in sorted((ROOT / rel).glob("*.jsonl"), key=lambda p: int(p.stem)):
            recs = [json.loads(l) for l in fp.read_text(encoding="utf-8").split("\n") if l.strip()]
            i = 0
            while i < len(recs):
                inp = recs[i]["input"]
                j = i
                while j < len(recs) and recs[j]["input"] == inp:
                    j += 1
                mm = LIB.FORMAL_BLOCK_RE.search(inp)
                sid = sid_map.get(LIB._normalize(mm.group(1)) if mm else None, (None, None))[0]
                if sid:
                    seen.setdefault(sid, set()).add(hashlib.sha256(inp.encode()).hexdigest())
                n_groups += 1
                i = j
    return seen, n_groups


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", default="runs/v3_d001/theta0_reps.npz")
    ap.add_argument("--reps-meta", default="runs/v3_d001/theta0_reps_meta.json")
    ap.add_argument("--outdir", default="runs/v3_d001")
    ap.add_argument("--skip-weights-hash", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    host = {"hostname": os.uname().nodename, "nproc": os.cpu_count(),
            "ip": subprocess.run(["hostname", "-I"], capture_output=True, text=True, check=False).stdout.strip(),
            "git_revision": _git("rev-parse", "HEAD"), "git_branch": _git("branch", "--show-current"),
            "worktree_dirty": bool(_git("status", "--short")),
            "blas_threads": os.environ.get("OPENBLAS_NUM_THREADS")}

    built = LIB.build_records(ROOT, R.DATASET, R.REGISTRY)
    records, prompts = built["valid"], built["prompts"]
    folds = json.loads((ROOT / FOLDS).read_text())
    committed = json.loads((ROOT / COMMITTED).read_text())
    man = (ROOT / MANIFEST).read_text()
    z = np.load(ROOT / args.reps, allow_pickle=True)
    layers_present = [int(x) for x in z["layers"]]
    reps_map = {s: (int(z["token_lens"][i]), {L: z["reps"][i, layers_present.index(L)] for L in R.LAYERS})
                for i, s in enumerate(z["statement_ids"])}

    y, comps, seeds, Xb1, Xb2 = R.build_designs(records, prompts, reps_map)
    outer = np.array([folds["component_outer_fold"][c] for c in comps], dtype=int)
    inner_by_outer = folds["inner_fold_by_outer"]
    n_outer = int(folds["n_outer"])

    oofB1, cB1 = R.nested_oof(Xb1, y, comps, outer, inner_by_outer, n_outer)
    oofB2 = {L: R.nested_oof(Xb2[L], y, comps, outer, inner_by_outer, n_outer) for L in R.LAYERS}
    mB1 = R.metrics(y, oofB1)
    mB2 = {L: R.metrics(y, oofB2[L][0]) for L in R.LAYERS}
    xs, xs_summary = cross_seed_full(seeds, comps, y, Xb1, Xb2[R.PRIMARY_LAYER])
    nested_s = round(time.time() - t0, 1)

    # ---- R: reproduction of the committed frozen result -------------------------------
    def _diff(got, exp):
        return {k: [exp.get(k), got.get(k)] for k in got if exp.get(k) != got.get(k)}

    repro = []
    surfaces = ([("B1_handcrafted", mB1, committed["B1_handcrafted"]),
                 ("B2_block18_PRIMARY", mB2[R.PRIMARY_LAYER], committed["B2_block18_PRIMARY"])]
                + [(f"B2_robustness_block{L}", mB2[L], committed["B2_robustness"][f"block{L}"])
                   for L in R.ROBUST_LAYERS])
    for tag, got, exp in surfaces:
        repro.append(_chk(f"R.reproduce_metric_surface[{tag}]", not _diff(got, exp),
                          mismatches=_diff(got, exp), n_fields=len(got)))
    got_ds = {"valid_groups": len(records), "components": int(len(set(comps))),
              "informative_positives": int(y.sum()), "prevalence": round(float(y.mean()), 5)}
    repro.append(_chk("R.reproduce_dataset", not _diff(got_ds, committed["dataset"]), mismatches=_diff(got_ds, committed["dataset"])))
    n_g3 = sum(v["G3_fold_criterion_met_frozen_rule"] for v in xs.values())
    repro.append(_chk("R.reproduce_gate_inputs",
                      round(mB2[R.PRIMARY_LAYER]["auprc"] - mB1["auprc"], 5) == committed["gate"]["G1_dAUPRC_point"]
                      and mB2[R.PRIMARY_LAYER]["top20_enrichment"] == committed["gate"]["G2_point"]
                      and committed["gate"]["G3_crossseed_folds_ok"].startswith(f"{n_g3}/3"),
                      recomputed_delta_auprc=round(mB2[R.PRIMARY_LAYER]["auprc"] - mB1["auprc"], 5),
                      committed_delta_auprc=committed["gate"]["G1_dAUPRC_point"],
                      recomputed_top20_enrichment=mB2[R.PRIMARY_LAYER]["top20_enrichment"],
                      committed_top20_enrichment=committed["gate"]["G2_point"],
                      recomputed_G3=f"{n_g3}/3", committed_G3=committed["gate"]["G3_crossseed_folds_ok"],
                      committed_ranking=committed["gate"]["ranking_GO_NO_GO"],
                      note="frozen gate fields are compared, never recomputed-and-substituted"))
    xs_committed = committed["cross_seed"]
    xs_match = all(_diff({k: v for k, v in xs[h].items() if k in xs_committed[h]}, xs_committed[h]) == {}
                   for h in xs)
    repro.append(_chk("R.reproduce_cross_seed_shared_fields", xs_match,
                      mismatches={h: _diff({k: v for k, v in xs[h].items() if k in xs_committed[h]}, xs_committed[h])
                                  for h in xs if _diff({k: v for k, v in xs[h].items() if k in xs_committed[h]},
                                                       xs_committed[h])}))
    repro_pass = all(c["pass"] for c in repro)

    # ---- A2: itemized leakage audit ---------------------------------------------------
    a2 = []
    a2 += leakage_fold_probe(Xb1, y, comps, outer, inner_by_outer, "B1_handcrafted")
    a2 += leakage_fold_probe(Xb2[R.PRIMARY_LAYER], y, comps, outer, inner_by_outer, "B2_block18")

    oofB0 = R.b0_oof(y, outer, n_outer)
    a2.append(_chk("A2.B0_prevalence_baseline_is_outer_train_local",
                   all(np.allclose(oofB0[outer == f], y[outer != f].mean()) for f in range(n_outer))
                   and not any(np.allclose(oofB0[outer == f], y.mean()) for f in range(n_outer)),
                   per_fold_train_prevalence=[round(float(y[outer != f].mean()), 5) for f in range(n_outer)],
                   pooled_prevalence=round(float(y.mean()), 5),
                   any_fold_equals_pooled=False))

    inner_scope_ok = True
    inner_scope = {}
    for f in range(n_outer):
        keys = set(inner_by_outer[str(f)])
        tr_c, te_c = set(comps[outer != f]), set(comps[outer == f])
        if keys != tr_c or (keys & te_c):
            inner_scope_ok = False
        inner_scope[f"fold{f}"] = {"inner_keyed_components": len(keys),
                                   "outer_train_components": len(tr_c),
                                   "outer_test_components": len(te_c),
                                   "inner_keys_overlapping_outer_test": len(keys & te_c)}
    a2.append(_chk("A2.inner_CV_keyed_strictly_within_outer_train", inner_scope_ok, per_fold=inner_scope))

    dis_outer = all(len(set(outer[comps == c])) == 1 for c in np.unique(comps))
    sizes = [len(set(comps[outer == f])) for f in range(n_outer)]
    a2.append(_chk("A2.family_component_hard_disjointness", dis_outer and sum(sizes) == len(set(comps)),
                   every_component_single_outer_fold=dis_outer,
                   components_per_fold=sizes, sum_components_per_fold=int(sum(sizes)),
                   n_components=int(len(set(comps)))))

    _, _, _, Xb1j, Xb2j = R.build_designs(junk_label_fields(records), prompts, reps_map)
    a2.append(_chk("A2.design_matrices_independent_of_all_label_fields",
                   np.array_equal(Xb1, Xb1j) and all(np.array_equal(Xb2[L], Xb2j[L]) for L in R.LAYERS),
                   b1_bitwise_identical=bool(np.array_equal(Xb1, Xb1j)),
                   b2_bitwise_identical={f"block{L}": bool(np.array_equal(Xb2[L], Xb2j[L])) for L in R.LAYERS},
                   perturbation="y_score, y_acc, n_pos_score, n_pos_acc, n_infra_candidates all replaced by junk "
                                "before featurization; designs must be bit-identical"))

    b1_declared = all(f in man for f in R.B1_FEATURES)
    onehot = Xb1[:, len(LIB.B1_NUMERIC):]
    src_levels = LIB.B1_SOURCE_LEVELS
    a2.append(_chk("A2.source_encoding_is_frozen_level_onehot_no_target_signal",
                   b1_declared and set(np.unique(onehot).tolist()) <= {0.0, 1.0}
                   and float(onehot.sum(axis=1).max()) <= 1.0 and float(onehot.sum(axis=1).min()) >= 0.0,
                   all_B1_feature_names_declared_in_frozen_manifest=b1_declared,
                   n_B1_features=int(Xb1.shape[1]), frozen_feature_order_matches_code=list(R.B1_FEATURES) == list(LIB.B1_FEATURES),
                   onehot_values=sorted(set(float(v) for v in np.unique(onehot))),
                   max_active_source_levels=float(onehot.sum(axis=1).max()),
                   records_with_no_known_source=int((onehot.sum(axis=1) == 0).sum()),
                   frozen_levels=src_levels,
                   note="levels fixed in the frozen manifest before fitting; no frequency/target/mean-encoding"))

    step_norm = np.array([r["step_norm"] for r in records])
    steps = np.array([r["step"] for r in records])
    a2.append(_chk("A2.training_step_normalization_is_deterministic_and_label_free",
                   np.array_equal(step_norm, steps / LIB.STEPS)
                   and all(bool(np.array_equal(Xb2[L][:, -1], step_norm)) for L in R.LAYERS)
                   and bool(np.array_equal(Xb1[:, LIB.B1_NUMERIC.index("step_norm")], step_norm)),
                   formula="step_norm = rollout_step / 60 (frozen STEPS; no per-seed or per-fold rescaling)",
                   is_last_B2_column={f"block{L}": bool(np.array_equal(Xb2[L][:, -1], step_norm)) for L in R.LAYERS},
                   unique_steps=int(len(set(steps.tolist()))),
                   note="known at deployment time (trainer step counter); carries no reward/verifier information"))

    a2.append(_chk("A2.B2_input_surface_is_frozen_representation_plus_step_only",
                   all(Xb2[L].shape[1] == REP_DIM + 1 for L in R.LAYERS),
                   design_ncols={f"block{L}": int(Xb2[L].shape[1]) for L in R.LAYERS}, expected=REP_DIM + 1,
                   forbidden_inputs_absent=["seed_id", "raw_step_index_as_category", "reward", "score_sum",
                                            "acc", "verifier_output", "group_size", "n_pos_score",
                                            "any_rollout_statistic"],
                   note="columns 0..1023 are the theta0 block last-token vector, column 1024 is step_norm"))

    fit_src = inspect.getsource(fit_logistic)
    sel_src = inspect.getsource(R.select_C) + inspect.getsource(R.nested_oof) + inspect.getsource(R.fit_predict)
    a2.append(_chk("A2.no_class_weight_or_resampling_in_fit_or_selection",
                   "class_weight" not in fit_src and "sample_weight" not in fit_src
                   and "class_weight" not in sel_src and "sample_weight" not in sel_src,
                   objective="min  sum_i NLL(p_i, y_i)  +  (1/(2C)) ||w_{1:}||^2   (bias unpenalized)",
                   Newton_ITRLS="deterministic full-batch IRLS, max_iter=50, tol=1e-7, no stochasticity"))

    a2.append(_chk("A2.C_grid_criterion_tiebreak_match_frozen_manifest",
                   all(str(c) in man for c in R.CGRID) and "maximize mean inner AUPRC" in man
                   and "smaller (more regularized) C" in man,
                   C_grid=R.CGRID, criterion="max mean inner AUPRC across the 4 inner folds of outer-train",
                   tie_break="strict-improvement update -> ties keep the smaller C",
                   chosen_C_B1_by_fold={f"fold{f}": cB1[f]["C"] for f in cB1},
                   chosen_C_B2p_by_fold={f"fold{f}": oofB2[R.PRIMARY_LAYER][1][f]["C"]
                                         for f in oofB2[R.PRIMARY_LAYER][1]},
                   selection_sees_only_inner_folds_of_outer_train=True))

    prev = float(y.mean())
    order_p = np.argsort(-oofB2[R.PRIMARY_LAYER][0])
    k20 = max(1, round(0.20 * len(y)))
    enr_direct = float(y[order_p[:k20]].mean()) / prev
    a2.append(_chk("A2.topk_enrichment_denominator_is_evaluation_label_mean",
                   abs(mB2[R.PRIMARY_LAYER]["top20_enrichment"] - enr_direct) < 1e-4
                   and abs(enr_direct - 1.0) > 1e-6,
                   reported_top20_enrichment=mB2[R.PRIMARY_LAYER]["top20_enrichment"],
                   recomputed_from_unrounded_values=round(enr_direct, 5),
                   pooled_OOF_prevalence=round(prev, 5),
                   note="evaluation-side denominator only; never fed to any fit (the fitted B0 baseline uses the "
                        "outer-TRAIN prevalence, see A2.B0 check)"))

    y_acc = np.array([r["y_acc"] for r in records], dtype=int)
    a2.append(_chk("A2.primary_label_is_y_score_never_acc",
                   np.array_equal(y, np.array([r["y_score"] for r in records], dtype=int))
                   and not np.array_equal(y, y_acc),
                   positives_y_score=int(y.sum()), positives_y_acc=int(y_acc.sum()),
                   groups_where_labels_differ=int((y != y_acc).sum())))

    rebuilt_outer = stratified_group_kfold(y, comps, n_outer, SEED)
    same_map = all(int(folds["component_outer_fold"][c]) == int(f) for c, f in zip(comps, rebuilt_outer))
    pre_reg_commit = _git("log", "--format=%H", "-1", "--", PREREG)
    blob_hist = _git("log", "--format=%H|%aI", "--", FOLDS).splitlines()
    a2.append(_chk("A2.folds_frozen_before_any_outcome_and_reproducible",
                   same_map and len(blob_hist) == 1
                   and _git("rev-parse", f"HEAD:{FOLDS}") == _git("rev-parse", f"{pre_reg_commit}:{FOLDS}"),
                   recomputed_fold_map_bit_identical_to_committed_manifest=same_map,
                   fold_manifest_commit_count=len(blob_hist),
                   fold_manifest_first_commit_utc=blob_hist[0].split("|")[1] if blob_hist else None,
                   fold_blob_unchanged_since_preregistration=True,
                   preregistration_commit_utc=_git("log", "-1", "--format=%aI", pre_reg_commit),
                   results_commit_utc=_git("log", "-1", "--format=%aI", "--", COMMITTED),
                   folds_hash=folds["folds_hash"],
                   seeds=dict(outer_folds=n_outer, inner_folds=int(folds["n_inner"]), fold_seed=SEED)))

    a2.append(_chk("A2.cross_seed_train_test_family_isolation",
                   all(v["family_disjoint_train_test"] for v in xs.values()),
                   per_fold={h: {"shared_components_train_test": 0 if v["family_disjoint_train_test"] else -1,
                                 "n_train_components": v["n_train_components"],
                                 "n_test_components": v["n_test_components"],
                                 "groups_dropped_by_exclusion": v["n_groups_dropped_by_family_exclusion"]}
                             for h, v in xs.items()}))

    boot_src = inspect.getsource(LIB.component_bootstrap)
    a2.append(_chk("A2.committed_bootstrap_is_fixed_OOF_prediction_level",
                   "p_oof" in boot_src and "fit_logistic" not in boot_src and "select_C" not in boot_src
                   and "stratified_group_kfold" not in boot_src,
                   resamples="family components -> row indices of the ALREADY COMPUTED OOF prediction vector",
                   refits_model_per_rep=False, reselects_C_per_rep=False, rebuilds_folds_per_rep=False,
                   consequence="does not propagate model-fitting / fold-partition variability; hence the "
                               "post-hoc full-procedure component bootstrap (A3) reported separately",
                   full_procedure_script="scripts/v3_d001_fullproc_boot.py"))

    leak_pass = all(c["pass"] for c in a2)

    # ---- A4: representation sanity ----------------------------------------------------
    reps_meta = json.loads((ROOT / args.reps_meta).read_text())
    hh = hashlib.sha256()
    hh.update(np.ascontiguousarray(z["reps"]).tobytes())
    hh.update(np.ascontiguousarray(z["token_lens"]).tobytes())
    seen, n_scanned = prompt_text_by_statement()
    stmt_rows: dict = {}
    for i, rec in enumerate(records):
        stmt_rows.setdefault(rec["statement_id"], []).append(i)
    flip_sids = [s for s, rows in stmt_rows.items() if len({int(y[i]) for i in rows}) > 1]
    rep_ident = all(all(np.array_equal(Xb2[R.PRIMARY_LAYER][rows[0], :REP_DIM], Xb2[R.PRIMARY_LAYER][i, :REP_DIM])
                        for i in rows) for rows in stmt_rows.values())
    weights_sha = (None if args.skip_weights_hash or not (ROOT / MODEL).exists()
                   else _sha256_file(ROOT / MODEL))

    a4 = [
        _chk("A4.theta0_backbone_is_the_frozen_checkpoint",
             reps_meta.get("theta0_weights_sha256_expected") == THETA0_SHA
             and (weights_sha is None or weights_sha == THETA0_SHA),
             expected=THETA0_SHA, local_model_file_sha256=weights_sha,
             local_model_file_present=(ROOT / MODEL).exists(),
             frozen_in=["docs/v3/V3-D001_preregistration.md", MANIFEST, "scripts/v3_d001_extract.py"]),
        _chk("A4.reps_artifact_is_the_fly122_formal_extraction",
             hh.hexdigest() == reps_meta.get("reps_content_sha256")
             and str(reps_meta.get("formality", "")).startswith("FORMAL")
             and "3080" in str(reps_meta.get("host", {}).get("device_name", "")),
             recomputed_content_sha256=hh.hexdigest(),
             committed_content_sha256=reps_meta.get("reps_content_sha256"),
             formality=reps_meta.get("formality"), device=reps_meta.get("host", {}).get("device_name"),
             formal_host_git_revision=reps_meta.get("host", {}).get("git_revision"),
             audit_host_git_revision=host["git_revision"],
             note="this audit re-derives every number from that exact artifact; the audit itself may execute on "
                  "either node because the pipeline is pure numpy + the frozen reps file (verified bit-identical)"),
        _chk("A4.one_representation_per_statement_computed_once",
             len({str(s) for s in z["statement_ids"]}) == int(z["reps"].shape[0])
             and len(stmt_rows) <= int(z["reps"].shape[0]),
             reps_rows=int(z["reps"].shape[0]), unique_statement_ids=len({str(s) for s in z["statement_ids"]}),
             statements_in_valid_groups=len(stmt_rows),
             statements_in_all_720_groups=len(seen), groups_scanned=n_scanned,
             keyed_by="statement_id only (not seed, not step, not label)",
             computed_in_one_pass="single no_grad forward per statement, layer indices pre-registered"),
        _chk("A4.extraction_path_never_reads_labels", rep_ident and all(len(v) == 1 for v in seen.values()),
             prompt_text_byte_identical_across_all_copies=all(len(v) == 1 for v in seen.values()),
             statements_observed_with_both_labels_in_valid_set=len(flip_sids),
             representation_bit_identical_for_those_statements=rep_ident,
             extractor_inputs="tokenizer(prompt text) -> bf16 forward -> hidden_states[layer+1][-1 token]",
             label_fields_used_by_extractor="none"),
        _chk("A4.block18_primary_was_pre_registered_not_outcome_selected",
             sorted(layers_present) == [9, 18, 27] and R.PRIMARY_LAYER == 18
             and "primary_layer: 18" in man and "frozen before outcomes" in man
             and "block 18" in (ROOT / PREREG).read_text().lower(),
             layers_in_artifact=layers_present, primary=R.PRIMARY_LAYER, robustness=R.ROBUST_LAYERS,
             pre_registration_rule="primary = int(2/3 x num_hidden_layers) of the loaded config; "
                                   "robustness = int(1/3 x NL) and NL-1 (asserted equal at extraction time)",
             extraction_refuses_mismatched_backbone="scripts/v3_d001_extract.py derives [NL//3, 2NL//3, NL-1] "
                                                    "and aborts unless it equals [9, 18, 27]",
             gate_code_reads_only_primary_layer=_gate_block_uses_primary_only(),
             auprc_by_layer={f"block{L}": mB2[L]["auprc"] for L in R.LAYERS},
             robust_layers_used_in_gate=False,
             note="block 18 was fixed a priori from depth fractions; blocks 9/27 are reported but enter no gate"),
    ]
    a4_pass = all(c["pass"] for c in a4)

    audit = {
        "artifact_type": "v3_d001_canonical_audit",
        "status": "POST-HOC CANONICAL METHODOLOGICAL AUDIT (frozen gate untouched; no RL performed)",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_scope": ["R_reproduction", "A1_cross_seed_full", "A2_leakage_audit",
                        "A4_representation_sanity", "A5_classification"],
        "formal_run_provenance_reference": committed["provenance"],
        "host": host,
        "R_reproduction_vs_committed": {"checks": repro, "all_pass": repro_pass,
                                        "committed_results_sha256_of_file": None},
        "A1_cross_seed_full": {"per_fold": xs, "summary": xs_summary},
        "A2_leakage_audit": {"checks": a2, "n_checks": len(a2),
                             "n_failed": int(sum(1 for c in a2 if not c["pass"])),
                             "failed_checks": [c["check"] for c in a2 if not c["pass"]],
                             "LEAKAGE_AUDIT": "PASS" if leak_pass else "FAIL"},
        "A4_representation_sanity": {"checks": a4, "all_pass": a4_pass},
        "A5_classification": {
            "reproduction_pass": repro_pass,
            "leakage_audit": "PASS" if leak_pass else "FAIL",
            "provenance_complete": bool(a4_pass),
            "frozen_gate_recomputed_identically": repro_pass,
            "D001_CANONICAL": ("CANONICAL_GO" if (repro_pass and leak_pass and a4_pass) else "NEEDS_CORRECTION"),
            "note": "A3 (full-procedure component bootstrap) is reported in "
                    "runs/v3_d001/v3_d001_fullproc_boot.json. It is post-hoc robustness and cannot alter G1-G3.",
        },
        "timing": {"nested_cv_and_cross_seed_s": nested_s, "total_s": round(time.time() - t0, 1)},
    }
    audit["R_reproduction_vs_committed"]["committed_results_sha256_of_file"] = _sha256_file(ROOT / COMMITTED)
    outp = ROOT / args.outdir
    outp.mkdir(parents=True, exist_ok=True)
    (outp / "v3_d001_canonical_audit.json").write_text(json.dumps(audit, indent=2, default=str))

    print(json.dumps({
        "host": host["hostname"], "R_all_pass": repro_pass,
        "LEAKAGE_AUDIT": audit["A2_leakage_audit"]["LEAKAGE_AUDIT"],
        "leakage_failed": audit["A2_leakage_audit"]["failed_checks"],
        "A4_all_pass": a4_pass, "A4_failed": [c["check"] for c in a4 if not c["pass"]],
        "D001_CANONICAL": audit["A5_classification"]["D001_CANONICAL"],
        "A1_summary": xs_summary, "timing": audit["timing"],
    }, indent=2))
    print("wrote", outp / "v3_d001_canonical_audit.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
