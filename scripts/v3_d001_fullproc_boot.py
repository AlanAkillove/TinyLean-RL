#!/usr/bin/env python3
"""V3-D001 A3 — POST-HOC robustness: full-procedure family-component bootstrap.

The committed V3-D001 CIs use a FIXED-OOF-PREDICTION component bootstrap: it resamples
family components among the rows of the already-computed out-of-fold prediction vector, so
it propagates sampling variability of the EVALUATION but not of the FITTING procedure
(fold partition, inner-CV regularization selection, Newton refit).

This script adds the honest, more expensive variant requested by the owner:

    resample family components (with replacement)
      -> rebuild the nested family-grouped folds from scratch (outer 5 / inner 4, frozen seeds)
      -> reselect C by inner family-grouped CV inside each bootstrap outer-train block
      -> refit B1 and B2(block 18) with the frozen L2-logistic Newton path
      -> regenerate out-of-fold predictions
      -> recompute dAUPRC(B2-B1) and the top-20 enrichment pair

It is ROBUSTNESS ONLY. It reads the frozen artifacts and never writes them; it cannot and
does not change the frozen G1/G2/G3 gate or the committed V3-D001 result.

Output: runs/v3_d001/v3_d001_fullproc_boot.json   (+ stdout summary)
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # per-worker: solve is single-thread optimal here

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import v3_d001_run as R                     # noqa: E402  (env guard must precede numpy)
from v3_d001_lib import (                   # noqa: E402
    average_precision,
    brier,
    build_records,
    stratified_group_kfold,
    topk_enrichment,
)

SEED = R.SEED                              # 20260923, the frozen pipeline seed
N_OUTER = 5
N_INNER = 4
G = {}                                     # fork-inherited read-only design


def build_shared():
    built = build_records(ROOT, R.DATASET, R.REGISTRY)
    records, prompts = built["valid"], built["prompts"]
    z = np.load(ROOT / "runs/v3_d001/theta0_reps.npz", allow_pickle=True)
    layers = [int(x) for x in z["layers"]]
    reps_map = {s: (int(z["token_lens"][i]),
                    {L: z["reps"][i, layers.index(L)] for L in R.LAYERS})
                for i, s in enumerate(z["statement_ids"])}
    y, comps, _seeds, Xb1, Xb2 = R.build_designs(records, prompts, reps_map)
    uniq = np.unique(comps)
    G.update(dict(y=y, comps=comps, Xb1=Xb1, Xb2=Xb2[R.PRIMARY_LAYER],
                  uniq=uniq, comp_rows={c: np.where(comps == c)[0] for c in uniq}))


def rebuild_folds(ys, cs):
    """Nested family-grouped folds rebuilt from scratch on a bootstrap sample."""
    outer = stratified_group_kfold(ys, cs, N_OUTER, SEED)
    inner = {}
    for f in range(N_OUTER):
        tr = outer != f
        ig = cs[tr]
        if len(np.unique(ig)) < N_INNER:
            sub = np.zeros(int(tr.sum()), dtype=int)
        else:
            sub = stratified_group_kfold(ys[tr], ig, N_INNER, SEED + 1000 + f)
        inner[str(f)] = {str(c): int(v) for c, v in zip(ig, sub)}
    return outer, inner


def _enrich2(ys, p):
    return float(topk_enrichment(ys, p, 0.20))


def run_rep(seed_rep: int) -> dict:
    y, comps, Xb1, Xb2, uniq = G["y"], G["comps"], G["Xb1"], G["Xb2"], G["uniq"]
    rng = np.random.default_rng(seed_rep)
    draw = rng.integers(0, len(uniq), len(uniq))
    rows = np.concatenate([G["comp_rows"][uniq[d]] for d in draw])
    ys, cs = y[rows], comps[rows]
    try:
        outer, inner = rebuild_folds(ys, cs)
        oof1, _ = R.nested_oof(Xb1[rows], ys, cs, outer, inner, N_OUTER)
        oof2, _ = R.nested_oof(Xb2[rows], ys, cs, outer, inner, N_OUTER)
    except np.linalg.LinAlgError:
        return {"seed_rep": seed_rep, "ok": False, "why": "lin alg"}
    ap1, ap2 = average_precision(ys, oof1), average_precision(ys, oof2)
    if not (np.isfinite(ap1) and np.isfinite(ap2)):
        return {"seed_rep": seed_rep, "ok": False, "why": "degenerate AUPRC (no positives)"}
    prev = float(ys.mean())
    e1, e2 = _enrich2(ys, oof1), _enrich2(ys, oof2)
    return {
        "seed_rep": seed_rep, "ok": True,
        "n_rows": int(len(rows)), "n_unique_components": int(len(np.unique(cs))),
        "prevalence": round(prev, 5),
        "positives": int(ys.sum()),
        "B1_auprc": round(float(ap1), 5), "B2_auprc": round(float(ap2), 5),
        "delta_auprc": round(float(ap2 - ap1), 5),
        "B1_top20_enrichment": None if prev == 0 else round(e1, 5),
        "B2_top20_enrichment": None if prev == 0 else round(e2, 5),
        "delta_top20_enrichment": None if prev == 0 else round(e2 - e1, 5),
        "B2_top20_enrichment_ge_1.75": bool(prev > 0 and e2 >= 1.75),
        "delta_brier": round(float(brier(ys, oof2) - brier(ys, oof1)), 5),
        "C_reselected": True, "folds_rebuilt": True,
    }


def _worker(seed_rep):
    t = time.time()
    r = run_rep(seed_rep)
    r["seconds"] = round(time.time() - t, 2)
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="runs/v3_d001/v3_d001_fullproc_boot.json")
    ap.add_argument("--committed", default="experiments/manifests/v3/V3-D001_results.json")
    args = ap.parse_args()

    build_shared()
    y = G["y"]
    committed = json.loads(open(ROOT / args.committed).read())

    # reference: the original (non-resampled) nested-CV point estimate, recomputed here
    folds = json.loads(open(ROOT / "experiments/manifests/v3/v3_d001_folds.json").read())
    outer = np.array([folds["component_outer_fold"][c] for c in G["comps"]], dtype=int)
    oof1, _ = R.nested_oof(G["Xb1"], y, G["comps"], outer, folds["inner_fold_by_outer"], N_OUTER)
    oof2, _ = R.nested_oof(G["Xb2"], y, G["comps"], outer, folds["inner_fold_by_outer"], N_OUTER)
    ref = {"B1_auprc": round(float(average_precision(y, oof1)), 5),
           "B2_auprc": round(float(average_precision(y, oof2)), 5),
           "delta_auprc": round(float(average_precision(y, oof2) - average_precision(y, oof1)), 5),
           "B2_top20_enrichment": round(_enrich2(y, oof2), 5),
           "delta_top20_enrichment": round(_enrich2(y, oof2) - _enrich2(y, oof1), 5),
           "delta_brier": round(float(brier(y, oof2) - brier(y, oof1)), 5)}

    seeds = [SEED + 1 + i for i in range(args.reps)]
    t0 = time.time()
    with mp.Pool(args.workers) as pool:
        reps = pool.map(_worker, seeds, chunksize=1)
    ok = [r for r in reps if r["ok"]]
    dropped = len(reps) - len(ok)

    def q(vals, p):
        return round(float(np.quantile(vals, p)), 5)

    d = np.array([r["delta_auprc"] for r in ok], dtype=float)
    e2 = np.array([r["B2_top20_enrichment"] for r in ok if r["B2_top20_enrichment"] is not None], dtype=float)
    de = np.array([r["delta_top20_enrichment"] for r in ok if r["delta_top20_enrichment"] is not None], dtype=float)
    db = np.array([r["delta_brier"] for r in ok], dtype=float)

    out = {
        "artifact_type": "v3_d001_full_procedure_component_bootstrap",
        "status": "POST-HOC ROBUSTNESS ONLY — does NOT modify the frozen V3-D001 gate or committed result",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": {"hostname": os.uname().nodename, "nproc": os.cpu_count(),
                 "workers": args.workers, "blas_threads_per_worker": os.environ["OPENBLAS_NUM_THREADS"],
                 "git_revision": subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()},
        "classification_of_committed_ci": {
            "kind": "fixed-OOF-prediction family-component bootstrap (evaluation-level)",
            "resamples": "family components, holding out-of-fold predictions, folds and C fixed",
            "propagates": "label-sampling variability only",
            "does_not_propagate": ["nested fold-partition variability", "inner-CV C reselection",
                                   "Newton refit variability"],
            "committed_dAUPRC_ci": committed["bootstrap"]["dAUPRC_B2_minus_B1"],
        },
        "procedure": {
            "unit": "family component (433 in the frozen valid set)",
            "resampling": "components drawn with replacement, all rows of a drawn component carried",
            "per_rep": ["rebuild outer 5 grouped folds (seed 20260923)",
                        "rebuild inner 4 grouped folds per outer fold (seed 20260923+1000+f)",
                        "reselect C on inner family-grouped CV (grid + AUPRC criterion + tie->smaller C)",
                        "refit B1 (14 handcrafted) and B2 (block-18 theta0 rep + step) L2 logistic",
                        "regenerate OOF predictions on the bootstrap sample",
                        "recompute dAUPRC, top-20 enrichment pair, dBrier"],
            "rep_seeds": f"{SEED + 1} .. {SEED + args.reps} (deterministic)",
            "n_rep_requested": args.reps, "n_rep_valid": len(ok), "n_rep_dropped": dropped,
            "seed_range": [SEED + 1, SEED + args.reps],
        },
        "reference_original_sample": {**ref,
                                      "committed_B1_auprc": committed["B1_handcrafted"]["auprc"],
                                      "committed_B2_auprc": committed["B2_block18_PRIMARY"]["auprc"],
                                      "committed_delta_auprc": committed["gate"]["G1_dAUPRC_point"],
                                      "matches_committed": bool(
                                          ref["B1_auprc"] == committed["B1_handcrafted"]["auprc"]
                                          and ref["B2_auprc"] == committed["B2_block18_PRIMARY"]["auprc"])},
        "full_procedure_results": {
            "full_procedure_delta_AUPRC_bootstrap_mean": round(float(d.mean()), 5),
            "full_procedure_delta_AUPRC_median": q(d, 0.5),
            "full_procedure_delta_AUPRC_ci95_percentile": [q(d, 0.025), q(d, 0.975)],
            "sign_consistency_fraction_delta_gt_0": round(float((d > 0).mean()), 5),
            "n_reps_delta_le_0": int((d <= 0).sum()),
            "bootstrap_p_value_delta_le_0_one_sided": round(float((d <= 0).mean()), 5),
            "B2_auprc_ci95": [q(np.array([r["B2_auprc"] for r in ok]), 0.025),
                              q(np.array([r["B2_auprc"] for r in ok]), 0.975)],
            "B1_auprc_ci95": [q(np.array([r["B1_auprc"] for r in ok]), 0.025),
                              q(np.array([r["B1_auprc"] for r in ok]), 0.975)],
            "B2_top20_enrichment_ci95": [q(e2, 0.025), q(e2, 0.975)],
            "B2_top20_enrichment_point_ge_1.75_fraction": round(float((e2 >= 1.75).mean()), 5),
            "delta_top20_enrichment_ci95": [q(de, 0.025), q(de, 0.975)],
            "delta_brier_ci95": [q(db, 0.025), q(db, 0.975)],
            "delta_brier_fraction_negative": round(float((db < 0).mean()), 5),
            "median_prevalence_of_bootstrap_samples": q(np.array([r["prevalence"] for r in ok]), 0.5),
        },
        "comparison_to_frozen_claim": {
            "frozen_G1": committed["gate"]["G1_dAUPRC_lower_gt_0"],
            "frozen_dAUPRC_ci_lower": committed["bootstrap"]["dAUPRC_B2_minus_B1"]["lo95"],
            "full_procedure_ci_lower": q(d, 0.025),
            "full_procedure_ci_upper": q(d, 0.975),
            "width_frozen_ci": round(committed["bootstrap"]["dAUPRC_B2_minus_B1"]["hi95"]
                                      - committed["bootstrap"]["dAUPRC_B2_minus_B1"]["lo95"], 5),
            "width_full_procedure_ci": round(q(d, 0.975) - q(d, 0.025), 5),
            "full_procedure_ci_wider_than_frozen": bool((q(d, 0.975) - q(d, 0.025))
                                                        > (committed["bootstrap"]["dAUPRC_B2_minus_B1"]["hi95"]
                                                           - committed["bootstrap"]["dAUPRC_B2_minus_B1"]["lo95"])),
            "sign_agreement_with_frozen_G1": bool((q(d, 0.975) > 0) == bool(committed["gate"]["G1_dAUPRC_lower_gt_0"])),
            "gate_outcome_changed": False,
            "reading": "the frozen GO/NO-GO classification is taken from the committed fixed-OOF bootstrap and "
                       "stands as preregistered; this variant only states how much procedure uncertainty it "
                       "understated",
        },
        "timing": {"total_s": round(time.time() - t0, 1),
                   "mean_s_per_rep": round(float(np.mean([r["seconds"] for r in reps])), 2),
                   "throughput_reps_per_min": round(60.0 * len(reps) / max(time.time() - t0, 1e-9), 2)},
        "per_rep": reps,
    }
    path = ROOT / args.out
    os.makedirs(os.path.dirname(path), exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ["full_procedure_results", "comparison_to_frozen_claim",
                                          "reference_original_sample", "timing"]}, indent=2))
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
