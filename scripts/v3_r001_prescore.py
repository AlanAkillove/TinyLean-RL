#!/usr/bin/env python3
"""V3-R001 §6-§7 / §17 step 7 — freeze the controller's PROSPECTIVE predictions and the sample.

Owner §6 requires that every candidate theorem -- including the ones that have never been rolled
out and therefore have no label -- receives a q_i = P(informative | theorem_i, step_norm=0) from the
FROZEN D001 controller. "Undefined because unlabelled" is explicitly not allowed, and no q may be
produced after an outcome is seen. This script is the freezing step. It touches no verifier, generates
nothing, and takes no gradient.

What "frozen" means here, precisely:
  * feature surface: block-18 last-token theta0 vector + the step_norm slot (B2), and D001's 15
    handcrafted statement-only features (B1) -- both taken from V3-D001's own code, unchanged;
  * estimator: the same L2 logistic IRLS, the same standardization, the same C grid and the same
    inner-CV AUPRC selection criterion and splitter (v3_d001_run.select_C / v3_d001_lib);
  * coefficients: fit ONCE on D001's 686 historical labelled groups. Nothing about R001 enters the
    fit -- not its sample, not its representations' labels (it has none), not its pool hash.
  * step_norm = 0.0 for every candidate: the pre-update / initial-policy condition (owner §6).
  * robustness scorer: the average of D001's five frozen outer-fold models, i.e. the OOF ensemble.
    Reported as a check on the deployment head; the deployment head is primary and that choice is
    written here, before any outcome exists.

Outputs: experiments/manifests/v3/V3-R001_predictions.json  (raw reps stay gitignored on the host)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# cap BLAS threads before numpy imports, exactly as V3_D001_run does (same numerics, same cost)
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import v3_d001_run as R
import v3_r001_power as PW  # the frozen exact-power machinery; imported, never re-derived here
from v3_d001_lib import (
    average_precision,
    build_records,
    predict_proba_logistic,
    standardize_apply,
    standardize_fit,
    stratified_group_kfold,
)
from v3_d001_lib import fit_logistic as fit_logistic_head

POOL = "experiments/manifests/v3/V3-R001_family_clean_pool.json"
POWER = "experiments/manifests/v3/V3-R001_power.json"
FOLDS = "experiments/manifests/v3/v3_d001_folds.json"
D001_RESULTS = "experiments/manifests/v3/V3-D001_results.json"
RAW_PARQUET = "data/raw/kimina_promptset/data/train-00000-of-00001.parquet"
TRAIN_PARQUET = "data/processed/p3_promptset/prompt_sets/AI-MO/Kimina-Prover-Promptset/train.parquet"
REGISTRY = "experiments/manifests/v2/family_component_registry.json"
TOKENIZER = "models/weights/kimina_distill_0_6b"
D001_REPS = "runs/v3_d001/theta0_reps.npz"
D001_META = "runs/v3_d001/theta0_reps_meta.json"

# owner §7: prefer N=128, 192 only if justified; the pool readings are the owner's open §5 choice.
# every (reading, N) pair that the pool can supply is drawn and hashed HERE, so whichever one the
# owner approves was frozen before any rollout result existed.
CANDIDATE_NS = [56, 64, 86, 128, 192, 221]
STEP_NORM_USED = 0.0                                       # owner §6: pre-update / initial policy
DRAW_SEED = 20260924                                       # the directive date; fixed in code
SCORE_KEYS = ["q_B2_controller", "q_B1_handcrafted", "q_B2_fold_ensemble", "q_B1_fold_ensemble"]


def sha(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(json.dumps(p, sort_keys=True, separators=(",", ":"), default=str).encode())
    return h.hexdigest()


def draw_components(component_ids: list[str], n: int, seed: int = DRAW_SEED) -> list[str]:
    """Uniform draw without replacement of `n` family components.

    Order by SHA256("<seed>|<component_id>") instead of a library RNG on purpose: the frozen sample
    must be reproducible on any machine and any numpy version, so that "the sample was fixed before
    the result" stays checkable and not merely asserted.
    """
    keyed = sorted(component_ids, key=lambda c: hashlib.sha256(f"{seed}|{c}".encode()).hexdigest())
    return keyed[:n]


def load_reps(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as z:
        layers = [int(x) for x in z["layers"]]
        missing = [L for L in R.LAYERS if L not in layers]
        if missing:
            raise SystemExit(f"FATAL: {path.name} carries layers {layers} but the pre-registered "
                             f"surface needs {R.LAYERS} (missing {missing}); refusing to index reps "
                             "that are not there")
        # read each member exactly once: a compressed .npz re-inflates the whole array on every
        # slice access, which turned the per-statement map below into ~1800 full decompressions
        ids = [str(s) for s in z["statement_ids"]]
        token_lens = np.asarray(z["token_lens"])
        reps = np.asarray(z["reps"])
        return {
            "layers": layers,
            "ids": ids,
            "token_lens": token_lens,
            "reps": reps,
            "map": {sid: (int(token_lens[i]), {L: reps[i, layers.index(L)] for L in R.LAYERS})
                    for i, sid in enumerate(ids)},
        }


def candidate_matrix(cand_ids: list[str], cand_map: dict, source_of: dict, prompts: dict):
    """B1 and B2 design rows for the candidate theorems, at step_norm = STEP_NORM_USED."""
    n = len(cand_ids)
    Xb1 = np.zeros((n, len(R.B1_FEATURES)))
    Xb2 = np.zeros((n, R.REP_DIM + 1))
    for i, sid in enumerate(cand_ids):
        tl, rep = cand_map[sid]
        prompt = prompts[sid]
        if not R.formal_text(prompt):
            raise SystemExit(f"FATAL: no '# Formal Statement:' block in the prompt of {sid}")
        Xb1[i] = R.extract_b1(R.formal_text(prompt), tl, source_of.get(sid, "unknown"), STEP_NORM_USED)
        Xb2[i, :R.REP_DIM] = rep[R.PRIMARY_LAYER]
        Xb2[i, R.REP_DIM] = STEP_NORM_USED
    return Xb1, Xb2


def fit_predict(X, y, Xc, inner_lbl):
    """The frozen protocol's deployment fit: select C on inner family-grouped CV, then fit on all."""
    C, inner_auprc = R.select_C(X, y, inner_lbl)
    mu, sd = standardize_fit(X)
    w = fit_logistic_head(standardize_apply(X, mu, sd), y.astype(float), C)
    p_in = predict_proba_logistic(standardize_apply(X, mu, sd), w)
    p_cand = predict_proba_logistic(standardize_apply(Xc, mu, sd), w)
    return {"C": C, "inner_auprc": inner_auprc, "w": w, "mu": mu, "sd": sd,
            "p_in_sample": p_in, "p_cand": p_cand}


def fold_ensemble(X, y, comps, Xc, folds, n_outer):
    """Average prediction of D001's five frozen outer-fold models (each never saw the family it
    scored inside D001, and none of them ever saw a candidate family -- candidates are disjoint)."""
    outer = np.array([folds["component_outer_fold"][c] for c in comps], dtype=int)
    preds, chosen = [], {}
    for f in range(n_outer):
        tr = outer != f
        inner_lbl = np.array([folds["inner_fold_by_outer"][str(f)][c] for c in comps[tr]], dtype=int)
        fit = fit_predict(X[tr], y[tr], Xc, inner_lbl)
        preds.append(fit["p_cand"])
        chosen[str(f)] = {"C": fit["C"], "inner_auprc": fit["inner_auprc"]}
    return np.mean(np.vstack(preds), axis=0), chosen


def quantiles(vals) -> dict:
    v = sorted(vals)
    if not v:
        return {"n": 0}
    q = lambda f: v[min(len(v) - 1, int(f * len(v)))]
    return {"n": len(v), "min": round(q(0.0), 5), "p25": round(q(0.25), 5),
            "median": round(statistics.median(v), 5), "p75": round(q(0.75), 5),
            "p90": round(q(0.90), 5), "max": round(q(1.0 - 1e-9), 5),
            "mean": round(statistics.fmean(v), 5)}


def stratum_power(n: int, pi: float, block_fraction: float = 0.20) -> dict:
    """Exact power of the top-block test INSIDE one frozen source stratum, using V3-R001_power's
    frozen machinery (imported, never re-derived). Source is known before generation, so a stratum
    is a legitimate pre-specified conditioning set, not a post-hoc subgroup."""
    m = max(1, round(block_fraction * n))
    if n < 2 or m >= n:
        return {"n_in_stratum": n, "testable": False,
                "why": f"a {block_fraction:.0%} block of {n} theorems is the whole cohort"}
    r = PW.exact_power(n, m, pi, 1.0)
    mde = PW.min_detectable_e(n, m, pi)
    out = {"n_in_stratum": n, "testable": True, "block_size_top20pct": m,
           "prevalence_assumed_from_V1": round(pi, 4),
           "expected_positives_in_block": round(m * pi, 2),
           "realized_size_of_the_exact_test_under_H0": round(r["power"], 4),
           "min_detectable_enrichment_at_80pct_power": (round(mde, 4) if isinstance(mde, (int, float))
                                                        else mde)}
    for e in (2.0, 2.5, 3.47):
        p = PW.exact_power(n, m, pi, e)["power"]
        out[f"power_at_enrichment_{e}"] = round(p, 4) if p is not None else None
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=POOL)
    ap.add_argument("--d001-reps", default=D001_REPS)
    ap.add_argument("--d001-meta", default=D001_META)
    ap.add_argument("--cand-reps", default="runs/v3_r001/theta0_reps.npz")
    ap.add_argument("--cand-meta", default="runs/v3_r001/theta0_reps_meta.json")
    ap.add_argument("--out", default="experiments/manifests/v3/V3-R001_predictions.json")
    ap.add_argument("--allow-non-formal", action="store_true",
                    help="debug escape hatch: score representations from a NON-FORMAL (off-fly122) "
                         "extraction. Refuses to write inside experiments/manifests/, so a tooling "
                         "run can never be committed as the frozen prediction.")
    args = ap.parse_args()

    if not (ROOT / args.cand_reps).exists():
        print(f"ERROR: candidate reps missing ({args.cand_reps}). Run scripts/v3_r001_extract.py (§17 "
              f"step 7) first.", file=sys.stderr)
        return 2
    for needed in (args.d001_reps, args.d001_meta, args.cand_meta, D001_RESULTS, FOLDS, POWER, POOL):
        if not (ROOT / needed).exists():
            raise SystemExit(f"FATAL: required input missing: {needed}")

    cand_meta = json.loads((ROOT / args.cand_meta).read_text())
    non_formal = not str(cand_meta.get("formality", "")).startswith("FORMAL")
    if non_formal:
        if not args.allow_non_formal:
            raise SystemExit("FATAL: the candidate reps are tagged NON-FORMAL. Run "
                             "scripts/v3_r001_extract.py --formal on fly122 (owner §16), or pass "
                             "--allow-non-formal for a tooling check that must not be committed.")
        if (ROOT / args.out).resolve().is_relative_to(ROOT / "experiments"):
            raise SystemExit("FATAL: a NON-FORMAL scoring must not land in experiments/manifests/; "
                             "point --out somewhere under runs/ instead.")

    pool = json.loads((ROOT / args.pool).read_text())
    folds = json.loads((ROOT / FOLDS).read_text())
    power = json.loads((ROOT / POWER).read_text())
    SRC_PI = {k: float(v) for k, v in power["prevalence_anchors"]["source_prevalence_V1"].items()}
    d001 = json.loads((ROOT / D001_RESULTS).read_text())
    hist_meta = json.loads((ROOT / args.d001_meta).read_text())
    hist = load_reps(ROOT / args.d001_reps)
    cand = load_reps(ROOT / args.cand_reps)
    if hist["layers"] != cand["layers"] or R.PRIMARY_LAYER not in cand["layers"]:
        raise SystemExit(f"FATAL: rep layer mismatch {hist['layers']} vs {cand['layers']}")
    if cand_meta.get("prompt_reconstruction_vs_rollout_inputs", {}).get("pass") is not True:
        raise SystemExit("FATAL: the candidate extraction did not pass its own prompt-reconstruction "
                         "check; refusing to score representations of unverified provenance")
    if cand_meta.get("extraction_recipe_check", {}).get("pass") is not True:
        raise SystemExit("FATAL: the candidate extraction did not pass the recipe-identity check "
                         "against V3-D001's formal reps")

    cand_ids = cand["ids"]
    union = sorted(pool["extraction_union"])
    if cand_ids != union:
        raise SystemExit(f"FATAL: extracted {len(cand_ids)} statements but the pool union has "
                         f"{len(union)}; refusing to score a different set than §5 froze")
    leaked = sorted(set(cand_ids) & set(hist["ids"]))
    if leaked:
        raise SystemExit(f"FATAL: {len(leaked)} candidates already appear in the labelled D001 rep "
                         f"set ({leaked[:5]}) -- the pool would not be family-clean")

    # --- historical training surface, built exactly as V3-D001 built it ---------------------------
    built = build_records(ROOT, RAW_PARQUET, REGISTRY)
    records = built["valid"]
    y, comps, _seeds, Xb1h, Xb2h_all = R.build_designs(records, built["prompts"], hist["map"])
    Xb2h = Xb2h_all[R.PRIMARY_LAYER]                       # block 18 = D001's pre-registered primary
    step_norm_support = [float(min(r["step_norm"] for r in records)),
                         float(max(r["step_norm"] for r in records))]
    outer_ref = np.array([folds["component_outer_fold"][c] for c in comps], dtype=int)
    recomputed = stratified_group_kfold(y, comps, int(folds["n_outer"]), int(folds["seed"]))
    fold_alignment = {
        "n_records": len(records),
        "matches_frozen_manifest": bool(np.array_equal(outer_ref, recomputed)),
        "n_components": len(set(comps)),
    }
    if not fold_alignment["matches_frozen_manifest"]:
        raise SystemExit(f"FATAL: record order no longer reproduces the frozen fold manifest: "
                         f"{fold_alignment}")
    ds = d001["dataset"]
    fit_surface = {"n_groups": len(records), "n_positives": int(y.sum()),
                   "n_components": len(set(comps)), "prevalence": round(float(y.mean()), 4)}
    if any(fit_surface[k] != v for k, v in
           [("n_groups", ds["valid_groups"]), ("n_positives", ds["informative_positives"]),
            ("n_components", ds["components"]), ("prevalence", ds["prevalence"])]):
        raise SystemExit(f"FATAL: the fitting surface drifted away from V3-D001's: {fit_surface} vs "
                         f"{ds}")

    # --- candidate families: resolved before anything expensive runs ------------------------------
    comp_of = {str(c["component_id"]): c for c in json.loads((ROOT / REGISTRY).read_text())["components"]}
    stmt_to_comp: dict[str, str] = {}
    for cid, c in comp_of.items():
        for m in c["member_statement_ids"]:
            stmt_to_comp[m] = cid
    unresolved = [s for s in cand_ids if s not in stmt_to_comp]
    if unresolved:
        raise SystemExit(f"FATAL: {len(unresolved)} candidates have no family component; the "
                         f"component-clustered CI would be undefined for them")
    # every reading's "one theorem per component" representative must really be a member of the
    # component the frozen V2 registry says it belongs to -- otherwise the sampling unit and the
    # bootstrap unit would silently disagree with the pool artifact that §5 froze.
    bad_reps = [(r, c, s) for r, p in pool["pools_by_reading"].items()
                for c, s in p["component_to_candidate"].items()
                if stmt_to_comp.get(s) != c]
    if bad_reps:
        raise SystemExit(f"FATAL: {len(bad_reps)} pool representatives are not registry members of "
                         f"their component, e.g. {bad_reps[:3]}")
    # the stronger form of the family-clean promise: not one candidate may sit in a family V1 rolled
    # out, or "prospective" would mean the same leakage the D001 audit was created to rule out.
    cand_comps = {stmt_to_comp[s] for s in cand_ids}
    hits = sorted(cand_comps & {str(c) for c in comps})
    if hits:
        raise SystemExit(f"FATAL: {len(hits)} candidate family components were used by V1/D001 "
                         f"({hits[:5]}) -- pool is not family-clean, refusing to pre-score")

    # --- candidate prompts / covariates ----------------------------------------------------------
    raw = pd.read_parquet(ROOT / RAW_PARQUET)
    formal = dict(zip(raw["statement_id"].astype(str), raw["formal_statement"].astype(str)))
    source_of = dict(zip(raw["statement_id"].astype(str), raw["source"].astype(str)))
    msgs: dict[str, list] = {}
    df = pd.read_parquet(ROOT / TRAIN_PARQUET)
    for sid, p in zip(df["statement_id"].astype(str), df["prompt"].tolist(), strict=True):
        msgs.setdefault(sid, p.tolist() if hasattr(p, "tolist") else list(p))
    from transformers import AutoTokenizer
    from v3_r001_extract import prompt_of  # the same rendering the GPU pass verified
    tok = AutoTokenizer.from_pretrained(str(ROOT / TOKENIZER))
    prompts = {sid: prompt_of(tok, msgs[sid]) for sid in cand_ids}
    lens_mismatch = [s for s in cand_ids
                     if len(tok(prompts[s], add_special_tokens=False)["input_ids"])
                     != cand["map"][s][0]]
    if lens_mismatch:
        raise SystemExit(f"FATAL: reconstructed prompt tokenizes to a different length than the "
                         f"extraction for {lens_mismatch[:5]}")

    Xb1c, Xb2c = candidate_matrix(cand_ids, cand["map"], source_of, prompts)
    src_uncovered = sorted({source_of.get(s, "unknown") for s in cand_ids} - set(SRC_PI))
    if src_uncovered:
        raise SystemExit(f"FATAL: candidate sources with no V1 prevalence anchor: {src_uncovered} "
                         f"(known: {sorted(SRC_PI)})")

    # --- the two frozen deployment heads ---------------------------------------------------------
    inner_lbl = stratified_group_kfold(y, comps, int(folds["n_inner"]), int(folds["seed"]))
    b2 = fit_predict(Xb2h, y, Xb2c, inner_lbl)
    b1 = fit_predict(Xb1h, y, Xb1c, inner_lbl)
    ens_b2, ens_b2_C = fold_ensemble(Xb2h, y, comps, Xb2c, folds, int(folds["n_outer"]))
    ens_b1, ens_b1_C = fold_ensemble(Xb1h, y, comps, Xb1c, folds, int(folds["n_outer"]))

    # Replication guard: re-fitting D001's own per-fold protocol must reproduce its frozen chosen C
    # and inner AUPRC exactly. If it does not, this script is NOT running the frozen controller and
    # every q below would be from an unregistered model.
    protocol_replication = {}
    for arm, got, ref in (("B2", ens_b2_C, d001["B2_block18_PRIMARY"]["chosen_C_by_outer_fold"]),
                          ("B1", ens_b1_C, d001["B1_handcrafted"]["chosen_C_by_outer_fold"])):
        bad = [f for f in got if got[f] != ref[f]]
        protocol_replication[arm] = {"folds_reproduced": len(got) - len(bad), "n_folds": len(got),
                                     "mismatched_folds": bad}
        if bad:
            raise SystemExit(f"FATAL: {arm} does not reproduce D001's frozen per-fold selection: "
                             f"fold {bad[0]} got {got[bad[0]]} vs frozen {ref[bad[0]]}")

    in_sample = {
        "B2": round(float(average_precision(y, b2["p_in_sample"])), 5),
        "B1": round(float(average_precision(y, b1["p_in_sample"])), 5),
        "d001_oof_reference": {"B2_auprc": d001["B2_block18_PRIMARY"]["auprc"],
                               "B1_auprc": d001["B1_handcrafted"]["auprc"]},
        "note": ("in-sample AUPRC on the groups the head is fitted on; NOT the prospective estimate "
                 "and NOT comparable to D001's OOF numbers, which are reported alongside it only so "
                 "that a much larger in-sample value reads as expected rather than as news. It "
                 "confirms the head is a working fitter on its own training data."),
    }

    candidates = {}
    for i, sid in enumerate(cand_ids):
        candidates[sid] = {
            "component_id": stmt_to_comp[sid],
            "source": source_of.get(sid, "unknown"),
            "prompt_token_count": int(cand["map"][sid][0]),
            "formal_char_count": len(formal.get(sid, "")),
            "step_norm_used": STEP_NORM_USED,
            "q_B2_controller": round(float(b2["p_cand"][i]), 6),
            "q_B1_handcrafted": round(float(b1["p_cand"][i]), 6),
            "q_B2_fold_ensemble": round(float(ens_b2[i]), 6),
            "q_B1_fold_ensemble": round(float(ens_b1[i]), 6),
            "has_historical_label": False,
        }
    missing_q = [s for s, c in candidates.items()
                 if not all(0.0 <= c[k] <= 1.0 and np.isfinite(c[k]) for k in SCORE_KEYS)]
    if missing_q:
        raise SystemExit(f"FATAL: {len(missing_q)} candidate theorems without a usable q "
                         f"(owner §6 forbids: {missing_q[:5]})")

    # --- prospective samples, one per (reading, N) the owner may still choose --------------------
    samples = {}
    for reading, p in pool["pools_by_reading"].items():
        cap = len(p["candidate_component_ids"])
        for n in CANDIDATE_NS:
            if n > cap:
                continue
            drawn = draw_components(sorted(p["candidate_component_ids"]), n)
            sids = [p["component_to_candidate"][c] for c in drawn]
            a = [candidates[s]["q_B2_controller"] for s in sids]
            b = [candidates[s]["q_B1_handcrafted"] for s in sids]
            mix = {src: sum(1 for x in sids if candidates[x]["source"] == src)
                   for src in sorted({candidates[x]["source"] for x in sids})}
            conc = sum(1 for i in range(n) for j in range(i + 1, n) if (a[i] - a[j]) * (b[i] - b[j]) > 0)
            disc = sum(1 for i in range(n) for j in range(i + 1, n) if (a[i] - a[j]) * (b[i] - b[j]) < 0)
            key = f"{reading}|N={n}"
            samples[key] = {
                "reading": reading, "N": n, "capacity": cap,
                "component_ids": drawn, "statement_ids": sids,
                "sample_sha256": sha({"reading": reading, "seed": DRAW_SEED, "components": drawn}),
                "q_B2_vector_sha256": sha(a),
                "q_B1_vector_sha256": sha(b),
                "source_mix": mix,
                "prompt_token_count": quantiles([candidates[s]["prompt_token_count"] for s in sids]),
                "q_summary_B2": quantiles(a),
                "q_summary_B1": quantiles(b),
                "arm_agreement_pairwise_sign": round((conc - disc) / max(conc + disc, 1), 4),
                "arm_agreement_note": ("fraction of unordered pairs on which q_B2 and q_B1 order the "
                                       "two theorems the same way, ties excluded; a diagnostic that "
                                       "the two arms are not the same score, not a test statistic"),
                "power_within_source_stratum": {
                    src: stratum_power(cnt, SRC_PI[src]) for src, cnt in
                    sorted(mix.items())},
            }
    undrawn = {r: "pool capacity 0 components under this reading -- nothing to draw"
               for r, p in pool["pools_by_reading"].items() if not p["candidate_component_ids"]}
    union_set = set(union)
    for key, s in samples.items():
        if len(set(s["component_ids"])) != s["N"] or len(set(s["statement_ids"])) != s["N"]:
            raise SystemExit(f"FATAL: {key} is not one theorem per component")
        if not set(s["statement_ids"]) <= union_set:
            raise SystemExit(f"FATAL: {key} draws a statement outside the frozen pool union")

    top = round(0.20 * len(cand_ids))
    ranked = {k: sorted(cand_ids, key=lambda x, k=k: -candidates[x][k])[:top]
              for k in ("q_B2_controller", "q_B1_handcrafted")}
    result = {
        "artifact_type": "v3_r001_prospective_predictions",
        "status": ("FROZEN_BEFORE_ANY_OUTCOME — controller scores and prospective samples committed "
                   "before a single R001 rollout exists; no label, no generation, no verifier"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorized_by": "owner directive 2026-09-24 §6, §7, §10 and §17 step 7",
        "host": {
            "hostname": subprocess.run(["hostname"], capture_output=True, text=True,
                                       check=False).stdout.strip(),
            "git_revision": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                           capture_output=True, text=True,
                                           check=False).stdout.strip(),
            "numpy": np.__version__,
        },
        "question": ("Does the frozen D001 controller give every family-clean candidate a prospective "
                     "q, and is the sample that will be drawn fixed before any outcome?"),
        "controller": {
            "identity": ("V3-D001's frozen protocol: theta0 block-18 last-token representation + "
                         "step_norm slot -> standardize -> L2 logistic (Newton/IRLS), C selected by "
                         "inner family-grouped CV on AUPRC over D001's grid, from v3_d001_run.py"),
            "fit_data": {**fit_surface,
                         "provenance": "V1 seed1/2/3 rollouts reconstructed by v3_d001_lib.build_records",
                         "d001_frozen_dataset_reference": ds},
            "step_norm_at_scoring": STEP_NORM_USED,
            "step_norm_caveat": (
                f"the fit data's step_norm spans {step_norm_support[0]:.4f}-{step_norm_support[1]:.2f} "
                "(V1 logged from the first optimizer step), so the owner's pre-update value 0.0 sits "
                "just below that support. Both heads are linear in the features, and 0.0 is the SAME "
                "value for every candidate, so it shifts all logits by one constant: it moves the "
                "absolute q level, not the ranking that §9's statistic depends on."),
            "no_refit_promise": ("these coefficients are never updated by R001 data; nothing about "
                                 "R001 entered the fit, and no further fitting happens afterwards"),
            "inner_partition_for_deployment_C": (
                f"stratified_group_kfold(y, components, k={folds['n_inner']}, seed={folds['seed']}) "
                "recomputed over ALL fitting groups: v3_d001_folds.json stores one inner map PER "
                "outer fold, and a full-data deployment fit has no held-out outer fold, so none of "
                "those frozen inner maps applies to it. Same splitter, same seed, same criterion, "
                "same C grid as D001's selection rule."),
            "B2_deployment": {"C": b2["C"], "inner_auprc": b2["inner_auprc"],
                              "w_sha256": sha(list(np.round(b2["w"], 12))),
                              "standardization_sha256": sha([list(np.round(b2["mu"], 12)),
                                                             list(np.round(b2["sd"], 12))])},
            "B1_deployment": {"C": b1["C"], "inner_auprc": b1["inner_auprc"],
                              "w_sha256": sha(list(np.round(b1["w"], 12)))},
            "fold_ensemble_C": {"B2": ens_b2_C,
                                "B1": ens_b1_C,
                                "d001_frozen_per_outer_fold_C": {
                                    "B2": d001["B2_block18_PRIMARY"]["chosen_C_by_outer_fold"],
                                    "B1": d001["B1_handcrafted"]["chosen_C_by_outer_fold"]},
                                "role": ("robustness scorer only; the deployment head is primary. The "
                                         "ensemble C values here should equal D001's frozen per-fold "
                                         "values -- same splitter, same seed, same grid, and no "
                                         "candidate data enters either fit.")},
            "in_sample_check": in_sample,
            "frozen_protocol_replication": {
                **protocol_replication,
                "meaning": ("the per-fold C / inner-AUPRC values above are recomputed here and compared "
                            "to V3-D001's frozen artifact; the script refuses to score anything unless "
                            "all 5 folds of BOTH arms match, so q cannot come from a silently different "
                            "model than the one that earned the D001 GO")},
        },
        "inputs": {
            "pool_artifact": args.pool, "pool_hash_primary": pool["pool_hash"],
            "pool_hashes_by_reading": {r: v["pool_hash"] for r, v in pool["pools_by_reading"].items()},
            "candidate_reps": args.cand_reps,
            "candidate_reps_content_sha256": cand_meta.get("reps_content_sha256"),
            "candidate_reps_formality": cand_meta.get("formality"),
            "candidate_reps_recipe_checks": {
                "prompt_reconstruction_vs_rollout_inputs":
                    cand_meta.get("prompt_reconstruction_vs_rollout_inputs"),
                "extraction_recipe_check": cand_meta.get("extraction_recipe_check")},
            "historical_reps": args.d001_reps,
            "historical_reps_content_sha256": hist_meta.get("reps_content_sha256"),
            "historical_reps_formality": hist_meta.get("formality"),
            "historical_reps_host": hist_meta.get("host", {}).get("device_name"),
            "power_artifact": POWER,
            "folds_manifest": FOLDS,
            "d001_results": "experiments/manifests/v3/V3-D001_results.json",
        },
        "coverage": {
            "n_candidates": len(cand_ids),
            "n_with_q": sum(1 for c in candidates.values()
                            if all(0.0 <= c[k] <= 1.0 and np.isfinite(c[k]) for k in SCORE_KEYS)),
            "n_missing_q": len(missing_q),
            "owner_6_assertion": ("every unlabelled theorem has a q; no theorem is scored as "
                                  "'undefined because it has no historical label'"),
            "candidate_readings_covered": {r: len(p["candidate_statement_ids"])
                                           for r, p in pool["pools_by_reading"].items()},
            "component_ids_resolved": len(cand_ids) - len(unresolved),
            "n_candidate_components": len(cand_comps),
            "family_clean_check": ("zero candidate components among the 433 that V1/D001 trained on "
                                   "(asserted, and the script exits if it ever fails)"),
            "readings_with_no_sample_drawn": undrawn,
        },
        "candidate_scores": candidates,
        "prospective_samples": samples,
        "pre_outcome_source_concentration": {
            "measured_on": ("frozen q and the frozen source label of every candidate -- no rollout, "
                            "so this is available NOW and is why the stratified test is designed now "
                            "rather than after seeing labels (owner §11, §14)"),
            "candidates_by_source": {src: sum(1 for c in candidates.values() if c["source"] == src)
                                     for src in sorted(SRC_PI)},
            "mean_q_by_source": {
                arm: {src: round(statistics.fmean([c[k] for c in candidates.values()
                                                   if c["source"] == src]), 5) for src in SRC_PI}
                for arm, k in (("B2", "q_B2_controller"), ("B1", "q_B1_handcrafted"))},
            "top20pct_block_source_mix_of_union": {
                arm: {src: sum(1 for x in ranked[k] if candidates[x]["source"] == src)
                      for src in SRC_PI}
                for arm, k in (("B2", "q_B2_controller"), ("B1", "q_B1_handcrafted"))},
            "consequence": (
                "the pooled top-20% enrichment test cannot separate a controller that ranks genuine "
                "informativeness from one that merely ranks synthetic-source, because the selected "
                "block is source-homogeneous before any outcome exists. So owner §11's source audit "
                "is not a robustness appendix: the preregistration carries a WITHIN-STRATUM test as a "
                "co-primary, and a stratum with too few expected positives is reported as descriptive "
                "rather than as a null result."),
            "within_synthetic_stratum_prevalence_sensitivity": {
                f"pi={pi}": stratum_power(samples["consumed_only|N=128"]["source_mix"]["synthetic"], pi)
                for pi in (SRC_PI["synthetic"], 0.25, 0.20, 0.15)},
            "caveat_on_pi": ("V1's per-source prevalence (0.3444 synthetic) was measured on theorems "
                             "V1 chose to re-draw up to 54 times, so it is an optimistic anchor; the "
                             "sensitivity column above is why the stratified design is not built on "
                             "that single number."),
        },
        "draw_rule": {
            "unit": "family component (owner §7: one component -> its deterministic theorem "
                    "representative; one theorem per family, which is also what makes the power "
                    "analysis exact)",
            "mechanism": "order components by SHA256(f\"{DRAW_SEED}|{component_id}\") and take the "
                         "first N -- a uniform permutation with no library-RNG version dependence",
            "seed": DRAW_SEED, "seed_chosen_as": "the date of the owner directive, fixed in code",
            "all_options_frozen": ("a sample exists for every (reading, N) the pool can supply, so "
                                   "the owner's choice cannot have followed an outcome; exactly one "
                                   "will be designated the analysis sample at launch"),
        },
        "reference_ranking_for_launch_planning": {
            "n_ranked": len(cand_ids),
            "block_fraction": 0.20,
            "top20pct_of_the_candidate_union_by_q_B2": [
                {"statement_id": s, "component_id": candidates[s]["component_id"],
                 "source": candidates[s]["source"], "q_B2": candidates[s]["q_B2_controller"]}
                for s in ranked["q_B2_controller"]],
            "warning": ("this ranks all candidates of all five readings, so its denominator is the "
                        "union, not any analysis sample. It is a pre-outcome ranking diagnostic ONLY. "
                        "The analysis sample is the uniform draw above; this list is not used to "
                        "select theorems, and after a rollout no q may be re-ordered, re-fit or "
                        "re-drawn (owner §7)."),
        },
        "gate_design_from_power": {
            "source": POWER,
            "design_level_named_by_power": "see docs/v3/V3-R001_power.md; the gate is written in the "
                                           "preregistration and names N and the block fraction",
            "b2_vs_b1_significance_required": False,
            "reason": power["delta_auprc_verdict"]["conclusion"],
        },
        "prohibitions_respected": [
            "no rollout, no generation, no verifier call, no n=8 sampling, no label produced",
            "no RL optimizer step and no gradient anywhere in this script",
            "no refit after freezing; no new controller task, MLP, extra layer or mean pooling",
            "no sample chosen after seeing an outcome -- no outcome exists yet",
            "no frozen V2 / V3-D001 artifact modified; all of them read-only inputs",
        ],
    }
    outp = ROOT / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(result, indent=2))
    print(json.dumps({
        "status": result["status"], "n_candidates": len(cand_ids),
        "C": {"B2": b2["C"], "B1": b1["C"]},
        "in_sample_check": in_sample,
        "coverage": {k: v for k, v in result["coverage"].items() if k != "candidate_readings_covered"},
        "samples": {k: {"capacity": v["capacity"], "sample_sha256": v["sample_sha256"],
                        "source_mix": v["source_mix"]} for k, v in samples.items()},
        "q_range_B2": [min(c["q_B2_controller"] for c in candidates.values()),
                       max(c["q_B2_controller"] for c in candidates.values())],
    }, indent=2))
    print("wrote", outp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
