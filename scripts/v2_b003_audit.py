#!/usr/bin/env python
"""V2-B003 post-run audit (owner directive 2026-09-20, before the formal gate reading).

Four audits. None of them changes the preregistered protocol, the fixed 4+4
seed split, the runner, or the B003 gate definition; the preregistered point
estimates stay exactly as produced by v2_b003_analyze.py.

1. positioning - the primary method is a CROSS-FITTED EMPIRICAL RESPONSE
   ORACLE: it uses each theorem's 4 fit-seed outcomes to decide the allocation
   and the other 4 seeds to evaluate it independently, so it remains a
   non-deployable theorem-specific Oracle. Its role is to test the cross-seed
   stability of the compute-response signal, NOT to represent the performance
   of any future predictor.
2. fold-level coverage - besides the overall n_ib >= 6 gate, report the valid
   trial distribution of each 4-replicate half and count theorem-budget cells
   with fewer than 3/4 valid trials in a half. Low missingness is reported
   normally; widespread missingness flips the gate status to
   coverage-limited/inconclusive (missing cells are never silently zero).
3. bootstrap audit - the analyzer's bootstrap holds the allocation fixed
   (conditional CI, theorem-level resampling). This audit adds a
   FULL-PROCEDURE robustness analysis: inside every bootstrap replicate the
   fit-half response is re-estimated, the knapsack allocation is re-solved,
   and the eval half is re-scored (resampling unit = theorem; each theorem is
   its own family component in the family-clean design).
4. headline scope - all headline numbers are scoped to the strict
   family-unseen hard/OOD reserve population; absolute expected solves/gain
   are reported alongside relative gains (rare events inflate percentages).

Resampling unit note: B003 has exactly one representative per family
component, so theorem-level resampling IS component-level resampling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import v2_b003_analyze as an

FOLD_HALF = 4
BAD_CELL_N = 3  # "fewer than 3/4 valid trials in a half"
BAD_SHARE_THRESHOLD = 0.05  # >5% of (theorem, budget) cells per half => coverage-limited
FULL_BOOT_N = 1000
FULL_BOOT_SEED = 20261001
DEFAULT_ROLLOUTS = "experiments/results/v2_b003_rollouts.jsonl"
DEFAULT_ANALYSIS = "experiments/results/v2_b003_analysis.json"
DEFAULT_OUTPUT = "experiments/results/v2_b003_audit.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def primary_subset(y: np.ndarray) -> np.ndarray:
    _, n_valid = an.counts(y)
    return np.array(
        [i for i in range(y.shape[0]) if bool((n_valid[i] >= an.COVERAGE_GATE).all())],
        dtype=np.int64,
    )


def fold_coverage(y: np.ndarray, subset: np.ndarray) -> dict:
    per_fold: dict[str, dict] = {}
    for name, reps in (("fold_A", an.FOLD_A), ("fold_B", an.FOLD_B)):
        n_half = (y[:, reps, :] != an.MISSING).sum(axis=1)[subset]  # (n_sub, 5)
        hist = Counter(int(v) for v in n_half.ravel())
        total_cells = int(n_half.size)
        below3 = int((n_half < BAD_CELL_N).sum())
        per_fold[name] = {
            "hist_valid_trials": {str(i): int(hist.get(i, 0)) for i in range(FOLD_HALF + 1)},
            "cells": total_cells,
            "cells_below_3": below3,
            "share_below_3": below3 / total_cells if total_cells else None,
            "share_below_4": (
                float((n_half < FOLD_HALF).sum()) / total_cells if total_cells else None
            ),
        }
    worst = max(
        per_fold["fold_A"]["share_below_3"] or 0.0,
        per_fold["fold_B"]["share_below_3"] or 0.0,
    )
    return {
        "bad_cell_n": BAD_CELL_N,
        "half_size": FOLD_HALF,
        "threshold_share_below_3": BAD_SHARE_THRESHOLD,
        "per_fold": per_fold,
        "gate_coverage_status": (
            "coverage-limited/inconclusive" if worst > BAD_SHARE_THRESHOLD else "coverage-ok"
        ),
        "note": (
            "each half serves as the fit half in one cross-fitting direction and the eval "
            "half in the other; missing cells are excluded from n, never counted as 0"
        ),
    }


def full_procedure_bootstrap(
    y: np.ndarray,
    subset: np.ndarray,
    tokens: int,
    allow_skip: bool,
    n_boot: int,
    seed: int,
) -> dict:
    n_sub = len(subset)
    y_sub = y[subset]
    cap_units = n_sub * (tokens // 512)
    rng = np.random.default_rng(seed)
    cf_rate = np.empty(n_boot, dtype=np.float64)
    uni_rate = np.empty(n_boot, dtype=np.float64)
    uni_alloc = np.full(n_sub, tokens, dtype=np.int64)
    started = time.perf_counter()
    for b in range(n_boot):
        idx = rng.integers(0, n_sub, size=n_sub)
        yb = y_sub[idx]
        # fit half A -> allocate -> eval half B
        p_a = an.p_hat(
            (yb[:, an.FOLD_A, :] == 1).sum(axis=1), (yb[:, an.FOLD_A, :] != an.MISSING).sum(axis=1)
        )
        _, alloc_a = an.dp_allocate(p_a, cap_units, allow_skip)
        v_a, _ = an.contribution_vector(yb, alloc_a, an.FOLD_B)
        # fit half B -> allocate -> eval half A
        p_b = an.p_hat(
            (yb[:, an.FOLD_B, :] == 1).sum(axis=1), (yb[:, an.FOLD_B, :] != an.MISSING).sum(axis=1)
        )
        _, alloc_b = an.dp_allocate(p_b, cap_units, allow_skip)
        v_b, _ = an.contribution_vector(yb, alloc_b, an.FOLD_A)
        cf_rate[b] = 0.5 * (v_a.mean() + v_b.mean())
        u_a, _ = an.contribution_vector(yb, uni_alloc, an.FOLD_B)
        u_b, _ = an.contribution_vector(yb, uni_alloc, an.FOLD_A)
        uni_rate[b] = 0.5 * (u_a.mean() + u_b.mean())
    abs_gain = (cf_rate - uni_rate) * n_sub
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_gain = np.where(uni_rate > 0, (cf_rate - uni_rate) / uni_rate, np.nan)
    rel_valid = rel_gain[~np.isnan(rel_gain)]
    return {
        "n_boot": n_boot,
        "seed": seed,
        "uniform_budget": tokens,
        "uniform_rate_mean": float(uni_rate.mean()),
        "cf_rate_mean": float(cf_rate.mean()),
        "abs_gain_mean": float(abs_gain.mean()),
        "abs_gain_ci95": [float(np.percentile(abs_gain, 2.5)), float(np.percentile(abs_gain, 97.5))],
        "rel_gain_mean": float(np.mean(rel_valid)) if len(rel_valid) else None,
        "rel_gain_ci95": (
            [float(np.percentile(rel_valid, 2.5)), float(np.percentile(rel_valid, 97.5))]
            if len(rel_valid)
            else None
        ),
        "wall_seconds": round(time.perf_counter() - started, 1),
        "note": (
            "fit-half re-estimation + knapsack re-solve + eval-half re-scoring inside every "
            "replicate; theorem=component resampling; the preregistered point estimate is "
            "unchanged"
        ),
    }


def headline_table(analysis: dict) -> list[dict]:
    rows = []
    for tokens in an.BUDGETS:
        point = analysis["budget_points"][str(tokens)]
        cf = point["cross_fitted"]
        rows.append(
            {
                "uniform_budget": tokens,
                "uniform_expected_solved": point["uniform_expected_solved"],
                "cf_oracle_expected_solved": cf["value"],
                "absolute_gain": point["cross_fitted_gain"]["absolute"],
                "relative_gain": point["cross_fitted_gain"]["relative"],
                "conditional_ci95_relative_gain": point["bootstrap"]["rel_gain_ci95"],
                "conditional_ci95_absolute_gain": point["bootstrap"]["abs_gain_ci95"],
                "savings_allocated_cap_pct": point["savings_cross_fitted"][
                    "allocated_cap_savings_pct"
                ],
                "plug_in_expected_solved_reference": point["plug_in"]["expected_solved"],
                "n_theorems": analysis["config"]["primary_set_size"],
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="V2-B003 post-run audit (four-point directive).")
    parser.add_argument("--rollouts", default=DEFAULT_ROLLOUTS)
    parser.add_argument("--analysis", default=DEFAULT_ANALYSIS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--full-boot-n", type=int, default=FULL_BOOT_N)
    parser.add_argument("--seed", type=int, default=FULL_BOOT_SEED)
    parser.add_argument("--skip-allowed", action="store_true", help="audit the triage space instead")
    args = parser.parse_args()

    def resolve(path: str) -> Path:
        candidate = Path(path)
        return candidate if candidate.is_absolute() else ROOT / candidate

    rollouts_path = resolve(args.rollouts)
    analysis_path = resolve(args.analysis)
    y, _, _ = an.load_rollouts(rollouts_path)
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    subset = primary_subset(y)

    out: dict = {
        "experiment": "V2-B003",
        "audit_directive": "owner 2026-09-20: four-point post-run audit before the formal gate reading",
        "inputs": {
            "rollouts": str(rollouts_path),
            "rollouts_sha256": sha256_file(rollouts_path),
            "analysis": str(analysis_path),
            "analysis_sha256": sha256_file(analysis_path),
        },
        "positioning": {
            "primary_method": "cross-fitted empirical response Oracle",
            "description": (
                "uses each theorem's 4 fit-seed outcomes to decide the allocation and the "
                "other 4 seeds to evaluate it independently; it remains a non-deployable "
                "theorem-specific Oracle. Its role is to test the cross-seed stability of "
                "the compute-response signal, not to represent the performance of any "
                "future predictor"
            ),
            "is_predictor_performance_claim": False,
        },
        "fold_coverage": fold_coverage(y, subset),
        "bootstrap_audit": {
            "conditional_ci": {
                "source": "v2_b003_analyze.py bootstrap",
                "label": "conditional CI",
                "description": (
                    "allocation held fixed after a single fit-half solve; resampling is "
                    "theorem-level, i.e. component-level in the family-clean design"
                ),
            },
            "full_procedure": {},
        },
        "headline_scope": (
            "strict family-unseen hard/OOD reserve population (192 components); NOT "
            "generalizable to the full Promptset distribution; absolute expected solves and "
            "absolute gains are reported alongside relative gains because rare events "
            "inflate percentage-only views"
        ),
        "headline_table": headline_table(analysis),
    }

    n_boot = args.full_boot_n
    for tokens in an.BUDGETS:
        out["bootstrap_audit"]["full_procedure"][str(tokens)] = full_procedure_bootstrap(
            y, subset, tokens, args.skip_allowed, n_boot, args.seed + an.BUDGETS.index(tokens)
        )

    output_path = resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    coverage = out["fold_coverage"]
    print(f"[audit] primary set: {len(subset)} theorems; gate coverage: {coverage['gate_coverage_status']}")
    for name in ("fold_A", "fold_B"):
        stat = coverage["per_fold"][name]
        print(
            f"[audit] {name}: cells<3 = {stat['cells_below_3']}/{stat['cells']} "
            f"({100 * (stat['share_below_3'] or 0):.2f}%)"
        )
    for tokens in an.BUDGETS:
        fp = out["bootstrap_audit"]["full_procedure"][str(tokens)]
        rel = fp["rel_gain_ci95"]
        print(
            f"[audit] N*{tokens}: full-procedure abs {fp['abs_gain_mean']:.3f} "
            f"{fp['abs_gain_ci95']} | rel {fp['rel_gain_mean'] if fp['rel_gain_mean'] is None else format(fp['rel_gain_mean'], '.3f')} "
            f"{rel} ({fp['wall_seconds']}s)"
        )
    print(f"[audit] artifact: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
