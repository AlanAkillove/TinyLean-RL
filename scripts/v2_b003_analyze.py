#!/usr/bin/env python
"""V2-B003 analyzer: multi-seed p_i(b) estimates + Expected Oracle gate.

Frozen protocol (experiments/manifests/v2/V2-B003.yaml). Definitions:

- probability estimates: for every theorem-budget cell, k_ib = successes,
  n_ib = valid trials (model-outcome verdicts only; verifier infra outcomes are
  missing and excluded), p_hat_ib = k_ib / n_ib. No monotonicity is forced.
  The PRIMARY analysis set S6 = theorems whose n_ib >= 6 for every budget;
  low-coverage cells are reported explicitly and never silently treated as 0.
- Expected Oracle, both statistics:
  * plug-in: all K=8 estimates -> optimize; explicitly a finite-sample
    optimistic upper bound (same trajectories fit and evaluate);
  * cross-fitted (PRIMARY GATE): fixed folds A = replicates 0-3, B = 4-7;
    A estimates and allocates, B independently evaluates, then B->A swap; the
    reported value is the average of the two folds. The split is frozen.
- action spaces: no-skip selective-upgrade (b_i in {512..4096}, the >=512 floor
  kept) is PRIMARY; skip-allowed triage (b_i = 0 additionally permitted) is
  reported separately and never mixed into the primary gain.
- optimization: exact 0/1 knapsack over 512-token units (each theorem takes
  exactly one budget; total allocated cap <= N * uniform budget), deterministic
  tie-breaking toward the smaller budget.
- comparisons at total budgets N*{512,1024,2048,3072,4096}: absolute expected
  solved, absolute gain, relative gain, same-solved allocated-token savings
  (smallest uniform budget whose expected solved covers the oracle value on the
  same evaluation folds; discrete rounding noted), and theorem-level bootstrap
  uncertainty (1000 resamples, fixed seed; the allocation is held fixed, so
  this measures evaluation uncertainty, not re-optimization uncertainty).

Evaluation math (expected solved, fold-averaged): each theorem contributes the
mean realized success over the valid evaluation-fold trajectories at its
allocated budget; skipped or wholly-missing theorems contribute 0 and are
counted in eval_missing.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BUDGETS = (512, 1024, 2048, 3072, 4096)
BUDGET_UNITS = (1, 2, 4, 6, 8)  # in 512-token units
MISSING = -1
FOLD_A = (0, 1, 2, 3)
FOLD_B = (4, 5, 6, 7)
COVERAGE_GATE = 6
BOOTSTRAP_SEED = 20261001
BOOTSTRAP_N = 1000
DEFAULT_ROLLOUTS = "experiments/results/v2_b003_rollouts.jsonl"
DEFAULT_OUTPUT = "experiments/results/v2_b003_analysis.json"


def load_rollouts(path: Path) -> tuple[np.ndarray, dict[int, int], list[dict[str, Any]]]:
    """Return (y, rank_to_index, records_meta).

    y shape (n_theorems, 8, 5) with values 1 (verified), 0 (model-outcome
    failure), -1 (missing / infra). ranks must be a contiguous 1..n set.
    """
    by_rank: dict[int, dict[int, dict[str, Any]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            rank = int(record["theorem_rank"])
            replicate = int(record["replicate"])
            by_rank.setdefault(rank, {})[replicate] = record
    if not by_rank:
        raise ValueError(f"no records in {path}")
    ranks = sorted(by_rank)
    expected = list(range(1, len(ranks) + 1))
    if ranks != expected:
        raise ValueError(f"ranks are not contiguous 1..{len(ranks)}: found {len(ranks)} ranks")
    n = len(ranks)
    y = np.full((n, 8, len(BUDGETS)), MISSING, dtype=np.int8)
    meta: list[dict[str, Any]] = []
    for rank in ranks:
        row = by_rank[rank]
        if sorted(row) != list(range(8)):
            raise ValueError(f"rank {rank} does not have replicates 0..7: {sorted(row)}")
        first = row[0]
        meta.append(
            {
                "rank": rank,
                "statement_id": first["statement_id"],
                "component_id": first["component_id"],
                "component_size": first["component_size"],
            }
        )
        for replicate, record in row.items():
            prefixes = sorted(record["prefixes"], key=lambda p: int(p["budget"]))
            if [int(p["budget"]) for p in prefixes] != list(BUDGETS):
                raise ValueError(f"rank {rank} k{replicate}: budgets do not match {BUDGETS}")
            for b_index, prefix in enumerate(prefixes):
                if prefix["verified"]:
                    y[rank - 1, replicate, b_index] = 1
                elif prefix["verify_status"] in (
                    "verifier_timeout",
                    "verifier_server_error",
                    "not_checked",
                ):
                    y[rank - 1, replicate, b_index] = MISSING
                else:
                    y[rank - 1, replicate, b_index] = 0
    rank_to_index = {rank: rank - 1 for rank in ranks}
    return y, rank_to_index, meta


def counts(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """successes k and valid trials n, shape (n, 5), full K=8."""
    valid = y != MISSING
    k = (y == 1).sum(axis=1).astype(np.int64)
    n = valid.sum(axis=1).astype(np.int64)
    return k, n


def p_hat(k: np.ndarray, n: np.ndarray) -> np.ndarray:
    p = np.zeros(k.shape, dtype=np.float64)
    np.divide(k, n, out=p, where=n > 0)
    return p


def dp_allocate(p: np.ndarray, cap_units: int, allow_skip: bool) -> tuple[float, np.ndarray]:
    """Exact knapsack: every theorem takes exactly one budget (or 0 if allowed).

    Returns (best expected solved, allocation in tokens per theorem).
    Deterministic tie-breaking: smaller budget wins.
    """
    n = p.shape[0]
    neg = -1e18
    dp = np.full(cap_units + 1, neg, dtype=np.float64)
    dp[0] = 0.0
    choices = np.zeros((n, cap_units + 1), dtype=np.int8)
    options: list[tuple[int, int]] = []
    if allow_skip:
        options.append((0, -1))
    options.extend(zip(BUDGET_UNITS, range(len(BUDGETS))))
    for i in range(n):
        dp_new = np.full(cap_units + 1, neg, dtype=np.float64)
        best_opt = np.zeros(cap_units + 1, dtype=np.int8)
        for opt_index, (weight, b_index) in enumerate(options):
            value = float(p[i, b_index]) if b_index >= 0 else 0.0
            if weight == 0:
                cand = dp + value
                update = cand > dp_new
                dp_new[update] = cand[update]
                best_opt[update] = opt_index
            else:
                if weight > cap_units:
                    continue  # this budget cannot fit under the current cap
                cand = np.full(cap_units + 1, neg, dtype=np.float64)
                cand[weight:] = dp[: cap_units + 1 - weight] + value
                update = cand > dp_new
                dp_new[update] = cand[update]
                best_opt[update] = opt_index
        choices[i] = best_opt
        dp = dp_new
    s_best = int(np.argmax(dp))
    alloc = np.zeros(n, dtype=np.int64)
    s = s_best
    for i in range(n - 1, -1, -1):
        opt_index = int(choices[i, s])
        weight, b_index = options[opt_index]
        alloc[i] = 0 if b_index < 0 else BUDGETS[b_index]
        s -= weight
    return float(dp[s_best]), alloc


def contribution_vector(
    y: np.ndarray,
    alloc_tokens: np.ndarray,
    replicates: tuple[int, ...],
) -> tuple[np.ndarray, int]:
    """Per-theorem mean realized success over valid trajectories of a fold.

    Returns (vector of per-theorem expectations; allocation 0 contributes 0),
    and the number of theorems with no valid trajectory at their budget.
    """
    n = y.shape[0]
    b_index_of = {b: i for i, b in enumerate(BUDGETS)}
    out = np.zeros(n, dtype=np.float64)
    missing = 0
    for i in range(n):
        b = int(alloc_tokens[i])
        if b == 0:
            continue
        b_index = b_index_of[b]
        vals = [y[i, r, b_index] for r in replicates if y[i, r, b_index] != MISSING]
        if not vals:
            missing += 1
            continue
        out[i] = float(np.mean(vals))
    return out, missing


def fold_mean(
    y: np.ndarray,
    alloc_tokens: np.ndarray,
    replicates: tuple[int, ...],
    subset: np.ndarray,
) -> tuple[float, int]:
    vector, missing = contribution_vector(y, alloc_tokens, replicates)
    return float(vector[subset].mean()), missing


def uniform_alloc(n: int, tokens: int) -> np.ndarray:
    return np.full(n, tokens, dtype=np.int64)


def cross_fitted(
    y: np.ndarray,
    subset: np.ndarray,
    cap_units: int,
    allow_skip: bool,
) -> dict[str, Any]:
    """Cross-fitted Expected Oracle: A estimates/allocates, B evaluates; swap."""
    results: dict[str, Any] = {}
    fold_specs = (
        ("A_to_B", FOLD_A, FOLD_B),
        ("B_to_A", FOLD_B, FOLD_A),
    )
    per_fold: list[dict[str, Any]] = []
    for name, fit_reps, eval_reps in fold_specs:
        k_fit = (y[:, fit_reps, :] == 1).sum(axis=1).astype(np.int64)[subset]
        n_fit = (y[:, fit_reps, :] != MISSING).sum(axis=1).astype(np.int64)[subset]
        p_fit = p_hat(k_fit, n_fit)
        _, alloc_sub = dp_allocate(p_fit, cap_units, allow_skip)
        alloc = np.zeros(y.shape[0], dtype=np.int64)
        alloc[subset] = alloc_sub
        value_rate, missing = fold_mean(y, alloc, eval_reps, subset)
        per_fold.append(
            {
                "direction": name,
                "eval_value": value_rate * len(subset),
                "eval_value_rate": value_rate,
                "eval_missing_theorems": missing,
                "alloc_cap_tokens": int(alloc.sum()),
            }
        )
    results["folds"] = per_fold
    results["value"] = float(np.mean([f["eval_value"] for f in per_fold]))
    results["alloc_cap_tokens_mean"] = float(np.mean([f["alloc_cap_tokens"] for f in per_fold]))
    results["alloc_cap_tokens_per_fold"] = [f["alloc_cap_tokens"] for f in per_fold]
    results["eval_missing_theorems"] = int(sum(f["eval_missing_theorems"] for f in per_fold))
    return results


def uniform_table(
    y: np.ndarray,
    subset: np.ndarray,
    allow_skip: bool,
) -> dict[str, Any]:
    n_sub = len(subset)
    table: dict[str, Any] = {}
    for tokens in BUDGETS:
        alloc = np.zeros(y.shape[0], dtype=np.int64)
        alloc[subset] = uniform_alloc(n_sub, tokens)
        fold_values = []
        missing_total = 0
        for reps in (FOLD_A, FOLD_B):
            value, missing = fold_mean(y, alloc, reps, subset)
            fold_values.append(value)
            missing_total += missing
        table[str(tokens)] = {
            "expected_solved": float(np.mean(fold_values) * n_sub),
            "fold_values": fold_values,
            "eval_missing_theorems": int(missing_total),
            "alloc_cap_tokens": int(n_sub * tokens),
        }
    return table


def savings_for(
    oracle_value: float,
    oracle_cap_tokens: float,
    uniform: dict[str, Any],
    n_sub: int,
) -> dict[str, Any]:
    for tokens in BUDGETS:
        entry = uniform[str(tokens)]
        if entry["expected_solved"] >= oracle_value - 1e-12:
            uniform_cap = n_sub * tokens
            saved = 1.0 - oracle_cap_tokens / uniform_cap if uniform_cap else None
            return {
                "uniform_budget_reference": tokens,
                "uniform_cap_tokens": uniform_cap,
                "oracle_cap_tokens": oracle_cap_tokens,
                "allocated_cap_savings_pct": None if saved is None else round(100.0 * saved, 2),
                "note": "smallest uniform budget whose fold-averaged expected solved covers the oracle value (discrete; conservative)",
            }
    return {
        "uniform_budget_reference": None,
        "uniform_cap_tokens": None,
        "oracle_cap_tokens": oracle_cap_tokens,
        "allocated_cap_savings_pct": None,
        "note": "oracle value exceeds uniform at every budget point",
    }


def bootstrap_uncertainty(
    d_oracle: np.ndarray,
    d_uniform: np.ndarray,
    scale: float = 1.0,
) -> dict[str, Any]:
    n = len(d_oracle)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, n, size=(BOOTSTRAP_N, n))
    diff = d_oracle - d_uniform
    boot_abs = diff[idx].mean(axis=1)
    boot_uniform = d_uniform[idx].mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        boot_rel = np.where(boot_uniform > 0, boot_abs / boot_uniform, np.nan)
    rel_valid = boot_rel[~np.isnan(boot_rel)]
    return {
        "n_boot": BOOTSTRAP_N,
        "seed": BOOTSTRAP_SEED,
        "abs_gain_mean": float(diff.mean()) * scale,
        "abs_gain_ci95": [
            float(np.percentile(boot_abs, 2.5)) * scale,
            float(np.percentile(boot_abs, 97.5)) * scale,
        ],
        "rel_gain_mean": float(diff.mean() / d_uniform.mean()) if d_uniform.mean() > 0 else None,
        "rel_gain_ci95": (
            [float(np.percentile(rel_valid, 2.5)), float(np.percentile(rel_valid, 97.5))]
            if len(rel_valid)
            else None
        ),
        "note": "theorem-level bootstrap; allocation held fixed (evaluation uncertainty, not re-optimization); absolute values are expected solved counts",
    }


def compute_response(y: np.ndarray, subset: np.ndarray) -> dict[str, Any]:
    k, n = counts(y)
    p = p_hat(k, n)
    p_sub = p[subset]
    p4096 = p_sub[:, 4]
    p512 = p_sub[:, 0]
    sensitivity = p4096 - p512
    hist = Counter(round(v * 8) for v in p4096)
    p4096_hist = {str(i): int(hist.get(i, 0)) for i in range(9)}
    marginal = []
    for j in range(len(BUDGETS) - 1):
        marginal.append(
            {
                "from": BUDGETS[j],
                "to": BUDGETS[j + 1],
                "mean_delta_p": float((p_sub[:, j + 1] - p_sub[:, j]).mean()),
            }
        )
    non_decreasing = 0
    non_increasing = 0
    non_monotonic = 0
    for row in p_sub:
        diffs = np.diff(row)
        if np.all(diffs >= -1e-12):
            non_decreasing += 1
        elif np.all(diffs <= 1e-12):
            non_increasing += 1
        else:
            non_monotonic += 1
    def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
        if a.std() == 0 or b.std() == 0:
            return None
        return float(np.corrcoef(a, b)[0, 1])

    def _rank(v: np.ndarray) -> np.ndarray:
        order = np.argsort(v, kind="stable")
        ranks = np.empty(len(v), dtype=np.float64)
        ranks[order] = np.arange(len(v), dtype=np.float64)
        return ranks

    pearson = _pearson(p4096, sensitivity)
    spearman = _pearson(_rank(p4096), _rank(sensitivity))
    bins = []
    edges = [0.0, 0.125, 0.25, 0.5, 0.75, 0.875, 1.0001]
    for lo, hi in pairwise(edges):
        mask = (p4096 >= lo) & (p4096 < hi)
        bins.append(
            {
                "p4096_range": [lo, min(hi, 1.0)],
                "n": int(mask.sum()),
                "mean_sensitivity": float(sensitivity[mask].mean()) if mask.any() else None,
            }
        )
    return {
        "p4096_hist_k_of_8": p4096_hist,
        "sensitivity_mean": float(sensitivity.mean()),
        "sensitivity_nonzero_count": int((np.abs(sensitivity) > 1e-12).sum()),
        "marginal_gains": marginal,
        "monotonic_counts": {
            "non_decreasing": non_decreasing,
            "non_increasing": non_increasing,
            "non_monotonic": non_monotonic,
            "note": "empirical K=8 curves; not a claim about true monotonicity",
        },
        "correlations": {
            "pearson_difficulty_sensitivity": pearson,
            "spearman_difficulty_sensitivity": spearman,
            "note": "tests hard != compute-sensitive",
        },
        "difficulty_bins": bins,
    }


def gate_reading(relative: float | None) -> str:
    if relative is None:
        return "n/a"
    if relative < 0.15:
        return "no_go_band (<15%) - NO-GO B004 if repeated across budget points"
    if relative < 0.30:
        return "middle_band (15-30%) - heuristic/XGBoost only; MLP not approved"
    return "strong_band (>=30%) - strong GO if reproduced across budget points and absolute gain substantial"


def analyze(y: np.ndarray, meta: list[dict[str, Any]], allow_skip: bool) -> dict[str, Any]:
    n = y.shape[0]
    k, n_valid = counts(y)
    low_coverage = []
    for i in range(n):
        for b_index in range(len(BUDGETS)):
            if int(n_valid[i, b_index]) < COVERAGE_GATE:
                low_coverage.append(
                    {
                        "rank": int(meta[i]["rank"]),
                        "budget": BUDGETS[b_index],
                        "k": int(k[i, b_index]),
                        "n": int(n_valid[i, b_index]),
                    }
                )
    subset = np.array(
        [i for i in range(n) if bool((n_valid[i] >= COVERAGE_GATE).all())], dtype=np.int64
    )
    n_sub = len(subset)
    uniform = uniform_table(y, subset, allow_skip)
    out: dict[str, Any] = {
        "experiment": "V2-B003",
        "config": {
            "budgets": list(BUDGETS),
            "folds": {"A": list(FOLD_A), "B": list(FOLD_B)},
            "coverage_gate": COVERAGE_GATE,
            "skip_allowed_reported_separately": True,
            "n_theorems": n,
            "primary_set_size": int(n_sub),
        },
        "coverage": {
            "low_coverage_cells": low_coverage,
            "n_low_coverage_cells": len(low_coverage),
            "valid_trials_hist": {
                str(v): int((n_valid == v).sum()) for v in range(9)
            },
        },
        "uniform": uniform,
        "budget_points": {},
    }
    p_full = p_hat(k, n_valid)
    for tokens in BUDGETS:
        cap_units = n_sub * (tokens // 512)
        point: dict[str, Any] = {"total_budget": tokens, "total_cap_tokens": n_sub * tokens}
        # plug-in (optimistic reference)
        _value_plugin, alloc_plugin = dp_allocate(p_full[subset], cap_units, allow_skip)
        alloc_full = np.zeros(n, dtype=np.int64)
        alloc_full[subset] = alloc_plugin
        d_plugin, missing_plugin = contribution_vector(
            y, alloc_full, tuple(range(8))
        )
        point["plug_in"] = {
            "expected_solved": float(d_plugin[subset].mean() * n_sub),
            "alloc_cap_tokens": int(alloc_full.sum()),
            "note": "finite-sample optimistic upper bound (fit and evaluated on the same K=8)",
            "eval_missing_theorems": missing_plugin,
        }
        # cross-fitted (primary gate)
        cf = cross_fitted(y, subset, cap_units, allow_skip)
        point["cross_fitted"] = cf
        # comparisons
        uniform_solved = uniform[str(tokens)]["expected_solved"]
        point["uniform_expected_solved"] = uniform_solved
        cf_rel = (
            (cf["value"] - uniform_solved) / uniform_solved if uniform_solved > 0 else None
        )
        plugin_rel = (
            (point["plug_in"]["expected_solved"] - uniform_solved) / uniform_solved
            if uniform_solved > 0
            else None
        )
        point["cross_fitted_gain"] = {
            "absolute": float(cf["value"] - uniform_solved),
            "relative": None if cf_rel is None else float(cf_rel),
            "gate_reading": gate_reading(cf_rel),
        }
        point["plug_in_gain"] = {
            "absolute": float(point["plug_in"]["expected_solved"] - uniform_solved),
            "relative": None if plugin_rel is None else float(plugin_rel),
        }
        point["savings_cross_fitted"] = savings_for(
            cf["value"], cf["alloc_cap_tokens_mean"], uniform, n_sub
        )
        point["savings_plug_in"] = savings_for(
            point["plug_in"]["expected_solved"], float(alloc_full.sum()), uniform, n_sub
        )
        # bootstrap: per-theorem contributions, fold-averaged
        alloc_ab = np.zeros((2, n), dtype=np.int64)
        # reconstruct per-fold allocations for the bootstrap contributions
        k_a = (y[:, FOLD_A, :] == 1).sum(axis=1).astype(np.int64)[subset]
        n_a = (y[:, FOLD_A, :] != MISSING).sum(axis=1).astype(np.int64)[subset]
        _, alloc_a_sub = dp_allocate(p_hat(k_a, n_a), cap_units, allow_skip)
        k_b = (y[:, FOLD_B, :] == 1).sum(axis=1).astype(np.int64)[subset]
        n_b = (y[:, FOLD_B, :] != MISSING).sum(axis=1).astype(np.int64)[subset]
        _, alloc_b_sub = dp_allocate(p_hat(k_b, n_b), cap_units, allow_skip)
        alloc_ab[0, subset] = alloc_a_sub
        alloc_ab[1, subset] = alloc_b_sub
        d_oracle = 0.5 * (
            contribution_vector(y, alloc_ab[0], FOLD_B)[0]
            + contribution_vector(y, alloc_ab[1], FOLD_A)[0]
        )
        alloc_uniform_full = uniform_alloc_subset(n, subset, tokens)
        d_uniform = 0.5 * (
            contribution_vector(y, alloc_uniform_full, FOLD_B)[0]
            + contribution_vector(y, alloc_uniform_full, FOLD_A)[0]
        )
        point["bootstrap"] = bootstrap_uncertainty(
            d_oracle[subset], d_uniform[subset], scale=n_sub
        )
        out["budget_points"][str(tokens)] = point
    out["compute_response"] = compute_response(y, subset)
    out["gate_summary"] = {
        "basis": "cross-fitted no-skip Expected Oracle (primary gate)",
        "per_budget_relative_gain": {
            str(tokens): out["budget_points"][str(tokens)]["cross_fitted_gain"]["relative"]
            for tokens in BUDGETS
        },
        "informational": "band labels are informational; the GO/NO-GO decision belongs to the owner",
    }
    return out


def uniform_alloc_subset(n: int, subset: np.ndarray, tokens: int) -> np.ndarray:
    alloc = np.zeros(n, dtype=np.int64)
    alloc[subset] = tokens
    return alloc


def main() -> int:
    parser = argparse.ArgumentParser(description="V2-B003 analyzer (Expected Oracle gate).")
    parser.add_argument("--rollouts", default=DEFAULT_ROLLOUTS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-allowed", action="store_true", help="triage space (reported separately; default off)")
    args = parser.parse_args()

    def resolve(path: str) -> Path:
        candidate = Path(path)
        return candidate if candidate.is_absolute() else ROOT / candidate

    rollouts_path = resolve(args.rollouts)
    y, _, meta = load_rollouts(rollouts_path)
    analysis = analyze(y, meta, allow_skip=args.skip_allowed)
    output_path = resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[V2-B003] records: {y.shape[0]} theorems x {y.shape[1]} replicates")
    print(f"[V2-B003] primary set (n>= {COVERAGE_GATE} everywhere): {analysis['config']['primary_set_size']}/{y.shape[0]}")
    print(f"[V2-B003] low-coverage cells: {analysis['coverage']['n_low_coverage_cells']}")
    for tokens in BUDGETS:
        point = analysis["budget_points"][str(tokens)]
        cf = point["cross_fitted_gain"]
        rel = cf["relative"]
        print(
            f"  N*{tokens}: uniform {point['uniform_expected_solved']:.3f} | "
            f"cross-fitted {point['cross_fitted']['value']:.3f} "
            f"(+{cf['absolute']:.3f}, {('%.1f%%' % (100 * rel)) if rel is not None else 'n/a'}) | "
            f"plug-in {point['plug_in']['expected_solved']:.3f}"
        )
    print(f"[V2-B003] gate summary (cross-fitted no-skip): {analysis['gate_summary']['per_budget_relative_gain']}")
    print(f"[V2-B003] artifact: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
