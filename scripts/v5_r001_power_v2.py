#!/usr/bin/env python3
"""V5-R001 §9 (owner directive) — POWER AUDIT V2 for the two preregistered endpoints.

Owner §9 fixes the two endpoints and the audit plan, and forbids freezing a practical threshold
here ("do not freeze a practical threshold" — the numbers inform the owner's freeze decision):

  * binary endpoint: pass@n per theorem (solved iff >=1 of n candidates verified), paired exact
    one-sided McNemar on the discordant theorems;
  * continuous endpoint: s_i(M) = verified_candidates_i(M) / evaluation_n, with
    Delta_rate = mean_i[ s_i(T) - s_i(C) ], unit = theorem = family component.

Audit plan implemented here:
  1. exact McNemar power for the binary endpoint at N = 128 / 192 / 256 across a discordance grid;
  2. Monte-Carlo paired simulation on HISTORICAL EMPIRICAL outcome vectors (E018, E023 and the
     V3-R001 theta0 surface) for BOTH endpoints, at N = 128 / 192 / 256, under two treatment
     models (additive rate shift; multiplicative odds);
  3. paired family bootstrap CI and the exact paired sign-flip test for the rate endpoint
     (valid because under H0 the per-theorem rate difference is symmetric about zero);
  4. minimum detectable effect at 80% power per (pool, N, model, endpoint), by interpolation.

Everything is CPU-only, deterministic (frozen seeds), reads only frozen historical artifacts and
writes experiments/manifests/v5/v5_r001_power_v2.json. No model, no GPU, no holdout contact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

OUT_DEFAULT = "experiments/manifests/v5/v5_r001_power_v2.json"

E018_BASE = "experiments/results/e018_base.json"
E018_STEP30 = "experiments/results/e018_step30.json"
E023_BASE = "experiments/results/e023_holdout_base.json"
V3R001_RAW = "runs/v3_r001_rollout_archive_from_fly122/attempt2_complete_20260925T0839Z/v3_r001_raw_rollout.jsonl"

N_GRID = [128, 192, 256]
EVAL_N = 8
DELTA_GRID = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10]
MODELS = ["additive", "multiplicative"]
ALPHA = 0.05
TARGET_POWER = 0.80
MC_REPS = 2000
SEED_MC = 20260927
BOOTSTRAP_B = 999
SEED_BOOT = 20260926
BOOTSTRAP_LEVEL = 0.95
# the bootstrap-CI power view is run on the planning N and a reduced effect grid (cost control)
BOOT_N = 192
BOOT_DELTAS = [0.02, 0.04, 0.06, 0.08, 0.10]
# binary-endpoint analytic grid
RHO_GRID = [0.09, 0.12, 0.13, 0.20]
JEFFREYS_ALPHA = 0.5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_revision() -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()


# --------------------------------------------------------------------------
# empirical pools (per-theorem verified counts from frozen historical artifacts)
# --------------------------------------------------------------------------


def pool_from_records(records: list[dict], key: str = "verified") -> dict:
    counts: dict[str, list[int]] = defaultdict(list)
    for record in records:
        counts[str(record["statement_id"])].append(1 if record[key] else 0)
    k = np.array([sum(v) for v in counts.values()], dtype=float)
    n = np.array([len(v) for v in counts.values()], dtype=float)
    if len(set(n.tolist())) != 1:
        raise SystemExit(f"ragged sample counts in pool: {sorted(set(n.tolist()))}")
    return {"k": k, "n_hist": int(n[0]), "theorems": len(k)}


def load_e018(path: str) -> dict:
    data = json.loads((ROOT / path).read_text())
    return pool_from_records(data["records"])


def load_e023() -> dict:
    data = json.loads((ROOT / E023_BASE).read_text())
    return pool_from_records(data["records"])


def load_v3r001() -> dict:
    rows = []
    with (ROOT / V3R001_RAW).open(encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return pool_from_records(rows)


def pool_stats(pool: dict) -> dict:
    k, n = pool["k"], pool["n_hist"]
    p = (k + JEFFREYS_ALPHA) / (n + 2 * JEFFREYS_ALPHA)
    return {
        "theorems": pool["theorems"],
        "samples_per_theorem": n,
        "verified_candidates": int(k.sum()),
        "candidate_verified_rate": round(float(k.sum() / (len(k) * n)), 4),
        "theorems_solved_at_least_1": int((k >= 1).sum()),
        "pass_at_n": round(float((k >= 1).mean()), 4),
        "jeffreys_p_hat": {
            "mean": round(float(p.mean()), 4),
            "zero_count_theorems": int((k == 0).sum()),
            "saturated_theorems": int((k == n).sum()),
        },
    }


# --------------------------------------------------------------------------
# exact tests
# --------------------------------------------------------------------------


def sign_flip_exact_p(diff_counts: np.ndarray, n: int) -> float:
    """Exact paired sign-flip (permutation) p-value for mean(diff_counts) > 0.

    ``diff_counts[i] = k_T(i) - k_C(i)`` in [-n, n]. Under H0 each difference is symmetric about
    zero (difference of two iid Binomials), so the null distribution of the sum is the exact
    convolution of ``counts[m]`` independent Rademacher sums of magnitude m. Deterministic and
    exact - no Monte Carlo - and ties/discreteness are handled natively.
    """

    mags = np.abs(diff_counts).astype(int)
    observed = int(diff_counts.sum())
    if observed <= 0:
        return 1.0
    counts = np.bincount(mags, minlength=n + 1)
    pmf = np.array([1.0])
    min_value = 0
    for magnitude in range(1, n + 1):
        c = int(counts[magnitude])
        if c == 0:
            continue
        values = magnitude * (2 * np.arange(c + 1) - c)
        probs = np.array([math.comb(c, x) for x in range(c + 1)], dtype=float) / 2**c
        kernel = np.zeros(values.max() - values.min() + 1)
        kernel[values - values.min()] = probs
        pmf = np.convolve(pmf, kernel)
        min_value += int(values.min())
    index = observed - min_value
    if index <= 0:
        return 1.0
    if index > len(pmf):
        return 0.0
    return float(pmf[index:].sum() / pmf.sum())


def mcnemar_exact_greater(n_against: int, n_favor: int) -> float:
    if n_against < 0 or n_favor < 0:
        raise ValueError("discordant counts must be non-negative")
    total = n_against + n_favor
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, x) for x in range(n_favor, total + 1))
    return tail / 2**total


def _binomial_pmf(p: float, n: int) -> list[float]:
    return [math.comb(n, k) * p**k * (1.0 - p) ** (n - k) for k in range(n + 1)]


def _mcnemar_critical_table(max_total: int) -> list[int]:
    """crit[t] = smallest k_favor with exact p <= alpha given t = k_favor + k_against."""

    crit = [0] * (max_total + 1)
    for t in range(max_total + 1):
        row = [math.comb(t, j) for j in range(t + 1)]
        suffix = 0
        best = t + 1
        for j in range(t, -1, -1):
            suffix += row[j]
            if suffix / 2**t <= ALPHA:
                best = j
            else:
                break
        crit[t] = best
    return crit


def analytic_mcnemar_power_exact(n_theorems: int, rho: float, delta: float) -> float:
    """Power of the one-sided exact McNemar test given discordance rho and true gain delta.

    n_favor ~ Bin(N, q_favor) and n_against ~ Bin(N, q_against) independently, with
    q_favor = (rho + delta)/2 and q_against = (rho - delta)/2 (so E[favor - against] = N*delta and
    E[favor + against] = N*rho). Mirrors `scripts/v4_p001_power.py::exact_rule_power` without the
    delta_min condition, on the same exact binomial machinery as `mcnemar_exact_greater`.
    """

    if delta >= rho:
        q_favor, q_against = rho, 0.0
    else:
        q_favor = (rho + delta) / 2.0
        q_against = (rho - delta) / 2.0
    pf = _binomial_pmf(q_favor, n_theorems)
    pa = _binomial_pmf(q_against, n_theorems)
    crit = _mcnemar_critical_table(2 * n_theorems)
    power = 0.0
    for k_against, w_against in enumerate(pa):
        if w_against == 0.0:
            continue
        for k_favor, w_favor in enumerate(pf):
            if w_favor and k_favor >= crit[k_against + k_favor]:
                power += w_against * w_favor
    return power


def detectable_delta_analytic(n_theorems: int, rho: float) -> float | None:
    """Smallest delta (0.1pp resolution) whose exact power reaches the 80% target."""

    lo, hi = 0.0, rho
    if analytic_mcnemar_power_exact(n_theorems, rho, hi) < TARGET_POWER:
        return None
    for _ in range(40):
        mid = (lo + hi) / 2.0
        if analytic_mcnemar_power_exact(n_theorems, rho, mid) >= TARGET_POWER:
            hi = mid
        else:
            lo = mid
    return round(hi, 4)


# --------------------------------------------------------------------------
# simulation
# --------------------------------------------------------------------------


def treatment_rates(p_c: np.ndarray, delta: float, model: str) -> np.ndarray:
    if model == "additive":
        return np.clip(p_c + delta, 0.0, 1.0)
    if model == "multiplicative":
        base_mean = float(p_c.mean())
        odds = p_c / np.clip(1.0 - p_c, 1e-9, None)
        factor = 1.0
        for _ in range(200):  # solve the odds factor so the mean shift equals delta
            candidate = (odds * factor) / (1.0 + odds * factor)
            shift = float(candidate.mean() - base_mean)
            if abs(shift - delta) < 1e-7:
                break
            factor *= 1.0 + max(-0.5, min(0.5, (delta - shift)))
        return np.clip((odds * factor) / (1.0 + odds * factor), 0.0, 1.0)
    raise ValueError(f"unknown model {model}")


def bootstrap_ci_lower(diffs: np.ndarray, rng: np.random.Generator, reps: int = BOOTSTRAP_B) -> float:
    n = len(diffs)
    idx = rng.integers(0, n, size=(reps, n))
    means = diffs[idx].mean(axis=1)
    return float(np.quantile(means, (1.0 - BOOTSTRAP_LEVEL) / 2.0))


def simulate(
    pool: dict,
    n_theorems: int,
    delta: float,
    model: str,
    reps: int,
    seed: int,
    with_bootstrap: bool = False,
) -> dict:
    k, n_hist = pool["k"], pool["n_hist"]
    p_hat = (k + JEFFREYS_ALPHA) / (n_hist + 2 * JEFFREYS_ALPHA)
    rng = np.random.default_rng(seed)
    boot_rng = np.random.default_rng(seed + 1)
    powers_mcnemar = 0
    powers_signflip = 0
    powers_bootstrap = 0
    delta_rate_sum = 0.0
    delta_bin_sum = 0.0
    for _ in range(reps):
        idx = rng.integers(0, len(p_hat), size=n_theorems)
        p_c = p_hat[idx]
        p_t = treatment_rates(p_c, delta, model)
        k_c = rng.binomial(EVAL_N, p_c)
        k_t = rng.binomial(EVAL_N, p_t)
        diffs = (k_t - k_c).astype(int)
        delta_rate_sum += float(diffs.mean() / EVAL_N)
        # binary endpoint
        b_c = k_c >= 1
        b_t = k_t >= 1
        favor = int(np.sum(~b_c & b_t))
        against = int(np.sum(b_c & ~b_t))
        delta_bin_sum += float((b_t.astype(float) - b_c.astype(float)).mean())
        if mcnemar_exact_greater(against, favor) <= ALPHA:
            powers_mcnemar += 1
        # rate endpoint
        if sign_flip_exact_p(diffs, EVAL_N) <= ALPHA:
            powers_signflip += 1
        if with_bootstrap and bootstrap_ci_lower(diffs / EVAL_N, boot_rng) > 0.0:
            powers_bootstrap += 1
    result = {
        "theorems": n_theorems,
        "delta_target": delta,
        "model": model,
        "reps": reps,
        "mean_delta_rate": round(delta_rate_sum / reps, 5),
        "mean_delta_pass_at_n": round(delta_bin_sum / reps, 5),
        "power_mcnemar_binary": round(powers_mcnemar / reps, 4),
        "power_signflip_rate": round(powers_signflip / reps, 4),
    }
    if with_bootstrap:
        result["power_bootstrap_ci_rate"] = round(powers_bootstrap / reps, 4)
    return result


def min_detectable(points: list[dict], power_key: str) -> float | None:
    """First grid delta whose power reaches the target, linear refinement to one grid step."""

    for i, point in enumerate(points):
        if point[power_key] >= TARGET_POWER:
            if i == 0:
                return point["delta_target"]
            lo, hi = points[i - 1], point
            if hi[power_key] == lo[power_key]:
                return hi["delta_target"]
            frac = (TARGET_POWER - lo[power_key]) / (hi[power_key] - lo[power_key])
            return round(lo["delta_target"] + frac * (hi["delta_target"] - lo["delta_target"]), 4)
    return None


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=OUT_DEFAULT)
    parser.add_argument("--reps", type=int, default=MC_REPS)
    args = parser.parse_args()

    pools = {
        "E018_theta0_n8": load_e018(E018_BASE),
        "E018_step30_n8": load_e018(E018_STEP30),
        "E023_theta0_n4": load_e023(),
        "V3R001_theta0_n8": load_v3r001(),
    }
    report: dict = {
        "artifact_type": "v5_r001_power_v2",
        "created_utc": utc_now(),
        "git_revision": git_revision(),
        "owner_directive": {
            "section": "V5-R001 pre-launch directive §9 (power audit v2)",
            "endpoints": {
                "binary": "pass@n per theorem, paired exact one-sided McNemar",
                "rate": "Delta_rate = mean_i[verified/n (T) - verified/n (C)], unit = theorem/component",
            },
            "practical_threshold_frozen": "NO (owner §9: do not freeze a practical threshold here)",
        },
        "settings": {
            "eval_n": EVAL_N,
            "N_grid": N_GRID,
            "delta_grid": DELTA_GRID,
            "treatment_models": MODELS,
            "alpha": ALPHA,
            "target_power": TARGET_POWER,
            "mc_reps": args.reps,
            "seed_mc": SEED_MC,
            "bootstrap_reps": BOOTSTRAP_B,
            "seed_bootstrap": SEED_BOOT,
            "bootstrap_level": BOOTSTRAP_LEVEL,
            "jeffreys_alpha": JEFFREYS_ALPHA,
            "pool_resampling": "N theorems drawn with replacement from the pool's empirical p_hat "
            "distribution; for N above the pool size this is a mild conservatism (duplicate "
            "theorems are perfectly correlated)",
            "families": "one theorem per family component (H1 construction), so the theorem "
            "bootstrap IS the family bootstrap",
        },
        "inputs": {
            "E018_theta0_n8": E018_BASE,
            "E018_step30_n8": E018_STEP30,
            "E023_theta0_n4": E023_BASE,
            "V3R001_theta0_n8": V3R001_RAW,
        },
        "input_hashes": {
            name: sha256_file(ROOT / path)
            for name, path in {
                "E018_theta0_n8": E018_BASE,
                "E018_step30_n8": E018_STEP30,
                "E023_theta0_n4": E023_BASE,
                "V3R001_theta0_n8": V3R001_RAW,
            }.items()
        },
        "pool_stats": {name: pool_stats(pool) for name, pool in pools.items()},
        "analytic_mcnemar_binary": {},
        "simulation": {},
        "minimum_detectable_delta_rate": {},
        "recommendation": {},
    }

    # 1) analytic binary-endpoint table (deterministic, cross-check for rev.2 §14.6)
    for n_theorems in N_GRID:
        report["analytic_mcnemar_binary"][str(n_theorems)] = {
            f"rho_{rho}": {
                "detectable_delta_80": detectable_delta_analytic(n_theorems, rho),
                "power_at_delta_0.05": round(analytic_mcnemar_power_exact(n_theorems, rho, 0.05), 4),
                "power_at_delta_0.08": round(analytic_mcnemar_power_exact(n_theorems, rho, 0.08), 4),
            }
            for rho in RHO_GRID
        }

    # 2) paired Monte-Carlo simulation, both endpoints
    for pool_name, pool in pools.items():
        report["simulation"][pool_name] = {}
        for n_theorems in N_GRID:
            points = []
            for model in MODELS:
                points = [
                    simulate(pool, n_theorems, delta, model, args.reps, SEED_MC, with_bootstrap=False)
                    for delta in DELTA_GRID
                ]
                report["simulation"][pool_name][f"N{n_theorems}"] = {
                    "points": {model: points},
                    "min_detectable": {
                        f"delta_rate_80pct_{model}": min_detectable(points, "power_signflip_rate"),
                        f"delta_pass_at_n_80pct_{model}": min_detectable(points, "power_mcnemar_binary"),
                    },
                }
                if n_theorems == BOOT_N:
                    boot_points = [
                        simulate(
                            pool, n_theorems, delta, model, args.reps, SEED_MC + 7, with_bootstrap=True
                        )
                        for delta in BOOT_DELTAS
                    ]
                    report["simulation"][pool_name][f"N{n_theorems}"]["bootstrap_view"] = {
                        "model": model,
                        "points": boot_points,
                    }

    # 3) headline recommendation inputs (NO threshold frozen)
    for pool_name, pool in pools.items():
        entry = report["minimum_detectable_delta_rate"].setdefault(pool_name, {})
        for n_theorems in N_GRID:
            block = report["simulation"][pool_name][f"N{n_theorems}"]["min_detectable"]
            entry[f"N{n_theorems}"] = block

    report["recommendation"] = {
        "recommended_N": "to be set by the owner; N=192 is the planning target and N=256 buys "
        "about a 1 - sqrt(192/256) reduction in the detectable rate delta",
        "proposed_practical_gate": "NOT FROZEN. The audit supports reporting the rate endpoint "
        "with its paired bootstrap CI and the exact sign-flip p-value, and treating the binary "
        "endpoint as the confirmatory view; a threshold choice needs S1's measured between-arm "
        "noise first (owner §17).",
        "practical_threshold_frozen": "NO",
    }
    report["self_test"] = self_test()

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[power-v2] wrote {out_path}")
    for pool_name in pools:
        row = report["minimum_detectable_delta_rate"][pool_name]
        print(f"  {pool_name}: {json.dumps(row)}")
    return 0


def self_test() -> dict:
    """Deterministic correctness probes for the exact sign-flip machinery."""

    rng = np.random.default_rng(12345)
    # brute force over all sign patterns for a tiny case
    diffs = np.array([3, -1, 2, 0, -2], dtype=int)
    exact = sign_flip_exact_p(diffs, 8)
    total = 0
    hits = 0
    for mask in range(2 ** len(diffs)):
        value = 0
        for i, d in enumerate(diffs):
            value += d if (mask >> i) & 1 == 0 else -d
        total += 1
        if value >= int(diffs.sum()):
            hits += 1
    brute = hits / total
    # a null sample should not reject at anything like the nominal rate
    null_rejects = 0
    reps = 200
    for _ in range(reps):
        k_c = rng.binomial(8, 0.15, size=64)
        k_t = rng.binomial(8, 0.15, size=64)
        if sign_flip_exact_p(k_t - k_c, 8) <= ALPHA:
            null_rejects += 1
    return {
        "sign_flip_vs_bruteforce": {"exact": round(exact, 6), "brute_force": round(brute, 6),
                                    "delta": round(abs(exact - brute), 9)},
        "null_false_positive_rate_64": round(null_rejects / reps, 4),
        "mcnemar_known_value": round(mcnemar_exact_greater(2, 8), 6),
    }


if __name__ == "__main__":
    raise SystemExit(main())
