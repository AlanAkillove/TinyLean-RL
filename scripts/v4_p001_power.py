#!/usr/bin/env python3
"""V4-P001 §S.9 — power sanity check for N = 128 under the frozen gates.

This is a design check, not a prediction: it answers "if the verifier diagnostic really improved repair
success by X points, would 128 paired theorems and the §N thresholds detect it?" Nothing here reads a
V4 outcome (there are none), and no scenario value is an estimate of theta0's repair probability.

Two independent computations of the McNemar component are reported:
  * exact enumeration of the discordant-pair distribution (no simulation);
  * a simulation of the complete §N gate set (Delta thresholds + one-sided exact McNemar + paired
    bootstrap CI) that uses the same statistics as the frozen analysis code.

Output: experiments/manifests/v4/v4_p001_power.json
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tinylean_rl.evaluation.v4_stats import (
    ALPHA,
    DELTA_CA_MIN,
    DELTA_CB_MIN,
    N_PRIMARY,
    mcnemar_exact_greater,
    paired_gate,
)

OUT = "experiments/manifests/v4/v4_p001_power.json"
SIM_REPS = 2000
BOOTSTRAP_REPS = 1000
SIM_SEED = 20260925


def exact_mcnemar_power(p_arm: float, p_other: float, n: int = N_PRIMARY, alpha: float = ALPHA) -> float:
    """P(one-sided exact McNemar rejects) when the compared arm succeeds with ``p_other``.

    Discordant counts are independent Binomials: ``K_favor ~ Bin(n, p_other*(1-p_arm))`` (the compared
    arm passes where the baseline failed) and ``K_against ~ Bin(n, p_arm*(1-p_other))``.
    """

    q_favor = p_other * (1.0 - p_arm)
    q_against = p_arm * (1.0 - p_other)
    power = 0.0
    for k_favor in range(n + 1):
        p_favor = math.comb(n, k_favor) * q_favor**k_favor * (1 - q_favor) ** (n - k_favor)
        if p_favor < 1e-14:
            continue
        for k_against in range(n + 1):
            p_against = math.comb(n, k_against) * q_against**k_against * (1 - q_against) ** (n - k_against)
            if p_against < 1e-14:
                continue
            if mcnemar_exact_greater(k_against, k_favor) <= alpha:
                power += p_favor * p_against
    return power


def simulate_full_gate(p_baseline: float, p_other: float, rng: np.random.Generator) -> dict:
    """Share of simulated cohorts where the complete §N condition set is met."""

    passes = {DELTA_CA_MIN: 0, DELTA_CB_MIN: 0, "all": 0}
    deltas = []
    for _ in range(SIM_REPS):
        x_baseline = rng.random(N_PRIMARY) < p_baseline
        x_other = rng.random(N_PRIMARY) < p_other
        diffs = x_other.astype(np.int8) - x_baseline.astype(np.int8)
        delta = float(diffs.mean())
        deltas.append(delta)
        n_favor = int((~x_baseline & x_other).sum())
        n_against = int((x_baseline & ~x_other).sum())
        p_value = mcnemar_exact_greater(n_against, n_favor)
        resampled = diffs[rng.integers(0, N_PRIMARY, size=(BOOTSTRAP_REPS, N_PRIMARY))].mean(axis=1)
        ci_lower = float(np.percentile(resampled, 2.5))
        ok = delta >= DELTA_CA_MIN and p_value <= ALPHA and ci_lower > 0
        passes[DELTA_CA_MIN] += int(ok)
        passes[DELTA_CB_MIN] += int(ok)
        passes["all"] += int(ok)
    return {
        "reps": SIM_REPS,
        "power_full_gate": round(passes["all"] / SIM_REPS, 4),
        "mean_delta": round(float(np.mean(deltas)), 4),
        "delta_p05": round(float(np.percentile(deltas, 5)), 4),
        "delta_p95": round(float(np.percentile(deltas, 95)), 4),
    }


def attainable_p_table(max_favor: int = 24, max_against: int = 4) -> dict:
    """Smallest favorable discordant count reaching p <= 0.05 for each opposing count."""

    table = {}
    for n_against in range(max_against + 1):
        row = []
        for n_favor in range(max_favor + 1):
            if mcnemar_exact_greater(n_against, n_favor) <= ALPHA:
                row.append(n_favor)
                break
        table[str(n_against)] = row[0] if row else None
    return table


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=OUT)
    args = parser.parse_args()

    artifact: dict = {
        "artifact_type": "v4_p001_power_sanity",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": subprocess.run(["hostname"], capture_output=True, text=True, check=False).stdout.strip(),
        "git_revision": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip(),
        "note": "design check under stated hypothetical repair probabilities; not a prediction of any V4 outcome",
        "n_primary": N_PRIMARY,
        "alpha": ALPHA,
        "delta_thresholds": {"C_vs_A": DELTA_CA_MIN, "C_vs_B": DELTA_CB_MIN},
        "delta_threshold_in_pairs_at_n128": {
            "C_vs_A": DELTA_CA_MIN * N_PRIMARY,
            "C_vs_B": DELTA_CB_MIN * N_PRIMARY,
        },
        "min_favorable_discordant_for_p005": attainable_p_table(),
    }

    rng = np.random.default_rng(SIM_SEED)
    scenarios = []
    for p_arm in (0.10, 0.15, 0.20):
        for delta in (0.05, 0.08, 0.10, 0.15):
            p_other = p_arm + delta
            if p_other > 1.0:
                continue
            sim = simulate_full_gate(p_arm, p_other, rng)
            scenarios.append(
                {
                    "p_baseline_arm": p_arm,
                    "p_compared_arm": round(p_other, 4),
                    "true_delta": delta,
                    "exact_mcnemar_power": round(exact_mcnemar_power(p_arm, p_other), 4),
                    **sim,
                }
            )
    artifact["scenarios"] = scenarios

    # cross-check: the frozen analysis code on one synthetic paired cohort
    rng_check = np.random.default_rng(SIM_SEED + 1)
    x_a = rng_check.random(N_PRIMARY) < 0.15
    x_c = rng_check.random(N_PRIMARY) < 0.25
    gate = paired_gate(list(map(bool, x_a)), list(map(bool, x_c)))
    artifact["frozen_code_cross_check"] = {
        "configuration": "simulated p_A=0.15, p_C=0.25, n=128",
        "delta": round(gate.delta, 6),
        "mcnemar_p": gate.mcnemar_p,
        "ci_lower": round(gate.ci_lower, 6),
        "n_favor": gate.n_favor,
        "n_against": gate.n_against,
    }

    destination = ROOT / args.out
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact, indent=2) + "\n")
    for scenario in scenarios:
        print(
            f"p_arm={scenario['p_baseline_arm']:.2f} delta={scenario['true_delta']:.2f} "
            f"exact_power={scenario['exact_mcnemar_power']:.3f} full_gate_power={scenario['power_full_gate']:.3f}"
        )
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
