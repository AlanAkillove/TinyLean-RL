#!/usr/bin/env python3
"""V4-P001 §S.9 / Amendment A §4 — four-arm power sanity check for N = 128 under the frozen gates.

This is a design check, not a prediction: it answers "if an arm really changed repair success by X
points, would 128 paired theorems and the §N thresholds detect it?" Nothing here reads a V4 outcome
(there are none), and no scenario value is an estimate of theta0's repair probability.

The three comparisons and their rules:

    C vs A   delta >= 0.08   and one-sided exact McNemar p <= 0.05 and paired bootstrap CI lower > 0
    C vs B   delta >= 0.05   and one-sided exact McNemar p <= 0.05 and paired bootstrap CI lower > 0
    C vs D   delta >  0      and one-sided exact McNemar p <= 0.05 and paired bootstrap CI lower > 0

C vs A and C vs B are the approved primary gates and their thresholds are unchanged. C vs D is the
Amendment A §3 mechanism rule: no effect-size threshold, a purely directional rule, so what matters
there is the *smallest true effect it can detect* and its null firing rate -- both reported below.

Two computations are reported for every comparison:

  * exact enumeration of the paired discordant-pair distribution (no simulation, no independence
    assumption: the McNemar power depends on the two discordant rates alone);
  * a simulation of the complete gate (Delta rule + one-sided exact McNemar + paired bootstrap CI)
    that uses the same statistics as the frozen analysis code, under an independent-arm model.

Correction carried by this rerun: the previous (three-arm) artifact reported the C-A conjunction
(Delta >= 0.08) for *both* its keys, so its C-B rows understated the C-B gate power. Each scenario now
reports the power of its own comparison's rule.

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
    Mechanism,
    Outcome,
    mcnemar_exact_greater,
    mechanism_label,
    paired_gate,
)

OUT = "experiments/manifests/v4/v4_p001_power.json"
SIM_REPS = 2000
BOOTSTRAP_REPS = 1000
SIM_SEED = 20260925
TARGET_POWER = 0.8
MECHANISM_LABEL_SEED = 20260927

#: The frozen rules, expressed once. ``delta_min`` with ``strict=True`` means "strictly greater".
RULES: dict[str, dict] = {
    "C_vs_A": {
        "delta_min": DELTA_CA_MIN,
        "strict": False,
        "summary": f"delta >= {DELTA_CA_MIN:.2f}, McNemar p <= {ALPHA}, bootstrap CI lower > 0",
    },
    "C_vs_B": {
        "delta_min": DELTA_CB_MIN,
        "strict": False,
        "summary": f"delta >= {DELTA_CB_MIN:.2f}, McNemar p <= {ALPHA}, bootstrap CI lower > 0",
    },
    "C_vs_D": {
        "delta_min": 0.0,
        "strict": True,
        "summary": f"delta > 0, McNemar p <= {ALPHA}, bootstrap CI lower > 0 (Amendment A §3, no threshold)",
    },
}

DISCOURDANCE_GRID = (0.05, 0.10, 0.15, 0.20, 0.30, 0.50)
DELTA_GRID = (0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10)
SIMULATED_DISCOURDANCE_CELLS = ((0.10, 0.03), (0.10, 0.05), (0.10, 0.08), (0.20, 0.05), (0.20, 0.08), (0.30, 0.10))


def _binomial_pmf(p: float, n: int = N_PRIMARY) -> list[float]:
    """Binomial pmf by recurrence (no big-int combinatorics, stable for n = 128)."""

    if not 0.0 <= p <= 1.0:
        raise ValueError(f"probability out of range: {p}")
    if p == 0.0:
        return [1.0] + [0.0] * n
    if p == 1.0:
        return [0.0] * n + [1.0]
    pmf = [0.0] * (n + 1)
    pmf[0] = (1.0 - p) ** n
    ratio = p / (1.0 - p)
    for k in range(1, n + 1):
        pmf[k] = pmf[k - 1] * ((n - k + 1) / k) * ratio
    return pmf


def exact_rule_power(
    q_favor: float,
    q_against: float,
    *,
    delta_min: float | None = None,
    strict: bool = False,
    n: int = N_PRIMARY,
    alpha: float = ALPHA,
) -> float:
    """Exact P(rule passes) given the two discordant rates (no independence assumption).

    ``n_favor ~ Bin(n, q_favor)`` counts theorems the compared arm repairs and the baseline does not;
    ``n_against ~ Bin(n, q_against)`` the opposite. The Delta condition is evaluated on the realized
    paired difference ``(n_favor - n_against) / n``; ``delta_min=None`` drops it entirely (pure
    McNemar rejection power).
    """

    if q_favor < 0.0 or q_against < 0.0:
        raise ValueError("discordant rates must be non-negative")
    pf = _binomial_pmf(q_favor, n)
    pa = _binomial_pmf(q_against, n)
    total = 0.0
    for k_against in range(n + 1):
        w_against = pa[k_against]
        if w_against == 0.0:
            continue
        for k_favor in range(n + 1):
            w_favor = pf[k_favor]
            if w_favor == 0.0:
                continue
            if delta_min is not None:
                delta = (k_favor - k_against) / n
                if delta <= delta_min if strict else delta < delta_min:
                    continue
            if mcnemar_exact_greater(k_against, k_favor) <= alpha:
                total += w_against * w_favor
    return total


def _legacy_exact_mcnemar_power(
    p_arm: float, p_other: float, n: int = N_PRIMARY, alpha: float = ALPHA
) -> float:
    """The previous (three-arm artifact) computation, kept only as a numerical cross-check."""

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


def exact_mcnemar_power(p_arm: float, p_other: float, n: int = N_PRIMARY, alpha: float = ALPHA) -> float:
    """Pure one-sided McNemar rejection power (no Delta condition), from the two arm probabilities."""

    return exact_rule_power(
        p_other * (1.0 - p_arm), p_arm * (1.0 - p_other), delta_min=None, n=n, alpha=alpha
    )


def discordant_rates(p_baseline: float, p_other: float) -> dict:
    """The (q_favor, q_against, q_total, delta) induced by two independent-arm probabilities."""

    q_favor = p_other * (1.0 - p_baseline)
    q_against = p_baseline * (1.0 - p_other)
    return {
        "q_favor": round(q_favor, 6),
        "q_against": round(q_against, 6),
        "q_total": round(q_favor + q_against, 6),
        "delta": round(p_other - p_baseline, 6),
    }


def rates_from_discordance(q_total: float, delta: float) -> tuple[float, float]:
    """Invert (q_total, delta) to the two discordant rates; rejects impossible combinations."""

    if delta > q_total + 1e-12:
        raise ValueError(f"|delta| <= q_total is required (delta={delta}, q_total={q_total})")
    return (q_total + delta) / 2.0, (q_total - delta) / 2.0


def simulate_full_gate(
    p_baseline: float, p_other: float, rule: dict, rng: np.random.Generator
) -> dict:
    """Share of simulated cohorts where the complete rule (Delta + McNemar + bootstrap CI) passes."""

    passes = 0
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
        if rule["strict"]:
            delta_ok = delta > rule["delta_min"]
        else:
            delta_ok = delta >= rule["delta_min"]
        passes += int(delta_ok and p_value <= ALPHA and ci_lower > 0)
    return {
        "reps": SIM_REPS,
        "power_full_gate": round(passes / SIM_REPS, 4),
        "mean_delta": round(float(np.mean(deltas)), 4),
        "delta_p05": round(float(np.percentile(deltas, 5)), 4),
        "delta_p95": round(float(np.percentile(deltas, 95)), 4),
    }


def simulate_full_gate_from_discordance(
    q_total: float, delta: float, rule: dict, rng: np.random.Generator
) -> dict:
    """Simulation of one (q_total, delta) cell under independent arms, for the CI conjunct.

    The independent-arm model fixes the marginal probabilities that reproduce the requested
    discordant rates: solving ``p_b - p_a = delta`` and ``p_a + p_b - 2 p_a p_b = q_total`` for the
    root pair inside [0, 1].
    """

    discriminant = (1.0 - delta) ** 2 - 2.0 * (q_total - delta)
    if discriminant < 0.0:
        raise ValueError(f"no independent-arm model for q_total={q_total}, delta={delta}")
    root = math.sqrt(discriminant)
    p_baseline = (1.0 - delta - root) / 2.0
    p_other = p_baseline + delta
    return simulate_full_gate(p_baseline, p_other, rule, rng)


def detectable_delta(
    q_total: float, rule: dict, *, target: float = TARGET_POWER, step: float = 0.005
) -> dict:
    """Smallest true Delta whose exact rule power reaches ``target``, and the power limit otherwise.

    The feasible Delta range for a total discordance ``q_total`` is ``0 <= delta <= q_total``
    (``delta = q_favor - q_against`` and ``q_favor + q_against = q_total``). When no feasible Delta
    reaches the target, the reason is recorded together with the power at the largest feasible Delta,
    so "the grid stopped" is never reported as "impossible".
    """

    grid = [round(step * index, 4) for index in range(int(q_total / step) + 1)]
    if not grid:
        return {"detectable_delta": None, "reason": "q_total_below_step", "power_at_max_feasible": None}
    best_power = 0.0
    for delta in grid:
        q_favor, q_against = rates_from_discordance(q_total, delta)
        power = exact_rule_power(q_favor, q_against, delta_min=rule["delta_min"], strict=rule["strict"])
        best_power = power
        if power >= target:
            return {"detectable_delta": delta, "reason": "target_reached", "power_at_max_feasible": round(power, 4)}
    return {
        "detectable_delta": None,
        "reason": "target_not_reached_within[0, q_total]",
        "power_at_max_feasible": round(best_power, 4),
        "max_feasible_delta": grid[-1],
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
        "note": "four-arm design check under stated hypothetical repair probabilities; not a prediction of any V4 outcome",
        "sim_seed": SIM_SEED,
        "sim_reps": SIM_REPS,
        "bootstrap_reps": BOOTSTRAP_REPS,
        "n_primary": N_PRIMARY,
        "alpha": ALPHA,
        "rules": {name: dict(rule) for name, rule in RULES.items()},
        "delta_thresholds": {"C_vs_A": DELTA_CA_MIN, "C_vs_B": DELTA_CB_MIN},
        "delta_threshold_in_pairs_at_n128": {
            "C_vs_A": DELTA_CA_MIN * N_PRIMARY,
            "C_vs_B": DELTA_CB_MIN * N_PRIMARY,
        },
        "correction_vs_previous_run": (
            "the previous three-arm artifact reported the C-A conjunction (delta >= 0.08) for both of "
            "its keys, so its C-B rows understated the C-B gate power; each scenario here reports the "
            "power of its own comparison's rule. The C-A and C-B thresholds themselves are unchanged."
        ),
        "min_favorable_discordant_for_p005": attainable_p_table(),
        "discordance_grid": {
            "values": list(DISCOURDANCE_GRID),
            "role": "sensitivity range, not an estimate: no V4 outcome exists, and the locked historical audit measures first-attempt failure rates, never the disagreement between two repair attempts",
            "why_it_is_the_right_parameter": "for the exact paired McNemar power only the two discordant rates matter, so the tables below hold without any independence assumption about the four arms",
        },
    }

    # --- 1. arm-probability scenarios: one comparison each, C-A and C-B on the original RNG stream --
    rng = np.random.default_rng(SIM_SEED)
    scenarios = []
    for comparison, deltas in (
        ("C_vs_A", (0.05, 0.08, 0.10, 0.15)),
        ("C_vs_B", (0.05, 0.08, 0.10, 0.15)),
        ("C_vs_D", (0.02, 0.03, 0.05, 0.08, 0.10)),
    ):
        rule = RULES[comparison]
        for p_arm in (0.10, 0.15, 0.20):
            for delta in deltas:
                p_other = p_arm + delta
                if p_other > 1.0:
                    continue
                sim = simulate_full_gate(p_arm, p_other, rule, rng)
                scenarios.append(
                    {
                        "comparison": comparison,
                        "rule": rule["summary"],
                        "p_baseline_arm": p_arm,
                        "p_compared_arm": round(p_other, 4),
                        "true_delta": delta,
                        "exact_mcnemar_power": round(exact_mcnemar_power(p_arm, p_other), 4),
                        "exact_rule_power": round(
                            exact_rule_power(
                                p_other * (1.0 - p_arm),
                                p_arm * (1.0 - p_other),
                                delta_min=rule["delta_min"],
                                strict=rule["strict"],
                            ),
                            4,
                        ),
                        **discordant_rates(p_arm, p_other),
                        **sim,
                    }
                )
    artifact["scenarios"] = scenarios

    # --- 2. model-free discordance tables: exact power and the detectable effect ------------------
    detectable = {}
    exact_table = {}
    for comparison, rule in RULES.items():
        rows = []
        for q_total in DISCOURDANCE_GRID:
            row = {"q_total": q_total}
            for delta in DELTA_GRID:
                if delta > q_total:
                    row[f"delta_{delta:.2f}"] = None
                    continue
                q_favor, q_against = rates_from_discordance(q_total, delta)
                row[f"delta_{delta:.2f}"] = round(
                    exact_rule_power(q_favor, q_against, delta_min=rule["delta_min"], strict=rule["strict"]),
                    4,
                )
            rows.append(row)
        exact_table[comparison] = rows
        detectable[comparison] = {
            str(q_total): detectable_delta(q_total, rule) for q_total in DISCOURDANCE_GRID
        }
    artifact["exact_rule_power_by_discordance"] = {
        "definition": "P(rule passes) as a function of the true total discordant rate and the true Delta; exact enumeration, no independence assumption",
        "delta_conditions": {name: RULES[name]["summary"] for name in RULES},
        "rows_by_comparison": exact_table,
    }
    artifact["detectable_effect"] = {
        "target_power": TARGET_POWER,
        "definition": "smallest true Delta (percentage points) whose exact rule power reaches target_power, by total discordant rate; feasible range is 0 <= delta <= q_total",
        "by_comparison": detectable,
        "note": "the C-D rule has no effect-size threshold by design; this table is what makes it interpretable",
    }

    # --- 2b. power at each comparison's own frozen threshold --------------------------------------
    threshold_rows = []
    for q_total in DISCOURDANCE_GRID:
        row = {"q_total": q_total}
        for comparison, rule in RULES.items():
            delta = rule["delta_min"] if rule["delta_min"] > 0 else DELTA_CB_MIN
            if delta > q_total:
                row[comparison] = None
                continue
            q_favor, q_against = rates_from_discordance(q_total, delta)
            row[comparison] = round(
                exact_rule_power(q_favor, q_against, delta_min=rule["delta_min"], strict=rule["strict"]),
                4,
            )
        threshold_rows.append(row)
    artifact["power_at_frozen_threshold"] = {
        "definition": (
            "exact rule power when the true Delta equals the comparison's own threshold (C-A 0.08, "
            "C-B 0.05), by total discordant rate; the C-D column uses Delta = 0.05 as a reference "
            "point only, because the C-D rule deliberately has no effect-size threshold"
        ),
        "rows": threshold_rows,
        "reading": (
            "the +8 pp C-A threshold is a conservatism choice, not a high-power choice: a true effect "
            "of exactly +8 pp clears the gate in well under half of cohorts at realistic discordance, "
            "so a GO requires the observed effect to exceed the threshold by a margin. This is "
            "reported as a design property; the thresholds themselves are unchanged."
        ),
    }

    # --- 3. the mechanism rule under the null: how often would it fire with a true Delta of zero? --
    null_rates = {}
    for q_total in DISCOURDANCE_GRID:
        q_favor, q_against = rates_from_discordance(q_total, 0.0)
        rule = RULES["C_vs_D"]
        null_rates[str(q_total)] = {
            "exact_rule_power_mcnemar_only": round(
                exact_rule_power(q_favor, q_against, delta_min=rule["delta_min"], strict=rule["strict"]),
                4,
            ),
            "simulated_full_gate_firing_rate": simulate_full_gate_from_discordance(
                q_total, 0.0, rule, np.random.default_rng(MECHANISM_LABEL_SEED + int(q_total * 1000))
            )["power_full_gate"],
        }
    artifact["mechanism_label_null_firing_rate"] = {
        "definition": "P(DIAGNOSTIC_SPECIFIC is emitted) when the true C - D effect is exactly zero; the exact column is an upper bound because the bootstrap CI adds a further constraint",
        "reps_per_cell": SIM_REPS,
        "by_discordance": null_rates,
    }

    # --- 4. numerical cross-check of the refactor against the previous exact computation -----------
    legacy_rows = []
    for p_arm in (0.10, 0.15, 0.20):
        for delta in (0.05, 0.08, 0.10, 0.15):
            legacy_rows.append(
                (
                    p_arm,
                    p_arm + delta,
                    _legacy_exact_mcnemar_power(p_arm, p_arm + delta),
                    exact_mcnemar_power(p_arm, p_arm + delta),
                )
            )
    artifact["exact_computation_cross_check"] = {
        "definition": "max |previous comb-based McNemar power - pmf-recurrence McNemar power| over the 12 original cells",
        "max_abs_difference": max(abs(row[2] - row[3]) for row in legacy_rows),
    }

    # --- 5. frozen analysis code cross-checks -----------------------------------------------------
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

    rng_specificity = np.random.default_rng(SIM_SEED + 2)
    x_c_specificity = rng_specificity.random(N_PRIMARY) < 0.28
    x_d = rng_specificity.random(N_PRIMARY) < 0.18
    gate_cd = paired_gate(list(map(bool, x_c_specificity)), list(map(bool, x_d)))
    label = mechanism_label(Outcome.A_VERIFIER_SPECIFIC_GO, gate_cd)
    artifact["frozen_code_cross_check_specificity"] = {
        "configuration": "simulated p_C=0.28, p_D=0.18, n=128, classification held at the GO branch",
        "delta": round(gate_cd.delta, 6),
        "mcnemar_p": gate_cd.mcnemar_p,
        "ci_lower": round(gate_cd.ci_lower, 6),
        "n_favor": gate_cd.n_favor,
        "n_against": gate_cd.n_against,
        "mechanism_label": label.value,
        "label_matches_rule": (label is Mechanism.DIAGNOSTIC_SPECIFIC)
        == (gate_cd.delta > 0 and gate_cd.mcnemar_p <= ALPHA and gate_cd.ci_lower > 0),
    }

    destination = ROOT / args.out
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact, indent=2) + "\n")
    for scenario in scenarios:
        print(
            f"{scenario['comparison']} p_arm={scenario['p_baseline_arm']:.2f} "
            f"delta={scenario['true_delta']:.2f} q_total={scenario['q_total']:.3f} "
            f"exact_rule={scenario['exact_rule_power']:.3f} full_gate={scenario['power_full_gate']:.3f}"
        )
    print("detectable delta at power 0.8:", json.dumps(detectable))
    print("null firing rate of the C-D rule:", json.dumps(null_rates))
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
