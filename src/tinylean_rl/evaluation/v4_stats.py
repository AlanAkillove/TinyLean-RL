"""V4-P001 primary statistics and the frozen outcome taxonomy (owner §K, §L, §M, §N; Amendment A).

Everything is paired at the theorem level. Per theorem ``i`` the four arms give a binary success
indicator under one success definition (canonical score == 1 with the same format-gated Lean
verification), and the primary endpoints are

    Delta_repair     = mean(C_i - A_i)     Arm C against a plain fresh retry
    Delta_diagnostic = mean(C_i - B_i)     Arm C against self-revision without the diagnostic

each with a one-sided exact paired McNemar test on the discordant pairs and a paired theorem
bootstrap 95 % confidence interval. ``B - A`` is reported descriptively only; the specificity
endpoint ``Delta_specificity = mean(C_i - D_i)`` (Amendment A §3) is reported with the same
statistics under a directional rule without an effect-size threshold, and a *mechanism label* is
appended separately from the scientific classification so the two are never conflated. AUPRC and any
predictive metric are explicitly not endpoints here.

The guards are checked in the frozen order (Amendment A §6): provenance, then the data guard, then
the differential-censoring guard, then the primary gates. Fewer than 103 complete valid quadruplets
means ``D_INCONCLUSIVE_BY_DATA``; a censoring range above 0.05 across arms means
``D_INCONCLUSIVE_BY_DIFFERENTIAL_CENSORING``; the GO taxonomy is never evaluated when either guard
fails. Infra-censored arms are *missing*, never failures -- the infra-as-failure reading is reported
as a sensitivity analysis only and never redefines the primary endpoint.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

N_PRIMARY = 128
DATA_GUARD_MIN = 103
CENSORING_RANGE_MAX = 0.05

DELTA_CA_MIN = 0.08
DELTA_CB_MIN = 0.05
ALPHA = 0.05
BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 20260925
BOOTSTRAP_LEVEL = 0.95


class Outcome(str, Enum):
    """Frozen §N taxonomy plus the Amendment A §5 censoring guard."""

    A_VERIFIER_SPECIFIC_GO = "A_VERIFIER_SPECIFIC_GO"
    B_SELF_REVISION_ONLY = "B_SELF_REVISION_ONLY"
    C_NO_REPAIR_GAIN = "C_NO_REPAIR_GAIN"
    D_INCONCLUSIVE_BY_DATA = "D_INCONCLUSIVE_BY_DATA"
    D_INCONCLUSIVE_BY_DIFFERENTIAL_CENSORING = "D_INCONCLUSIVE_BY_DIFFERENTIAL_CENSORING"


class Mechanism(str, Enum):
    """Amendment A §3 mechanism label; appended to, never merged with, the classification."""

    DIAGNOSTIC_SPECIFIC = "DIAGNOSTIC_SPECIFIC"
    DIAGNOSTIC_NONSPECIFIC = "DIAGNOSTIC_NONSPECIFIC"
    NOT_EVALUABLE = "NOT_EVALUABLE"


@dataclass(frozen=True)
class Gate:
    delta: float
    mcnemar_p: float
    ci_lower: float
    n_favor: int
    n_against: int

    @property
    def passed(self) -> bool:
        return self.delta >= 0 and self.ci_lower > 0


def mcnemar_exact_greater(n_against: int, n_favor: int) -> float:
    """One-sided exact McNemar p-value: P(X >= n_favor) with X ~ Binomial(n_against + n_favor, 1/2).

    ``n_favor`` counts theorems where the first arm fails and the second passes (the direction the
    alternative predicts), ``n_against`` the opposite. Exact integer arithmetic; no scipy.
    """

    if n_against < 0 or n_favor < 0:
        raise ValueError("discordant counts must be non-negative")
    total = n_against + n_favor
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, k) for k in range(n_favor, total + 1))
    return tail / 2**total


def paired_bootstrap_ci(
    diffs: list[float],
    *,
    reps: int = BOOTSTRAP_REPS,
    seed: int = BOOTSTRAP_SEED,
    level: float = BOOTSTRAP_LEVEL,
) -> tuple[float, float]:
    """Percentile bootstrap over theorems (the pairing unit), with a frozen RNG seed."""

    if not diffs:
        raise ValueError("no paired differences to bootstrap")
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(reps):
        total = 0.0
        for _ in range(n):
            total += diffs[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    alpha = (1.0 - level) / 2.0
    lower = means[max(0, min(len(means) - 1, int(alpha * len(means))))]
    upper = means[max(0, min(len(means) - 1, int((1.0 - alpha) * len(means)) - 1))]
    return lower, upper


def paired_gate(a: list[bool], b: list[bool]) -> Gate:
    """Gate numbers for ``mean(b - a)`` with A as the baseline arm."""

    if len(a) != len(b):
        raise ValueError("paired arms must have equal length")
    if not a:
        raise ValueError("empty arms")
    diffs = [float(int(y) - int(x)) for x, y in zip(a, b)]
    n_favor = sum(1 for x, y in zip(a, b) if not x and y)
    n_against = sum(1 for x, y in zip(a, b) if x and not y)
    lower, _upper = paired_bootstrap_ci(diffs)
    return Gate(
        delta=sum(diffs) / len(diffs),
        mcnemar_p=mcnemar_exact_greater(n_against, n_favor),
        ci_lower=lower,
        n_favor=n_favor,
        n_against=n_against,
    )


def paired_gate_infra_as_failure(
    a: Sequence[bool | None], b: Sequence[bool | None]
) -> tuple[Gate, int]:
    """Amendment A §5 sensitivity: infra-censored (``None``) counts as failure on that arm.

    Reported separately from the primary endpoint and never used for the classification. Returns
    the gate and the number of pairs the reading had to complete with a censored arm (a
    data-completeness diagnostic, not a scientific quantity).
    """

    if len(a) != len(b):
        raise ValueError("paired arms must have equal length")
    if not a:
        raise ValueError("empty arms")
    completed = sum(1 for x, y in zip(a, b) if x is None or y is None)
    gate = paired_gate([bool(x) for x in a], [bool(y) for y in b])
    return gate, completed


def censoring_range(rates: Mapping[str, float]) -> float:
    """Amendment A §5: max(r_A, r_B, r_C, r_D) - min(r_A, r_B, r_C, r_D), computed over arms."""

    if not rates:
        raise ValueError("no arm censoring rates given")
    values = [float(rate) for rate in rates.values()]
    if any(rate < 0.0 or rate > 1.0 for rate in values):
        raise ValueError(f"censoring rates must lie in [0, 1]: {rates}")
    return max(values) - min(values)


def _primary_go(gate_ca: Gate, gate_cb: Gate) -> bool:
    return (
        gate_ca.delta >= DELTA_CA_MIN
        and gate_ca.mcnemar_p <= ALPHA
        and gate_ca.ci_lower > 0
        and gate_cb.delta >= DELTA_CB_MIN
        and gate_cb.mcnemar_p <= ALPHA
        and gate_cb.ci_lower > 0
    )


def decide_outcome(
    n_complete: int,
    censoring_rates: Mapping[str, float],
    gate_ca: Gate,
    gate_cb: Gate,
) -> Outcome:
    """The frozen decision, in order (Amendment A §6): N guard, censoring guard, then the gates."""

    if n_complete < DATA_GUARD_MIN:
        return Outcome.D_INCONCLUSIVE_BY_DATA
    if censoring_range(censoring_rates) > CENSORING_RANGE_MAX:
        return Outcome.D_INCONCLUSIVE_BY_DIFFERENTIAL_CENSORING
    if _primary_go(gate_ca, gate_cb):
        return Outcome.A_VERIFIER_SPECIFIC_GO
    if gate_ca.delta >= DELTA_CA_MIN and gate_ca.mcnemar_p <= ALPHA and gate_ca.ci_lower > 0:
        return Outcome.B_SELF_REVISION_ONLY
    return Outcome.C_NO_REPAIR_GAIN


def mechanism_label(outcome: Outcome, gate_cd: Gate) -> Mechanism:
    """Amendment A §3: DIAGNOSTIC_SPECIFIC iff C > D directionally under the paired evidence rule.

    The label is *not* part of the GO gate and carries no effect-size threshold: a large C - A or
    C - B gain with no C - D evidence means ``DIAGNOSTIC_NONSPECIFIC`` (the claim must then be
    "supplying verifier diagnostic text improved repair", never "the model understood the
    theorem-specific diagnostic"). ``NOT_EVALUABLE`` when a guard ended the study.
    """

    if outcome in {
        Outcome.D_INCONCLUSIVE_BY_DATA,
        Outcome.D_INCONCLUSIVE_BY_DIFFERENTIAL_CENSORING,
    }:
        return Mechanism.NOT_EVALUABLE
    specific = gate_cd.delta > 0 and gate_cd.mcnemar_p <= ALPHA and gate_cd.ci_lower > 0
    return Mechanism.DIAGNOSTIC_SPECIFIC if specific else Mechanism.DIAGNOSTIC_NONSPECIFIC


def min_discordant_for_p(n_against: int, alpha: float = ALPHA, limit: int = 40) -> int | None:
    """Smallest favorable discordant count that reaches ``p <= alpha`` given ``n_against``."""

    for n_favor in range(limit + 1):
        if mcnemar_exact_greater(n_against, n_favor) <= alpha:
            return n_favor
    return None


__all__ = [
    "ALPHA",
    "BOOTSTRAP_LEVEL",
    "BOOTSTRAP_REPS",
    "BOOTSTRAP_SEED",
    "CENSORING_RANGE_MAX",
    "DATA_GUARD_MIN",
    "DELTA_CA_MIN",
    "DELTA_CB_MIN",
    "N_PRIMARY",
    "Gate",
    "Mechanism",
    "Outcome",
    "censoring_range",
    "decide_outcome",
    "mcnemar_exact_greater",
    "mechanism_label",
    "min_discordant_for_p",
    "paired_bootstrap_ci",
    "paired_gate",
    "paired_gate_infra_as_failure",
]
