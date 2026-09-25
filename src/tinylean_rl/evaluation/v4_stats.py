"""V4-P001 primary statistics and the frozen outcome taxonomy (owner §K, §L, §M, §N).

Everything is paired at the theorem level. Per theorem ``i`` the three arms give a binary success
indicator under one success definition (canonical score == 1 with the same format-gated Lean
verification), and the primary endpoints are

    Delta_repair    = mean(C_i - A_i)      Arm C against a plain fresh retry
    Delta_diagnostic = mean(C_i - B_i)     Arm C against self-revision without the diagnostic

each with a one-sided exact paired McNemar test on the discordant pairs and a paired theorem
bootstrap 95 % confidence interval. ``B - A`` is reported descriptively only. AUPRC and any
predictive metric are explicitly not endpoints here.

The data guard (§M) is applied before the taxonomy: fewer than 103 complete valid triplets means
``INCONCLUSIVE_BY_DATA`` and the GO taxonomy is never evaluated. Infra-censored arms are *missing*,
never failures, and a theorem is never replaced across families once second-stage outcomes exist.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import Enum

N_PRIMARY = 128
DATA_GUARD_MIN = 103

DELTA_CA_MIN = 0.08
DELTA_CB_MIN = 0.05
ALPHA = 0.05
BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 20260925
BOOTSTRAP_LEVEL = 0.95


class Outcome(str, Enum):
    """Frozen §N taxonomy. Only A authorizes an owner decision about future verifier-integrated work."""

    A_VERIFIER_SPECIFIC_GO = "A_VERIFIER_SPECIFIC_GO"
    B_SELF_REVISION_ONLY = "B_SELF_REVISION_ONLY"
    C_NO_REPAIR_GAIN = "C_NO_REPAIR_GAIN"
    D_INCONCLUSIVE_BY_DATA = "D_INCONCLUSIVE_BY_DATA"


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


def decide_outcome(n_complete: int, gate_ca: Gate, gate_cb: Gate) -> Outcome:
    """The frozen §N decision, in order. The data guard is checked before any gate is read."""

    if n_complete < DATA_GUARD_MIN:
        return Outcome.D_INCONCLUSIVE_BY_DATA
    verifier_specific = (
        gate_ca.delta >= DELTA_CA_MIN
        and gate_ca.mcnemar_p <= ALPHA
        and gate_ca.ci_lower > 0
        and gate_cb.delta >= DELTA_CB_MIN
        and gate_cb.mcnemar_p <= ALPHA
        and gate_cb.ci_lower > 0
    )
    if verifier_specific:
        return Outcome.A_VERIFIER_SPECIFIC_GO
    if gate_ca.delta >= DELTA_CA_MIN and gate_ca.mcnemar_p <= ALPHA and gate_ca.ci_lower > 0:
        return Outcome.B_SELF_REVISION_ONLY
    return Outcome.C_NO_REPAIR_GAIN


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
    "DATA_GUARD_MIN",
    "DELTA_CA_MIN",
    "DELTA_CB_MIN",
    "N_PRIMARY",
    "Gate",
    "Outcome",
    "decide_outcome",
    "mcnemar_exact_greater",
    "min_discordant_for_p",
    "paired_bootstrap_ci",
    "paired_gate",
]
