"""V4-P001 seed derivation (owner §D, §J).

Two frozen seed streams, both pure functions of a frozen constant:

    first-stage screening   seed = V4_BASE_SEED + (screening_rank - 1)          rank 1..640
    second-stage paired     seed = V4_BASE_SEED + SECOND_STAGE_OFFSET + (i - 1) i    1..128

The second-stage seed is a function of the frozen formal rank *only*, and all three arms share it
(common random numbers, §J): ``second_stage_seed`` is called with the same argument for A, B and C, so
the three arms differ by their prompt and by nothing else. No outcome, source, family, error category
or diagnostic label enters either formula, which is what makes the invariance test meaningful: the
same theorem gets the same seed whatever the arm, the labels or the diagnostic say.
"""

from __future__ import annotations

V4_BASE_SEED = 20260925
SECOND_STAGE_OFFSET = 1_000_000
N_PRIMARY = 128
MAX_SCREENING = 640
MAX_SEED = 2**31 - 1


def _check(rank: int, upper: int, label: str) -> None:
    if not isinstance(rank, int) or isinstance(rank, bool):
        raise TypeError(f"{label} must be an int, got {type(rank)!r}")
    if not 1 <= rank <= upper:
        raise ValueError(f"{label} {rank} outside 1..{upper}")


def first_stage_seed(screening_rank: int) -> int:
    """Generation seed of the single first-attempt roll-out at a frozen screening rank."""

    _check(screening_rank, MAX_SCREENING, "screening_rank")
    return V4_BASE_SEED + (screening_rank - 1)


def second_stage_seed(formal_rank: int) -> int:
    """Shared generation seed of the A/B/C second-stage candidates at a frozen formal rank."""

    _check(formal_rank, N_PRIMARY, "formal_rank")
    return V4_BASE_SEED + SECOND_STAGE_OFFSET + (formal_rank - 1)


def first_stage_seeds(n: int) -> list[int]:
    _check(n, MAX_SCREENING, "n")
    return [first_stage_seed(rank) for rank in range(1, n + 1)]


def second_stage_seeds(n: int = N_PRIMARY) -> list[int]:
    _check(n, N_PRIMARY, "n")
    return [second_stage_seed(rank) for rank in range(1, n + 1)]


__all__ = [
    "MAX_SCREENING",
    "MAX_SEED",
    "N_PRIMARY",
    "SECOND_STAGE_OFFSET",
    "V4_BASE_SEED",
    "first_stage_seed",
    "first_stage_seeds",
    "second_stage_seed",
    "second_stage_seeds",
]
