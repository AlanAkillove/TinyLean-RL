"""V4-P001 balanced arm-order schedule (Amendment A, owner §7).

Executing every theorem as a fixed ``A -> B -> C -> D`` sequence would confound arm identity with
temporal/server state: Arm C would always meet a warmer server than Arm A. The schedule below
freezes a balanced execution order over the 128 formal ranks, built from the 24 permutations of the
four arms:

* formal ranks 1..120 use the 24 permutations in lexicographic order, five complete cycles, which is
  exactly position-balanced and predecessor-balanced over that block;
* formal ranks 121..128 use the eight rotations of the two edge-disjoint directed 4-cycles
  ``(A,B,C,D)`` and ``(A,D,C,B)`` -- two Latin squares sharing no ordered pair -- which restores the
  per-position counts to exactly 32 per arm.

Result (checked in tests): every arm appears in every position exactly 32 times, and every ordered
predecessor pair appears 30-33 times out of an ideal 32. The schedule is a pure function of the
formal theorem rank; no source, error category, diagnostic or outcome enters it.
"""

from __future__ import annotations

import hashlib
import itertools
import json

N_PRIMARY = 128
POSITIONS = 4
ARM_ORDER = (
    "A_FRESH_RETRY",
    "B_SELF_REVISION",
    "C_VERIFIER_REPAIR",
    "D_MISMATCHED_DIAGNOSTIC",
)
ARM_INDEX = {arm: index for index, arm in enumerate(ARM_ORDER)}

#: All 24 permutations of four arms, lexicographic -- 120 rows = five complete cycles.
_PERMUTATIONS: tuple[tuple[int, ...], ...] = tuple(itertools.permutations(range(POSITIONS)))
_CYCLES = 120
#: Rotations of two *edge-disjoint* directed 4-cycles, ``(A,B,C,D)`` and ``(A,D,C,B)``; together
#: they contribute exactly two of each arm to each position (completing the 120-row block to 32 per
#: arm and position) and keep every ordered predecessor pair within 30..33 of an ideal 32, because
#: the two cycles share no ordered pair.
_EXTRA: tuple[tuple[int, ...], ...] = (
    (0, 1, 2, 3),
    (1, 2, 3, 0),
    (2, 3, 0, 1),
    (3, 0, 1, 2),
    (0, 3, 2, 1),
    (3, 2, 1, 0),
    (2, 1, 0, 3),
    (1, 0, 3, 2),
)


def _permutation(row: int) -> tuple[int, ...]:
    if not 0 <= row < N_PRIMARY:
        raise ValueError(f"formal rank row {row} outside 0..{N_PRIMARY - 1}")
    return _PERMUTATIONS[row % len(_PERMUTATIONS)] if row < _CYCLES else _EXTRA[row - _CYCLES]


def arm_order(formal_rank: int) -> list[str]:
    """The frozen generation order of the four arms at one formal rank (1-based)."""

    return [ARM_ORDER[index] for index in _permutation(formal_rank - 1)]


def schedule() -> list[dict]:
    return [
        {"formal_rank": rank, "order": arm_order(rank)} for rank in range(1, N_PRIMARY + 1)
    ]


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def schedule_hash() -> str:
    return hashlib.sha256(canonical_json(schedule()).encode("utf-8")).hexdigest()


def balance_report() -> dict:
    """Position counts per arm and ordered predecessor-pair counts, for the audit."""

    rows = schedule()
    positions = {arm: {index: 0 for index in range(1, POSITIONS + 1)} for arm in ARM_ORDER}
    predecessors: dict[str, int] = {}
    for row in rows:
        for index, arm in enumerate(row["order"], start=1):
            positions[arm][index] += 1
        for first, second in zip(row["order"], row["order"][1:]):
            key = f"{first}->{second}"
            predecessors[key] = predecessors.get(key, 0) + 1
    return {
        "arms": list(ARM_ORDER),
        "positions_per_arm": {arm: [positions[arm][index] for index in range(1, POSITIONS + 1)]
                              for arm in ARM_ORDER},
        "predecessor_pairs": dict(sorted(predecessors.items())),
        "predecessor_min": min(predecessors.values()),
        "predecessor_max": max(predecessors.values()),
    }


__all__ = [
    "ARM_INDEX",
    "ARM_ORDER",
    "N_PRIMARY",
    "POSITIONS",
    "arm_order",
    "balance_report",
    "canonical_json",
    "schedule",
    "schedule_hash",
]
