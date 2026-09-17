"""Deterministic fixed-set selection helpers for P3-C (E018)."""

from __future__ import annotations

import random
from collections.abc import Sequence


def select_fixed_set(eligible_ids: Sequence[str], n: int, seed: int) -> list[str]:
    """Deterministically sample ``n`` ids from ``eligible_ids``.

    Determinism relies on the caller passing the eligible ids in a stable,
    dataset-defined order (the parquet dedup order); this function adds no
    ordering of its own. The same inputs always yield the same selection, and
    the selection must be sealed before any evaluation result is observed.
    """

    if n < 1:
        raise ValueError("n must be positive")
    if n > len(eligible_ids):
        raise ValueError(f"cannot select {n} ids from {len(eligible_ids)} eligible ids")
    return random.Random(seed).sample(list(eligible_ids), n)
