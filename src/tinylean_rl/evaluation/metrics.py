"""Small, dependency-free metrics used by the P2/P3 evaluation harness."""

from __future__ import annotations

import math


def pass_at_k(successes: int, samples: int, k: int) -> float:
    """Compute the standard unbiased Pass@k estimate from ``c`` successes in n."""

    if not 0 <= successes <= samples:
        raise ValueError("successes must be between 0 and samples")
    if not 1 <= k <= samples:
        raise ValueError("k must be between 1 and samples")
    if samples - successes < k:
        return 1.0
    return 1.0 - math.comb(samples - successes, k) / math.comb(samples, k)


def group_rates(success_counts: list[int], samples_per_group: int) -> dict[str, float]:
    """Return all-zero, mixed and all-one group rates."""

    if not success_counts:
        return {"all_zero": 0.0, "mixed": 0.0, "all_one": 0.0}
    n = len(success_counts)
    return {
        "all_zero": sum(count == 0 for count in success_counts) / n,
        "mixed": sum(0 < count < samples_per_group for count in success_counts) / n,
        "all_one": sum(count == samples_per_group for count in success_counts) / n,
    }

