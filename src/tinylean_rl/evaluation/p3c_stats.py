"""Statistical helpers for the P3-C fixed-set evaluation (E018).

The pairing unit is the *theorem* (never the candidate). See
``docs/experiment_log.md`` E018 and the P3-C protocol: per-theorem
``c_i(t)`` counts of verified candidates drive the paired bootstrap and the
McNemar solved-indicator analysis.
"""

from __future__ import annotations

import math
import random
import statistics
from collections import Counter
from collections.abc import Sequence

from tinylean_rl.evaluation.metrics import group_rates, pass_at_k

TAXONOMY_CATEGORIES = (
    "verified",
    "truncated_no_complete_lean_block",
    "nontruncated_format_invalid",
    "lean_parse_error",
    "lean_semantic_error",
    "verifier_error",
)

# Heuristic parse-vs-semantic split for Lean rejection messages (diagnostic only).
_PARSE_HINTS = (
    "unexpected token",
    "expected",
    "unknown identifier",
    "unknown constant",
    "unknown tactic",
    "invalid syntax",
    "parser",
    "unterminated",
)


def classify_candidate(*, truncated: bool, format_ok: bool, verify_status: str, lean_message: str | None) -> str:
    """Map one candidate to the E018 error taxonomy."""

    if verify_status == "verified":
        return "verified"
    if verify_status == "verifier_error":
        return "verifier_error"
    if not format_ok:
        return "truncated_no_complete_lean_block" if truncated else "nontruncated_format_invalid"
    message = (lean_message or "").lower()
    if any(hint in message for hint in _PARSE_HINTS):
        return "lean_parse_error"
    return "lean_semantic_error"


def _percentile(sorted_values: list[float], fraction: float) -> float:
    index = min(len(sorted_values) - 1, max(0, round(fraction * len(sorted_values)) - 1))
    return sorted_values[index]


def candidate_metrics(records: Sequence[dict], samples_per_theorem: int) -> dict:
    """Aggregate per-candidate and per-theorem statistics for one checkpoint.

    Records must expose ``theorem_index``, ``verified``, ``truncated``,
    ``format_ok``, ``verify_status``, ``lean_message``, ``generated_tokens``
    and ``taxonomy``.
    """

    total = len(records)
    if total == 0:
        return {"theorems": 0, "candidates": 0}
    counts: dict[int, int] = {}
    for record in records:
        index = record["theorem_index"]
        counts[index] = counts.get(index, 0) + int(record["verified"])
    theorem_count = len(counts)
    c_values = [counts[index] for index in sorted(counts)]

    def mean_pass_at(k: int) -> float:
        effective = min(k, samples_per_theorem)
        return sum(pass_at_k(c, samples_per_theorem, effective) for c in c_values) / theorem_count

    lengths = sorted(float(record["generated_tokens"]) for record in records)
    taxonomy = Counter(record["taxonomy"] for record in records)
    return {
        "theorems": theorem_count,
        "candidates": total,
        "verified_candidates": sum(c_values),
        "candidate_verified_rate": sum(c_values) / total,
        "mean_c_over_n": sum(c_values) / (theorem_count * samples_per_theorem),
        "theorems_solved_at_least_1": sum(1 for c in c_values if c >= 1),
        "theorems_solved_at_least_4": sum(1 for c in c_values if c >= 4),
        "theorems_solved_all": sum(1 for c in c_values if c == samples_per_theorem),
        "group_rates": group_rates(c_values, samples_per_theorem),
        "igr": group_rates(c_values, samples_per_theorem)["mixed"],
        "pass_at_1": mean_pass_at(1),
        "pass_at_4": mean_pass_at(4),
        "pass_at_8": mean_pass_at(8),
        "response_length": {
            "mean": statistics.fmean(lengths),
            "median": statistics.median(lengths),
            "p95": _percentile(lengths, 0.95),
            "max": lengths[-1],
        },
        "truncation_rate": sum(record["truncated"] for record in records) / total,
        "verifier_error_rate": taxonomy.get("verifier_error", 0) / total,
        "taxonomy": {category: taxonomy.get(category, 0) for category in TAXONOMY_CATEGORIES},
        "per_theorem_counts": {str(index): counts[index] for index in sorted(counts)},
    }


def paired_bootstrap(
    deltas: Sequence[float],
    n_resamples: int = 10_000,
    seed: int = 20260917,
    alpha: float = 0.05,
) -> dict:
    """Percentile bootstrap CI for the mean of theorem-level paired deltas."""

    if not deltas:
        return {"n": 0}
    rng = random.Random(seed)
    n = len(deltas)
    means = []
    for _ in range(n_resamples):
        total = 0.0
        for _ in range(n):
            total += deltas[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    low_index = max(0, round(alpha / 2 * n_resamples) - 1)
    high_index = min(n_resamples - 1, round((1 - alpha / 2) * n_resamples) - 1)
    return {
        "n": n,
        "mean_delta": statistics.fmean(deltas),
        "median_delta": statistics.median(deltas),
        "ci_low": means[low_index],
        "ci_high": means[high_index],
        "n_resamples": n_resamples,
        "seed": seed,
    }


def cluster_bootstrap(
    deltas_by_cluster: Sequence[Sequence[float]],
    n_resamples: int = 10_000,
    seed: int = 20260917,
    alpha: float = 0.05,
) -> dict:
    """Percentile bootstrap over clusters (families): resample clusters with
    replacement, carrying all theorems of a drawn cluster.

    Post-hoc robustness tool (V2-A001 family-cluster bootstrap): the formal
    selection rule always uses the theorem-level ``paired_bootstrap`` and is
    never affected by this function's output.
    """

    if not any(deltas_by_cluster):
        return {
            "n_clusters": 0,
            "n_theorems": 0,
            "mean_delta": 0.0,
            "median_delta": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "n_resamples": n_resamples,
            "seed": seed,
        }
    rng = random.Random(seed)
    n_clusters = len(deltas_by_cluster)
    means = []
    for _ in range(n_resamples):
        total = 0.0
        count = 0
        for _ in range(n_clusters):
            cluster = deltas_by_cluster[rng.randrange(n_clusters)]
            total += sum(cluster)
            count += len(cluster)
        means.append(total / count)
    means.sort()
    low_index = max(0, round(alpha / 2 * n_resamples) - 1)
    high_index = min(n_resamples - 1, round((1 - alpha / 2) * n_resamples) - 1)
    pooled = [delta for cluster in deltas_by_cluster for delta in cluster]
    return {
        "n_clusters": n_clusters,
        "n_theorems": len(pooled),
        "mean_delta": statistics.fmean(pooled),
        "median_delta": statistics.median(pooled),
        "ci_low": means[low_index],
        "ci_high": means[high_index],
        "n_resamples": n_resamples,
        "seed": seed,
    }


def win_tie_loss(baseline: Sequence[int], treatment: Sequence[int]) -> dict:
    """Theorem-level win/tie/loss counts between two checkpoints."""

    wins = sum(1 for base, treat in zip(baseline, treatment) if treat > base)
    ties = sum(1 for base, treat in zip(baseline, treatment) if treat == base)
    losses = sum(1 for base, treat in zip(baseline, treatment) if treat < base)
    return {"win": wins, "tie": ties, "loss": losses}


def mcnemar_exact(baseline_solved: Sequence[bool], treatment_solved: Sequence[bool]) -> dict:
    """Exact two-sided McNemar test on the solved indicator s_i(t)."""

    diff = [
        (bool(base), bool(treat))
        for base, treat in zip(baseline_solved, treatment_solved)
        if bool(base) != bool(treat)
    ]
    newly_solved = sum(1 for base, treat in diff if treat and not base)
    newly_lost = sum(1 for base, treat in diff if base and not treat)
    n = newly_solved + newly_lost
    if n == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(n, i) for i in range(min(newly_solved, newly_lost) + 1)) / 2**n
        p_value = min(1.0, 2 * tail)
    return {
        "newly_solved": newly_solved,
        "newly_lost": newly_lost,
        "still_solved": sum(1 for base, treat in zip(baseline_solved, treatment_solved) if base and treat),
        "still_unsolved": sum(1 for base, treat in zip(baseline_solved, treatment_solved) if not base and not treat),
        "discordant_pairs": n,
        "p_value_two_sided": p_value,
    }
