"""Tests for the P3-C statistics helpers."""

from __future__ import annotations

import pytest

from tinylean_rl.evaluation.p3c_stats import (
    candidate_metrics,
    classify_candidate,
    mcnemar_exact,
    paired_bootstrap,
    win_tie_loss,
)


def _record(
    theorem_index: int,
    sample: int,
    verified: bool,
    *,
    truncated: bool = False,
    format_ok: bool = True,
    verify_status: str = "lean_error",
    lean_message: str = "",
    tokens: int = 100,
) -> dict:
    return {
        "theorem_index": theorem_index,
        "sample": sample,
        "verified": verified,
        "truncated": truncated,
        "format_ok": format_ok,
        "verify_status": verify_status,
        "lean_message": lean_message,
        "generated_tokens": tokens,
        "taxonomy": classify_candidate(
            truncated=truncated,
            format_ok=format_ok,
            verify_status=verify_status,
            lean_message=lean_message,
        ),
    }


def test_classify_candidate_taxonomy():
    assert (
        classify_candidate(truncated=False, format_ok=True, verify_status="verified", lean_message="")
        == "verified"
    )
    assert (
        classify_candidate(truncated=True, format_ok=False, verify_status="lean_error", lean_message="")
        == "truncated_no_complete_lean_block"
    )
    assert (
        classify_candidate(truncated=False, format_ok=False, verify_status="lean_error", lean_message="")
        == "nontruncated_format_invalid"
    )
    assert (
        classify_candidate(
            truncated=False, format_ok=True, verify_status="lean_error", lean_message="unexpected token 'foo'"
        )
        == "lean_parse_error"
    )
    assert (
        classify_candidate(
            truncated=False, format_ok=True, verify_status="lean_error", lean_message="tactic 'rfl' failed"
        )
        == "lean_semantic_error"
    )
    assert (
        classify_candidate(truncated=False, format_ok=True, verify_status="verifier_error", lean_message="")
        == "verifier_error"
    )


def test_candidate_metrics_known_values():
    records = [
        _record(0, 0, True, verify_status="verified"),
        _record(0, 1, True, verify_status="verified"),
        _record(1, 0, False),
        _record(1, 1, False, truncated=True, format_ok=False),
    ]
    metrics = candidate_metrics(records, 2)
    assert metrics["candidates"] == 4
    assert metrics["candidate_verified_rate"] == pytest.approx(0.5)
    assert metrics["theorems_solved_at_least_1"] == 1
    assert metrics["theorems_solved_all"] == 1
    assert metrics["group_rates"]["all_one"] == pytest.approx(0.5)
    assert metrics["group_rates"]["all_zero"] == pytest.approx(0.5)
    assert metrics["igr"] == 0.0
    assert metrics["pass_at_1"] == pytest.approx(0.5)
    assert metrics["pass_at_8"] == pytest.approx(0.5)
    assert metrics["taxonomy"]["verified"] == 2
    assert metrics["taxonomy"]["truncated_no_complete_lean_block"] == 1
    assert metrics["taxonomy"]["lean_semantic_error"] == 1


def test_paired_bootstrap_deterministic_and_zero_ci():
    deltas = [0.25, -0.25, 0.0, 0.5]
    first = paired_bootstrap(deltas, n_resamples=500, seed=7)
    second = paired_bootstrap(deltas, n_resamples=500, seed=7)
    assert first == second
    assert first["n"] == 4
    assert first["ci_low"] <= first["mean_delta"] <= first["ci_high"]
    zeros = paired_bootstrap([0.0] * 8, n_resamples=200, seed=1)
    assert zeros["ci_low"] == 0.0
    assert zeros["ci_high"] == 0.0


def test_win_tie_loss_counts():
    assert win_tie_loss([1, 2, 3, 4], [2, 2, 1, 4]) == {"win": 1, "tie": 2, "loss": 1}


def test_mcnemar_exact_known_values():
    result = mcnemar_exact([False] * 5, [True] * 5)
    assert result["newly_solved"] == 5
    assert result["newly_lost"] == 0
    assert result["p_value_two_sided"] == pytest.approx(0.0625)
    same = mcnemar_exact([True, False], [True, False])
    assert same["p_value_two_sided"] == 1.0
