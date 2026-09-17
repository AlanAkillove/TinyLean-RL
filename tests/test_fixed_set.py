"""Tests for the P3-C fixed-set selection helper."""

from __future__ import annotations

import pytest

from tinylean_rl.evaluation.fixed_set import select_fixed_set


def _ids(n: int) -> list[str]:
    return [f"stmt-{index:04d}" for index in range(n)]


def test_selection_is_deterministic():
    eligible = _ids(500)
    first = select_fixed_set(eligible, 64, seed=20260917)
    second = select_fixed_set(eligible, 64, seed=20260917)
    assert first == second
    assert len(first) == 64
    assert len(set(first)) == 64


def test_different_seed_changes_selection():
    eligible = _ids(500)
    assert select_fixed_set(eligible, 64, seed=20260917) != select_fixed_set(eligible, 64, seed=1)


def test_selection_respects_eligible_order_and_exclusions():
    eligible = _ids(200)
    excluded = set(eligible[:50])
    remaining = [identifier for identifier in eligible if identifier not in excluded]
    selection = select_fixed_set(remaining, 64, seed=20260917)
    assert excluded.isdisjoint(selection)
    assert set(selection) <= set(remaining)


def test_selection_rejects_invalid_sizes():
    eligible = _ids(10)
    with pytest.raises(ValueError):
        select_fixed_set(eligible, 0, seed=0)
    with pytest.raises(ValueError):
        select_fixed_set(eligible, 11, seed=0)
