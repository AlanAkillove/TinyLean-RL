"""Fixtures for the frozen R001 process-credit semantics (owner directive §1-§4, draft §7.5).

Pure CPU: no network, no model, no oracle server, no holdout. These are the parity/collision/
padding/censoring fixtures the preregistration requires before any GPU step. The collision
fixtures are pinned against the frozen offline surface (``v5_process_oracle.py::_token_credit``)
so the online and offline paths cannot drift apart.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from v5_process_oracle import _token_credit

from tinylean_rl.rl.grpo import compute_grpo_outcome_advantage
from tinylean_rl.rl.process_credit import (
    PHI_D1,
    PHI_D2,
    PHI_SUCCESS,
    CreditStats,
    assign_positions,
    compute_process_advantage,
    credit_phi,
    summarize_groups,
    tactic_labels,
)

RESPONSE_LENGTH = 8


def _batch(
    rewards: list[float], lengths: list[int] | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-padded token_level_rewards + response_mask; the outcome sits on token ``length-1``."""

    lengths = lengths or [RESPONSE_LENGTH] * len(rewards)
    token_rewards = torch.zeros(len(rewards), RESPONSE_LENGTH)
    mask = torch.zeros(len(rewards), RESPONSE_LENGTH)
    for row, (reward, length) in enumerate(zip(rewards, lengths)):
        for position in range(length):
            mask[row, position] = 1.0
        token_rewards[row, length - 1] = reward
    return token_rewards, mask


def _rec(
    label: str | None,
    status: str,
    *,
    token_index: int | None = None,
    span_in_response: bool | None = None,
) -> dict[str, Any]:
    """One oracle-style tactic record, exactly as ``derive_facts`` emits it."""

    mapping: dict[str, Any] = {"status": status}
    if token_index is not None:
        mapping["token_index"] = token_index
        mapping["token_id"] = 1000 + token_index
    if span_in_response is not None:
        mapping["span_in_response"] = span_in_response
    return {"label": label, "mapping": mapping}


def test_lambda_zero_equals_v1_bit_for_bit() -> None:
    rewards = [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0]
    index = ["g0"] * 4 + ["g1"] * 4
    token_rewards, mask = _batch(rewards)
    reference, _ = compute_grpo_outcome_advantage(
        token_rewards.clone(), mask.clone(), index, norm_adv_by_std_in_grpo=False
    )
    credit = [{2: "d1", 4: "d2"}, {1: "success"}, {}, {3: "d1"}]
    for row in range(4, 8):
        credit.append({} if row != 5 else {2: "success"})
    advantages, returns, row_keep, stats = compute_process_advantage(
        token_rewards.clone(), mask.clone(), index, [1] * 8, credit, lambda_process=0.0
    )
    assert torch.equal(advantages, reference)
    assert torch.equal(returns, reference)
    assert torch.equal(row_keep, torch.ones(8))
    assert stats.infra_censored == 0
    assert stats.group_skipped_infra == 0
    assert stats.credit_tokens == 0  # lambda = 0 applies no process term


def test_lambda_one_adds_phi_minus_gbar_once_per_position() -> None:
    token_rewards, mask = _batch([1.0, 0.0], lengths=[6, 6])
    index = ["g"] * 2
    advantage, _, _, _ = compute_process_advantage(
        token_rewards, mask, index, [1, 1], [{}, {}], lambda_process=0.0
    )
    # g_bar = 0.5; success row: A_outcome = 0.5
    assert advantage[0, 5].item() == pytest.approx(0.5)
    advantages, _, _, stats = compute_process_advantage(
        token_rewards, mask, index, [1, 1], [{2: "success", 4: "d1"}, {1: "d2"}], lambda_process=1.0
    )
    assert advantages[0, 2].item() == pytest.approx(0.5 + (PHI_SUCCESS - 0.5))
    assert advantages[0, 4].item() == pytest.approx(0.5 + (PHI_D1 - 0.5))
    assert advantages[0, 5].item() == pytest.approx(0.5)  # outcome token untouched
    assert advantages[1, 1].item() == pytest.approx(-0.5 + (PHI_D2 - 0.5))
    assert stats.credit_tokens == 3


def test_collision_precedence_is_single_writer() -> None:
    first = [
        _rec("success", "exact", token_index=3, span_in_response=True),
        _rec("d1", "contained", token_index=3, span_in_response=True),
    ]
    positions, stats = assign_positions(first)
    assert positions == {3: "d1"}
    assert stats.conflicts == 1
    assert stats.mappable_tactics == 2
    # d2 beats d1 in either order; re-writing the same label is not a conflict
    # (frozen ``_token_credit`` semantics) but still leaves exactly one writer.
    both = [
        _rec("d1", "exact", token_index=5, span_in_response=True),
        _rec("d2", "exact", token_index=5, span_in_response=True),
        _rec("d2", "exact", token_index=5, span_in_response=True),
    ]
    positions, stats = assign_positions(both)
    assert positions == {5: "d2"}
    assert stats.conflicts == 1
    reversed_order = [both[1], both[0]]
    positions, stats = assign_positions(reversed_order)
    assert positions == {5: "d2"}
    assert stats.conflicts == 1


def test_collision_advantage_matches_single_write() -> None:
    token_rewards, mask = _batch([0.0], lengths=[6])
    index = ["g", "g"]  # two rows so the group is usable
    token_rewards = torch.cat([token_rewards, token_rewards], dim=0)
    mask = torch.cat([mask, mask], dim=0)
    credit = [{3: "d1", 4: "d2"}, {}]
    # simulate the oracle having resolved row 0's position-3 collision to d2 upstream
    advantages, _, _, stats = compute_process_advantage(
        token_rewards, mask, index, [1, 1], credit, lambda_process=1.0
    )
    group_mean = 0.0
    assert advantages[0, 3].item() == pytest.approx(PHI_D1 - group_mean)
    assert advantages[0, 4].item() == pytest.approx(PHI_D2 - group_mean)
    assert stats.credit_tokens == 2


def test_padding_credit_is_dropped_and_counted() -> None:
    token_rewards, mask = _batch([0.0, 0.0], lengths=[4, 6])
    index = ["g"] * 2
    credit = [{5: "success"}, {}]  # position 5 is padding on row 0 (length 4)
    advantages, _, _, stats = compute_process_advantage(
        token_rewards, mask, index, [1, 1], credit, lambda_process=1.0
    )
    assert advantages[0, 5].item() == 0.0
    assert stats.dropped_by_mask == 1
    assert stats.credit_tokens == 0  # only written credit counts as a credit token


def test_out_of_range_position_is_counted_not_shifted() -> None:
    token_rewards, mask = _batch([0.0, 0.0], lengths=[6, 6])
    index = ["g"] * 2
    credit = [{9: "success", -1: "d1"}, {}]
    advantages, _, _, stats = compute_process_advantage(
        token_rewards, mask, index, [1, 1], credit, lambda_process=1.0
    )
    assert advantages.sum().item() == pytest.approx(0.0)
    assert stats.out_of_range == 2
    assert stats.credit_tokens == 0


def test_infra_censoring_excludes_from_gbar_and_masks_row() -> None:
    rewards = [1.0, 0.0, 0.0, 0.0]  # one valid success, three rows where two are infra
    token_rewards, mask = _batch(rewards)
    index = ["g"] * 4
    row_valid = [1, 1, 0, 0]
    advantages, _, row_keep, stats = compute_process_advantage(
        token_rewards, mask, index, row_valid, None, lambda_process=1.0
    )
    # g_bar over the two VALID rows only: (1 + 0)/2 = 0.5
    assert advantages[0, 7].item() == pytest.approx(0.5)
    assert advantages[1, 7].item() == pytest.approx(-0.5)
    assert row_keep.tolist() == [1.0, 1.0, 0.0, 0.0]
    assert advantages[2].abs().sum().item() == 0.0
    assert stats.infra_censored == 2
    assert stats.group_skipped_infra == 0


def test_group_with_fewer_than_two_valid_is_skipped() -> None:
    rewards = [1.0, 0.0, 0.0, 0.0, 0.0]
    token_rewards, mask = _batch(rewards)
    index = ["g0"] * 5 + ["g1"] * 3
    token_rewards = torch.cat([token_rewards, torch.zeros(3, RESPONSE_LENGTH)], dim=0)
    mask = torch.cat([mask, torch.zeros(3, RESPONSE_LENGTH)], dim=0)
    for row in range(5, 8):
        for position in range(6):
            mask[row, position] = 1.0
    row_valid = [1] + [0] * 4 + [1, 1, 0]
    advantages, _, row_keep, stats = compute_process_advantage(
        token_rewards, mask, index, row_valid, None, lambda_process=1.0
    )
    assert advantages[:5].abs().sum().item() == 0.0  # group g0 skipped entirely
    assert row_keep[:5].tolist() == [0.0] * 5
    assert row_keep[5:].tolist() == [1.0, 1.0, 0.0]
    assert stats.group_skipped_infra == 1
    report = summarize_groups(index, row_valid)
    assert report.groups == 2 and report.groups_skipped_infra == 1


def test_retokenization_mismatch_counts_and_drops_credit() -> None:
    records = [
        _rec("d2", "retokenization_mismatch"),
        _rec("d1", "exact", token_index=2, span_in_response=True),
        _rec("d1", "outside_response"),
        _rec("d2", "ambiguous"),
        _rec(None, "unmapped"),
        _rec(None, "response_only"),
        _rec("d1", "exact", token_index=4, span_in_response=False),
        _rec(None, "exact", token_index=6, span_in_response=True),
    ]
    positions, stats = assign_positions(records)
    assert positions == {2: "d1"}
    assert stats.retokenization_mismatch == 1
    assert stats.outside_response == 1
    assert stats.ambiguous == 1
    assert stats.unmapped == 2  # the unmapped status + the span outside the response
    assert stats.span_not_in_response == 1
    assert stats.mappable_tactics == 2  # only tactics whose span is inside the response


def test_tactic_labels_follow_the_frozen_oracle() -> None:
    assert tactic_labels(3, blamed_index=1, outcome_valid=True, outcome=0, infra=False) == [
        "d1",
        "d2",
        "d2",
    ]
    assert tactic_labels(2, blamed_index=0, outcome_valid=True, outcome=0, infra=False) == ["d2", "d2"]
    assert tactic_labels(2, blamed_index=None, outcome_valid=True, outcome=0, infra=False) == [None, None]
    assert tactic_labels(2, blamed_index=None, outcome_valid=True, outcome=1, infra=False) == [
        "success",
        "success",
    ]
    assert tactic_labels(2, blamed_index=0, outcome_valid=False, outcome=0, infra=True) == [None, None]


def test_credit_phi_constants() -> None:
    assert credit_phi({0: "success", 1: "d1", 2: "d2"}) == {0: PHI_SUCCESS, 1: PHI_D1, 2: PHI_D2}
    assert PHI_SUCCESS == 1.0 and PHI_D1 == -0.05 and PHI_D2 == -0.10


def test_stats_dict_is_complete() -> None:
    keys = set(CreditStats().as_dict())
    assert {"mappable_tactics", "ambiguous", "outside_response", "retokenization_mismatch"} <= keys
    assert {"conflicts", "credit_tokens", "dropped_by_mask", "out_of_range"} <= keys
    assert {"infra_censored", "group_skipped_infra", "no_code"} <= keys


def test_assign_positions_parity_with_frozen_oracle() -> None:
    records = [
        _rec("success", "exact", token_index=3, span_in_response=True),
        _rec("d1", "contained", token_index=3, span_in_response=True),
        _rec("d2", "exact", token_index=5, span_in_response=True),
        _rec("d1", "exact", token_index=5, span_in_response=True),
        _rec("success", "exact", token_index=7, span_in_response=True),
        _rec("success", "exact", token_index=7, span_in_response=True),
        _rec("d1", "exact", token_index=9, span_in_response=False),
        _rec("d2", "outside_response"),
        _rec(None, "ambiguous"),
        _rec(None, "unmapped"),
        _rec(None, "response_only"),
        _rec(None, "exact", token_index=11, span_in_response=True),
    ]
    positions, stats = assign_positions(records)
    oracle_credit = _token_credit(records, infra=False, blamed=None)
    oracle_positions: dict[int, str] = {}
    for label, key in (
        ("success", "success_token_positions"),
        ("d1", "d1_token_positions"),
        ("d2", "d2_token_positions"),
    ):
        for token_index, _token_id in oracle_credit[key]:
            oracle_positions[int(token_index)] = label
    assert positions == oracle_positions
    assert stats.mappable_tactics == oracle_credit["n_mapped"]
    assert stats.exact == oracle_credit["n_exact"]
    assert stats.contained == oracle_credit["n_contained"]
    assert stats.span_not_in_response == oracle_credit["n_span_not_in_response"]
    assert stats.outside_response == oracle_credit["n_outside_response"]
    assert stats.conflicts == oracle_credit["n_credit_conflicts"]
