"""Fixtures for the R001 S2 self-check (owner directive §13, draft §10.4 S2).

The self-check is the in-run proof that the shared R001 execution stack writes the frozen
semantics onto the advantage tensor: λ = 0 parity with the pinned DrGRPO re-derivation on every
infra-free group, and λ = 1 equal to the frozen per-token additions ``λ (φ − ḡ)`` at the credited
positions only. These fixtures drive the pure function directly, including the *failing*
directions: a treatment step must not run on an unverified advantage path, so every drift must
raise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import tinylean_rl.rl.r001_selfcheck as SC
from tinylean_rl.rl.process_credit import PHI, compute_process_advantage
from tinylean_rl.rl.r001_selfcheck import SELFCHECK_CONFIG_KEY, check_advantage_paths

WIDTH = 8


def _tensors(
    rewards: list[float],
    index: list[str],
    row_valid: list[int],
    credit: list[dict],
    *,
    lambda_process: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full-width-masked batch with the outcome on the last token; λ = 1 by default."""

    bs = len(rewards)
    token_rewards = torch.zeros(bs, WIDTH)
    mask = torch.ones(bs, WIDTH)
    for row, reward in enumerate(rewards):
        token_rewards[row, WIDTH - 1] = reward
    advantages, returns, row_keep, _ = compute_process_advantage(
        token_rewards, mask, index, row_valid, credit, lambda_process=lambda_process
    )
    return token_rewards, mask, advantages, returns, row_keep


def _check(
    token_rewards: torch.Tensor,
    mask: torch.Tensor,
    index: list[str],
    row_valid: list[int],
    credit: list[dict],
    advantages: torch.Tensor,
    returns: torch.Tensor,
    row_keep: torch.Tensor,
    *,
    lambda_process: float = 1.0,
) -> dict:
    return check_advantage_paths(
        token_rewards,
        mask,
        index,
        row_valid,
        credit,
        advantages,
        returns,
        row_keep,
        lambda_process=lambda_process,
    )


def test_selfcheck_config_key_is_the_frozen_name() -> None:
    assert SELFCHECK_CONFIG_KEY == "r001_selfcheck"


def test_clean_batch_passes_and_reports_the_parity_scope() -> None:
    index = ["g", "g", "g", "g", "h", "h"]
    rewards = [1.0, 1.0, 0.0, 0.0, 1.0, 0.0]
    row_valid = [1, 1, 1, 1, 1, 1]
    credit = [{2: "success"}, {1: "d1"}, {}, {5: "d2"}, {0: "success"}, {}]
    tensors = _tensors(rewards, index, row_valid, credit)

    payload = _check(*tensors[:2], index, row_valid, credit, *tensors[2:])

    assert payload["passed"] is True and payload["problems"] == []
    assert payload["rows"] == 6 and payload["groups"] == 2
    assert payload["parity_groups"] == 2 and payload["parity_rows"] == 6
    assert payload["credit_tokens_expected"] == 4
    assert payload["lambda0_equals_pinned_drgrpo"] is True
    assert payload["lambda1_equals_frozen_additions"] is True
    assert payload["max_abs_lambda0_delta"] == 0.0 and payload["max_abs_lambda1_delta"] == 0.0
    assert payload["lambda_process"] == 1.0


def test_frozen_additions_are_phi_minus_gbar_at_the_mapped_positions() -> None:
    index = ["g", "g"]
    rewards = [1.0, 0.0]
    credit = [{2: "success"}, {1: "d2"}]
    token_rewards, mask, advantages, returns, row_keep = _tensors(rewards, index, [1, 1], credit)

    g_bar = 0.5
    assert advantages[0, WIDTH - 1].item() == pytest.approx(0.5)
    assert advantages[0, 2].item() == pytest.approx(0.5 + (PHI["success"] - g_bar))
    assert advantages[1, 1].item() == pytest.approx(-0.5 + (PHI["d2"] - g_bar))

    payload = _check(token_rewards, mask, index, [1, 1], credit, advantages, returns, row_keep)
    assert payload["credit_tokens_expected"] == 2


def test_infra_rows_and_skipped_groups_are_checked_for_exact_zeroing() -> None:
    """λ = 0 parity is claimed per infra-free group; censored slots must be exactly zero."""

    index = ["g"] * 4 + ["h"] * 2
    rewards = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    row_valid = [1, 1, 0, 0, 1, 0]
    credit = [{2: "success"}, {1: "d1"}, {}, {}, {0: "success"}, {}]
    tensors = _tensors(rewards, index, row_valid, credit)

    payload = _check(*tensors[:2], index, row_valid, credit, *tensors[2:])

    assert payload["passed"] is True
    assert payload["parity_groups"] == 1 and payload["parity_rows"] == 2
    assert payload["credit_tokens_expected"] == 2  # the skipped group contributes nothing


def test_lambda_zero_divergence_is_caught(monkeypatch) -> None:
    """If the shared stack ever drifts from the pinned DrGRPO path the check must fail loudly."""

    index = ["g", "g"]
    rewards = [1.0, 0.0]
    credit = [{2: "success"}, {}]
    token_rewards, mask, advantages, returns, row_keep = _tensors(rewards, index, [1, 1], credit)

    real = SC.compute_process_advantage

    def drifted(*args, **kwargs):
        out_advantages, out_returns, out_keep, out_stats = real(*args, **kwargs)
        out_advantages[0, 0] += 0.25
        return out_advantages, out_returns, out_keep, out_stats

    monkeypatch.setattr(SC, "compute_process_advantage", drifted)
    with pytest.raises(RuntimeError, match="lambda=0") as excinfo:
        _check(token_rewards, mask, index, [1, 1], credit, advantages, returns, row_keep)
    assert "'passed': False" in str(excinfo.value)


def test_nonzeroed_infra_and_skipped_rows_are_caught(monkeypatch) -> None:
    index = ["g"] * 4
    rewards = [1.0, 0.0, 0.0, 0.0]
    row_valid = [1, 0, 0, 0]  # fewer than two valid rows: the whole group is skipped
    credit = [{}, {}, {}, {}]
    token_rewards, mask, advantages, returns, row_keep = _tensors(rewards, index, row_valid, credit)

    real = SC.compute_process_advantage

    def drifted(*args, **kwargs):
        out_advantages, out_returns, out_keep, out_stats = real(*args, **kwargs)
        out_advantages[2].fill_(0.1)
        out_keep[3] = 1.0
        return out_advantages, out_returns, out_keep, out_stats

    monkeypatch.setattr(SC, "compute_process_advantage", drifted)
    with pytest.raises(RuntimeError, match="is not zeroed"):
        _check(token_rewards, mask, index, row_valid, credit, advantages, returns, row_keep)


def test_returns_must_be_bit_identical_to_advantages() -> None:
    index = ["g", "g"]
    credit = [{2: "success"}, {}]
    token_rewards, mask, advantages, returns, row_keep = _tensors([1.0, 0.0], index, [1, 1], credit)

    returns = returns.clone()
    returns[0, WIDTH - 1] += 1e-3
    with pytest.raises(RuntimeError, match="returns differ from advantages"):
        _check(token_rewards, mask, index, [1, 1], credit, advantages, returns, row_keep)


def test_credit_outside_the_mask_or_the_width_is_caught() -> None:
    index = ["g", "g"]
    token_rewards, mask, advantages, returns, row_keep = _tensors(
        [1.0, 0.0], index, [1, 1], [{}, {}]
    )
    mask = mask.clone()
    mask[0, 3] = 0.0  # position 3 exists but is not a response token

    for credit, match in (
        ([{3: "success"}, {}], "outside the response mask"),
        ([{99: "d1"}, {}], "out of range"),
    ):
        with pytest.raises(RuntimeError, match=match):
            _check(token_rewards, mask, index, [1, 1], credit, advantages, returns, row_keep)


def test_a_tampered_treatment_tensor_is_caught() -> None:
    """Dropping one frozen addition must fail the check even with a clean λ = 0 path."""

    index = ["g", "g"]
    credit = [{2: "success"}, {1: "d2"}]
    token_rewards, mask, advantages, returns, row_keep = _tensors([1.0, 0.0], index, [1, 1], credit)

    advantages = advantages.clone()
    advantages[1, 1] -= PHI["d2"] - 0.5
    with pytest.raises(RuntimeError, match="frozen additions"):
        _check(token_rewards, mask, index, [1, 1], credit, advantages, returns, row_keep)
