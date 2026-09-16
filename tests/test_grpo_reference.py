"""Reference tests for the ported GRPO objective and the cached-batch loader.

Hand-computed expectations only; the synthetic fixture lives in
``tests/fixtures/grpo_batch`` and never contains real rollouts.
"""

import math
import shutil
from pathlib import Path

import pytest
import torch

from tinylean_rl.rl.batch import build_training_batch, load_cached_rollout_batch
from tinylean_rl.rl.grpo import (
    PINNED_VERL_COMMIT,
    VALID_LOSS_AGG_MODES,
    agg_loss,
    compute_grpo_outcome_advantage,
    compute_policy_loss_vanilla,
    masked_mean,
    masked_sum,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "grpo_batch"


def make_outcome_batch(rewards, response_mask=None, group_ids=None):
    """Build (token_level_rewards, response_mask, index) with rewards on the last token."""

    bsz = len(rewards)
    mask = torch.ones(bsz, 3) if response_mask is None else response_mask
    token_level_rewards = torch.zeros(bsz, 3)
    for i, reward in enumerate(rewards):
        token_level_rewards[i, -1] = reward
    index = list(range(bsz)) if group_ids is None else group_ids
    return token_level_rewards, mask, index


def test_pinned_commit_constant():
    assert PINNED_VERL_COMMIT == "e16b605e8186614c685875c9b57eb19e841b521a"


def test_masked_sum_and_mean():
    values = torch.tensor([[1.0, 2.0, 3.0]])
    mask = torch.tensor([[1.0, 0.0, 1.0]])
    assert masked_sum(values, mask).item() == pytest.approx(4.0)
    assert masked_mean(values, mask).item() == pytest.approx(2.0)


def test_advantage_hand_computed_mean_only():
    # rewards [0, 1, 0, 1] in two groups -> mean 0.5 -> advantages -0.5 / +0.5
    token_level_rewards, response_mask, index = make_outcome_batch(
        [0.0, 1.0, 0.0, 1.0], group_ids=["a", "a", "b", "b"]
    )
    advantages, returns = compute_grpo_outcome_advantage(
        token_level_rewards, response_mask, index, norm_adv_by_std_in_grpo=False
    )
    expected = torch.tensor([-0.5, 0.5, -0.5, 0.5])
    for row in range(4):
        assert torch.allclose(advantages[row], expected[row].expand(3))
    assert torch.allclose(advantages, returns)


def test_advantage_identical_group_is_zero():
    token_level_rewards, response_mask, index = make_outcome_batch(
        [1.0, 1.0, 0.0, 0.0], group_ids=["a", "a", "b", "b"]
    )
    advantages, _ = compute_grpo_outcome_advantage(
        token_level_rewards, response_mask, index, norm_adv_by_std_in_grpo=False
    )
    assert torch.allclose(advantages, torch.zeros_like(advantages))


def test_advantage_single_member_group_uses_zero_mean():
    token_level_rewards, response_mask, index = make_outcome_batch([1.0], group_ids=["solo"])
    advantages, _ = compute_grpo_outcome_advantage(
        token_level_rewards, response_mask, index, norm_adv_by_std_in_grpo=False
    )
    assert torch.allclose(advantages, torch.ones_like(advantages))


def test_advantage_respects_response_mask():
    response_mask = torch.tensor(
        [[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]]
    )
    token_level_rewards, _, index = make_outcome_batch(
        [0.0, 1.0, 0.0, 1.0], response_mask=response_mask, group_ids=["a", "a", "b", "b"]
    )
    advantages, _ = compute_grpo_outcome_advantage(
        token_level_rewards, response_mask, index, norm_adv_by_std_in_grpo=False
    )
    expected = torch.tensor([-0.5, 0.5, -0.5, 0.5]).unsqueeze(-1) * response_mask
    assert torch.allclose(advantages, expected)


def test_advantage_std_normalisation_switch():
    token_level_rewards, response_mask, index = make_outcome_batch(
        [0.0, 1.0, 0.0, 1.0], group_ids=["a", "a", "b", "b"]
    )
    advantages, _ = compute_grpo_outcome_advantage(
        token_level_rewards, response_mask, index, norm_adv_by_std_in_grpo=True
    )
    # torch.std is unbiased by default: std([0, 1]) = sqrt(0.5)
    expected = 0.5 / (math.sqrt(0.5) + 1e-6)
    assert advantages[0, 0].item() == pytest.approx(-expected, rel=1e-5)
    assert advantages[1, 0].item() == pytest.approx(expected, rel=1e-5)


def test_agg_loss_modes_hand_computed():
    loss_mat = torch.tensor([[1.0, 2.0, 0.0], [4.0, 0.0, 6.0]])
    loss_mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    assert agg_loss(loss_mat, loss_mask, "token-mean").item() == pytest.approx(13 / 4)
    assert agg_loss(loss_mat, loss_mask, "seq-mean-token-sum").item() == pytest.approx(6.5)
    assert agg_loss(loss_mat, loss_mask, "seq-mean-token-mean").item() == pytest.approx(3.25)
    assert agg_loss(loss_mat, loss_mask, "seq-mean-token-sum-norm").item() == pytest.approx(13 / 3)
    with pytest.raises(ValueError):
        agg_loss(loss_mat, loss_mask, "no-such-mode")
    assert "seq-mean-token-sum-norm" in VALID_LOSS_AGG_MODES


def test_agg_loss_seq_norm_ignores_masked_out_values():
    loss_mat = torch.tensor([[5.0, 5.0], [1.0, 2.0]])
    loss_mask = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    baseline = agg_loss(loss_mat, loss_mask, "seq-mean-token-sum-norm")
    perturbed = loss_mat.clone()
    perturbed[0, 1] = 500.0
    assert agg_loss(perturbed, loss_mask, "seq-mean-token-sum-norm").item() == pytest.approx(
        baseline.item()
    )
    assert baseline.item() == pytest.approx(8 / 2)


def test_policy_loss_ratio_one_equals_negative_mean_advantage():
    old_log_prob = torch.zeros(1, 4)
    log_prob = torch.zeros(1, 4)
    advantages = torch.full((1, 4), 0.5)
    response_mask = torch.ones(1, 4)
    loss, clipfrac, ppo_kl, clipfrac_lower = compute_policy_loss_vanilla(
        old_log_prob,
        log_prob,
        advantages,
        response_mask,
        clip_ratio_low=0.2,
        clip_ratio_high=0.3,
        loss_agg_mode="seq-mean-token-sum-norm",
    )
    assert loss.item() == pytest.approx(-0.5)
    assert clipfrac.item() == pytest.approx(0.0)
    assert ppo_kl.item() == pytest.approx(0.0)
    assert clipfrac_lower.item() == pytest.approx(0.0)


def test_policy_loss_mixed_sign_advantages_cancel_at_ratio_one():
    old_log_prob = torch.zeros(1, 2)
    log_prob = torch.zeros(1, 2)
    advantages = torch.tensor([[1.0, -1.0]])
    response_mask = torch.ones(1, 2)
    loss, clipfrac, _, _ = compute_policy_loss_vanilla(
        old_log_prob, log_prob, advantages, response_mask, loss_agg_mode="seq-mean-token-sum-norm"
    )
    assert loss.item() == pytest.approx(0.0)
    assert clipfrac.item() == pytest.approx(0.0)


def test_policy_loss_upper_clip_boundary():
    old_log_prob = torch.zeros(1, 1)
    log_prob = torch.log(torch.tensor([[1.5]]))
    advantages = torch.tensor([[1.0]])
    response_mask = torch.ones(1, 1)
    loss, clipfrac, ppo_kl, _ = compute_policy_loss_vanilla(
        old_log_prob,
        log_prob,
        advantages,
        response_mask,
        clip_ratio_low=0.2,
        clip_ratio_high=0.3,
        loss_agg_mode="seq-mean-token-sum-norm",
    )
    # ratio 1.5 is clamped to 1.3 -> loss = -1 * 1.3, clip fraction 1
    assert loss.item() == pytest.approx(-1.3)
    assert clipfrac.item() == pytest.approx(1.0)
    assert ppo_kl.item() == pytest.approx(-math.log(1.5), rel=1e-6)


def test_policy_loss_dual_clip_lower_bound():
    old_log_prob = torch.zeros(1, 1)
    log_prob = torch.log(torch.tensor([[4.0]]))
    advantages = torch.tensor([[-1.0]])
    response_mask = torch.ones(1, 1)
    loss, clipfrac, _, clipfrac_lower = compute_policy_loss_vanilla(
        old_log_prob,
        log_prob,
        advantages,
        response_mask,
        clip_ratio_low=0.2,
        clip_ratio_high=0.3,
        clip_ratio_c=3.0,
        loss_agg_mode="seq-mean-token-sum-norm",
    )
    # A < 0 and ratio 4 > clip_ratio_c 3 -> loss capped at 3, dual-clip active
    assert loss.item() == pytest.approx(3.0)
    assert clipfrac.item() == pytest.approx(0.0)
    assert clipfrac_lower.item() == pytest.approx(1.0)


def test_policy_loss_gradients_are_finite():
    log_prob = torch.tensor([[0.3]], requires_grad=True)
    old_log_prob = torch.zeros(1, 1)
    advantages = torch.tensor([[1.0]])
    response_mask = torch.ones(1, 1)
    loss, _, _, _ = compute_policy_loss_vanilla(
        old_log_prob, log_prob, advantages, response_mask, loss_agg_mode="seq-mean-token-sum-norm"
    )
    loss.backward()
    assert log_prob.grad is not None
    assert torch.isfinite(log_prob.grad).all()


def test_load_cached_rollout_batch_fixture():
    prompts, rollouts, rewards, metadata = load_cached_rollout_batch(FIXTURE_DIR)
    assert len(prompts) == 2
    assert len(rollouts) == 4
    assert len(rewards) == 4
    assert [row["candidate_id"] for row in rollouts] == [
        row["candidate_id"] for row in rewards
    ]
    assert metadata["samples_per_theorem"] == 2


def test_load_cached_rollout_batch_rejects_misaligned(tmp_path):
    broken = tmp_path / "broken"
    shutil.copytree(FIXTURE_DIR, broken)
    rewards_lines = (broken / "rewards.jsonl").read_text(encoding="utf-8").splitlines()
    (broken / "rewards.jsonl").write_text("\n".join(rewards_lines[:3]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_cached_rollout_batch(broken)


def test_build_training_batch_from_fixture():
    prompts, rollouts, rewards, _ = load_cached_rollout_batch(FIXTURE_DIR)
    batch = build_training_batch(prompts, rollouts, rewards, pad_token_id=0)
    input_ids = batch["input_ids"]
    assert input_ids.shape == (4, 9)
    assert input_ids[0].tolist() == [10, 11, 12, 13, 100, 101, 0, 0, 0]
    assert input_ids[3].tolist() == [20, 21, 22, 23, 24, 201, 202, 203, 204]
    assert batch["attention_mask"][0].tolist() == [1, 1, 1, 1, 1, 1, 0, 0, 0]
    assert batch["response_mask"][0].tolist() == [0, 0, 0, 0, 1, 1, 0, 0, 0]
    assert batch["response_lengths"] == [2, 3, 1, 4]
    assert batch["uids"] == [
        "fixture-theorem-0",
        "fixture-theorem-0",
        "fixture-theorem-1",
        "fixture-theorem-1",
    ]
    totals = batch["token_level_rewards"].sum(dim=-1)
    assert totals.tolist() == pytest.approx([1.0, 0.0, 0.0, 1.0])


def test_build_training_batch_truncates_response():
    prompts, rollouts, rewards, _ = load_cached_rollout_batch(FIXTURE_DIR)
    batch = build_training_batch(prompts, rollouts, rewards, pad_token_id=0, max_response_length=2)
    assert batch["response_lengths"] == [2, 2, 1, 2]
    assert batch["input_ids"][3, 5:7].tolist() == [201, 202]
    # The reward must land on the last surviving response token.
    assert batch["token_level_rewards"][3, 6].item() == pytest.approx(1.0)
    assert batch["token_level_rewards"].sum(dim=-1).tolist() == pytest.approx([1.0, 0.0, 0.0, 1.0])


def test_advantage_on_fixture_batch_centres_per_group():
    prompts, rollouts, rewards, _ = load_cached_rollout_batch(FIXTURE_DIR)
    batch = build_training_batch(prompts, rollouts, rewards, pad_token_id=0)
    advantages, _ = compute_grpo_outcome_advantage(
        batch["token_level_rewards"],
        batch["response_mask"],
        batch["uids"],
        norm_adv_by_std_in_grpo=False,
    )
    # rewards [1, 0, 0, 1]: both groups have mean 0.5 -> +-0.5 per valid token
    first_valid = torch.tensor([4, 4, 5, 5])
    expected = torch.tensor([0.5, -0.5, -0.5, 0.5])
    row_values = torch.tensor([advantages[row, first_valid[row]].item() for row in range(4)])
    assert row_values.tolist() == pytest.approx(expected.tolist())
    # mean-only centring makes each group's mean advantage zero
    for rows in ([0, 1], [2, 3]):
        assert row_values[rows].mean().item() == pytest.approx(0.0)
