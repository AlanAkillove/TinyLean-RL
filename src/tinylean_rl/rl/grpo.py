"""GRPO / DrGRPO objective ported line-by-line from the pinned Kimina-Prover-RL tree.

Source: ``third_party/kimina-prover-rl`` at commit
``e16b605e8186614c685875c9b57eb19e841b521a`` (2025-08-14, "Add recipe (#44)"):

- ``compute_grpo_outcome_advantage`` -> ``verl/trainer/ppo/core_algos.py`` L261-324
- ``agg_loss``                       -> ``verl/trainer/ppo/core_algos.py`` L703-736
- ``compute_policy_loss_vanilla``    -> ``verl/trainer/ppo/core_algos.py`` L816-889
- ``masked_sum`` / ``masked_mean``   -> ``verl/utils/torch_functional.py`` L163-185

Numerical outputs must match the pinned source exactly; any deviation is a bug.
The P3-A configuration follows ``recipe/kimina_prover_rl/kimina_prover_0.6B.sh``:
``norm_adv_by_std_in_grpo=False`` (DrGRPO mean-only centring), asymmetric clipping
``clip_ratio_low=0.2`` / ``clip_ratio_high=0.3``, dual-clip ``clip_ratio_c=3.0``,
``loss_agg_mode="seq-mean-token-sum-norm"``, and no KL / entropy terms.
This module deliberately has no verl/trl dependency.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Sequence

import torch

PINNED_VERL_COMMIT = "e16b605e8186614c685875c9b57eb19e841b521a"

VALID_LOSS_AGG_MODES = (
    "token-mean",
    "seq-mean-token-sum",
    "seq-mean-token-mean",
    "seq-mean-token-sum-norm",
)


def masked_sum(values: torch.Tensor, mask: torch.Tensor, axis=None) -> torch.Tensor:
    """Sum ``values`` over the true entries of ``mask`` (torch_functional.py L163)."""

    valid_values = torch.where(mask.bool(), values, 0.0)
    return (valid_values * mask).sum(axis=axis)


def masked_mean(values: torch.Tensor, mask: torch.Tensor, axis=None) -> torch.Tensor:
    """Mean of ``values`` over the true entries of ``mask`` (torch_functional.py L171)."""

    totals = masked_sum(values, mask, axis)
    return totals / (mask.sum(axis=axis) + 1e-8)


def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: Sequence[Hashable],
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GRPO/DrGRPO outcome advantage (core_algos.py L261).

    ``token_level_rewards`` and ``response_mask`` have shape
    ``(bs, response_length)``; ``index`` labels the rows that share one prompt.
    With ``norm_adv_by_std_in_grpo=False`` this is the DrGRPO mean-only centring
    used by the Kimina 0.6B recipe (advantage is not divided by the group std).
    Returns ``(advantages, returns)`` which are identical for outcome supervision.
    """

    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str) -> torch.Tensor:
    """Aggregate per-token losses into a scalar (core_algos.py L703).

    ``seq-mean-token-sum-norm`` divides by ``loss_mask.shape[-1]`` (the padded
    response width), which is the DrGRPO-style normaliser used in the recipe.
    """

    if loss_agg_mode == "token-mean":
        loss = masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)  # token-mean
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.sum(seq_losses) / loss_mask.shape[-1]
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


def compute_policy_loss_vanilla(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_ratio: float = 0.2,
    clip_ratio_low: float | None = None,
    clip_ratio_high: float | None = None,
    clip_ratio_c: float = 3.0,
    loss_agg_mode: str = "token-mean",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Clipped policy objective with dual-clip (core_algos.py L816).

    The recipe passes ``clip_ratio_low=0.2`` / ``clip_ratio_high=0.3`` /
    ``clip_ratio_c=3.0`` / ``loss_agg_mode="seq-mean-token-sum-norm"``.
    Returns ``(pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower)``.
    """

    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if clip_ratio_low is None:
        clip_ratio_low = clip_ratio
    if clip_ratio_high is None:
        clip_ratio_high = clip_ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    pg_clipfrac = masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower
