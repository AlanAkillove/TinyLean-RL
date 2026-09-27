"""R001 S2 self-check (owner directive §13): the online advantage tensors vs the frozen semantics.

§13 requires that, before any treatment result is read, the *shared execution stack* proves two
things on the same frozen batch:

* **λ = 0 parity** - with the process term switched off, the online path reproduces the
  V1-equivalent DrGRPO outcome objective *bit for bit* on every infra-free group. The reference
  here is ``tinylean_rl.rl.grpo.compute_grpo_outcome_advantage`` (the frozen re-derivation of
  ``core_algos.py`` L261, pinned by the V5 recon tests) evaluated per group, which is exactly the
  control semantics of §7.2. Groups that contain an infrastructure-censored candidate are excluded
  from this comparison *by design*: there the frozen intervention deliberately differs from V1
  (censored rows leave ``g_bar`` and their policy loss is masked, §4), so bit equality is not the
  claim.
* **λ = 1 construction** - the actually-written advantages differ from the λ = 0 tensors exactly by
  the frozen per-token additions ``λ * (φ - g_bar)`` at the credited positions, with ``g_bar``
  recomputed here independently from the outcome scores (not read back from the producing code),
  and only where the response mask is 1.

The check is a pure function over tensors so fixtures can drive it directly; the training path
calls it (gated by the ``r001_selfcheck`` config key) and both *raises* on any failure - a
treatment step must not run on an unverified advantage path - and records the payload in
``data.meta_info["r001_selfcheck"]``.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import Any

import torch

from tinylean_rl.rl.grpo import compute_grpo_outcome_advantage
from tinylean_rl.rl.process_credit import PHI, compute_process_advantage

SELFCHECK_CONFIG_KEY = "r001_selfcheck"


def _groups(index: Sequence[Hashable]) -> dict[Hashable, list[int]]:
    groups: dict[Hashable, list[int]] = {}
    for row, key in enumerate(index):
        groups.setdefault(key, []).append(row)
    return groups


def _max_abs_delta(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual - expected).abs().max().item()) if actual.numel() else 0.0


def check_advantage_paths(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: Sequence[Hashable],
    row_valid: Sequence[int],
    credit_positions: Sequence[Mapping[int, str]],
    advantages: torch.Tensor,
    returns: torch.Tensor,
    row_keep: torch.Tensor,
    *,
    lambda_process: float,
) -> dict[str, Any]:
    """Verify the written advantages against the frozen semantics; raise on any failure."""

    bs, response_length = token_level_rewards.shape
    if row_keep.shape != (bs,):
        raise RuntimeError(f"row_keep shape {tuple(row_keep.shape)} != ({bs},)")
    groups = _groups(index)
    zero, _, keep_zero, _ = compute_process_advantage(
        token_level_rewards,
        response_mask,
        index,
        list(row_valid),
        credit_positions,
        lambda_process=0.0,
    )

    problems: list[str] = []
    scores = token_level_rewards.sum(dim=-1)
    group_mean: dict[Hashable, float] = {}
    parity_groups = 0
    parity_rows = 0
    max_zero_delta = 0.0
    for key, rows in groups.items():
        valid_rows = [row for row in rows if int(row_valid[row]) > 0]
        infra_rows = [row for row in rows if int(row_valid[row]) <= 0]
        if len(valid_rows) < 2:
            # GROUP_SKIPPED_INFRA semantics: nothing may be written for any row of this group.
            for row in rows:
                if float(zero[row].abs().max()) != 0.0 or float(keep_zero[row]) != 0.0:
                    problems.append(f"skipped group {key!r} row {row} is not zeroed")
            continue
        group_mean[key] = float(torch.mean(torch.stack([scores[row] for row in valid_rows])))
        for row in infra_rows:
            if float(zero[row].abs().max()) != 0.0 or float(keep_zero[row]) != 0.0:
                problems.append(f"infra row {row} (group {key!r}) is not zeroed")
        reference, _ = compute_grpo_outcome_advantage(
            token_level_rewards[valid_rows],
            response_mask[valid_rows],
            [key] * len(valid_rows),
            norm_adv_by_std_in_grpo=False,
        )
        parity_groups += 1
        parity_rows += len(valid_rows)
        if not torch.equal(zero[valid_rows], reference):
            delta = _max_abs_delta(zero[valid_rows], reference)
            max_zero_delta = max(max_zero_delta, delta)
            problems.append(f"lambda=0 group {key!r} differs from pinned DrGRPO by {delta}")
        if keep_zero[valid_rows].tolist() != [1.0] * len(valid_rows):
            problems.append(f"lambda=0 group {key!r} row_keep is not all ones")

    if torch.equal(returns, advantages) is False:
        problems.append(f"returns differ from advantages by {_max_abs_delta(returns, advantages)}")

    expected = zero.clone()
    credit_expected = 0
    for row in range(bs):
        mean = group_mean.get(index[row])
        if mean is None:
            continue
        for position, label in dict(credit_positions[row] or {}).items():
            position = int(position)
            if position < 0 or position >= response_length:
                problems.append(f"row {row} credit position {position} is out of range")
                continue
            if float(response_mask[row, position]) == 0.0:
                problems.append(f"row {row} credit position {position} is outside the response mask")
                continue
            expected[row, position] += lambda_process * (PHI[label] - mean)
            credit_expected += 1

    max_one_delta = _max_abs_delta(advantages, expected)
    if max_one_delta != 0.0:
        problems.append(f"lambda={lambda_process} tensors differ from the frozen additions by {max_one_delta}")

    payload: dict[str, Any] = {
        "rows": bs,
        "groups": len(groups),
        "parity_groups": parity_groups,
        "parity_rows": parity_rows,
        "lambda_process": float(lambda_process),
        "lambda0_equals_pinned_drgrpo": not any("lambda=0" in problem for problem in problems),
        "lambda1_equals_frozen_additions": max_one_delta == 0.0,
        "max_abs_lambda0_delta": max_zero_delta,
        "max_abs_lambda1_delta": max_one_delta,
        "credit_tokens_expected": credit_expected,
        "problems": problems,
        "passed": not problems,
    }
    if problems:
        raise RuntimeError(f"R001 self-check failed: {payload}")
    return payload


__all__ = ["SELFCHECK_CONFIG_KEY", "check_advantage_paths"]
