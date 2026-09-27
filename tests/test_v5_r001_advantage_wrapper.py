"""Fixtures for the R001 advantage wrapper (draft §7.2, owner directive §13 S2 parity).

Pure CPU: a minimal DataProto stand-in (``batch`` dict + ``non_tensor_batch`` + ``meta_info``)
exercises the exact call shape the pinned trainer uses at ``ray_trainer.py:1303``:
all-keyword dispatch with ``config=self.config.algorithm``. The λ=0 parity fixture compares
bit-for-bit against ``compute_grpo_outcome_advantage`` under the same frozen batch.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tinylean_rl.rl.grpo import compute_grpo_outcome_advantage
from tinylean_rl.rl.process_credit import LAMBDA_PRIMARY
from tinylean_rl.rl.r001_advantage import (
    R001_ESTIMATOR,
    build_wrapper,
    compute_advantage_r001,
    install,
    lambda_from_config,
    parse_credit_payload,
    register_stub,
    sum_online_counters,
)

RESPONSE_LENGTH = 8

#: The pinned trainer passes ``algorithm.norm_adv_by_std_in_grpo`` (False in every frozen
#: R001/V1 launch config); the R001 path refuses True rather than silently ignoring it.
NORM_ADV = {"norm_adv_by_std_in_grpo": False}


class FakeData:
    """Just enough of ``DataProto`` for the wrapper: batch dict, non-tensor dict, meta_info."""

    def __init__(
        self,
        attention_mask: torch.Tensor,
        non_tensor_batch: dict,
        meta_info: dict | None = None,
    ) -> None:
        self.batch = {"response_mask": attention_mask}
        self.non_tensor_batch = non_tensor_batch
        self.meta_info = {} if meta_info is None else meta_info


def _batch(rewards: list[float], lengths: list[int] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-padded token_level_rewards + response_mask; the outcome sits on token ``length-1``."""

    lengths = lengths or [RESPONSE_LENGTH] * len(rewards)
    token_rewards = torch.zeros(len(rewards), RESPONSE_LENGTH)
    mask = torch.zeros(len(rewards), RESPONSE_LENGTH)
    for row, (reward, length) in enumerate(zip(rewards, lengths)):
        for position in range(length):
            mask[row, position] = 1.0
        token_rewards[row, length - 1] = reward
    return token_rewards, mask


def _data(
    rewards: list[float],
    index: list[str],
    payload: list | None = None,
    lengths: list[int] | None = None,
) -> tuple[FakeData, torch.Tensor, torch.Tensor]:
    token_rewards, mask = _batch(rewards, lengths)
    non_tensor: dict = {"uid": np.array(index, dtype=object)}
    if payload is not None:
        non_tensor["process_credit"] = np.array(payload, dtype=object)
    data = FakeData(mask.clone(), non_tensor)
    return data, token_rewards, mask


def test_other_estimator_passes_through_untouched() -> None:
    data, token_rewards, _ = _data([1.0, 0.0], ["g", "g"])
    data.batch["token_level_rewards"] = token_rewards
    seen: list[dict] = []

    def original(data, adv_estimator, gamma=None, lam=None, num_repeat=None,
                 norm_adv_by_std_in_grpo=True, config=None):
        seen.append(
            {
                "adv_estimator": adv_estimator,
                "gamma": gamma,
                "lam": lam,
                "num_repeat": num_repeat,
                "norm_adv_by_std_in_grpo": norm_adv_by_std_in_grpo,
                "config": config,
            }
        )
        return data

    result = compute_advantage_r001(
        data,
        "grpo",
        gamma=1.0,
        lam=1.0,
        num_repeat=4,
        norm_adv_by_std_in_grpo=False,
        config={"algorithm": "grpo"},
        original=original,
    )
    assert result is data
    assert seen == [
        {
            "adv_estimator": "grpo",
            "gamma": 1.0,
            "lam": 1.0,
            "num_repeat": 4,
            "norm_adv_by_std_in_grpo": False,
            "config": {"algorithm": "grpo"},
        }
    ]
    assert "advantages" not in data.batch  # nothing invented on the pass-through path
    assert "r001_stats" not in data.meta_info


def _r001(data, config, **kwargs):
    return compute_advantage_r001(data, R001_ESTIMATOR, config=config, **NORM_ADV, **kwargs)


def test_missing_payload_degenerates_to_v1_equivalent_outcome_objective() -> None:
    rewards = [1.0, 1.0, 0.0, 0.0]
    index = ["g"] * 4
    data, token_rewards, mask = _data(rewards, index, payload=None)
    data.batch["token_level_rewards"] = token_rewards
    reference, _ = compute_grpo_outcome_advantage(
        token_rewards.clone(), mask.clone(), index, norm_adv_by_std_in_grpo=False
    )

    _r001(data, {"r001_lambda": 0.0})

    assert torch.equal(data.batch["advantages"], reference)
    assert torch.equal(data.batch["returns"], reference)
    assert torch.equal(data.batch["response_mask"], mask)  # no infra rows -> mask untouched
    assert data.meta_info["r001_stats"]["infra_censored"] == 0
    assert data.meta_info["r001_stats"]["credit_tokens"] == 0


def test_lambda_zero_with_credit_present_is_still_v1_equivalent() -> None:
    """S2 parity under the frozen batch: credit payload present, TREATMENT(λ=0) == CONTROL."""

    rewards = [1.0, 0.0]
    index = ["g", "g"]
    payload = [{"valid": 1, "positions": {2: "d2"}}, {"valid": 1, "positions": {1: "d1"}}]
    data, token_rewards, mask = _data(rewards, index, payload)
    data.batch["token_level_rewards"] = token_rewards
    reference, _ = compute_grpo_outcome_advantage(
        token_rewards.clone(), mask.clone(), index, norm_adv_by_std_in_grpo=False
    )

    compute_advantage_r001(data, R001_ESTIMATOR, config={"r001_lambda": 0.0}, **NORM_ADV)

    assert torch.equal(data.batch["advantages"], reference)


def test_lambda_one_applies_phi_minus_gbar_at_mapped_positions() -> None:
    rewards = [1.0, 0.0]
    index = ["g", "g"]
    payload = [{"valid": 1, "positions": {2: "success", 4: "d1"}}, {"valid": 1, "positions": {1: "d2"}}]
    data, token_rewards, _ = _data(rewards, index, payload)
    data.batch["token_level_rewards"] = token_rewards

    compute_advantage_r001(data, R001_ESTIMATOR, config={"r001_lambda": 1.0}, **NORM_ADV)

    advantages = data.batch["advantages"]
    assert advantages[0, 7].item() == pytest.approx(0.5)
    assert advantages[0, 2].item() == pytest.approx(0.5 + (1.0 - 0.5))
    assert advantages[0, 4].item() == pytest.approx(0.5 + (-0.05 - 0.5))
    assert advantages[1, 1].item() == pytest.approx(-0.5 + (-0.10 - 0.5))
    assert data.meta_info["r001_stats"]["credit_tokens"] == 3


def test_infra_row_is_censored_in_both_arms() -> None:
    rewards = [1.0, 0.0, 0.0, 0.0]
    index = ["g"] * 4
    payload = [
        {"valid": 1, "positions": {2: "success"}},
        {"valid": 1, "positions": {}},
        {"valid": 0, "status": "infra", "positions": {}},
        {"valid": 0, "status": "infra", "positions": {}},
    ]
    for arm in (0.0, 1.0):
        data, token_rewards, mask = _data(rewards, index, payload)
        data.batch["token_level_rewards"] = token_rewards
        compute_advantage_r001(data, R001_ESTIMATOR, config={"r001_lambda": arm}, **NORM_ADV)

        # ḡ = 0.5 over the two VALID rows only, identical in both arms
        assert data.batch["advantages"][0, 7].item() == pytest.approx(0.5)
        assert data.batch["advantages"][1, 7].item() == pytest.approx(-0.5)
        assert data.batch["advantages"][2].abs().sum().item() == 0.0
        assert data.batch["advantages"][3].abs().sum().item() == 0.0
        # response policy loss for censored rows is masked to zero
        assert data.batch["response_mask"][2].abs().sum().item() == 0.0
        assert data.batch["response_mask"][3].abs().sum().item() == 0.0
        assert torch.equal(data.batch["response_mask"][:2], mask[:2])
        stats = data.meta_info["r001_stats"]
        assert stats["infra_censored"] == 2
        assert stats["group_skipped_infra"] == 0
        assert stats["credit_tokens"] == (1 if arm == 1.0 else 0)


def test_group_with_fewer_than_two_valid_is_skipped_entirely() -> None:
    rewards = [1.0, 0.0, 0.0]
    index = ["g"] * 3
    payload = [
        {"valid": 1, "positions": {}},
        {"valid": 0, "positions": {}},
        {"valid": 0, "positions": {}},
    ]
    data, token_rewards, _ = _data(rewards, index, payload)
    data.batch["token_level_rewards"] = token_rewards

    compute_advantage_r001(data, R001_ESTIMATOR, config={"r001_lambda": 1.0}, **NORM_ADV)

    assert data.batch["advantages"].abs().sum().item() == 0.0
    assert data.batch["response_mask"].abs().sum().item() == 0.0
    assert data.meta_info["r001_stats"]["group_skipped_infra"] == 1


def test_parse_credit_payload_shapes() -> None:
    row_valid, credit = parse_credit_payload(None, 3)
    assert row_valid == [1, 1, 1]
    assert credit == [{}, {}, {}]

    payload = [
        {"valid": 1, "positions": {2: "d1"}},
        {},
        {"valid": 0, "status": "infra"},
    ]
    row_valid, credit = parse_credit_payload(np.array(payload, dtype=object), 3)
    assert row_valid == [1, 1, 0]
    assert credit == [{2: "d1"}, {}, {}]

    # a short payload never fabricates censoring or credit for the missing rows
    row_valid, credit = parse_credit_payload([{"valid": 0}], 3)
    assert row_valid == [0, 1, 1]
    assert credit == [{}, {}, {}]


def test_lambda_from_config_reads_the_frozen_key() -> None:
    assert lambda_from_config(None) == LAMBDA_PRIMARY == 1.0
    assert lambda_from_config({"r001_lambda": 0.0}) == 0.0
    assert lambda_from_config(types.SimpleNamespace(r001_lambda=0.0)) == 0.0

    class OmegaConfLike:
        def get(self, key, default=None):
            return 0.0 if key == "r001_lambda" else default

    assert lambda_from_config(OmegaConfLike()) == 0.0


def test_install_patches_module_global_and_delegates() -> None:
    calls: list[str] = []

    def original(data, adv_estimator, gamma=None, lam=None, num_repeat=None,
                 norm_adv_by_std_in_grpo=True, config=None):
        calls.append(str(adv_estimator))
        return data

    module = types.SimpleNamespace(compute_advantage=original)
    previous = install(module)
    assert previous is original
    assert module.compute_advantage is not original

    data, token_rewards, _ = _data([1.0, 0.0], ["g", "g"])
    data.batch["token_level_rewards"] = token_rewards
    module.compute_advantage(
        data,
        adv_estimator="grpo",
        gamma=1.0,
        lam=1.0,
        num_repeat=4,
        norm_adv_by_std_in_grpo=False,
        config=None,
    )
    assert calls == ["grpo"]

    module.compute_advantage(data, adv_estimator=R001_ESTIMATOR, config={"r001_lambda": 0.0}, **NORM_ADV)
    assert "advantages" in data.batch
    # a second install must not stack wrappers
    wrapper = module.compute_advantage
    install(module)
    assert module.compute_advantage is wrapper


def test_register_stub_refuses_unwrapped_dispatch() -> None:
    registered: dict = {}

    def register_fn(name):
        def decorator(fn):
            registered[name] = fn
            return fn

        return decorator

    register_stub(register_fn)
    assert R001_ESTIMATOR in registered
    with pytest.raises(RuntimeError, match="R001 wrapper"):
        registered[R001_ESTIMATOR](token_level_rewards=None, response_mask=None, config=None)


def test_build_wrapper_requires_original_for_foreign_estimator() -> None:
    wrapper = build_wrapper(None)
    data, token_rewards, _ = _data([1.0, 0.0], ["g", "g"])
    data.batch["token_level_rewards"] = token_rewards
    with pytest.raises(RuntimeError, match="delegate"):
        wrapper(data, adv_estimator="grpo")


def test_std_normalized_configuration_is_refused() -> None:
    data, token_rewards, _ = _data([1.0, 0.0], ["g", "g"])
    data.batch["token_level_rewards"] = token_rewards
    with pytest.raises(ValueError, match="norm_adv_by_std_in_grpo"):
        compute_advantage_r001(
            data,
            R001_ESTIMATOR,
            norm_adv_by_std_in_grpo=True,
            config={"r001_lambda": 1.0},
        )


def test_selfcheck_hook_verifies_written_tensors() -> None:
    """S2 wiring: with ``r001_selfcheck`` the wrapper validates before anything reads the tensors."""

    rewards = [1.0, 0.0]
    index = ["g", "g"]
    payload = [{"valid": 1, "positions": {2: "success"}}, {"valid": 1, "positions": {1: "d2"}}]

    data, token_rewards, _ = _data(rewards, index, payload)
    data.batch["token_level_rewards"] = token_rewards
    _r001(data, {"r001_lambda": 1.0, "r001_selfcheck": True})
    check = data.meta_info["r001_selfcheck"]
    assert check["passed"] is True
    assert check["parity_groups"] == 1 and check["parity_rows"] == 2
    assert check["lambda0_equals_pinned_drgrpo"] and check["lambda1_equals_frozen_additions"]
    assert check["credit_tokens_expected"] == 2

    # the hook stays off unless configured
    data, token_rewards, _ = _data(rewards, index, payload)
    data.batch["token_level_rewards"] = token_rewards
    _r001(data, {"r001_lambda": 1.0})
    assert "r001_selfcheck" not in data.meta_info


def test_online_mapping_counters_are_summed_into_the_stats() -> None:
    rewards = [1.0, 0.0]
    index = ["g", "g"]
    payload = [{"valid": 1, "positions": {}}, {"valid": 1, "positions": {}}]
    data, token_rewards, _ = _data(rewards, index, payload)
    data.batch["token_level_rewards"] = token_rewards
    data.non_tensor_batch["process_counters"] = np.array(
        [
            {"mappable_tactics": 2, "ambiguous": 1, "conflicts": 0, "no_code": 0},
            {"mappable_tactics": 1, "ambiguous": 0, "conflicts": 1, "no_code": 1},
        ],
        dtype=object,
    )

    _r001(data, {"r001_lambda": 0.0})

    stats = data.meta_info["r001_stats"]
    assert stats["online_rows"] == 2
    assert stats["online_mappable_tactics"] == 3
    assert stats["online_ambiguous"] == 1
    assert stats["online_conflicts"] == 1
    assert stats["online_no_code"] == 1
    assert sum_online_counters(None) == {}
