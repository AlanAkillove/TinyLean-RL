"""R001 advantage wrapper: the single write path into ``batch["advantages"]``.

Both arms of V5-R001 run the *same* registered estimator name (``r001_process``) through the
*wrapper* installed here; the arm difference is the frozen constant ``r001_lambda``
(CONTROL = 0.0, TREATMENT = 1.0). Rationale in ``docs/v5/V5-R001_preregistration_draft.md`` §7.2:
the stock ``grpo`` branch of the pinned trainer takes its group mean over *all* rows and cannot
exclude infrastructure-censored candidates from ``g_bar`` (owner directive §4), so both arms go
through ``src/tinylean_rl/rl/process_credit.py`` instead, whose lambda = 0 path is pinned
bit-for-bit to ``compute_grpo_outcome_advantage`` on non-infra batches.

The wrapper is installed by the R001 training entry (``scripts/v5_r001_train.py``) onto
``verl.trainer.ppo.ray_trainer`` before ``fit()`` runs. Three things happen here and nowhere else:

1. the per-row credit payload produced by the online reward path
   (``data.non_tensor_batch["process_credit"]``) is parsed into ``(row_valid, credit_positions)``;
2. ``compute_process_advantage`` builds the advantages and the ``row_keep`` mask;
3. ``row_keep`` is written back into ``response_mask`` (owner §4.3: a censored candidate's
   response policy loss is masked to zero — it contributes neither gradient nor denominator),
   and the per-step counters are recorded in ``data.meta_info["r001_stats"]`` (the frozen
   :class:`CreditStats` tensor-stage counters plus, when the reward path provided them, the
   per-candidate online mapping counters summed under ``online_*`` keys — owner §14);
4. when the ``r001_selfcheck`` config key is set (S2 on fly122, owner §13), the just-written
   tensors are verified against the frozen semantics
   (``src/tinylean_rl/rl/r001_selfcheck.py``) *before* anything downstream reads them; a failure
   raises, never warns.

Every R001 call also prints one ``R001_STATS {json}`` line so the §14 counter set is visible in
the run log of both arms.

Nothing here invents credit: an empty payload or ``r001_lambda = 0`` degenerates to the
V1-equivalent outcome-only objective, and any other estimator name passes through to the pinned
trainer's own ``compute_advantage`` untouched.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from tinylean_rl.rl.process_credit import (
    LAMBDA_PRIMARY,
    CreditStats,
    compute_process_advantage,
)
from tinylean_rl.rl.r001_selfcheck import SELFCHECK_CONFIG_KEY, check_advantage_paths

R001_ESTIMATOR = "r001_process"
LAMBDA_CONFIG_KEY = "r001_lambda"


def _config_get(config: Any, key: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(config, key, default)


def lambda_from_config(config: Any) -> float:
    """The frozen arm constant; one scalar, read in exactly one place."""

    return float(_config_get(config, LAMBDA_CONFIG_KEY, LAMBDA_PRIMARY))


def parse_credit_payload(payload: Sequence[Mapping[str, Any]] | None, n_rows: int) -> tuple[list[int], list[dict[int, str]]]:
    """``non_tensor_batch["process_credit"]`` -> ``(row_valid, credit_positions)``.

    Each entry is the per-candidate dict the online adapter emits:
    ``{"valid": 1|0, "status": str, "positions": {token_position: label}}``. ``valid = 0`` marks
    an infrastructure-censored candidate (owner §4.3). Missing payload / missing entries mean
    "all rows valid, no process credit", which is exactly the V1-equivalent configuration.
    """

    row_valid = [1] * n_rows
    credit: list[dict[int, str]] = [{} for _ in range(n_rows)]
    if payload is None:
        return row_valid, credit
    for row, entry in enumerate(payload):
        if row >= n_rows or not entry:
            continue
        row_valid[row] = int(entry.get("valid", 1))
        positions = entry.get("positions") or {}
        credit[row] = {int(position): str(label) for position, label in positions.items()}
    return row_valid, credit


def sum_online_counters(rows: Sequence[Mapping[str, Any]] | None) -> dict[str, int]:
    """Sum the per-candidate ``process_counters`` payloads of one batch (owner §14)."""

    if rows is None:
        return {}
    totals: dict[str, int] = {}
    for row in rows:
        for key, value in dict(row).items():
            totals[str(key)] = totals.get(str(key), 0) + int(value)
    return totals


def compute_advantage_r001(
    data: Any,
    adv_estimator: Any,
    gamma: float | None = None,
    lam: float | None = None,
    num_repeat: int | None = None,
    norm_adv_by_std_in_grpo: bool = True,
    config: Any = None,
    original: Any = None,
) -> Any:
    """Wrapper for ``verl.trainer.ppo.ray_trainer.compute_advantage``.

    Any estimator other than ``r001_process`` is forwarded to ``original`` unchanged (the pinned
    trainer's own function), so a non-R001 run cannot be altered by importing this module.
    """

    if str(adv_estimator) != R001_ESTIMATOR:
        if original is None:
            raise RuntimeError("R001 wrapper reached without the pinned compute_advantage to delegate to")
        return original(
            data,
            adv_estimator=adv_estimator,
            gamma=gamma,
            lam=lam,
            num_repeat=num_repeat,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            config=config,
        )

    if norm_adv_by_std_in_grpo:
        raise ValueError(
            "R001 runs must pin algorithm.norm_adv_by_std_in_grpo=False: the frozen advantage is "
            "A_outcome = g - g_bar with no std normalization (owner directive §1). Refusing to "
            "silently ignore a std-normalized configuration."
        )

    token_level_rewards = data.batch["token_level_rewards"]
    response_mask = data.batch["response_mask"]
    index = data.non_tensor_batch["uid"]
    payload = data.non_tensor_batch.get("process_credit")
    row_valid, credit_positions = parse_credit_payload(payload, token_level_rewards.shape[0])

    lambda_value = lambda_from_config(config)
    stats = CreditStats()
    advantages, returns, row_keep, stats = compute_process_advantage(
        token_level_rewards,
        response_mask,
        index,
        row_valid,
        credit_positions,
        lambda_process=lambda_value,
        stats=stats,
    )

    if _config_get(config, SELFCHECK_CONFIG_KEY, False):
        selfcheck = check_advantage_paths(
            token_level_rewards,
            response_mask,
            index,
            row_valid,
            credit_positions,
            advantages,
            returns,
            row_keep,
            lambda_process=lambda_value,
        )
        data.meta_info["r001_selfcheck"] = selfcheck
        print("R001_SELFCHECK " + json.dumps(selfcheck, sort_keys=True), flush=True)

    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    if float(row_keep.min()) < 1.0:
        data.batch["response_mask"] = response_mask * row_keep.unsqueeze(-1).to(response_mask)
    stats_dict = stats.as_dict()
    counters = data.non_tensor_batch.get("process_counters")
    if counters is not None:
        stats_dict["online_rows"] = len(counters)
        for key, value in sum_online_counters(counters).items():
            stats_dict[f"online_{key}"] = value
    data.meta_info["r001_stats"] = stats_dict
    print("R001_STATS " + json.dumps(stats_dict, sort_keys=True), flush=True)
    return data


def install(ray_trainer_module: Any) -> Any:
    """Patch ``ray_trainer.compute_advantage`` with the wrapper; return the previous object.

    Idempotent: a second install on an already-wrapped module is a no-op, so the S0-S3 smokes can
    call the entry twice without stacking wrappers.
    """

    previous = ray_trainer_module.compute_advantage
    if getattr(previous, "_r001_wrapper", False):
        return previous
    wrapper = build_wrapper(previous)
    wrapper._r001_wrapper = True
    ray_trainer_module.compute_advantage = wrapper
    return previous


def build_wrapper(original: Any) -> Any:
    """A closure over ``original`` with the pinned trainer's exact call signature."""

    def wrapper(
        data,
        adv_estimator,
        gamma=None,
        lam=None,
        num_repeat=None,
        norm_adv_by_std_in_grpo=True,
        config=None,
    ):
        return compute_advantage_r001(
            data,
            adv_estimator,
            gamma=gamma,
            lam=lam,
            num_repeat=num_repeat,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            config=config,
            original=original,
        )

    return wrapper


def _unreachable(**_kwargs):
    raise RuntimeError(
        f"estimator {R001_ESTIMATOR!r} must be dispatched through the R001 wrapper "
        "(scripts/v5_r001_train.py installs it on verl.trainer.ppo.ray_trainer) - "
        "refusing to fall back to a different advantage semantics"
    )


def register_stub(register_fn: Any) -> None:
    """Register the R001 estimator name so a misconfigured run fails loud at dispatch time.

    The name is only ever reached if the trainer dispatches it *without* the wrapper (wrong entry
    module), in which case there is no credit payload and silently falling back to another
    estimator would be a silent scientific change. Refusing is the correct behaviour.

    The stub is module-level so repeated calls are idempotent: ``core_algos.register_adv_est``
    raises when a name is re-registered with a *different* function object.
    """

    register_fn(R001_ESTIMATOR)(_unreachable)


__all__ = [
    "LAMBDA_CONFIG_KEY",
    "R001_ESTIMATOR",
    "build_wrapper",
    "compute_advantage_r001",
    "install",
    "lambda_from_config",
    "parse_credit_payload",
    "register_stub",
    "sum_online_counters",
]
