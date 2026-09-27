#!/usr/bin/env python3
"""R001 worker-side installation (owner directive §11-§15; draft §7.4, §10.4).

Three pieces, all of which must run inside the Ray worker process that executes the *pinned*
trainer (`verl.trainer.main_ppo.TaskRunner.run`), because that is where the reward manager is
looked up by name and where `RayPPOTrainer.fit` resolves `compute_advantage` as a module global:

1. ``R001BatchRewardManager`` (registered as ``r001_batch``): the pinned ``BatchRewardManager``
   with one addition - each row's generated token ids are handed to the reward function. The
   frozen process credit lives on the training token frame and the round-trip gate must see the ids
   the model actually produced, which the plain ``batch`` manager never passes;
2. ``install_r001_stack()``: patches ``verl.trainer.ppo.ray_trainer.compute_advantage`` with the
   frozen R001 wrapper, registers the ``r001_process`` estimator stub, and asserts the manager is
   registered in *this* process. Idempotent by construction (the wrapper and the stub both refuse
   to double-install);
3. ``make_r001_task_runner(actor_class)``: derives the pinned ``TaskRunner`` so that (2) runs
   before ``load_reward_manager``. Ray forbids subclassing an actor class
   (``ActorClassInheritanceException`` on 2.48), so the derivation goes through the undecorated
   class Ray keeps in ``__ray_metadata__.modified_class`` and re-applies the pinned scheduling
   options verbatim.

``scripts/v5_r001_train.py`` is the entry point that wires (3) into ``main_ppo`` and then calls the
pinned hydra ``main`` unchanged.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "scripts", ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tinylean_rl.rl.r001_advantage import R001_ESTIMATOR, install, register_stub

REWARD_MANAGER_NAME = "r001_batch"

#: ``ray.remote`` keyword options the pinned actor class may carry; re-applied on derivation.
_REMOTE_OPTION_ATTRS = (
    "num_cpus",
    "num_gpus",
    "memory",
    "object_store_memory",
    "resources",
    "accelerator_type",
    "max_restarts",
    "max_task_retries",
    "max_concurrency",
    "max_pending_calls",
    "concurrency_groups",
    "runtime_env",
    "scheduling_strategy",
    "label_selector",
    "max_calls",
)

_MANAGER_CLS: type | None = None


def reward_manager_class() -> type:
    """The pinned batch manager with token ids; built once per process (registry identity)."""

    global _MANAGER_CLS
    if _MANAGER_CLS is not None:
        return _MANAGER_CLS

    from verl.workers.reward_manager.batch import BatchRewardManager
    from verl.workers.reward_manager.registry import register

    @register(REWARD_MANAGER_NAME)
    class R001BatchRewardManager(BatchRewardManager):
        """``BatchRewardManager`` that also passes the generated token ids to the reward fn."""

        def verify(self, data: Any) -> Any:
            prompt_len = data.batch["prompts"].shape[-1]
            response_ids = data.batch["responses"]
            valid_response_lengths = data.batch["attention_mask"][:, prompt_len:].sum(dim=-1)

            responses_str = []
            token_id_rows = []
            for row in range(len(data)):
                valid_len = int(valid_response_lengths[row].item())
                row_ids = response_ids[row][:valid_len]
                token_id_rows.append([int(token_id) for token_id in row_ids])
                responses_str.append(self.tokenizer.decode(row_ids, skip_special_tokens=True))

            ground_truths = [
                item.non_tensor_batch["reward_model"].get("ground_truth", None) for item in data
            ]
            data_sources = data.non_tensor_batch[self.reward_fn_key]
            extras = data.non_tensor_batch.get("extra_info", [None] * len(data))

            return self.compute_score(
                data_sources=data_sources,
                solution_strs=responses_str,
                ground_truths=ground_truths,
                extra_infos=extras,
                token_ids=token_id_rows,
                **self.reward_kwargs,
            )

    _MANAGER_CLS = R001BatchRewardManager
    return _MANAGER_CLS


def install_r001_stack() -> dict[str, Any]:
    """Install (idempotently) the advantage wrapper, the estimator stub and the manager."""

    from verl.trainer.ppo import core_algos, ray_trainer

    manager_cls = reward_manager_class()
    install(ray_trainer)
    register_stub(core_algos.register_adv_est)

    from verl.workers.reward_manager import get_reward_manager_cls

    registered = get_reward_manager_cls(REWARD_MANAGER_NAME)
    if registered is not manager_cls:
        raise RuntimeError(
            f"{REWARD_MANAGER_NAME} is registered as {registered}, not {manager_cls}"
        )
    state = {
        "compute_advantage_wrapped": bool(
            getattr(ray_trainer.compute_advantage, "_r001_wrapper", False)
        ),
        "reward_manager": REWARD_MANAGER_NAME,
        "adv_estimator": R001_ESTIMATOR,
        "manager_module": manager_cls.__module__,
    }
    print("R001_STACK " + json.dumps(state, sort_keys=True), flush=True)
    return state


def gpu_memory_snapshot() -> dict[str, Any]:
    """Peak/resident GPU and CPU memory of *this* worker process (S1 evidence, owner §12).

    The hybrid engine runs vLLM and the FSDP actor in one process, so the torch peaks cover both
    the rollout engine and the training phases. Diagnostics only: never raises.
    """

    snapshot: dict[str, Any] = {}
    try:
        import resource

        snapshot["max_rss_mib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    except (ImportError, OSError) as exc:  # pragma: no cover - platform diagnostics
        snapshot["rss_error"] = repr(exc)
    try:
        import torch

        if torch.cuda.is_available():
            snapshot["peak_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
            snapshot["peak_reserved_mib"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)
            free, total = torch.cuda.mem_get_info()
            snapshot["free_mib"] = round(free / 2**20, 1)
            snapshot["total_mib"] = round(total / 2**20, 1)
    except (ImportError, RuntimeError, AssertionError) as exc:  # pragma: no cover - host dependent
        snapshot["cuda_error"] = repr(exc)
    return snapshot


def make_r001_task_runner(actor_class: Any, *, install_fn: Any = install_r001_stack) -> Any:
    """Derive the pinned ``TaskRunner`` so ``run`` installs the R001 stack before anything else.

    Ray identifies a submitted actor class by ``__module__`` + ``__qualname__``, so the derived
    class is named ``R001TaskRunner`` inside *this* module and rebuilt in the worker by importing
    it - the same code path the driver took.
    """

    import ray

    metadata = getattr(actor_class, "__ray_metadata__", None)
    plain = getattr(metadata, "modified_class", None)
    if plain is None:
        raise RuntimeError(
            f"{actor_class!r} does not expose __ray_metadata__.modified_class; cannot derive the "
            "R001 TaskRunner. Ray's actor-class introspection changed - fix the derivation instead "
            "of running without the stack."
        )

    class R001TaskRunner(plain):  # type: ignore[misc, valid-type]
        def run(self, config: Any) -> Any:
            install_fn()
            try:
                return super().run(config)
            finally:
                # Also printed on failure: an OOM must leave the peak numbers in the log for the
                # S1 one-knob-per-retry diff table (owner §12).
                print("R001_MEM " + json.dumps(gpu_memory_snapshot(), sort_keys=True), flush=True)

    R001TaskRunner.__name__ = "R001TaskRunner"
    R001TaskRunner.__qualname__ = "R001TaskRunner"
    R001TaskRunner.__module__ = __name__
    options = {
        name: getattr(metadata, name)
        for name in _REMOTE_OPTION_ATTRS
        if getattr(metadata, name, None) is not None
    }
    return ray.remote(**options)(R001TaskRunner) if options else ray.remote(R001TaskRunner)


__all__ = [
    "REWARD_MANAGER_NAME",
    "gpu_memory_snapshot",
    "install_r001_stack",
    "make_r001_task_runner",
    "reward_manager_class",
]


def main(argv: list[str]) -> int:
    """``--check``: the S0 static probe - install the stack in *this* process and report it."""

    if argv[1:] != ["--check"]:
        print("usage: python scripts/v5_r001_manager.py --check", file=sys.stderr)
        return 2
    state = install_r001_stack()
    print("R001_STACK_CHECK " + json.dumps(state, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
