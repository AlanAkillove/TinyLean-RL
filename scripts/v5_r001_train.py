#!/usr/bin/env python3
"""R001 training entry: the pinned trainer, plus the worker-side R001 stack.

The pinned launcher runs ``python -m verl.trainer.main_ppo``. R001 needs the advantage wrapper,
the ``r001_process`` estimator stub and the ``r001_batch`` reward manager to exist *inside* the
Ray actor process that runs the trainer, which the pinned ``main_ppo`` gives no hook for (its
``TaskRunner`` is created from a module global and has no plug-in point). So this entry:

1. imports the pinned module, replaces ``main_ppo.TaskRunner`` with the derived
   ``R001TaskRunner`` (``scripts/v5_r001_manager.py``) whose ``run`` installs the stack first;
2. calls the pinned hydra ``main`` unchanged, with the launcher's overrides still on ``sys.argv``
   (the decorated function resolves its ``config_path`` relative to the pinned module, so the exact
   ``ppo_trainer`` config is used).

Everything else - ``ray.init``, the runtime env, the worker layout, the trainer itself - stays the
pinned code path.

Ray fetches the *derived* actor class in the worker by module name (``v5_r001_manager``), and the
worker later imports the reward module (which needs ``kimina_prover_rl``), so the worker process
must see these directories on ``PYTHONPATH``. Ray workers inherit this process's environment at
``ray.init`` time, and the pinned ``main_ppo`` only *adds* its own ``env_vars``, so
``ensure_worker_imports`` repairs ``os.environ`` before ``main_ppo.main`` is called; the same
directories are exported by ``scripts/run_v5_r001.sh`` for the driver side.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECIPE_DIR = ROOT / "third_party" / "kimina-prover-rl" / "recipe" / "kimina_prover_rl"
WORKER_PYTHONPATH_DIRS = (ROOT / "src", ROOT / "scripts", RECIPE_DIR)
for _path in (ROOT / "scripts", ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def ensure_worker_imports() -> str:
    """Put the worker-side import roots on ``PYTHONPATH`` (idempotent; returns the merged value)."""

    required = [str(path) for path in WORKER_PYTHONPATH_DIRS]
    current = [part for part in os.environ.get("PYTHONPATH", "").split(os.pathsep) if part]
    merged = [*[path for path in required if path not in current], *current]
    os.environ["PYTHONPATH"] = os.pathsep.join(merged)
    return os.environ["PYTHONPATH"]


def main() -> int:
    from v5_r001_manager import make_r001_task_runner, reward_manager_class
    from verl.trainer import main_ppo

    worker_pythonpath = ensure_worker_imports()
    manager_cls = reward_manager_class()
    runner = make_r001_task_runner(main_ppo.TaskRunner)
    descriptor = runner.__ray_metadata__.actor_creation_function_descriptor
    pinned = main_ppo.TaskRunner
    main_ppo.TaskRunner = runner
    print(
        "R001_ENTRY "
        + json.dumps(
            {
                "pinned_task_runner": str(pinned),
                "runner": f"{descriptor.module_name}.{descriptor.class_name}",
                "reward_manager": manager_cls.__name__,
                "worker_pythonpath": worker_pythonpath,
                "argv": sys.argv[1:],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    main_ppo.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
