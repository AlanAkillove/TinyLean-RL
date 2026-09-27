"""Fixture tests for the worker-side R001 installation (owner §11, §13; draft §7.4).

``scripts/v5_r001_manager.py`` has three jobs and each is pinned here without touching the pinned
trainer's runtime:

* ``reward_manager_class()`` registers ``r001_batch`` exactly once and returns the same class;
* ``gpu_memory_snapshot()`` is diagnostics only - it must return a dict on any host;
* ``make_r001_task_runner()`` refuses to derive a runner when Ray's actor metadata is missing
  (fail loud, never run without the stack) and otherwise produces a ``R001TaskRunner`` in
  ``v5_r001_manager``.

The last one is only fully verifiable with a live Ray worker (the class is fetched by module name);
that check is gated behind ``TINYLEAN_R001_RAY_ACTOR_TEST=1`` because it starts a local Ray
instance.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "scripts", ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import v5_r001_manager as M


def test_reward_manager_is_registered_once() -> None:
    pytest.importorskip("verl")
    from verl.workers.reward_manager import get_reward_manager_cls

    first = M.reward_manager_class()
    second = M.reward_manager_class()
    assert first is second, "the registry keys on class identity"
    assert M.REWARD_MANAGER_NAME == "r001_batch"
    assert get_reward_manager_cls(M.REWARD_MANAGER_NAME) is first
    assert first.__name__ == "R001BatchRewardManager"


def test_install_stack_is_idempotent() -> None:
    pytest.importorskip("verl")
    from verl.trainer.ppo import ray_trainer

    before = ray_trainer.compute_advantage
    first = M.install_r001_stack()
    second = M.install_r001_stack()

    assert first["compute_advantage_wrapped"] and second["compute_advantage_wrapped"]
    assert ray_trainer.compute_advantage is not before
    assert M.install_r001_stack()["reward_manager"] == M.REWARD_MANAGER_NAME


def test_gpu_memory_snapshot_never_raises() -> None:
    snapshot = M.gpu_memory_snapshot()
    assert isinstance(snapshot, dict)
    assert "rss_error" not in snapshot and "cuda_error" not in snapshot
    if sys.platform.startswith("linux"):
        assert snapshot["max_rss_mib"] > 0


def test_ensure_worker_imports_puts_the_paths_on_pythonpath(monkeypatch) -> None:
    """Ray workers import the derived class by module name, so the entry repairs ``PYTHONPATH``."""

    import v5_r001_train as T

    monkeypatch.setenv("PYTHONPATH", "/existing-keep-me")
    merged = T.ensure_worker_imports()
    parts = merged.split(os.pathsep)
    for directory in T.WORKER_PYTHONPATH_DIRS:
        assert str(directory) in parts
    assert parts[-1] == "/existing-keep-me", "existing entries must survive, order aside"
    assert T.ensure_worker_imports() == merged, "idempotent"
    assert len(parts) == len(set(parts))


def test_runner_derivation_requires_ray_metadata() -> None:
    pytest.importorskip("ray")

    with pytest.raises(RuntimeError, match="modified_class"):
        M.make_r001_task_runner(object)


def test_runner_derivation_names_the_remote_class() -> None:
    pytest.importorskip("ray")

    class Metadata:
        num_cpus = 1
        num_gpus = 0

    class Plain:
        def run(self, config):
            return config

    class Actor:
        __ray_metadata__ = Metadata()

    Actor.__ray_metadata__.modified_class = Plain
    runner = M.make_r001_task_runner(Actor, install_fn=lambda: None)
    descriptor = runner.__ray_metadata__.actor_creation_function_descriptor
    assert (descriptor.module_name, descriptor.class_name) == ("v5_r001_manager", "R001TaskRunner")


@pytest.mark.skipif(
    os.environ.get("TINYLEAN_R001_RAY_ACTOR_TEST") != "1",
    reason="starts a local Ray instance; set TINYLEAN_R001_RAY_ACTOR_TEST=1 to run",
)
def test_runner_installs_before_the_pinned_run() -> None:
    """The real contract: the worker imports the derived class by name, then ``install`` runs first.

    Without the entry's ``PYTHONPATH`` repair the worker dies with ``ModuleNotFoundError:
    v5_r001_manager`` (Ray fetches an actor class defined in a function by module name), which is
    exactly why ``v5_r001_train.ensure_worker_imports`` runs before ``ray.init``.
    """

    import tempfile

    import ray
    import v5_r001_manager as MM
    import v5_r001_train as T

    class Metadata:
        num_cpus = 1
        num_gpus = 0

    def install_probe():
        # Both closures resolve ``MM`` by module name, so the flag is visible in the worker.
        MM.R001_INSTALL_PROBE = True

    class Plain:
        def run(self, config):
            if not getattr(MM, "R001_INSTALL_PROBE", False):
                raise RuntimeError("the pinned run was reached before the R001 stack was installed")
            return config

    class Actor:
        __ray_metadata__ = Metadata()

    Actor.__ray_metadata__.modified_class = Plain
    runner = M.make_r001_task_runner(Actor, install_fn=install_probe)
    T.ensure_worker_imports()
    ray.init(
        num_cpus=1,
        include_dashboard=False,
        _temp_dir=tempfile.mkdtemp(prefix="r001-ray-", dir="/tmp"),
    )
    try:
        assert ray.get(runner.remote().run.remote("cfg")) == "cfg"
    finally:
        ray.shutdown()
