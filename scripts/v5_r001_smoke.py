#!/usr/bin/env python3
"""R001 engineering smokes S0-S3 (owner directive §10-§14; preregistration draft §10).

Four stages, each writing one JSON artifact under ``--out-dir`` (default
``experiments/results/v5_r001_smoke``) plus the raw run log it was derived from:

* **S0** - configuration/static audit of *both* arms, no GPU: the pinned hydra config is composed
  per arm with the exact launcher overrides and audited against the frozen expectations; the two
  resolved configs must differ in ``algorithm.r001_lambda`` and nowhere else; the parameter table
  is measured on a meta-device model; the worker-side stack and the reward-path identity are probed
  in subprocesses. No model outcome is produced or recorded.
* **S1** - memory/OOM/wall-clock feasibility, <= 3 optimizer steps on consumed training prompts.
  Every attempt is appended to ``attempts.json`` with an explicit knob diff; a retry must change
  exactly one *allowed* memory knob (owner §12). No scientific metric may be interpreted.
* **S2** - arm parity, <= 1 optimizer step per condition: condition ``lambda0`` (λ = 0) and
  condition ``lambda1`` (λ = 1), both with ``r001_selfcheck=True``, which proves on the *same
  online batch* that the treatment zero-path equals the pinned DrGRPO reference bit-for-bit and
  that the λ = 1 tensors equal the frozen additions. Loss/gradient deltas are recorded across the
  two conditions with their scope stated (different sampled continuations).
* **S3** - online info-tree/mapping dry run: one step with the oracle archive enabled, collecting
  the seven required counters (``mapped``, ``ambiguous``, ``outside_response``,
  ``retokenization_mismatch``, ``conflict``, ``infra_censored``, ``no_code``) plus the per-candidate
  archive records.

Every stage: consumed training prompts only (no capability holdout, no sealed reserve), no formal
scientific result, checkpoints off (``trainer.save_freq=-1``). The trainer runs as a subprocess of
this driver so a CUDA context never leaks between stages; the R001 JSON lines printed by the entry,
the worker stack, the reward path and the self-check are parsed out of the captured log.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"
RECIPE_DIR = ROOT / "third_party" / "kimina-prover-rl" / "recipe" / "kimina_prover_rl"
ENTRY = SCRIPTS / "v5_r001_train.py"
REWARD_FILE = SCRIPTS / "v5_r001_reward.py"
MANAGER_FILE = SCRIPTS / "v5_r001_manager.py"
DATASET = ROOT / "data" / "processed" / "p3_promptset" / "prompt_sets"
TRAIN_PARQUET = DATASET / "AI-MO" / "Kimina-Prover-Promptset" / "train.parquet"
MODEL_PATH = ROOT / "models" / "weights" / "kimina_distill_0_6b"
DEFAULT_OUT = ROOT / "experiments" / "results" / "v5_r001_smoke"
STAGES = ("s0", "s1", "s2", "s3")

ARM_LAMBDA = {"control": 0.0, "treatment": 1.0}
ORACLE_ENDPOINT = "http://127.0.0.1:8020"

R001_LINE_PREFIXES = (
    "R001_ENTRY",
    "R001_STACK",
    "R001_REWARD",
    "R001_STATS",
    "R001_SELFCHECK",
    "R001_MEM",
)

CODE_FILES = (
    "scripts/v5_r001_train.py",
    "scripts/v5_r001_manager.py",
    "scripts/v5_r001_reward.py",
    "scripts/v5_r001_smoke.py",
    "scripts/v5_r001_online_credit.py",
    "scripts/v5_process_oracle.py",
    "scripts/v5_p001_spec.py",
    "src/tinylean_rl/rl/r001_advantage.py",
    "src/tinylean_rl/rl/r001_selfcheck.py",
    "src/tinylean_rl/rl/process_credit.py",
    "src/tinylean_rl/rl/grpo.py",
)

#: The candidate 10 GB configuration (draft §10.5); every entry is an owner-allowed memory knob
#: and applies identically to both arms.
BASELINE_KNOBS: dict[str, str] = {
    "actor_rollout_ref.model.enable_gradient_checkpointing": "True",
    "actor_rollout_ref.model.enable_activation_offload": "True",
    "actor_rollout_ref.actor.fsdp_config.param_offload": "True",
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload": "True",
    "actor_rollout_ref.actor.use_dynamic_bsz": "True",
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": "1",
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": "5120",
    "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz": "True",
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": "1",
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu": "5120",
    "actor_rollout_ref.rollout.gpu_memory_utilization": "0.30",
}

#: Owner §12: only these may move between S1 retries (memory engineering, never the science).
ALLOWED_KNOBS = {
    "actor_rollout_ref.model.enable_gradient_checkpointing",
    "actor_rollout_ref.model.enable_activation_offload",
    "actor_rollout_ref.model.use_remove_padding",
    "actor_rollout_ref.model.use_fused_kernels",
    "actor_rollout_ref.model.fused_kernel_options.impl_backend",
    "actor_rollout_ref.actor.fsdp_config.param_offload",
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload",
    "actor_rollout_ref.actor.use_dynamic_bsz",
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu",
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu",
    "actor_rollout_ref.actor.entropy_from_logits_with_chunking",
    "actor_rollout_ref.actor.entropy_checkpointing",
    "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz",
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu",
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu",
    "actor_rollout_ref.rollout.gpu_memory_utilization",
    "actor_rollout_ref.rollout.enforce_eager",
    "actor_rollout_ref.rollout.free_cache_engine",
}


# --------------------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------------------


def rel(path: Path) -> str:
    """Artifacts are recorded relative to ROOT when they live under it (they are read from here)."""

    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


def code_stamp() -> dict[str, Any]:
    return {
        "git_head": git_head(),
        "files": {name: sha256_file(ROOT / name) for name in CODE_FILES},
    }


def package_version(name: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return None


def environment_snapshot() -> dict[str, Any]:
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "host": socket.gethostname(),
        "python": sys.version.split()[0],
        "torch": package_version("torch"),
        "vllm": package_version("vllm"),
        "ray": package_version("ray"),
        "transformers": package_version("transformers"),
        "gpu": out.stdout.strip() if out.returncode == 0 else None,
    }


def gpu_compute_apps() -> list[dict[str, Any]]:
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    apps: list[dict[str, Any]] = []
    if out.returncode != 0:
        return apps
    for line in out.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 3 and parts[0].isdigit():
            apps.append({"pid": int(parts[0]), "process": parts[1], "used_mib": int(parts[2])})
    return apps


def leftover_gpu_processes(min_mib: int = 1024) -> list[dict[str, Any]]:
    """A previous python run still holding the card would silently poison S1-S3 measurements."""

    return [
        app
        for app in gpu_compute_apps()
        if app["used_mib"] >= min_mib and "python" in app["process"].lower()
    ]


# --------------------------------------------------------------------------------------
# overrides and subprocess plumbing
# --------------------------------------------------------------------------------------


def base_overrides(
    *,
    arm: str,
    steps: int,
    rollout_dir: Path | None,
    selfcheck: bool = False,
    knobs: dict[str, str] | None = None,
) -> list[str]:
    """The frozen execution stack of both arms (draft §10.2/§10.5); only λ && knobs vary."""

    lam = ARM_LAMBDA[arm]
    overrides = [
        "algorithm.adv_estimator=r001_process",
        "algorithm.use_kl_in_reward=False",
        "algorithm.norm_adv_by_std_in_grpo=False",
        f"+algorithm.r001_lambda={lam!r}",
    ]
    if selfcheck:
        overrides.append("+algorithm.r001_selfcheck=True")
    overrides += [
        f"data.train_files=[{TRAIN_PARQUET}]",
        f"data.val_files=[{TRAIN_PARQUET}]",  # consumed training prompts; never iterated
        "data.train_batch_size=4",
        "data.max_prompt_length=1024",
        "data.max_response_length=4096",
        "+data.return_extra_info=True",
        "+data.multiturn=False",
        "data.return_raw_chat=True",
        "data.shuffle=False",  # identical prompt order across the S2 conditions
        "data.dataloader_num_workers=0",
        "data.filter_overlong_prompts=True",
        f"data.custom_cls.path={RECIPE_DIR / 'kimina_prover_rl' / 'dataset.py'}",
        "data.custom_cls.name=NuminaRLDataset",
        "data.truncation=error",
        f"actor_rollout_ref.model.path={MODEL_PATH}",
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.actor.optim.lr=2e-6",
        "actor_rollout_ref.actor.ppo_mini_batch_size=4",
        "actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm",
        "actor_rollout_ref.actor.use_kl_loss=False",
        "actor_rollout_ref.actor.kl_loss_coef=0.0",
        "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.actor.clip_ratio_low=0.2",
        "actor_rollout_ref.actor.clip_ratio_high=0.3",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.n=4",
        "actor_rollout_ref.rollout.max_num_batched_tokens=5120",
        "actor_rollout_ref.rollout.max_model_len=5120",
        "reward_model.reward_manager=r001_batch",
        "reward_model.launch_reward_fn_async=True",
        f"custom_reward_function.path={REWARD_FILE}",
        "custom_reward_function.name=reward",
        "+custom_reward_function.reward_kwargs.return_dict=True",
        "trainer.critic_warmup=0",
        "trainer.logger=[console,wandb]",
        "trainer.project_name=v5-r001-smoke",
        "trainer.experiment_name=r001-smoke",
        "trainer.n_gpus_per_node=1",
        "trainer.nnodes=1",
        "trainer.save_freq=-1",  # engineering checkpoints are never produced (owner §19)
        "trainer.test_freq=-1",
        "trainer.val_before_train=False",
        "trainer.total_epochs=1",
        f"trainer.total_training_steps={steps}",
    ]
    if rollout_dir is not None:
        overrides.append(f"trainer.rollout_data_dir={rollout_dir}")
    for key, value in {**BASELINE_KNOBS, **(knobs or {})}.items():
        overrides.append(f"{key}={value}")
    return overrides


def subprocess_env(*, archive_dir: Path | None = None, endpoint: str | None = None) -> dict[str, str]:
    env = dict(os.environ)
    paths = [str(SRC), str(SCRIPTS), str(RECIPE_DIR)]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env["WANDB_MODE"] = env.get("WANDB_MODE", "offline")
    env["LEAN_SERVER_API_URL"] = env.get("LEAN_SERVER_API_URL", ORACLE_ENDPOINT)
    env["TINYLEAN_R001_ORACLE_ENDPOINT"] = endpoint or env.get(
        "TINYLEAN_R001_ORACLE_ENDPOINT", ORACLE_ENDPOINT
    )
    if archive_dir is not None:
        env["TINYLEAN_R001_ORACLE_ARCHIVE"] = str(archive_dir)
    return env


class VramSampler(threading.Thread):
    """nvidia-smi `memory.used` poller; records the peak while a stage runs."""

    def __init__(self, interval: float = 2.0) -> None:
        super().__init__(daemon=True)
        self.interval = interval
        self.samples: list[int] = []
        self.peak_mib: int | None = None
        # Thread already defines ``_stop`` (used by join); shadowing it breaks join with a TypeError.
        self._stop_event = threading.Event()

    def run(self) -> None:  # pragma: no cover - thread body
        while not self._stop_event.is_set():
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=False,
            )
            if out.returncode == 0:
                try:
                    used = int(out.stdout.strip().splitlines()[0])
                except (IndexError, ValueError):
                    used = None
                if used is not None:
                    self.samples.append(used)
                    self.peak_mib = used if self.peak_mib is None else max(self.peak_mib, used)
            self._stop_event.wait(self.interval)

    def stop(self) -> None:
        self._stop_event.set()
        self.join(timeout=10)


def parse_r001_lines(text: str) -> dict[str, list[Any]]:
    found: dict[str, list[Any]] = {}
    for line in text.splitlines():
        line = line.strip()
        for prefix in R001_LINE_PREFIXES:
            if line.startswith(prefix + " "):
                payload = line[len(prefix) + 1 :]
                try:
                    value = json.loads(payload)
                except json.JSONDecodeError:
                    value = {"_unparsed": payload}
                found.setdefault(prefix, []).append(value)
    return found


_STEP_LINE = re.compile(r"^step:(\d+)\s*-\s*(.*)$")


def _to_number(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        match = re.search(r"\(([-0-9.eE+]+)\)", value)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
    return None


def parse_step_metrics(text: str) -> list[dict[str, float]]:
    """Console logger lines: ``step:N - key:value - key:value ...`` (aggregate_logger)."""

    rows: list[dict[str, float]] = []
    for line in text.splitlines():
        match = _STEP_LINE.match(line.strip())
        if match is None:
            continue
        entry: dict[str, float] = {"step": float(match.group(1))}
        for token in match.group(2).split(" - "):
            key, sep, value = token.partition(":")
            if not sep:
                continue
            number = _to_number(value)
            if number is not None:
                entry[key] = number
        rows.append(entry)
    return rows


def run_trainer(
    *,
    overrides: list[str],
    log_path: Path,
    archive_dir: Path | None = None,
    timeout_s: int | None = None,
) -> dict[str, Any]:
    """Run one trainer invocation and digest its log; the driver only ever observes it."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    sampler = VramSampler()
    sampler.start()
    started = time.perf_counter()
    timed_out = False
    with log_path.open("w", encoding="utf-8") as log:
        try:
            proc = subprocess.run(
                [sys.executable, "-u", str(ENTRY), *overrides],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=subprocess_env(archive_dir=archive_dir),
                timeout=timeout_s,
                check=False,
            )
            returncode: int | str = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            returncode = "TIMEOUT"
    sampler.stop()
    seconds = round(time.perf_counter() - started, 1)
    text = log_path.read_text(encoding="utf-8", errors="replace")
    oom = bool(re.search(r"CUDA out of memory|OutOfMemoryError", text))
    metrics = parse_step_metrics(text)
    return {
        "returncode": returncode,
        "timed_out": timed_out,
        "oom": oom,
        "seconds": seconds,
        "log": rel(log_path),
        "log_sha256": sha256_file(log_path),
        "peak_vram_mib": sampler.peak_mib,
        "r001": parse_r001_lines(text),
        "metrics": metrics,
        "saw_exit_code": returncode == 0,
    }


def run_capture(cmd: list[str], *, timeout_s: int = 900) -> dict[str, Any]:
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=subprocess_env(),
        timeout=timeout_s,
        check=False,
    )
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-20000:],
        "stderr": proc.stderr[-20000:],
    }


def write_artifact(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------------------
# S0 - configuration/static audit (both arms, no GPU)
# --------------------------------------------------------------------------------------


def hydra_config_dir() -> Path:
    import verl

    return Path(verl.__file__).resolve().parent / "trainer" / "config"


def compose_config(overrides: list[str]) -> Any:
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(hydra_config_dir()), version_base=None):
        return compose(config_name="ppo_trainer", overrides=list(overrides))


def _plain(value: Any) -> Any:
    from omegaconf import OmegaConf

    if OmegaConf.is_config(value) or isinstance(value, (list, tuple, dict)):
        try:
            return OmegaConf.to_container(OmegaConf.create({"v": value}), resolve=True)["v"]
        except (ValueError, TypeError, KeyError):  # unresolvable interpolation / non-container
            return str(value)
    return value if isinstance(value, (str, int, float, bool, type(None))) else str(value)


def _select(config: Any, key: str) -> Any:
    from omegaconf import OmegaConf

    value = OmegaConf.select(config, key, throw_on_missing=False)
    return _plain(value)


def _equal(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return bool(actual) is expected
    if isinstance(expected, (int, float)):
        try:
            return float(actual) == float(expected)
        except (TypeError, ValueError):
            return False
    return str(actual) == str(expected)


AUDIT_KEYS = (
    "algorithm.adv_estimator",
    "algorithm.norm_adv_by_std_in_grpo",
    "algorithm.use_kl_in_reward",
    "algorithm.r001_lambda",
    "algorithm.r001_selfcheck",
    "reward_model.reward_manager",
    "custom_reward_function.path",
    "custom_reward_function.name",
    "data.train_files",
    "data.val_files",
    "data.train_batch_size",
    "data.max_prompt_length",
    "data.max_response_length",
    "data.multiturn",
    "data.shuffle",
    "actor_rollout_ref.model.path",
    "actor_rollout_ref.model.use_remove_padding",
    "actor_rollout_ref.model.enable_gradient_checkpointing",
    "actor_rollout_ref.model.enable_activation_offload",
    "actor_rollout_ref.actor.fsdp_config.param_offload",
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload",
    "actor_rollout_ref.actor.use_dynamic_bsz",
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu",
    "actor_rollout_ref.actor.ppo_mini_batch_size",
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu",
    "actor_rollout_ref.actor.loss_agg_mode",
    "actor_rollout_ref.actor.use_kl_loss",
    "actor_rollout_ref.actor.entropy_coeff",
    "actor_rollout_ref.rollout.name",
    "actor_rollout_ref.rollout.n",
    "actor_rollout_ref.rollout.gpu_memory_utilization",
    "actor_rollout_ref.rollout.max_num_batched_tokens",
    "actor_rollout_ref.rollout.max_model_len",
    "actor_rollout_ref.rollout.tensor_model_parallel_size",
    "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz",
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu",
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu",
    "trainer.n_gpus_per_node",
    "trainer.nnodes",
    "trainer.save_freq",
    "trainer.test_freq",
    "trainer.val_before_train",
)


def arm_expectations(arm: str) -> dict[str, Any]:
    return {
        "algorithm.adv_estimator": "r001_process",
        "algorithm.norm_adv_by_std_in_grpo": False,
        "algorithm.use_kl_in_reward": False,
        "algorithm.r001_lambda": ARM_LAMBDA[arm],
        "algorithm.r001_selfcheck": False,
        "reward_model.reward_manager": "r001_batch",
        "custom_reward_function.path": str(REWARD_FILE),
        "custom_reward_function.name": "reward",
        "data.train_batch_size": 4,
        "data.max_prompt_length": 1024,
        "data.max_response_length": 4096,
        "data.multiturn": False,
        "data.shuffle": False,
        "actor_rollout_ref.actor.use_kl_loss": False,
        "actor_rollout_ref.actor.entropy_coeff": 0,
        "actor_rollout_ref.actor.loss_agg_mode": "seq-mean-token-sum-norm",
        "actor_rollout_ref.rollout.n": 4,
        "actor_rollout_ref.rollout.max_model_len": 5120,
        "actor_rollout_ref.rollout.tensor_model_parallel_size": 1,
        "trainer.save_freq": -1,
        "trainer.test_freq": -1,
        "trainer.val_before_train": False,
        "trainer.n_gpus_per_node": 1,
        "trainer.nnodes": 1,
    }


def parameter_table() -> dict[str, Any]:
    """Exact parameter count via a meta-device instantiation; no weights, no GPU."""

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(MODEL_PATH)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    parameters = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    dtype = str(getattr(config, "torch_dtype", "")).replace("torch.", "")
    gib = 2**30
    return {
        "parameters": parameters,
        "trainable_parameters": trainable,
        "parameter_dtype": dtype,
        "gradient_bytes": 2 * trainable,
        "master_weight_bytes": 4 * trainable,
        "optimizer_state_bytes": 8 * trainable,  # Adam m + v in fp32
        "static_total_bytes": 2 * trainable + 2 * trainable + 4 * trainable + 8 * trainable,
        "static_total_gib": round((2 + 2 + 4 + 8) * trainable / gib, 2),
        "model_config": {
            key: getattr(config, key, None)
            for key in (
                "architectures",
                "model_type",
                "hidden_size",
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
                "vocab_size",
                "tie_word_embeddings",
                "max_position_embeddings",
            )
        },
    }


def stage_s0(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out_dir) / "s0"
    stamp = code_stamp()
    violations: list[str] = []
    arms: dict[str, Any] = {}
    for arm in ("control", "treatment"):
        overrides = base_overrides(
            arm=arm, steps=1, rollout_dir=None, knobs=dict(args.knobs)
        )
        config = compose_config(overrides)
        resolved = {key: _select(config, key) for key in AUDIT_KEYS}
        expectations = arm_expectations(arm)
        for key, expected in expectations.items():
            if not _equal(resolved.get(key), expected):
                violations.append(f"{arm}: {key} = {resolved.get(key)!r}, expected {expected!r}")
        arms[arm] = {
            "lambda": ARM_LAMBDA[arm],
            "overrides": overrides,
            "resolved": resolved,
            "memory": {
                "param_offload": resolved.get("actor_rollout_ref.actor.fsdp_config.param_offload"),
                "optimizer_offload": resolved.get(
                    "actor_rollout_ref.actor.fsdp_config.optimizer_offload"
                ),
                "gradient_checkpointing": resolved.get(
                    "actor_rollout_ref.model.enable_gradient_checkpointing"
                ),
                "activation_offload": resolved.get(
                    "actor_rollout_ref.model.enable_activation_offload"
                ),
                "microbatch_sizes": {
                    "ppo_micro_batch_size_per_gpu": resolved.get(
                        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"
                    ),
                    "ppo_mini_batch_size": resolved.get(
                        "actor_rollout_ref.actor.ppo_mini_batch_size"
                    ),
                    "use_dynamic_bsz": resolved.get("actor_rollout_ref.actor.use_dynamic_bsz"),
                    "ppo_max_token_len_per_gpu": resolved.get(
                        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu"
                    ),
                    "log_prob_micro_batch_size_per_gpu": resolved.get(
                        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu"
                    ),
                    "log_prob_use_dynamic_bsz": resolved.get(
                        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz"
                    ),
                    "log_prob_max_token_len_per_gpu": resolved.get(
                        "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu"
                    ),
                },
                "rollout_engine": {
                    "name": resolved.get("actor_rollout_ref.rollout.name"),
                    "gpu_memory_utilization": resolved.get(
                        "actor_rollout_ref.rollout.gpu_memory_utilization"
                    ),
                    "max_num_batched_tokens": resolved.get(
                        "actor_rollout_ref.rollout.max_num_batched_tokens"
                    ),
                    "max_model_len": resolved.get("actor_rollout_ref.rollout.max_model_len"),
                    "tensor_model_parallel_size": resolved.get(
                        "actor_rollout_ref.rollout.tensor_model_parallel_size"
                    ),
                },
            },
        }

    arm_diff: dict[str, Any] = {}
    for key in sorted(set(arms["control"]["resolved"]) | set(arms["treatment"]["resolved"])):
        left = arms["control"]["resolved"].get(key)
        right = arms["treatment"]["resolved"].get(key)
        if left != right:
            arm_diff[key] = {"control": left, "treatment": right}
    if set(arm_diff) != {"algorithm.r001_lambda"}:
        violations.append(f"arms differ beyond algorithm.r001_lambda: {sorted(arm_diff)}")

    try:
        parameters = parameter_table()
    except (ImportError, RuntimeError, OSError) as exc:  # pragma: no cover - host dependent
        parameters = {"error": repr(exc)}
        violations.append(f"parameter table unavailable: {exc}")

    stack_probe = run_capture([sys.executable, str(MANAGER_FILE), "--check"])
    if stack_probe["returncode"] != 0:
        violations.append("worker-side stack probe failed (see checks.stack)")

    reward_probe = run_capture([sys.executable, str(REWARD_FILE), "--check"])
    identity: dict[str, Any] = {}
    if reward_probe["returncode"] == 0:
        try:
            identity = json.loads(reward_probe["stdout"])
        except json.JSONDecodeError:
            violations.append("reward identity probe did not print JSON")
    else:
        violations.append("reward identity probe failed (see checks.reward_identity)")
    if identity and identity.get("endpoint") != ORACLE_ENDPOINT:
        violations.append(f"reward endpoint {identity.get('endpoint')!r} != {ORACLE_ENDPOINT!r}")
    if identity and identity.get("oracle_container") is None:
        violations.append("reward identity is missing the oracle container")

    left = leftover_gpu_processes()
    artifact = {
        "stage": "s0",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "stamp": stamp,
        "environment": environment_snapshot(),
        "parameters": parameters,
        "arms": arms,
        "arm_diff": arm_diff,
        "checks": {
            "stack": stack_probe,
            "reward_identity": reward_probe,
            "reward_identity_parsed": identity,
            "leftover_gpu_processes": left,
        },
        "violations": violations,
        "passed": not violations,
    }
    write_artifact(out / "config_audit.json", artifact)
    return {
        "stage": "s0",
        "artifact": rel(out / "config_audit.json"),
        "passed": artifact["passed"],
        "violations": violations,
        "arm_diff": sorted(arm_diff),
    }


# --------------------------------------------------------------------------------------
# S1 - <= 3 optimizer steps, memory/OOM/wall-clock only
# --------------------------------------------------------------------------------------


def knob_diff(previous: dict[str, str], current: dict[str, str]) -> dict[str, dict[str, str]]:
    diff = {}
    for key in sorted(set(previous) | set(current)):
        if previous.get(key) != current.get(key):
            diff[key] = {"previous": previous.get(key), "current": current.get(key)}
    return diff


def append_attempt(path: Path, stamp: dict[str, Any], attempt: dict[str, Any]) -> list[dict[str, Any]]:
    if path.exists():
        record = json.loads(path.read_text(encoding="utf-8"))
        attempts = record.get("attempts", [])
    else:
        record, attempts = {"stamp": stamp, "attempts": []}, []
    previous_knobs = attempts[-1]["knobs"] if attempts else None
    diff = knob_diff(previous_knobs or {}, attempt["knobs"])
    attempt["diff_from_previous"] = diff
    if previous_knobs is not None:
        if not diff:
            raise SystemExit("S1 retry changes no knob; nothing to measure")
        if len(diff) > 1:
            raise SystemExit(
                f"S1 retry must change exactly one knob (owner §12); got {sorted(diff)}"
            )
        forbidden = sorted(key for key in diff if key not in ALLOWED_KNOBS)
        if forbidden:
            raise SystemExit(f"S1 retry changes non-allowed knobs: {forbidden}")
    attempt["attempt"] = len(attempts) + 1
    attempts.append(attempt)
    record["attempts"] = attempts
    record["stamp"] = stamp
    write_artifact(path, record)
    return attempts


def stage_s1(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out_dir) / "s1"
    stamp = code_stamp()
    leftovers = leftover_gpu_processes()
    if leftovers:
        raise SystemExit(f"S1 blocked: python processes still hold the GPU: {leftovers}")
    steps = args.steps
    if not 1 <= steps <= 3:
        raise SystemExit(f"S1 steps must be 1..3 (owner §12), got {steps}")
    log_path = out / "logs" / f"attempt-{int(time.time())}.log"
    overrides = base_overrides(
        arm=args.arm, steps=steps, rollout_dir=None, knobs=dict(args.knobs)
    )
    result = run_trainer(overrides=overrides, log_path=log_path, timeout_s=args.timeout_s)
    metrics = result["metrics"]
    final = metrics[-1] if metrics else {}
    per_step = round(result["seconds"] / steps, 1)
    if result["oom"]:
        outcome = "FAILED_OOM"
    elif result["timed_out"]:
        outcome = "TIMEOUT"
    elif result["returncode"] != 0:
        outcome = "FAILED"
    else:
        outcome = "SUCCESS"
    attempt = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "stamp": stamp,
        "environment": environment_snapshot(),
        "arm": args.arm,
        "steps": steps,
        "knobs": {**BASELINE_KNOBS, **dict(args.knobs)},
        "command": [str(ENTRY), *overrides],
        "log": result["log"],
        "log_sha256": result["log_sha256"],
        "returncode": result["returncode"],
        "outcome": outcome,
        "oom": result["oom"],
        "seconds": result["seconds"],
        "seconds_per_step": per_step,
        "peak_vram_mib": result["peak_vram_mib"],
        "metrics_final_step": final,
        "metrics": metrics,
        "r001": result["r001"],
        "selfcheck": result["r001"].get("R001_SELFCHECK", []),
        "mem": result["r001"].get("R001_MEM", []),
    }
    attempts_path = out / "attempts.json"
    attempts = append_attempt(attempts_path, stamp, attempt)
    return {
        "stage": "s1",
        "artifact": rel(attempts_path),
        "attempt": attempt["attempt"],
        "outcome": outcome,
        "seconds": result["seconds"],
        "seconds_per_step": per_step,
        "peak_vram_mib": result["peak_vram_mib"],
        "oom": result["oom"],
        "diff_from_previous": attempt["diff_from_previous"],
        "log": result["log"],
        "attempts": len(attempts),
    }


# --------------------------------------------------------------------------------------
# S2 - arm parity, <= 1 optimizer step per condition
# --------------------------------------------------------------------------------------


def rollout_rows(directory: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def last_step_value(metrics: list[dict[str, float]], key: str) -> float | None:
    for row in reversed(metrics):
        if key in row:
            return row[key]
    return None


def run_condition(
    *,
    out: Path,
    name: str,
    arm: str,
    knobs: dict[str, str],
    timeout_s: int | None,
) -> dict[str, Any]:
    rollout_dir = out / "rollout" / name
    log_path = out / "logs" / f"{name}.log"
    overrides = base_overrides(
        arm=arm, steps=1, rollout_dir=rollout_dir, selfcheck=True, knobs=knobs
    )
    result = run_trainer(overrides=overrides, log_path=log_path, timeout_s=timeout_s)
    rows = rollout_rows(rollout_dir) if rollout_dir.exists() else []
    prompts = [row.get("input") for row in rows]
    return {
        "arm": arm,
        "lambda": ARM_LAMBDA[arm],
        "overrides": overrides,
        "log": result["log"],
        "log_sha256": result["log_sha256"],
        "returncode": result["returncode"],
        "oom": result["oom"],
        "seconds": result["seconds"],
        "peak_vram_mib": result["peak_vram_mib"],
        "selfcheck": result["r001"].get("R001_SELFCHECK", []),
        "stats": result["r001"].get("R001_STATS", []),
        "reward": result["r001"].get("R001_REWARD", []),
        "mem": result["r001"].get("R001_MEM", []),
        "pg_loss": last_step_value(result["metrics"], "actor/pg_loss"),
        "grad_norm": last_step_value(result["metrics"], "actor/grad_norm"),
        "rollout_dir": rel(rollout_dir),
        "rollout_rows": len(rows),
        "prompt_sha256": hashlib.sha256(
            json.dumps(prompts, sort_keys=True, default=str).encode()
        ).hexdigest(),
        "prompts": prompts,
    }


def stage_s2(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out_dir) / "s2"
    stamp = code_stamp()
    leftovers = leftover_gpu_processes()
    if leftovers:
        raise SystemExit(f"S2 blocked: python processes still hold the GPU: {leftovers}")
    knobs = dict(args.knobs)
    lambda0 = run_condition(out=out, name="lambda0", arm="control", knobs=knobs, timeout_s=args.timeout_s)
    lambda1 = run_condition(
        out=out, name="lambda1", arm="treatment", knobs=knobs, timeout_s=args.timeout_s
    )
    prompt_rows_identical = lambda0["prompts"] == lambda1["prompts"]

    sc0 = lambda0["selfcheck"][-1] if lambda0["selfcheck"] else {}
    sc1 = lambda1["selfcheck"][-1] if lambda1["selfcheck"] else {}
    stats1 = lambda1["stats"][-1] if lambda1["stats"] else {}
    parity = {
        "source": "r001_selfcheck per condition, same online batch",
        "lambda0_run": {
            key: sc0.get(key)
            for key in (
                "rows",
                "groups",
                "parity_groups",
                "parity_rows",
                "max_abs_lambda0_delta",
                "lambda0_equals_pinned_drgrpo",
                "credit_tokens_expected",
                "passed",
            )
        },
        "lambda1_run": {
            key: sc1.get(key)
            for key in (
                "rows",
                "groups",
                "parity_groups",
                "parity_rows",
                "max_abs_lambda0_delta",
                "lambda0_equals_pinned_drgrpo",
                "max_abs_lambda1_delta",
                "lambda1_equals_frozen_additions",
                "credit_tokens_expected",
                "passed",
            )
        },
    }
    lambda1_valid = bool(
        sc1.get("lambda1_equals_frozen_additions")
        and sc1.get("passed")
        and (sc1.get("credit_tokens_expected") or 0) > 0
        and (stats1.get("credit_tokens") or 0) > 0
    )
    loss_delta = (
        abs((lambda0["pg_loss"] or 0.0) - (lambda1["pg_loss"] or 0.0))
        if lambda0["pg_loss"] is not None and lambda1["pg_loss"] is not None
        else None
    )
    grad_delta = (
        abs((lambda0["grad_norm"] or 0.0) - (lambda1["grad_norm"] or 0.0))
        if lambda0["grad_norm"] is not None and lambda1["grad_norm"] is not None
        else None
    )
    violations = []
    if lambda0["returncode"] != 0:
        violations.append("lambda0 condition did not finish")
    if lambda1["returncode"] != 0:
        violations.append("lambda1 condition did not finish")
    if not sc0.get("passed"):
        violations.append("lambda0 self-check did not pass")
    if not sc1.get("passed"):
        violations.append("lambda1 self-check did not pass")
    if not lambda1_valid:
        violations.append(
            "lambda1 tensor construction not evidenced "
            f"(credit_tokens_expected={sc1.get('credit_tokens_expected')}, stats={stats1})"
        )
    if not prompt_rows_identical:
        violations.append("the two conditions did not see the same prompt rows")
    artifact = {
        "stage": "s2",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "stamp": stamp,
        "environment": environment_snapshot(),
        "conditions": {
            "lambda0": {k: v for k, v in lambda0.items() if k != "prompts"},
            "lambda1": {k: v for k, v in lambda1.items() if k != "prompts"},
        },
        "prompt_rows_identical": prompt_rows_identical,
        "prompt_sha256": {"lambda0": lambda0["prompt_sha256"], "lambda1": lambda1["prompt_sha256"]},
        "lambda0_parity": parity,
        "lambda1_tensor_valid": lambda1_valid,
        "loss_delta": loss_delta,
        "gradient_delta": grad_delta,
        "delta_scope": (
            "loss/gradient deltas compare two single-step conditions that share the prompt rows "
            "but not the sampled continuations; the same-batch equality claim is the "
            "advantage-tensor parity inside r001_selfcheck (lambda=0 vs pinned DrGRPO and "
            "lambda=1 vs the frozen additions)"
        ),
        "violations": violations,
        "passed": not violations,
    }
    write_artifact(out / "parity.json", artifact)
    return {
        "stage": "s2",
        "artifact": rel(out / "parity.json"),
        "passed": artifact["passed"],
        "lambda0_parity": parity["lambda0_run"],
        "lambda1_tensor_valid": lambda1_valid,
        "loss_delta": loss_delta,
        "gradient_delta": grad_delta,
        "prompt_rows_identical": prompt_rows_identical,
        "violations": violations,
    }


# --------------------------------------------------------------------------------------
# S3 - online info-tree/mapping dry run (with the oracle archive)
# --------------------------------------------------------------------------------------


COUNTER_KEYS = (
    "mapped",
    "ambiguous",
    "outside_response",
    "retokenization_mismatch",
    "conflict",
    "infra_censored",
    "no_code",
)


def stage_s3(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out_dir) / "s3"
    stamp = code_stamp()
    leftovers = leftover_gpu_processes()
    if leftovers:
        raise SystemExit(f"S3 blocked: python processes still hold the GPU: {leftovers}")
    archive_dir = out / "archive"
    rollout_dir = out / "rollout"
    log_path = out / "logs" / "online.log"
    overrides = base_overrides(
        arm=args.arm, steps=1, rollout_dir=rollout_dir, selfcheck=True, knobs=dict(args.knobs)
    )
    result = run_trainer(
        overrides=overrides, log_path=log_path, archive_dir=archive_dir, timeout_s=args.timeout_s
    )
    reward_lines = result["r001"].get("R001_REWARD", [])
    counters = {key: sum(int(line.get(key, 0)) for line in reward_lines) for key in COUNTER_KEYS}
    summary = {
        key: sum(int(line.get(key, 0)) for line in reward_lines)
        for key in ("rows", "submitted", "verified", "positions_total", "candidates", "valid")
    }
    statuses: dict[str, int] = {}
    for line in reward_lines:
        for status, count in dict(line.get("statuses", {})).items():
            statuses[status] = statuses.get(status, 0) + int(count)
    archive_files = sorted(rel(path) for path in archive_dir.glob("*.jsonl"))
    archive_records = 0
    archive_bytes = 0
    if archive_dir.exists():
        for path in archive_dir.glob("*.jsonl"):
            archive_bytes += path.stat().st_size
            archive_records += sum(
                1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
            )
    rows = rollout_rows(rollout_dir) if rollout_dir.exists() else []
    credit_rows = sum(
        1 for row in rows if isinstance(row.get("process_credit"), dict) and row["process_credit"]
    )
    violations = []
    if result["returncode"] != 0:
        violations.append("S3 run did not finish")
    if not reward_lines:
        violations.append("no R001_REWARD line was produced")
    if result["oom"]:
        violations.append("S3 run hit an OOM")
    if archive_records == 0 and summary.get("submitted", 0) > 0:
        violations.append("submitted candidates but the archive is empty")
    artifact = {
        "stage": "s3",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "stamp": stamp,
        "environment": environment_snapshot(),
        "arm": args.arm,
        "lambda": ARM_LAMBDA[args.arm],
        "log": result["log"],
        "log_sha256": result["log_sha256"],
        "returncode": result["returncode"],
        "seconds": result["seconds"],
        "peak_vram_mib": result["peak_vram_mib"],
        "counters": counters,
        "summary": summary,
        "statuses": statuses,
        "stats": result["r001"].get("R001_STATS", []),
        "selfcheck": result["r001"].get("R001_SELFCHECK", []),
        "archive": {
            "dir": rel(archive_dir),
            "files": archive_files,
            "records": archive_records,
            "bytes": archive_bytes,
        },
        "rollout": {
            "dir": rel(rollout_dir) if rollout_dir.exists() else None,
            "rows": len(rows),
            "rows_with_credit": credit_rows,
        },
        "violations": violations,
        "passed": not violations,
    }
    write_artifact(out / "online_counters.json", artifact)
    return {
        "stage": "s3",
        "artifact": rel(out / "online_counters.json"),
        "passed": artifact["passed"],
        "counters": counters,
        "summary": summary,
        "archive_records": archive_records,
        "violations": violations,
    }


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT),
        help="artifact directory (default experiments/results/v5_r001_smoke)",
    )
    parser.add_argument(
        "--knob",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="memory knob override (repeatable); S1 allows at most one per retry",
    )
    parser.add_argument("--timeout-s", type=int, default=None, help="per-run wall clock budget")
    parser.add_argument("--dry", action="store_true", help="print the commands, run nothing")
    parser.add_argument("--json", action="store_true", help="print the stage summary as JSON")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="R001 S0-S3 engineering smokes")
    sub = parser.add_subparsers(dest="stage", required=True, metavar="STAGE")

    p0 = sub.add_parser("s0", help="configuration/static audit, both arms, no GPU")
    add_common(p0)

    p1 = sub.add_parser("s1", help="<= 3 optimizer steps, memory/wall-clock only")
    add_common(p1)
    p1.add_argument("--arm", choices=sorted(ARM_LAMBDA), default="treatment")
    p1.add_argument("--steps", type=int, default=3)

    p2 = sub.add_parser("s2", help="arm parity, <= 1 step per condition")
    add_common(p2)

    p3 = sub.add_parser("s3", help="online mapping dry run with the oracle archive")
    add_common(p3)
    p3.add_argument("--arm", choices=sorted(ARM_LAMBDA), default="treatment")
    return parser


def parse_knobs(raw: list[str]) -> dict[str, str]:
    knobs: dict[str, str] = {}
    for item in raw:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise SystemExit(f"--knob expects KEY=VALUE, got {item!r}")
        knobs[key] = value
    return knobs


def main(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv[1:])
    args.knobs = parse_knobs(args.knob)
    args.out_dir = str(Path(args.out_dir).resolve())
    if args.dry:
        steps = getattr(args, "steps", 1)
        arm = getattr(args, "arm", "treatment")
        overrides = base_overrides(
            arm=arm, steps=steps, rollout_dir=Path(args.out_dir) / args.stage / "rollout",
            selfcheck=args.stage in ("s2", "s3"), knobs=args.knobs,
        )
        print(json.dumps({"stage": args.stage, "entry": str(ENTRY), "overrides": overrides}, indent=1))
        return 0
    handler = {"s0": stage_s0, "s1": stage_s1, "s2": stage_s2, "s3": stage_s3}[args.stage]
    summary = handler(args)
    print("R001_SMOKE " + json.dumps(summary, sort_keys=True, default=str), flush=True)
    return 0 if summary.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
