# Environment

## Design

TinyLean-RL keeps Python dependencies, model/data caches and experiment outputs inside the project directory. The intended training host is Linux with an NVIDIA driver, Docker Engine, NVIDIA Container Toolkit, Git/Git LFS and `uv`.

- Python environment: `.venv/` managed by `uv`.
- Lean verifier: Docker image `projectnumina/kimina-lean-server:2.0.0`.
- Kimina RL code: `third_party/kimina-prover-rl` submodule at the commit recorded by Git.
- Caches: `.cache/`, configured by `scripts/env.sh`.
- Large assets: `models/weights/`, `data/raw/`, `data/processed/`, `runs/`; all are ignored by Git.

## Linux host record

Audited 2026-09-17 on the lab training server before any P3-0 experiment; compact
machine record in [`../experiments/manifests/p3_0_environment.yaml`](../experiments/manifests/p3_0_environment.yaml)
and research decisions D001–D004 in [`research-decisions.md`](research-decisions.md).

| Component | Value |
|---|---|
| OS | Ubuntu 24.04.4 LTS (kernel 7.0.0-31-generic) |
| Machine | Dell PowerEdge T640, 64 logical CPUs, 62 GiB RAM, 2.8 TiB free disk |
| GPU | NVIDIA GeForce RTX 3090, 24576 MiB, driver 580.173.02 (CUDA 13.0 driver) |
| Python | uv-managed project `.venv` (3.12.3) |
| PyTorch | `2.7.0` (cu126 default wheels) — aligned with the pinned VERL matrix, D004 |
| uv | 0.12.13, user-level install (`~/.local/bin/uv`) |
| Docker | Engine 29.8.0 + Compose v5.5.1 with NVIDIA Container Toolkit (`nvidia` runtime registered) |
| Network | `huggingface.co` and `registry-1.docker.io` are unreachable from this host |

Host-specific network workarounds (do not change pinned revisions):

- Hugging Face downloads require `HF_ENDPOINT=https://hf-mirror.com` and
  `HF_HUB_DISABLE_XET=1`; `scripts/env.sh` now defaults to both on this host (D002).
- The Docker daemon has no direct Docker Hub access; the pinned Lean server image was
  side-loaded with `docker save` / `docker load` (D003).
- The Compose network is pinned to `10.201.0.0/24` (`infra/lean-server/compose.yaml`)
  because automatic subnet allocation collided with the campus LAN.
- The pinned recipe's dataset hook calls `wandb.log()` after every trainer step, so
  runs use the wandb logger; `scripts/env.sh` defaults `WANDB_MODE=offline` (no account
  needed, run data lands under the ignored `runs/wandb/`).

## Linux setup

```bash
git clone --recurse-submodules <repository-url> TinyLean-RL
cd TinyLean-RL
source scripts/env.sh
uv sync
cp .env.example .env
docker compose -f infra/lean-server/compose.yaml up -d
bash scripts/doctor.sh
```

For inference dependencies, use the project environment:

```bash
uv sync --extra inference
```

The inference extra pins `torch==2.7.0` (the CUDA 12.6 default PyPI wheels), aligned with the pinned VERL training matrix (D004). If a host cannot use CUDA 12.6 wheels, do not silently substitute a different wheel; record the host-specific compatibility decision first.

Do not install PyTorch, vLLM, VERL or Lean globally for this project. Kimina-Prover-RL remains the reference implementation; do not replace it with an unpinned upstream VERL install during P0.

## P3 runbook (Linux RTX 3090: P3-0 → P3-A)

P2.5 is frozen at tag `p2.5-win-complete`; the migration handoff lives in [`p3_linux_handoff.md`](p3_linux_handoff.md). P3-0 (Linux migration & on-policy calibration) comes first; P3-A — the first real on-policy RL step — runs only after P3-0 freezes its config. Everything must run on Linux with Docker Engine and an NVIDIA GPU. The gated command chain is:

```bash
git clone --recurse-submodules <repository-url> TinyLean-RL
cd TinyLean-RL
source scripts/env.sh
uv sync --all-extras
cp .env.example .env
docker compose -f infra/lean-server/compose.yaml up -d
bash scripts/doctor.sh                     # environment gate: VRAM floor, ray/vllm imports, Lean warm-up latency, manifest gate
bash scripts/run_p3_smoke.sh --dry         # provisional P3-A runner: print the generated command first
# P3-0: Promptset rollout calibration (temp 1.0) + full-FT memory probe -> freeze the P3-A config
bash scripts/run_p3_smoke.sh               # P3-A smoke, only after the P3-0 freeze
```

`scripts/run_p3_smoke.sh` refuses to start unless: the host is Linux, the Docker daemon is up, the 0.6B checkpoint and the promptset parquet exist, `experiments/manifests/p2_5_complete.yaml` marks P2.5 complete, the pinned recipe file exists, the Lean server answers `/health` (the 2.0.0 image disables `/openapi.json` in prod mode; both the runner and `doctor.sh` probe `/health` on Linux), and `verl` is importable. It then runs `prepare_data.py` (idempotent), warms the Lean server via `scripts/prewarm_lean_server.py` (cold REPL rebuilds are the main source of dropped batch verification results), and executes `python3 -m verl.trainer.main_ppo` with the audited single-GPU overrides (n=4, max_prompt_length 1024 / max_response_length 4096, DrGRPO without KL, multiturn off) for 2-5 optimizer steps, and saves one checkpoint at the final step (trainer.save_freq = steps) so the save/reload path can be validated. The runner is provisional: not yet Linux-validated, and the training mode (full-parameter vs LoRA) is frozen during P3-0. `--steps 1` is the P3-0 full-FT memory probe.

The `training` extra carries the pinned VERL matrix (D004): `vllm==0.9.1`, `ray`, the flash-attn wheel matching torch 2.7 / cp312 / cxx11abiTRUE, and the `third_party/kimina-prover-rl` submodule installed editable (its declared ceiling `vllm<=0.9.1` forces `torch==2.7.0`). Always sync with `uv sync --all-extras` — uv prunes packages that the selected extras no longer require, and the training stack must survive every re-sync (`uv run` keeps the full environment). While the GitHub release CDN is unreachable from this host, `uv lock` re-validation of the pinned flash-attn wheel can fail; retry once the CDN is reachable (the wheel stays cached, and `uv run --no-sync` works against the installed environment).

`scripts/doctor.sh` enforces a 23 GiB GPU memory floor for the smoke host. The single-device update-step footprint measured by `scripts/lora_step_probe.py` is recorded in `experiments/results/p2_5_lora_step_probe.json`; use it (together with the audit) when sizing the Linux host. Every override and its pinned-commit justification lives in `docs/p3_config_audit.md`; read it before changing any value in the smoke script.

For the bounded P3-B learning pilot, `scripts/run_p3_pilot.sh` reuses the same gates and the frozen configuration with explicit bounds: `--steps` (default 30, hard range 1..500, never auto-extended), `--n` (group size; `--n 8` restores the official baseline), `--save-freq` (default 10, keeps the last 3 checkpoints) and per-step rollout dumps under `runs/p3b_pilot/rollout_data` for the IGR_t / Z_t / O_t analysis. `scripts/run_rl_pilot.sh` remains the coarse gate for any longer run.

## Windows development note

The current repository may be inspected and edited from Windows with `scripts/env.ps1`. The Docker Lean server and eventual Kimina RL execution are Linux/NVIDIA targets. A Windows workstation without Docker is therefore expected to show warnings in `doctor.sh`; it is not evidence that the Linux training environment is ready. Windows-only workarounds (`env.ps1`, GBK console reconfigures, WDDM shared-memory notes, the system proxy) never enter the default Linux path of `scripts/env.sh`, `scripts/doctor.sh` or `scripts/run_p3_smoke.sh`.

## Environment variables

The canonical names and project-local defaults live in `scripts/env.sh` and `.env.example`. Secrets such as `HF_TOKEN` and `LEAN_SERVER_API_KEY` must be kept in the ignored `.env` file or supplied by the runtime secret manager.
