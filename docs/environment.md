# Environment

## Design

TinyLean-RL keeps Python dependencies, model/data caches and experiment outputs inside the project directory. The intended training host is Linux with an NVIDIA driver, Docker Engine, NVIDIA Container Toolkit, Git/Git LFS and `uv`.

- Python environment: `.venv/` managed by `uv`.
- Lean verifier: Docker image `projectnumina/kimina-lean-server:2.0.0`.
- Kimina RL code: `third_party/kimina-prover-rl` submodule at the commit recorded by Git.
- Caches: `.cache/`, configured by `scripts/env.sh`.
- Large assets: `models/weights/`, `data/raw/`, `data/processed/`, `runs/`; all are ignored by Git.

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

The inference extra pins the CUDA 12.8 PyTorch wheel (`torch==2.9.0+cu128`) through the explicit PyTorch package index. If a host cannot use CUDA 12.8, do not silently substitute a different wheel; record the host-specific compatibility decision first.

Do not install PyTorch, vLLM, VERL or Lean globally for this project. Kimina-Prover-RL remains the reference implementation; do not replace it with an unpinned upstream VERL install during P0.

## P3 runbook (Linux RTX 3090: P3-0 → P3-A)

P2.5 is frozen at tag `p2.5-win-complete`; the migration handoff lives in [`p3_linux_handoff.md`](p3_linux_handoff.md). P3-0 (Linux migration & on-policy calibration) comes first; P3-A — the first real on-policy RL step — runs only after P3-0 freezes its config. Everything must run on Linux with Docker Engine and an NVIDIA GPU. The gated command chain is:

```bash
git clone --recurse-submodules <repository-url> TinyLean-RL
cd TinyLean-RL
source scripts/env.sh
uv sync --extra inference --extra verifier --extra data
cp .env.example .env
docker compose -f infra/lean-server/compose.yaml up -d
bash scripts/doctor.sh                     # environment gate: VRAM floor, ray/vllm imports, Lean warm-up latency, manifest gate
bash scripts/run_p3_smoke.sh --dry         # provisional P3-A runner: print the generated command first
# P3-0: Promptset rollout calibration (temp 1.0) + full-FT memory probe -> freeze the P3-A config
bash scripts/run_p3_smoke.sh               # P3-A smoke, only after the P3-0 freeze
```

`scripts/run_p3_smoke.sh` refuses to start unless: the host is Linux, the Docker daemon is up, the 0.6B checkpoint and the promptset parquet exist, `experiments/manifests/p2_5_complete.yaml` marks P2.5 complete, the pinned recipe file exists, the Lean server answers `/openapi.json`, and `verl` is importable. It then runs `prepare_data.py` (idempotent), warms the Lean server via `scripts/prewarm_lean_server.py` (cold REPL rebuilds are the main source of dropped batch verification results), and executes `python3 -m verl.trainer.main_ppo` with the audited single-GPU overrides (n=4, max_prompt_length 1024 / max_response_length 4096, DrGRPO without KL, multiturn off) for 2-5 optimizer steps, and saves one checkpoint at the final step (trainer.save_freq = steps) so the save/reload path can be validated. The runner is provisional: not yet Linux-validated, and the training mode (full-parameter vs LoRA) is frozen during P3-0.

`scripts/doctor.sh` enforces a 23 GiB GPU memory floor for the smoke host. The single-device update-step footprint measured by `scripts/lora_step_probe.py` is recorded in `experiments/results/p2_5_lora_step_probe.json`; use it (together with the audit) when sizing the Linux host. Every override and its pinned-commit justification lives in `docs/p3_config_audit.md`; read it before changing any value in the smoke script.

For a longer P3-B pilot, `scripts/run_rl_pilot.sh` reuses the same P2.5 gate manifest and environment.

## Windows development note

The current repository may be inspected and edited from Windows with `scripts/env.ps1`. The Docker Lean server and eventual Kimina RL execution are Linux/NVIDIA targets. A Windows workstation without Docker is therefore expected to show warnings in `doctor.sh`; it is not evidence that the Linux training environment is ready. Windows-only workarounds (`env.ps1`, GBK console reconfigures, WDDM shared-memory notes, the system proxy) never enter the default Linux path of `scripts/env.sh`, `scripts/doctor.sh` or `scripts/run_p3_smoke.sh`.

## Environment variables

The canonical names and project-local defaults live in `scripts/env.sh` and `.env.example`. Secrets such as `HF_TOKEN` and `LEAN_SERVER_API_KEY` must be kept in the ignored `.env` file or supplied by the runtime secret manager.
