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

## Windows development note

The current repository may be inspected and edited from Windows with `scripts/env.ps1`. The Docker Lean server and eventual Kimina RL execution are Linux/NVIDIA targets. A Windows workstation without Docker is therefore expected to show warnings in `doctor.sh`; it is not evidence that the Linux training environment is ready.

## Environment variables

The canonical names and project-local defaults live in `scripts/env.sh` and `.env.example`. Secrets such as `HF_TOKEN` and `LEAN_SERVER_API_KEY` must be kept in the ignored `.env` file or supplied by the runtime secret manager.
