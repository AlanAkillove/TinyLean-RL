#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/env.sh" >/dev/null

PASS=0
WARN=0
FAIL=0

pass() { printf '[✓] %s\n' "$1"; PASS=$((PASS + 1)); }
warn() { printf '[!] %s\n' "$1"; WARN=$((WARN + 1)); }
fail() { printf '[✗] %s\n' "$1"; FAIL=$((FAIL + 1)); }

printf 'TinyLean-RL environment check\n\n'

if [[ "$(uname -s)" == "Linux" ]]; then pass "Linux"; else warn "Target runtime is Linux; detected $(uname -s)"; fi

if command -v git >/dev/null 2>&1; then pass "Git: $(git --version)"; else fail "Git"; fi
if command -v git-lfs >/dev/null 2>&1; then pass "Git LFS: $(git-lfs --version | head -n1)"; else warn "Git LFS not installed"; fi
if command -v uv >/dev/null 2>&1; then pass "uv: $(uv --version)"; else fail "uv"; fi

if command -v nvidia-smi >/dev/null 2>&1; then
  pass "NVIDIA: $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -n1)"
  gpu_mem_mib="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
  if [[ "$gpu_mem_mib" =~ ^[0-9]+$ ]]; then
    if (( gpu_mem_mib >= 23000 )); then
      pass "GPU memory: ${gpu_mem_mib} MiB (>= 24 GB P3-A target)"
    else
      warn "GPU memory: ${gpu_mem_mib} MiB is below the 24 GB P3-A target"
    fi
  fi
else
  warn "nvidia-smi unavailable (required for GPU training)"
fi

if command -v docker >/dev/null 2>&1; then
  if docker info >/dev/null 2>&1; then pass "Docker daemon"; else warn "Docker CLI found but daemon is unavailable"; fi
  runtimes="$(docker info --format '{{json .Runtimes}}' 2>/dev/null || true)"
  if [[ "$runtimes" == *nvidia* ]]; then pass "NVIDIA Container Runtime"; else warn "NVIDIA Container Runtime not visible to Docker"; fi
else
  warn "Docker unavailable (required for Lean server)"
fi

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  pass "Project-local Python: $ROOT/.venv/bin/python"
  if "$ROOT/.venv/bin/python" -c 'import torch; assert torch.cuda.is_available()' >/dev/null 2>&1; then pass "PyTorch CUDA"; else warn "PyTorch CUDA unavailable"; fi
  if "$ROOT/.venv/bin/python" -c 'import vllm' >/dev/null 2>&1; then pass "vLLM"; else warn "vLLM not installed in project environment"; fi
  if "$ROOT/.venv/bin/python" -c 'import ray' >/dev/null 2>&1; then pass "Ray"; else warn "Ray not installed (required by the VERL trainer)"; fi
  if "$ROOT/.venv/bin/python" -c 'import verl' >/dev/null 2>&1; then pass "VERL (pinned submodule)"; else warn "VERL not importable; install per docs/environment.md runbook"; fi
else
  warn "Project-local .venv not found; run: uv sync --extra inference"
fi

if [[ -f "$ROOT/third_party/kimina-prover-rl/.git" || -d "$ROOT/third_party/kimina-prover-rl/.git" ]]; then
  pass "Kimina-Prover-RL submodule"
else
  warn "Kimina-Prover-RL submodule not initialized"
fi

if command -v curl >/dev/null 2>&1; then
  # The 2.0.0 image disables /openapi.json in prod mode (404 on the Linux
  # host, observed 2026-09-17); /health is the reliable readiness endpoint.
  if curl --silent --show-error --fail --max-time 5 "$LEAN_SERVER_API_URL/health" >/dev/null 2>&1; then
    pass "Lean Server: $LEAN_SERVER_API_URL"
    if [[ -x "$ROOT/.venv/bin/python" ]]; then
      lean_latency="$("$ROOT/.venv/bin/python" - <<'PY' 2>/dev/null || true
import time

from tinylean_rl.verifier.kimina import verify_codes

code = "import Mathlib\ntheorem tinylean_doctor : (1 : Nat) = 1 := by rfl\n"
started = time.perf_counter()
verify_codes([code], custom_ids=["tinylean-doctor"])
print(f"{time.perf_counter() - started:.1f}")
PY
)"
      if [[ -n "$lean_latency" ]]; then
        if awk "BEGIN{exit !($lean_latency > 60)}"; then
          warn "Lean verify first request took ${lean_latency}s (cold REPL; run scripts/prewarm_lean_server.py before RL)"
        else
          pass "Lean verify latency: ${lean_latency}s"
        fi
      else
        warn "Lean verify probe failed (server reachable but /verify did not answer)"
      fi
    fi
  else
    warn "Lean Server unavailable at $LEAN_SERVER_API_URL"
  fi
else
  warn "curl unavailable; cannot check Lean Server"
fi

for dir in "$HF_HOME" "$TINYLEAN_MODEL_ROOT" "$TINYLEAN_DATA_ROOT"; do
  if [[ -d "$dir" && -w "$dir" ]]; then pass "Writable project path: $dir"; else warn "Not writable or missing: $dir"; fi
done

for model_dir in "$TINYLEAN_MODEL_ROOT/kimina_distill_0_6b" "$TINYLEAN_MODEL_ROOT/kimina_rl_0_6b"; do
  if [[ -d "$model_dir" ]]; then pass "Model directory: $model_dir"; else warn "Model not downloaded: $model_dir"; fi
done

if [[ -f "$ROOT/data/raw/kimina_promptset/data/train-00000-of-00001.parquet" ]]; then
  pass "Promptset parquet: data/raw/kimina_promptset/data/train-00000-of-00001.parquet"
else
  warn "Promptset parquet missing (python3 scripts/download_data.py --dataset kimina_promptset)"
fi

if [[ -f "$ROOT/experiments/manifests/p2_5_complete.yaml" ]]; then
  pass "P2.5 completion manifest"
else
  warn "P2.5 manifest missing: experiments/manifests/p2_5_complete.yaml (P3 ships only after P2.5)"
fi

printf '\nSummary: %d passed, %d warnings, %d failures\n' "$PASS" "$WARN" "$FAIL"
if (( FAIL > 0 )); then exit 1; fi
if (( WARN > 0 )); then exit 2; fi
exit 0

