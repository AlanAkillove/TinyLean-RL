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
else
  warn "Project-local .venv not found; run: uv sync --extra inference"
fi

if [[ -f "$ROOT/third_party/kimina-prover-rl/.git" || -d "$ROOT/third_party/kimina-prover-rl/.git" ]]; then
  pass "Kimina-Prover-RL submodule"
else
  warn "Kimina-Prover-RL submodule not initialized"
fi

if command -v curl >/dev/null 2>&1; then
  if curl --silent --show-error --fail --max-time 5 "$LEAN_SERVER_API_URL/openapi.json" >/dev/null 2>&1; then
    pass "Lean Server: $LEAN_SERVER_API_URL"
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

printf '\nSummary: %d passed, %d warnings, %d failures\n' "$PASS" "$WARN" "$FAIL"
if (( FAIL > 0 )); then exit 1; fi
if (( WARN > 0 )); then exit 2; fi
exit 0

