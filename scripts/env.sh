#!/usr/bin/env bash
set -u

# Source this file from any working directory:
#   source scripts/env.sh

TINYLEAN_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TINYLEAN_ROOT="$(cd "$TINYLEAN_SCRIPT_DIR/.." && pwd)"

export HF_HOME="${HF_HOME:-$TINYLEAN_ROOT/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$TINYLEAN_ROOT/.cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$TINYLEAN_ROOT/.cache/uv}"
export TORCH_HOME="${TORCH_HOME:-$TINYLEAN_ROOT/.cache/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$TINYLEAN_ROOT/.cache/triton}"
export RAY_TMPDIR="${RAY_TMPDIR:-$TINYLEAN_ROOT/.cache/ray}"
export WANDB_DIR="${WANDB_DIR:-$TINYLEAN_ROOT/runs/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$TINYLEAN_ROOT/.cache/wandb}"

export TINYLEAN_MODEL_ROOT="${TINYLEAN_MODEL_ROOT:-$TINYLEAN_ROOT/models/weights}"
export TINYLEAN_DATA_ROOT="${TINYLEAN_DATA_ROOT:-$TINYLEAN_ROOT/data/raw}"
export LEAN_SERVER_API_URL="${LEAN_SERVER_API_URL:-http://127.0.0.1:8000}"
export PYTHONPATH="$TINYLEAN_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$UV_CACHE_DIR" "$TORCH_HOME" \
  "$TRITON_CACHE_DIR" "$RAY_TMPDIR" "$WANDB_DIR" "$WANDB_CACHE_DIR" \
  "$TINYLEAN_MODEL_ROOT" "$TINYLEAN_DATA_ROOT" "$TINYLEAN_ROOT/runs"

echo "TinyLean-RL environment loaded: $TINYLEAN_ROOT"
