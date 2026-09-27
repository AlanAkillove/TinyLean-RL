#!/usr/bin/env bash
# R001 engineering smokes S0-S3 on the formal node (owner directive §10-§14; draft §10).
#
# STATUS: engineering feasibility only. Consumed training prompts only, no capability holdout,
# no sealed reserve, no formal scientific result, no checkpoints (trainer.save_freq=-1). S0-S3
# completion does NOT authorize R001; both arms of the formal run restart from theta0.
#
# Command chain (fly122, RTX 3080 10 GB):
#   source scripts/env.sh
#   docker start tinylean-rl-lean-oracle-v5        # the frozen 2.0.0 oracle on 127.0.0.1:8020
#   bash scripts/run_v5_r001.sh s0                 # config/static audit, both arms, no GPU
#   bash scripts/run_v5_r001.sh s1 --steps 1       # memory/OOM probe; retries: one knob at a time
#   bash scripts/run_v5_r001.sh s2                 # arm parity, 1 step per condition
#   bash scripts/run_v5_r001.sh s3                 # online mapping dry run + oracle archive
#
# Gates: Linux, Docker daemon + frozen oracle image digest, 0.6B model directory, Promptset
# train parquet, oracle /health, `import verl`, and (S1-S3) no leftover python process on the
# GPU. S0 additionally requires no GPU at all. Artifacts land in
# experiments/results/v5_r001_smoke/<stage>/; the stage summary is printed as one
# `R001_SMOKE {json}` line.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/env.sh" >/dev/null

PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  if [[ -x "$ROOT/.venv/bin/python" ]]; then
    PYTHON="$ROOT/.venv/bin/python"
  else
    PYTHON="python3"
  fi
fi

RECIPE_DIR="$ROOT/third_party/kimina-prover-rl/recipe/kimina_prover_rl"
TRAIN_PARQUET="$ROOT/data/processed/p3_promptset/prompt_sets/AI-MO/Kimina-Prover-Promptset/train.parquet"
MODEL_PATH="$TINYLEAN_MODEL_ROOT/kimina_distill_0_6b"
ORACLE_URL="${TINYLEAN_R001_ORACLE_ENDPOINT:-http://127.0.0.1:8020}"
ORACLE_CONTAINER="tinylean-rl-lean-oracle-v5"
ORACLE_IMAGE="projectnumina/kimina-lean-server:2.0.0"
ORACLE_DIGEST="sha256:588a2cbbd10da509ed13f53ac136f8463fabff02dfe4eca535e7c47ae6e3ffd9"
OUT_DIR="$ROOT/experiments/results/v5_r001_smoke"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_v5_r001.sh STAGE [stage options...]

  STAGE = s0 | s1 | s2 | s3
    s0                          configuration/static audit, both arms, no GPU
    s1 [--arm control|treatment] [--steps 1..3] [--knob KEY=VALUE ...]
    s2 [--knob KEY=VALUE ...]   arm parity, <= 1 optimizer step per condition
    s3 [--arm control|treatment] [--knob KEY=VALUE ...]   online mapping dry run

Also accepted by every stage: --out-dir DIR, --timeout-s N, --dry, --json.
Any `--knob` change between S1 attempts must touch exactly one allowed memory knob
(owner §12); the driver enforces this and records the diff table.
EOF
}

[[ $# -ge 1 ]] || { usage >&2; exit 2; }
STAGE="$1"; shift
case "$STAGE" in
  s0|s1|s2|s3) ;;
  -h|--help) usage; exit 0 ;;
  *) echo "Unknown stage: $STAGE" >&2; usage >&2; exit 2 ;;
esac

DRY=0
for arg in "$@"; do
  [[ "$arg" == "--dry" ]] && DRY=1
done

fail() { echo "R001 smoke blocked: $1" >&2; exit 2; }

if (( ! DRY )); then
  # --- Gates ----------------------------------------------------------------
  [[ "$(uname -s)" == "Linux" ]] || fail "requires Linux (detected $(uname -s))"
  if ! command -v docker >/dev/null 2>&1 || ! docker info >/dev/null 2>&1; then
    fail "Docker daemon unavailable; the frozen Lean oracle must run in Docker."
  fi
  [[ -d "$MODEL_PATH" ]] || fail "model missing: $MODEL_PATH"
  [[ -f "$TRAIN_PARQUET" ]] || fail "consumed training parquet missing: $TRAIN_PARQUET (run scripts/run_p3_smoke.sh once to prepare it)"
  [[ -f "$RECIPE_DIR/kimina_prover_rl/dataset.py" ]] \
    || fail "pinned recipe missing; run: git submodule update --init --recursive"
  if ! "$PYTHON" -c 'import verl' >/dev/null 2>&1; then
    fail "VERL is not importable with '$PYTHON'"
  fi
  if [[ "$STAGE" != "s0" ]]; then
    if ! docker ps --format '{{.Names}}' | grep -qx "$ORACLE_CONTAINER"; then
      fail "oracle container '$ORACLE_CONTAINER' is not running (docker start $ORACLE_CONTAINER)"
    fi
    digest="$(docker image inspect "$ORACLE_IMAGE" --format '{{index .RepoDigests 0}}' 2>/dev/null || true)"
    if [[ -n "$digest" && "$digest" != *"${ORACLE_DIGEST#sha256:}"* ]]; then
      fail "oracle image digest $digest does not match the frozen $ORACLE_DIGEST"
    fi
    if ! curl --silent --show-error --fail --max-time 5 "$ORACLE_URL/health" >/dev/null 2>&1; then
      fail "frozen oracle unreachable at $ORACLE_URL/health"
    fi
  fi
fi

export PYTHONPATH="$ROOT/src:$ROOT/scripts:$RECIPE_DIR${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export LEAN_SERVER_API_URL="${LEAN_SERVER_API_URL:-$ORACLE_URL}"
export TINYLEAN_R001_ORACLE_ENDPOINT="$ORACLE_URL"

echo "R001 smoke stage=$STAGE host=$(uname -n) oracle=$ORACLE_URL out=$OUT_DIR"
exec "$PYTHON" "$ROOT/scripts/v5_r001_smoke.py" "$STAGE" --out-dir "$OUT_DIR" "$@"
