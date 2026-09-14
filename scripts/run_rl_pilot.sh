#!/usr/bin/env bash
set -u

# Guarded P3 entry point. This script deliberately refuses to start until the
# human/research log records that P2 evaluation and the Linux verifier gate have
# passed. It also uses conservative single-GPU hypotheses rather than the
# upstream 8-GPU recipe unchanged.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/env.sh" >/dev/null

GATE_FILE="${TINYLEAN_P2_GATE_FILE:-$ROOT/experiments/manifests/p2_passed}"
if [[ ! -f "$GATE_FILE" ]]; then
  echo "P3 blocked: create $GATE_FILE only after P2 evaluator and verifier checks pass." >&2
  exit 2
fi

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "P3 requires Linux; refusing to start training on $(uname -s)." >&2
  exit 2
fi

if ! command -v docker >/dev/null 2>&1 || ! docker info >/dev/null 2>&1; then
  echo "P3 requires a running Docker daemon for Kimina Lean Server." >&2
  exit 2
fi

if [[ ! -d "$TINYLEAN_MODEL_ROOT/kimina_distill_0_6b" ]]; then
  echo "P3 model missing: $TINYLEAN_MODEL_ROOT/kimina_distill_0_6b" >&2
  exit 2
fi

if [[ ! -f "$ROOT/data/raw/kimina_promptset/data/train-00000-of-00001.parquet" ]]; then
  echo "P3 promptset missing: data/raw/kimina_promptset/data/train-00000-of-00001.parquet" >&2
  exit 2
fi

echo "P3 gate passed; use configs/rl/kimina_0.6b_pilot.yaml to review overrides."
echo "The actual trainer invocation is intentionally not automated until the pilot configuration is reviewed."
exit 0

