#!/usr/bin/env bash
set -u

# Guarded P3 entry point. This script deliberately refuses to start until the
# P2.5 Local RL Readiness manifest exists (promptset reward evidence, GRPO loss
# rehearsal, LoRA step probe, config audit). It also uses conservative
# single-GPU hypotheses rather than the upstream 8-GPU recipe unchanged.
# Provisional P3-A runner: run scripts/run_p3_smoke.sh only after P3-0 (Linux
# migration & on-policy calibration) has frozen the P3-A config; see
# docs/p3_linux_handoff.md.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/env.sh" >/dev/null

GATE_FILE="${TINYLEAN_P2_5_GATE_FILE:-$ROOT/experiments/manifests/p2_5_complete.yaml}"
if [[ ! -f "$GATE_FILE" ]]; then
  echo "P3 blocked: create $GATE_FILE only after all P2.5 stop conditions are met" >&2
  echo "(see docs/studies/rl_readiness.md)." >&2
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

echo "P3 gate passed; use configs/rl/kimina_0.6b_pilot.yaml to review the provisional overrides."
echo "P3-0 (Linux migration & on-policy calibration) comes first; see docs/p3_linux_handoff.md."
echo "The provisional P3-A smoke runs only after P3-0 freezes the config:"
echo "  bash scripts/run_p3_smoke.sh --steps 3   # not yet Linux-validated"
exit 0

