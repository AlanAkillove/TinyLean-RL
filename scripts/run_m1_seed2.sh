#!/usr/bin/env bash
# M1 seed2 replication runner (E020): fresh 60-step GRPO run from Distill with
# an independent RNG seed. Config identical to P3-B/E019 except the seed knob
# audited in docs/seed_control_audit.md:
#   +data.seed=20260918 (dataloader shuffle generator -> determines the prompt
#   sequence per step; the FSDP rollout path has NO configurable engine seed -
#   vLLM is seeded 0 for every run and diverges through the different request
#   streams, which the live sanity check verifies empirically).
# Pre-registered in experiments/manifests/m1_seed_replication.yaml; the seed
# value must NOT be changed based on results. Runs in an isolated directory
# runs/m1_seed2 (fresh start, no resume); save_freq 10 keeps checkpoints
# 10..60 (cleanup is lazy in this VERL build, as observed in E019).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/env.sh" >/dev/null

SEED2=20260918
RUN_DIR="$ROOT/runs/m1_seed2"
fail() { echo "seed2 replication blocked: $1" >&2; exit 2; }

if [[ -e "$RUN_DIR" && -n "$(ls -A "$RUN_DIR" 2>/dev/null)" ]]; then
  fail "$RUN_DIR already exists and is non-empty - use a fresh directory or resume explicitly"
fi

curl --silent --show-error --fail --max-time 5 "${LEAN_SERVER_API_URL:-http://127.0.0.1:8000}/health" >/dev/null \
  || fail "Lean server unhealthy"
CAP_NOW="$(docker inspect tinylean-rl-lean-server --format '{{.HostConfig.Memory}}' 2>/dev/null || echo missing)"
[[ "$CAP_NOW" == "42949672960" ]] || fail "container cap is '$CAP_NOW', expected 40 GiB"

echo "[E020 seed2] fresh 60-step replication, data.seed=$SEED2"
echo "  config: frozen P3-B recipe; only the dataloader seed differs (see audit)"
echo "  dir:    runs/m1_seed2 (fresh)"
echo "  sanity: after step 1, compare runs/m1_seed2/rollout_data/1.jsonl statement"
echo "          sequence against runs/p3b_pilot/rollout_data/1.jsonl - must differ"

exec bash "$SCRIPT_DIR/run_p3_pilot.sh" --steps 60 --n 8 --dir "$RUN_DIR" --skip-prewarm \
  +data.seed="$SEED2" \
  trainer.experiment_name=m1-seed2 "$@"
