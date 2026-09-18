#!/usr/bin/env bash
# M1 seed3 replication runner (E022): fresh 60-step GRPO run from Distill with
# the third pre-registered seed. Rules identical to seed2
# (experiments/manifests/m1_seed_replication.yaml): the only seed knob is
# +data.seed=20260919 (FSDP rollout engine seed is not configurable in this
# pinned build); fresh start, isolated dir runs/m1_seed3; no outcome-dependent
# stopping. Run only if the host stayed stable - checked by the caller.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/env.sh" >/dev/null

SEED3=20260919
RUN_DIR="$ROOT/runs/m1_seed3"
fail() { echo "seed3 replication blocked: $1" >&2; exit 2; }

if [[ -e "$RUN_DIR" && -n "$(ls -A "$RUN_DIR" 2>/dev/null)" ]]; then
  fail "$RUN_DIR already exists and is non-empty - use a fresh directory or resume explicitly"
fi

curl --silent --show-error --fail --max-time 5 "${LEAN_SERVER_API_URL:-http://127.0.0.1:8000}/health" >/dev/null \
  || fail "Lean server unhealthy"
CAP_NOW="$(docker inspect tinylean-rl-lean-server --format '{{.HostConfig.Memory}}' 2>/dev/null || echo missing)"
[[ "$CAP_NOW" == "42949672960" ]] || fail "container cap is '$CAP_NOW', expected 40 GiB"
free_gib="$(free -g | awk '/^Mem:/{print $7}')"
(( free_gib >= 8 )) || fail "only ${free_gib} GiB available memory - host unstable for a new run"

echo "[E022 seed3] fresh 60-step replication, data.seed=$SEED3"
echo "  pre-registered in experiments/manifests/m1_seed_replication.yaml (rules identical to seed2)"
echo "  dir: runs/m1_seed3 (fresh); sanity: step-1 statements must differ from seed1/seed2"

exec bash "$SCRIPT_DIR/run_p3_pilot.sh" --steps 60 --n 8 --dir "$RUN_DIR" --skip-prewarm \
  +data.seed="$SEED3" \
  trainer.experiment_name=m1-seed3 "$@"
