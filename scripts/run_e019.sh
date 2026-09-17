#!/usr/bin/env bash
# E019 runner - M1 Step30 -> Step60 training extension (bounded resume).
#
# Frozen P3-B configuration, no algorithm changes: this runner only verifies
# the resume preconditions and hands off to run_p3_pilot.sh with the TOTAL
# target of 60 steps. VERL's resume_mode=auto (default) restores the latest
# checkpoint in --dir, so the run continues at global_step 31 and stops at
# global_step 60 (E015 validated the same mechanism for global_step_3 -> 4).
#
# Preconditions enforced here (protocol sections VI and XI):
#   - runs/p3b_pilot/global_step_30 exists and IS the latest checkpoint;
#   - Lean server healthy (HTTP /health) AND warm (single trivial verify);
#   - container memory cap is the fixed 40 GiB verifier instrumentation value.
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

RUN_DIR="$ROOT/runs/p3b_pilot"
RESUME_CKPT="$RUN_DIR/global_step_30"
TARGET_STEPS=60
CAP_EXPECTED=42949672960 # 40 GiB

fail() { echo "E019 blocked: $1" >&2; exit 2; }

[[ -d "$RESUME_CKPT" ]] || fail "resume checkpoint missing: $RESUME_CKPT"
LATEST="$(ls -d "$RUN_DIR"/global_step_* 2>/dev/null | sed 's/.*global_step_//' | sort -n | tail -1)"
[[ "$LATEST" == "30" ]] || fail "latest checkpoint is global_step_$LATEST, expected 30 (already extended?)"

curl --silent --show-error --fail --max-time 5 "${LEAN_SERVER_API_URL:-http://127.0.0.1:8000}/health" >/dev/null \
  || fail "Lean server unhealthy"

CAP_NOW="$(docker inspect tinylean-rl-lean-server --format '{{.HostConfig.Memory}}' 2>/dev/null || echo missing)"
[[ "$CAP_NOW" == "$CAP_EXPECTED" ]] \
  || fail "container cap is '$CAP_NOW', expected 40 GiB ($CAP_EXPECTED); adjust with docker update, never compose recreate"

PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" - <<'PY' || fail "Lean warm probe failed"
import sys

from tinylean_rl.verifier.kimina import verify_code

response = verify_code(
    "import Mathlib\ntheorem tinylean_probe : 1 + 1 = 2 := by norm_num",
    custom_id="e019-warm-probe",
    timeout=300,
)
assert "results" in response, response
print("  lean warm probe: verified")
PY

echo "[E019] M1 Step30 -> Step60 training extension (frozen P3-B config)"
echo "  resume global_step = 30"
echo "  target global_step = 60"
echo "  checkpoints: global_step_40 / 50 / 60 (save_freq 10; last 3 kept)"
echo "  watch: VERL 'Training Progress' bar (live %, step, ETA) in this terminal;"
echo "         rollout dumps land in runs/p3b_pilot/rollout_data as 31..60.jsonl"

exec bash "$SCRIPT_DIR/run_p3_pilot.sh" --steps "$TARGET_STEPS" --n 8 --dir "$RUN_DIR" --skip-prewarm \
  trainer.experiment_name=e019-step60-extension "$@"
