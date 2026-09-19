#!/usr/bin/env bash
# V2-A001 - Track-A checkpoint-selection evaluation (fly90 canonical side).
#
# Evaluates the 7 physically available models on the frozen A3-primary set
# (512 theorems x 4 samples, canonical seed schedule, strict Kimina 2.0.0),
# sequentially, one model per evaluator invocation with --resume so that
# chunk-level partial states survive kills.
#
# Preconditions enforced here (fail-close):
#   - GPU has no other compute process;
#   - Lean server healthy + warm probe passes;
#   - container memory cap is the fixed 40 GiB instrumentation value;
#   - the frozen selection set hash matches the V2-A001 preregistration.
#
# Bounded retries: each model gets at most 2 attempts; two consecutive failed
# attempts of the same model stop the whole run (same-cause rule).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

LOG="$ROOT/.cache/v2_a001.log"
SET="$ROOT/experiments/manifests/v2/v2_a001_selection_set.json"
SET_SHA="f429ddd7c628c3ce36b9ee7309c1fc68c819a0e03d20b38f5fd44112d7853e86"
CAP_EXPECTED=42949672960 # 40 GiB
PY="$ROOT/.venv/bin/python"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }
fail() { log "V2-A001 blocked: $1"; exit 2; }

log "=== V2-A001 start ==="

busy="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)"
(( busy == 0 )) || fail "GPU busy ($busy compute process(es))"

curl --silent --show-error --fail --max-time 5 "${LEAN_SERVER_API_URL:-http://127.0.0.1:8000}/health" >/dev/null \
  || fail "Lean server unhealthy"

CAP_NOW="$(docker inspect tinylean-rl-lean-server --format '{{.HostConfig.Memory}}' 2>/dev/null || echo missing)"
[[ "$CAP_NOW" == "$CAP_EXPECTED" ]] \
  || fail "container cap is '$CAP_NOW', expected 40 GiB ($CAP_EXPECTED); adjust with docker update, never compose recreate"

[[ -f "$SET" ]] || fail "selection set missing: $SET"
SHA_NOW="$(sha256sum "$SET" | awk '{print $1}')"
[[ "$SHA_NOW" == "$SET_SHA" ]] || fail "selection set hash mismatch ($SHA_NOW, expected $SET_SHA)"

log "lean warm probe"
PYTHONPATH="$ROOT/src" "$PY" - <<'PY' || fail "Lean warm probe failed"
import sys

from tinylean_rl.verifier.kimina import verify_code

response = verify_code(
    "import Mathlib\ntheorem tinylean_probe : 1 + 1 = 2 := by norm_num",
    custom_id="v2-a001-warm-probe",
    timeout=300,
)
assert "results" in response, response
print("  lean warm probe: verified", file=sys.stderr)
PY

MODELS=(base step10 step20 step30 seed1_step60 seed2_step60 seed3_step60)
for model in "${MODELS[@]}"; do
  out="$ROOT/experiments/results/v2_a001_${model}.json"
  attempt=1
  while (( attempt <= 2 )); do
    log "model=$model attempt=$attempt -> $out"
    "$PY" scripts/p3c_checkpoint_eval.py \
      --checkpoint "$model" \
      --fixed-set "$SET" \
      --samples-per-theorem 4 \
      --temperature 1.0 --top-p 1.0 --max-new-tokens 4096 \
      --chunk-theorems 16 \
      --verify-workers 4 --verify-batch-size 4 \
      --resume \
      --output "$out" \
      --offline 2>&1 | tee -a "$LOG"
    rc=${PIPESTATUS[0]}
    if (( rc == 0 )); then
      log "model=$model OK"
      break
    fi
    log "model=$model attempt=$attempt failed rc=$rc"
    if (( attempt == 2 )); then
      fail "model=$model failed twice in a row (same-cause stop)"
    fi
    attempt=$((attempt + 1))
  done
done

log "=== V2-A001 evaluation complete: all 7 models evaluated ==="
