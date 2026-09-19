#!/usr/bin/env bash
# V2-B002 concurrent shards wrapper (concurrency amendment 2026-09-20).
# 4 disjoint rank shards, each process single-path batch=1 with the frozen
# parameters (token-level equivalence verified by v2_b002_concurrency_check.py).
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
LOG=experiments/results/v2_b002_run.log
mkdir -p experiments/results/shards
echo "[B002-shards] $(date -u +%FT%TZ) launching 4 shards (concurrency amendment)" >> "$LOG"
"$PY" scripts/prepare_b002_shards.py >> "$LOG" 2>&1
pids=()
for i in 0 1 2 3; do
    "$PY" scripts/v2_b002_generate.py \
        --shard-index "$i" --shard-count 4 \
        --output "experiments/results/shards/v2_b002_rollouts.shard${i}.jsonl" \
        --meta "experiments/results/shards/v2_b002_run_meta.shard${i}.jsonl" \
        > "experiments/results/shards/shard${i}.log" 2>&1 &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then status=1; fi
done
if [ "$status" -ne 0 ]; then
    echo "[B002-shards] $(date -u +%FT%TZ) at least one shard FAILED; not merging; rerun to resume" >> "$LOG"
    exit 1
fi
"$PY" scripts/merge_b002_shards.py >> "$LOG" 2>&1
echo "[B002-shards] $(date -u +%FT%TZ) all shards complete; merged" >> "$LOG"
