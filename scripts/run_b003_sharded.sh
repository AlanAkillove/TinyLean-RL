#!/usr/bin/env bash
# V2-B003 concurrent shards wrapper: 4 disjoint rank shards, each process
# single-path batch=1 with the frozen parameters (the same concurrency
# semantics verified token-identical 8/8 in the B002 amendment).
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
LOG=experiments/results/v2_b003_run.log
mkdir -p experiments/results/shards
echo "[B003-shards] $(date -u +%FT%TZ) launching 4 shards x 48 theorems x K=8" >> "$LOG"
pids=()
for i in 0 1 2 3; do
    "$PY" scripts/v2_b003_generate.py \
        --shard-index "$i" --shard-count 4 \
        --output "experiments/results/shards/v2_b003_rollouts.shard${i}.jsonl" \
        --meta "experiments/results/shards/v2_b003_run_meta.shard${i}.jsonl" \
        > "experiments/results/shards/b003_shard${i}.log" 2>&1 &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then status=1; fi
done
if [ "$status" -ne 0 ]; then
    echo "[B003-shards] $(date -u +%FT%TZ) at least one shard FAILED; not merging; rerun to resume" >> "$LOG"
    exit 1
fi
"$PY" scripts/merge_b003_shards.py >> "$LOG" 2>&1
echo "[B003-shards] $(date -u +%FT%TZ) all shards complete; merged" >> "$LOG"
