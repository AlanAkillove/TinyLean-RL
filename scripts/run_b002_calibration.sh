#!/usr/bin/env bash
# V2-B002 pipeline-calibration replay (diagnostic-only): 4 shards of the frozen
# 64-theorem E023 subset, run through the unmodified B002 runner.
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
SET=experiments/manifests/v2/v2_b002_calibration_set.json
OUTDIR=experiments/results/calibration
LOG=experiments/results/v2_b002_calibration_run.log
mkdir -p "$OUTDIR"
echo "[B002-cal] $(date -u +%FT%TZ) launching 4 shards (diagnostic replay)" >> "$LOG"
pids=()
for i in 0 1 2 3; do
    "$PY" scripts/v2_b002_generate.py --pilot "$SET" \
        --output "$OUTDIR/cal.shard${i}.jsonl" \
        --meta "$OUTDIR/cal.meta.shard${i}.jsonl" \
        --shard-index "$i" --shard-count 4 \
        > "$OUTDIR/shard${i}.log" 2>&1 &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then status=1; fi
done
if [ "$status" -ne 0 ]; then
    echo "[B002-cal] $(date -u +%FT%TZ) at least one shard FAILED" >> "$LOG"
    exit 1
fi
"$PY" - << 'EOF' >> "$LOG" 2>&1
import json
from pathlib import Path

out = Path("experiments/results/calibration")
recs = []
for i in range(4):
    recs += [json.loads(l) for l in (out / f"cal.shard{i}.jsonl").read_text().splitlines() if l.strip()]
recs.sort(key=lambda r: r["theorem_rank"])
assert len(recs) == 64, f"expected 64 records, got {len(recs)}"
body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs)
Path("experiments/results/v2_b002_calibration_rollouts.jsonl").write_text(body)
print("[B002-cal] merged 64 records -> v2_b002_calibration_rollouts.jsonl")
EOF
echo "[B002-cal] $(date -u +%FT%TZ) all shards complete" >> "$LOG"
