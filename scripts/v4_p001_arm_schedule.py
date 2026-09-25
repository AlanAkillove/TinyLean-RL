#!/usr/bin/env python3
"""V4-P001 Amendment A §7 — freeze the balanced arm-order schedule and its hash.

The schedule is a pure function of the formal theorem rank (1..128): no source, error category,
diagnostic or outcome enters it. It is frozen here, in the same pre-outcome state as the seeds, so
that the execution order of the four candidates of a theorem cannot be chosen after any second-stage
outcome exists.

Output: experiments/manifests/v4/v4_p001_arm_schedule.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tinylean_rl.evaluation.v4_schedule import (
    ARM_ORDER,
    N_PRIMARY,
    POSITIONS,
    balance_report,
    canonical_json,
    schedule,
    schedule_hash,
)

OUT = "experiments/manifests/v4/v4_p001_arm_schedule.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=OUT)
    args = parser.parse_args()

    rows = schedule()
    balance = balance_report()
    checks = {
        "exactly_one_candidate_per_arm_per_theorem": all(
            sorted(row["order"]) == sorted(ARM_ORDER) for row in rows
        ),
        "formal_ranks_contiguous_1_to_128": [row["formal_rank"] for row in rows]
        == list(range(1, N_PRIMARY + 1)),
        "position_counts_exactly_balanced": all(
            counts == [N_PRIMARY // POSITIONS] * POSITIONS
            for counts in balance["positions_per_arm"].values()
        ),
        "predecessor_counts_within_three_of_ideal": (
            balance["predecessor_min"] >= N_PRIMARY * 3 // 12 - 2
            and balance["predecessor_max"] <= N_PRIMARY * 3 // 12 + 1
        ),
    }
    failing = [name for name, ok in checks.items() if not ok]
    if failing:
        raise SystemExit(f"schedule balance failed: {failing}")

    artifact = {
        "artifact_type": "v4_p001_arm_order_schedule",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": subprocess.run(["hostname"], capture_output=True, text=True, check=False).stdout.strip(),
        "git_revision": subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip(),
        "note": (
            "frozen before any V4 formal generation; the schedule is a function of the formal theorem "
            "rank alone and is stored so that arm identity cannot be confounded with temporal or "
            "server state at analysis time"
        ),
        "amendment": "Amendment A §7 (owner decision, owner review of V4-P001 preregistration)",
        "n_primary": N_PRIMARY,
        "positions": POSITIONS,
        "arms": list(ARM_ORDER),
        "algorithm": {
            "block_1_120": "the 24 permutations of the four arms in lexicographic order, five complete cycles",
            "block_121_128": "the eight rotations of (A,B,C,D) and of (A,B,D,C) — two Latin squares",
            "why": (
                "the 120-row block is exactly position- and predecessor-balanced; the two Latin squares "
                "restore the per-position counts to exactly 32 per arm"
            ),
            "outcome_dependence": "none: formal theorem rank only",
        },
        "schedule": rows,
        "balance": balance,
        "schedule_hash": schedule_hash(),
        "schedule_canonical_sha256_cross_check": hashlib.sha256(
            canonical_json(rows).encode("utf-8")
        ).hexdigest(),
        "producer_sha256": sha256_file(ROOT / "src/tinylean_rl/evaluation/v4_schedule.py"),
        "checks": checks,
    }
    destination = ROOT / args.out
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact, indent=2) + "\n")
    print("schedule_hash:", artifact["schedule_hash"])
    print("positions per arm:", json.dumps(balance["positions_per_arm"]))
    print("predecessor pairs:", balance["predecessor_min"], "-", balance["predecessor_max"])
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
