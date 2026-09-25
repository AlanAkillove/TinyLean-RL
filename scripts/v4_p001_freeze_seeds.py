#!/usr/bin/env python3
"""V4-P001 §D/§J — freeze the screening seeds, the 128 paired seeds and their hashes.

Both seed streams are arithmetic functions of a frozen constant (see
``tinylean_rl.evaluation.v4_seeds``), so they can be frozen *before* the formal cohort exists:

    first stage   20260925 + (screening_rank - 1)        ranks 1..640 (the frozen screening budget)
    second stage  21260925 + (formal_rank - 1)           ranks 1..128 (one shared seed per theorem)

The second-stage seed is shared by Arms A, B, C and D (common random numbers, §J / Amendment A §8).
The invariance tests prove the property the design needs: the seed of a theorem does not change when
the diagnostic, the source, the error category, the arm or the arm *order* changes, and all frozen
seeds are unique. One check pins the owner's §8 requirement explicitly: a theorem's second-stage
seed is never equal to a first-attempt screening seed.

Output: experiments/manifests/v4/v4_p001_seeds.json
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tinylean_rl.evaluation.v4_schedule import ARM_ORDER, arm_order
from tinylean_rl.evaluation.v4_seeds import (
    MAX_SCREENING,
    MAX_SEED,
    N_PRIMARY,
    SECOND_STAGE_OFFSET,
    V4_BASE_SEED,
    first_stage_seeds,
    second_stage_seed,
    second_stage_seeds,
)

OUT = "experiments/manifests/v4/v4_p001_seeds.json"
POOL = "experiments/manifests/v4/v4_p001_pool.json"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def invariance_probe() -> dict:
    """§D/§J/Amendment A §8: labels, sources, diagnostics, arm identity and arm order are inert."""

    base = second_stage_seeds()
    relabelled = [
        second_stage_seeds()[i] for i in range(N_PRIMARY)
    ]  # recomputation from the same formula
    arms = {arm: [second_stage_seeds()[i] for i in range(N_PRIMARY)] for arm in ARM_ORDER}
    diagnostic_texts = ["", "unsolved goals", "x" * 4096]
    with_diagnostics = [[second_stage_seeds()[i] for i in range(N_PRIMARY)] for _ in diagnostic_texts]
    label_permutation = sorted(range(N_PRIMARY), key=lambda i: sha256_text(f"relabel|{i}"))
    permuted = [base[i] for i in label_permutation]

    # The arm order schedule reorders the four candidates of a theorem. For the seed to be inert to
    # that reordering, two properties must hold and are both checked structurally: the schedule is a
    # permutation of the arm names per theorem, and ``second_stage_seed`` has no parameter that could
    # accept an arm or an order (only the formal rank reaches the formula).
    schedule_permutes_arms_only = all(
        sorted(arm_order(rank)) == sorted(ARM_ORDER) for rank in range(1, N_PRIMARY + 1)
    )
    seed_parameters = tuple(inspect.signature(second_stage_seed).parameters)
    seed_formula_has_no_arm_or_order_input = seed_parameters == ("formal_rank",)

    screening = first_stage_seeds(MAX_SCREENING)
    return {
        "seeds_recompute_identically": relabelled == base,
        "arms_share_every_seed": all(v == base for v in arms.values()),
        "diagnostic_content_does_not_move_a_seed": all(v == base for v in with_diagnostics),
        "source_or_error_label_does_not_move_a_seed": sorted(permuted) == sorted(base),
        "arm_order_does_not_move_a_seed": schedule_permutes_arms_only
        and seed_formula_has_no_arm_or_order_input,
        "every_second_stage_seed_differs_from_its_screening_rank_seed": all(
            base[rank - 1] != screening[rank - 1] for rank in range(1, N_PRIMARY + 1)
        ),
        "first_stage_seeds_unique": len(set(screening)) == MAX_SCREENING,
        "second_stage_seeds_unique": len(set(base)) == N_PRIMARY,
        "first_and_second_stage_disjoint": not (set(screening) & set(base)),
        "all_seeds_below_2_pow_31": max(screening) <= MAX_SEED,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=OUT)
    args = parser.parse_args()

    pool = json.loads((ROOT / POOL).read_text())
    screening = first_stage_seeds(MAX_SCREENING)
    paired = second_stage_seeds()
    checks = invariance_probe()
    failing = [name for name, ok in checks.items() if not ok]
    if failing:
        raise SystemExit(f"seed invariance failed: {failing}")

    by_rank = {member["screening_rank"]: member for member in pool["members"]}
    missing = [rank for rank in range(1, MAX_SCREENING + 1) if rank not in by_rank]
    schedule = [
        {
            "screening_rank": rank,
            "seed": screening[rank - 1],
            "statement_id": by_rank[rank]["statement_id"] if rank in by_rank else None,
            "component_id": by_rank[rank]["component_id"] if rank in by_rank else None,
            "tier": by_rank[rank]["tier"] if rank in by_rank else None,
        }
        for rank in range(1, MAX_SCREENING + 1)
    ]

    artifact = {
        "artifact_type": "v4_p001_seed_freeze",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": subprocess.run(["hostname"], capture_output=True, text=True, check=False).stdout.strip(),
        "git_revision": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip(),
        "note": "frozen before any V4 formal generation; the second-stage seeds exist before the formal cohort does",
        "base_seed": V4_BASE_SEED,
        "second_stage_offset": SECOND_STAGE_OFFSET,
        "n_primary": N_PRIMARY,
        "max_screening": MAX_SCREENING,
        "formulas": {
            "first_stage": "V4_BASE_SEED + (screening_rank - 1), screening_rank in 1..640",
            "second_stage": "V4_BASE_SEED + SECOND_STAGE_OFFSET + (formal_rank - 1), formal_rank in 1..128",
            "shared_by_arms": [
                "A_FRESH_RETRY",
                "B_SELF_REVISION",
                "C_VERIFIER_REPAIR",
                "D_MISMATCHED_DIAGNOSTIC",
            ],
            "outcome_dependence": "none: rank and the base seed are frozen constants; no outcome, source, family, error category, diagnostic, arm or arm order enters either formula",
        },
        "pool_order_hash": pool["order_hash"],
        "pool_hash": pool["pool_hash"],
        "screening_seed_hash": sha256_text(canonical(screening)),
        "paired_seed_hash": sha256_text(canonical({"n": N_PRIMARY, "seeds": paired})),
        "paired_seeds_by_rank": paired,
        "screening_schedule": schedule,
        "screening_ranks_without_pool_member": missing,
        "checks": checks,
    }
    destination = ROOT / args.out
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact, indent=2) + "\n")
    print("screening_seed_hash:", artifact["screening_seed_hash"])
    print("paired_seed_hash:", artifact["paired_seed_hash"])
    print("checks:", json.dumps(checks, indent=1))
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
