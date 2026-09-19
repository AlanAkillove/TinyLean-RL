#!/usr/bin/env python
"""V2-B002 analysis: compute-response heterogeneity + Hindsight Oracle screening.

Reads the pilot rollouts (experiments/results/v2_b002_rollouts.jsonl) and reports
the four evidence blocks the V2-B002 go/no-go memo needs:

1. five-bit response patterns  (y512, y1024, y2048, y3072, y4096) with descriptive
   categories (easy / compute-sensitive / hopeless-at-4096 / late-success);
2. monotonic vs non-monotonic trajectories - no automatic "fixing"; non-monotonic
   cases are listed with their per-budget verification status for manual forensics;
3. pathwise Hindsight Oracle frontier vs uniform allocation, on TWO x-axes kept
   separate: (a) allocated max-token cap, (b) estimated actually-generated tokens;
   realized generation cost (tokens, generated seconds, wall estimate) is reported
   beside them; the oracle knows the realized outcome of the SAME trajectory and
   is a screening upper bound only - it is NOT achievable by any deployable policy;
4. verification-health summary (status by budget, infrastructure outcomes, runtimes).

No decision model, no XGBoost/MLP, no training: descriptive statistics only.
Output: experiments/results/v2_b002_analysis.json (gitignored raw + summary).
"""

from __future__ import annotations

import argparse
import bisect
import collections
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUDGETS = (512, 1024, 2048, 3072, 4096)
INFRA_STATUSES = {
    "verifier_timeout",
    "verifier_server_error",
    "verifier_unhealthy",
    "unresolved_infra_error",
}
CAP_GRID_STEP = 512


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision() -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()


def load_records(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def pattern_of(record: dict) -> str:
    by_budget = {entry["budget"]: entry for entry in record["prefixes"]}
    return "".join("1" if by_budget[b]["verified"] else "0" for b in BUDGETS)


def first_success(pattern: str):
    index = pattern.find("1")
    return None if index < 0 else BUDGETS[index]


def is_monotonic(pattern: str) -> bool:
    seen_success = False
    for char in pattern:
        if char == "1":
            seen_success = True
        elif seen_success:
            return False
    return True


def analyze_patterns(records: list[dict]) -> dict:
    patterns = [pattern_of(record) for record in records]
    counts = collections.Counter(patterns)
    first_success_dist = collections.Counter(
        "never" if first_success(p) is None else str(first_success(p)) for p in patterns
    )
    categories = {
        "easy_success_at_512": sum(1 for p in patterns if p[0] == "1"),
        "compute_sensitive": sum(1 for p in patterns if p[0] == "0" and "1" in p[1:]),
        "hopeless_at_4096": sum(1 for p in patterns if p[-1] == "0"),
        "always_success": sum(1 for p in patterns if p == "11111"),
        "never_success": sum(1 for p in patterns if p == "00000"),
        "late_success_ge_3072": sum(
            1 for p in patterns if first_success(p) in (BUDGETS[3], BUDGETS[4])
        ),
    }
    return {
        "pattern_counts": dict(sorted(counts.items())),
        "unique_patterns": len(counts),
        "categories": categories,
        "first_success_distribution": dict(sorted(first_success_dist.items())),
    }


def analyze_monotonicity(records: list[dict]) -> dict:
    non_monotonic = []
    monotonic_count = 0
    for record in records:
        pattern = pattern_of(record)
        if is_monotonic(pattern):
            monotonic_count += 1
        else:
            by_budget = {entry["budget"]: entry for entry in record["prefixes"]}
            non_monotonic.append(
                {
                    "theorem_rank": record["theorem_rank"],
                    "statement_id": record["statement_id"],
                    "component_id": record["component_id"],
                    "pattern": pattern,
                    "status_by_budget": {
                        str(b): by_budget[b]["verify_status"] for b in BUDGETS
                    },
                    "message_by_budget": {
                        str(b): (by_budget[b]["verify_message"] or "")[:200] for b in BUDGETS
                    },
                    "n_generated": record["n_generated"],
                    "hit_eos": record["hit_eos"],
                }
            )
    return {
        "monotonic": monotonic_count,
        "non_monotonic": len(non_monotonic),
        "monotonic_pct": round(100.0 * monotonic_count / max(1, len(records)), 3),
        "non_monotonic_cases": non_monotonic,
        "note": (
            "non-monotonic cases are reported as-is; token-prefix equivalence (V2-B001) does not "
            "imply verification monotonicity because extraction/assembly act on decoded text"
        ),
    }


def analyze_verification_health(records: list[dict]) -> dict:
    status_by_budget = {b: collections.Counter() for b in BUDGETS}
    runtime_by_budget = {b: [] for b in BUDGETS}
    infra_total = 0
    for record in records:
        for entry in record["prefixes"]:
            budget = entry["budget"]
            status_by_budget[budget][entry["verify_status"]] += 1
            if entry["verify_status"] in INFRA_STATUSES:
                infra_total += 1
            if entry["verification_runtime_seconds"] is not None:
                runtime_by_budget[budget].append(entry["verification_runtime_seconds"])
    runtime_summary = {}
    for budget, values in runtime_by_budget.items():
        if values:
            ordered = sorted(values)
            runtime_summary[str(budget)] = {
                "n": len(values),
                "mean": round(sum(values) / len(values), 3),
                "median": round(ordered[len(ordered) // 2], 3),
                "max": round(ordered[-1], 3),
            }
    return {
        "status_by_budget": {
            str(b): dict(sorted(status_by_budget[b].items())) for b in BUDGETS
        },
        "infrastructure_outcomes_total": infra_total,
        "verification_runtime_seconds": runtime_summary,
    }


def analyze_oracle(records: list[dict]) -> dict:
    """Pathwise Hindsight Oracle vs Uniform, on allocated-cap and actual-token axes.

    The oracle knows each theorem's realized five-budget outcome of THIS trajectory
    and picks the cheapest successful budget (multi-choice knapsack with unit
    profit; the greedy by min-success cost is exact). It is a screening upper
    bound, strictly stronger than any deployable policy (including the Expected
    Oracle) - the report must never claim an allocator can achieve it.
    """

    n = len(records)
    patterns = [pattern_of(record) for record in records]
    costs = [first_success(p) for p in patterns]
    solvable = sorted(c for c in costs if c is not None)
    never = n - len(solvable)

    uniform_solved = {
        str(budget): sum(1 for p in patterns if p[index] == "1")
        for index, budget in enumerate(BUDGETS)
    }

    cumulative = [0]
    for cost in solvable:
        cumulative.append(cumulative[-1] + cost)

    def oracle_solved_at(cap: int) -> int:
        return max(0, bisect.bisect_right(cumulative, cap) - 1)

    frontier = [{"cap": cumulative[k], "solved": k} for k in range(len(cumulative))]

    uniform_required = []
    for target in sorted({v for v in uniform_solved.values() if v > 0}):
        for index, budget in enumerate(BUDGETS):
            if uniform_solved[str(budget)] >= target:
                uniform_required.append(
                    {
                        "solved_target": target,
                        "uniform_budget": budget,
                        "required_allocated_cap": n * budget,
                    }
                )
                break

    uniform_points = []
    for index, budget in enumerate(BUDGETS):
        cap = n * budget
        uniform_at = uniform_solved[str(budget)]
        oracle_at = oracle_solved_at(cap)
        uniform_points.append(
            {
                "budget": budget,
                "allocated_cap": cap,
                "uniform_solved": uniform_at,
                "oracle_solved_at_same_cap": oracle_at,
                "oracle_gain_solved": oracle_at - uniform_at,
                "oracle_gain_pct_of_uniform": round(
                    100.0 * (oracle_at - uniform_at) / max(1, uniform_at), 2
                ),
            }
        )

    solved_full = uniform_solved[str(4096)]
    # A zero solved count has no meaningful "same solved" target: report None
    # instead of a vacuous 100% saving (the smoke edge case).
    oracle_cap_for_full = (
        cumulative[solved_full] if 0 < solved_full <= len(solvable) else None
    )
    cap_full = n * 4096
    savings = {
        "uniform_4096_solved": solved_full,
        "uniform_4096_allocated_cap": cap_full,
        "oracle_cap_for_same_solved": oracle_cap_for_full,
        "allocated_cap_savings_pct": (
            round(100.0 * (cap_full - oracle_cap_for_full) / cap_full, 2)
            if oracle_cap_for_full is not None
            else None
        ),
        "note": (
            "savings are in ALLOCATED max-token cap; actual generated tokens and runtime are "
            "reported separately and are not interchangeable with the cap"
        ),
    }

    actual_estimates = {}
    for budget in BUDGETS:
        estimated_tokens = sum(min(budget, r["n_generated"]) for r in records)
        estimated_seconds = sum(
            r["generation_runtime_seconds"] * (min(budget, r["n_generated"]) / max(1, r["n_generated"]))
            for r in records
        )
        actual_estimates[str(budget)] = {
            "hypothetical_actual_tokens_if_run_directly": estimated_tokens,
            "hypothetical_generation_seconds": round(estimated_seconds, 1),
        }

    realized = {
        "theorems": n,
        "generated_tokens_total": sum(r["n_generated"] for r in records),
        "generation_seconds_total": round(sum(r["generation_runtime_seconds"] for r in records), 1),
        "verification_seconds_total": round(
            sum(
                entry["verification_runtime_seconds"] or 0.0
                for r in records
                for entry in r["prefixes"]
            ),
            1,
        ),
        "wall_clock_note": (
            "realized cost is for the K=1 pilot screening design (one 4096 trajectory per "
            "theorem); a deployed adaptive policy would run one direct-budget generation per theorem"
        ),
    }

    return {
        "min_success_cost_distribution": {
            "512": costs.count(512),
            "1024": costs.count(1024),
            "2048": costs.count(2048),
            "3072": costs.count(3072),
            "4096": costs.count(4096),
            "never": never,
        },
        "uniform_solved": uniform_solved,
        "uniform_points": uniform_points,
        "oracle_frontier_breakpoints": frontier,
        "required_cap_by_solved_count": {
            "oracle": [
                {"solved": k, "required_allocated_cap": cumulative[k]}
                for k in range(len(cumulative))
            ],
            "uniform": uniform_required,
            "note": (
                "dual frontier for fixed-solved-count reporting: oracle = minimum allocated cap "
                "solving k theorems under the pathwise hindsight knapsack; uniform = smallest "
                "uniform budget reaching at least k solved (cap = n * budget)"
            ),
        },
        "savings": savings,
        "hypothetical_actual_compute_by_budget": actual_estimates,
        "realized_cost": realized,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="V2-B002 heterogeneity + Hindsight Oracle analysis.")
    parser.add_argument("--rollouts", default="experiments/results/v2_b002_rollouts.jsonl")
    parser.add_argument("--output", default="experiments/results/v2_b002_analysis.json")
    args = parser.parse_args()

    def resolve(path: str) -> Path:
        candidate = Path(path)
        return candidate if candidate.is_absolute() else ROOT / candidate

    rollouts_path = resolve(args.rollouts)
    records = load_records(rollouts_path)
    if not records:
        print("[ERROR] no records found", file=sys.stderr)
        return 2

    result = {
        "artifact_type": "v2_b002_analysis",
        "experiment": "V2-B002",
        "input": {
            "rollouts": str(rollouts_path),
            "sha256": sha256_file(rollouts_path),
            "n_records": len(records),
        },
        "response_patterns": analyze_patterns(records),
        "monotonicity": analyze_monotonicity(records),
        "verification_health": analyze_verification_health(records),
        "oracle": analyze_oracle(records),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
        "script": "scripts/v2_b002_analyze.py",
    }

    print(f"[V2-B002] records: {len(records)}")
    print(f"patterns: {result['response_patterns']['pattern_counts']}")
    print(f"unique patterns: {result['response_patterns']['unique_patterns']}")
    print(f"categories: {result['response_patterns']['categories']}")
    print(f"first success: {result['response_patterns']['first_success_distribution']}")
    print(
        f"monotonic: {result['monotonicity']['monotonic']} "
        f"({result['monotonicity']['monotonic_pct']}%), "
        f"non-monotonic: {result['monotonicity']['non_monotonic']}"
    )
    print("uniform vs oracle at the same allocated cap:")
    for point in result["oracle"]["uniform_points"]:
        print(
            f"  uniform-{point['budget']}: solved {point['uniform_solved']} -> oracle "
            f"{point['oracle_solved_at_same_cap']} (+{point['oracle_gain_solved']})"
        )
    print(f"savings: {result['oracle']['savings']}")
    print(f"verifier infra outcomes: {result['verification_health']['infrastructure_outcomes_total']}")

    output_path = resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"artifact: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
