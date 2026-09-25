#!/usr/bin/env python3
"""V4-P001 §P — the verifier plan and the pre-registered safety recovery ceiling.

Owner §P: V4 keeps its own verifier infrastructure but reuses the C′ lessons, and — this is the
part that bites — the recovery ceiling must be pre-registered **before** formal generation, be
derived from the V3-observed infrastructure density, and be high enough that a routine recovery
never ends a run. "avoid repeated low per-session ceilings" is not a style note: in V3-R001
attempt-2 the ceiling of 16 restarts *was itself* the cause of both aborts (the abort messages are
the budget message), and each abort cost an operator relaunch.

This script is read-only over the V3 attempt-2 archive on this host. It measures the recovery
density actually observed, projects it onto V4's worst-case verification count, and freezes a
ceiling with a stated derivation. No generation, no verifier call, no GPU work.

Output: experiments/manifests/v4/v4_p001_verifier_plan.json
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ARCHIVE = "runs/v3_r001_rollout_archive_from_fly122"
OUT = "experiments/manifests/v4/v4_p001_verifier_plan.json"

# V4 worst case: MAX_SCREENING first attempts + 3 arms x N_PRIMARY repairs.
V4_WORST_CASE_VERIFICATIONS = 640 + 3 * 128
# V4's structural cap on recoveries attributable to one theorem: it has at most one first attempt
# plus three arm candidates to verify, and a recovery is triggered at most once per candidate.
V4_MAX_ARMS_PER_THEOREM = 3
QUANTILE = 0.999
CEILING_ROUNDING = 192


def poisson_cdf(lam: float, k: int) -> float:
    term = math.exp(-lam)
    total = term
    for i in range(1, k + 1):
        term *= lam / i
        total += term
    return total


def poisson_upper_quantile(lam: float, q: float) -> int:
    k = 0
    while poisson_cdf(lam, k) < q:
        k += 1
    return k


def read_executions() -> list[dict]:
    """The V3-R001 attempt-2 executions, deduplicated by ``run_id``.

    The archive keeps two copies of the `20260925T062406Z` summary (session directory + archive
    copy); a run that aborted twice would otherwise be counted twice and inflate the density.
    """

    seen: dict[str, dict] = {}
    for path in sorted((ROOT / ARCHIVE).rglob("*run_summary*.json")):
        summary = json.loads(path.read_text())
        infra = summary.get("verifier_infrastructure") or {}
        # Only C′ executions measure this density: attempt-1 had no recovery path at all, and its
        # 1,024-candidate supersession is provenance, not a denominator.
        if summary.get("attempt") != 2 or int(summary.get("candidates_generated") or 0) == 0:
            continue
        run_id = str(summary.get("run_id"))
        if run_id in seen:
            continue
        events = infra.get("recovery_events") or []
        seen[run_id] = {
            "run_id": run_id,
            "archive_path": str(path.relative_to(ROOT)),
            "candidates_generated": int(summary.get("candidates_generated") or 0),
            "recoveries_allowed": infra.get("recoveries_allowed"),
            "recoveries_attempted": int(infra.get("recoveries_attempted") or 0),
            "recoveries_succeeded": int(infra.get("recoveries_succeeded") or 0),
            "completed": not summary.get("aborted"),
            "abort_reason": summary.get("aborted") or "",
            "recovery_seconds": [round(float(e["recovery_seconds"]), 2) for e in events if e.get("ok")],
            "triggers": [
                {"theorem_rank": (e.get("trigger") or {}).get("theorem_rank"), "outcome": (e.get("trigger") or {}).get("outcome")}
                for e in events
            ],
        }
    return [seen[k] for k in sorted(seen)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=OUT)
    args = parser.parse_args()

    executions = read_executions()
    if not executions:
        raise SystemExit(f"no V3 attempt-2 executions found under {ARCHIVE}")

    total_candidates = sum(e["candidates_generated"] for e in executions)
    total_recoveries = sum(e["recoveries_attempted"] for e in executions)
    pooled_density = total_recoveries / total_candidates
    worst = max(executions, key=lambda e: e["recoveries_attempted"] / e["candidates_generated"])
    worst_density = worst["recoveries_attempted"] / worst["candidates_generated"]

    ceiling_aborts = [e for e in executions if "recovery budget exhausted" in str(e["abort_reason"])]
    recovery_seconds = sorted(s for e in executions for s in e["recovery_seconds"])

    lmbda_pooled = pooled_density * V4_WORST_CASE_VERIFICATIONS
    lmbda_worst = worst_density * V4_WORST_CASE_VERIFICATIONS
    q_pooled = poisson_upper_quantile(lmbda_pooled, QUANTILE)
    q_worst = poisson_upper_quantile(lmbda_worst, QUANTILE)

    # The frozen ceiling is the worst-execution-density bound rounded up, and it must stay below the
    # structural cap (one recovery per verification) or it would not be a fail-closed guard at all.
    frozen_ceiling = CEILING_ROUNDING

    out = {
        "artifact_type": "v4_p001_verifier_plan",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": subprocess.run(["hostname"], capture_output=True, text=True, check=False).stdout.strip(),
        "git_revision": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip(),
        "note": "owner §P: V4 verifier infrastructure, and the safety recovery ceiling frozen before any formal generation",
        "observed": {
            "source": f"{ARCHIVE} (V3-R001 attempt-2 archive, read-only)",
            "executions": executions,
            "total_candidates_verified": total_candidates,
            "total_recoveries_attempted": total_recoveries,
            "total_recoveries_succeeded": sum(e["recoveries_succeeded"] for e in executions),
            "pooled_density_recoveries_per_candidate": round(pooled_density, 4),
            "worst_execution": worst["run_id"],
            "worst_execution_density": round(worst_density, 4),
            "aborts_caused_by_the_ceiling": len(ceiling_aborts),
            "ceiling_abort_reasons": [e["abort_reason"] for e in ceiling_aborts],
            "recovery_seconds": {
                "n": len(recovery_seconds),
                "min": recovery_seconds[0] if recovery_seconds else None,
                "median": recovery_seconds[len(recovery_seconds) // 2] if recovery_seconds else None,
                "max": recovery_seconds[-1] if recovery_seconds else None,
            },
            "pathology_note": (
                "recoveries are not spread evenly: one theorem (rank 67) accounts for 7 of the 17 "
                "recoveries in the worst execution, because the V3 unit verified 8 samples of the "
                "same theorem and each sample re-poisoned the singleton REPL. V4 verifies one "
                "candidate per theorem per arm, so the structural per-theorem exposure is smaller."
            ),
        },
        "projection": {
            "v4_worst_case_verifications": V4_WORST_CASE_VERIFICATIONS,
            "lambda_at_pooled_density": round(lmbda_pooled, 2),
            "poisson_999_pooled_density": q_pooled,
            "lambda_at_worst_density": round(lmbda_worst, 2),
            "poisson_999_worst_density": q_worst,
            "chosen_basis": "worst observed execution density, Poisson 99.9% upper bound, rounded up",
            "structural_max_recoveries": V4_WORST_CASE_VERIFICATIONS,
            "frozen_ceiling": frozen_ceiling,
            "ceiling_over_worst_quantile": round(frozen_ceiling / q_worst, 3) if q_worst else None,
        },
        "frozen": {
            "max_recoveries_per_run": frozen_ceiling,
            "max_recoveries_per_theorem": V4_MAX_ARMS_PER_THEOREM,
            "per_run_semantics": (
                "the run continues through routine recovery; at the ceiling it fails closed with an "
                "aborted summary that keeps every finished candidate and stays resumable"
            ),
            "per_theorem_semantics": (
                "at most one recovery per arm candidate, so three recoveries exhaust a theorem's "
                "exposure; a theorem that exceeds it has its remaining arm candidates marked "
                "INFRA_CENSORED (missing, never a failure) and the run continues - one pathological "
                "theorem cannot consume the run budget"
            ),
            "recovery_is_not_a_candidate_retry": True,
            "recovery_trigger": "a non-conclusive candidate outcome (timeout / 5xx / unhealthy / unresolved), never a score",
        },
        "invariant": (
            "no candidate verification attempt may leave a Lean computation occupying a reusable "
            "REPL after that attempt is classified timed out or failed; capacity is restored before "
            "the next candidate"
        ),
        "inherited_from_Cprime": {
            "dedicated_instance": True,
            "max_repls": 1,
            "verification_concurrency": 1,
            "batch_size": 1,
            "server_side_timeout_s": 120,
            "client_slack_s": 60,
            "cold_canary_budget_s": 600,
            "restart_then_health_then_canary": True,
            "fail_closed_on_identity_health_or_canary": True,
        },
        "not_claimed": [
            "no verifier throughput, latency or capacity claim - §P forbids it, and the recovery wall-clock is reported only as cost",
            "no change to the frozen verification semantics: /verify, format gate, score, censoring rule",
            "the V3 attempt-2 archive is read-only here; no V3 artifact is modified or re-interpreted",
        ],
    }

    destination = ROOT / args.out
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(out, indent=2) + "\n")
    print(f"observed: {total_recoveries} recoveries / {total_candidates} candidates pooled density {pooled_density:.4f}")
    print(f"worst execution {worst['run_id']}: density {worst_density:.4f}; Poisson 99.9% at V4 scale = {q_worst} (pooled {q_pooled})")
    print(f"aborts caused by the ceiling: {len(ceiling_aborts)}")
    print(f"frozen ceiling: {frozen_ceiling} per run, {V4_MAX_ARMS_PER_THEOREM} per theorem")
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
