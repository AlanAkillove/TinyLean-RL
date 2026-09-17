#!/usr/bin/env python
"""E018-D verifier-error adjudication (P3-C addendum).

Re-verifies every candidate whose E018 ``verify_status == "verifier_error"``
using the stored ``proof`` string (the exact code that was submitted), with no
re-generation: single-candidate requests against a warm server under a fixed
40 GiB container cap, timeout 120 s (same as the original single-retry path),
up to three attempts. (24 GiB was experimentally found to be below the normal
warm REPL-pool footprint ~26-30 GiB and killed the healthy pool; 40 GiB is the
current fixed verifier instrumentation setting.)

Classification per candidate (decision order):
  1. verified_on_recheck                  any attempt verifies
  2. deterministic_timeout_or_resource_   a real Lean response whose message carries a
     exhaustion                           timeout/resource hint (Lean-internal deterministic
                                          timeout), all attempts timing out, or a server
                                          message mentioning memory/OOM/kill
  3. deterministic_lean_failure           a plain Lean rejection (error message) seen
  4. transient_infrastructure_failure     mixed timeout + other infra errors, no verdict
  5. unresolved_verifier_error            only infra errors (no verdict, no timeout mix)

Only class 1 changes the corrected verified counts; classes 2-5 keep reward 0.
The artifact reports observed / corrected / optimistic (all 43 credited) /
pessimistic (nothing credited) deltas for step30 vs step0, and re-runs the
theorem-level paired bootstrap and exact McNemar on the corrected counts.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from promptset_rollout_probe import analyze_item, response_items

from tinylean_rl.evaluation.p3c_stats import (
    mcnemar_exact,
    paired_bootstrap,
    win_tie_loss,
)
from tinylean_rl.verifier.kimina import verify_code

CHECKPOINTS = {
    "step0": ROOT / "experiments/results/e018_base.json",
    "step10": ROOT / "experiments/results/e018_step10.json",
    "step20": ROOT / "experiments/results/e018_step20.json",
    "step30": ROOT / "experiments/results/e018_step30.json",
}
SAMPLES_PER_THEOREM = 8
_PROBE = "import Mathlib\ntheorem tinylean_probe : 1 + 1 = 2 := by norm_num"
_RESOURCE_HINTS = ("memory", "oom", "killed", "out of memory", "resource", "timeout")


def run_attempt(proof: str, custom_id: str, timeout: float) -> dict:
    started = time.perf_counter()
    try:
        items = response_items(verify_code(proof, custom_id=custom_id, timeout=timeout))
    except httpx.TimeoutException as exc:
        return {
            "kind": "timeout",
            "status": "client_timeout",
            "message": f"client timeout after {timeout:.0f}s: {exc}"[:300],
            "duration_s": round(time.perf_counter() - started, 2),
        }
    except httpx.HTTPError as exc:
        return {
            "kind": "http_error",
            "status": "http_error",
            "message": str(exc)[:300],
            "duration_s": round(time.perf_counter() - started, 2),
        }
    analysis = analyze_item(items[0] if items else None)
    if analysis["verified"]:
        kind = "verified"
    elif analysis["status"] == "lean_error":
        kind = "lean_error"
    else:
        kind = "verifier_error"
    return {
        "kind": kind,
        "status": "verified" if analysis["verified"] else analysis["status"],
        "message": str(analysis["lean_message"])[:300],
        "duration_s": round(time.perf_counter() - started, 2),
    }


def classify_candidate(attempts: list[dict]) -> str:
    kinds = [attempt["kind"] for attempt in attempts]
    if not kinds:
        return "unresolved_verifier_error"
    if "verified" in kinds:
        return "verified_on_recheck"
    # Lean itself answering with a timeout/resource hint is a candidate-induced
    # deterministic resource failure, not a plain proof rejection.
    lean_messages = [attempt["message"].lower() for attempt in attempts if attempt["kind"] == "lean_error"]
    if any(hint in message for message in lean_messages for hint in _RESOURCE_HINTS):
        return "deterministic_timeout_or_resource_exhaustion"
    if "lean_error" in kinds:
        return "deterministic_lean_failure"
    if any(hint in attempt["message"].lower() for attempt in attempts for hint in _RESOURCE_HINTS):
        return "deterministic_timeout_or_resource_exhaustion"
    if all(kind == "timeout" for kind in kinds):
        return "deterministic_timeout_or_resource_exhaustion"
    if "timeout" in kinds and any(kind != "timeout" for kind in kinds):
        return "transient_infrastructure_failure"
    return "unresolved_verifier_error"


def load_verifier_errors() -> list[dict]:
    candidates: list[dict] = []
    for label, path in CHECKPOINTS.items():
        artifact = json.loads(path.read_text(encoding="utf-8"))
        for record in artifact["records"]:
            if record["verify_status"] != "verifier_error":
                continue
            candidates.append(
                {
                    "checkpoint": label,
                    "theorem_index": record["theorem_index"],
                    "sample_index": record["sample_index"],
                    "sampling_seed": record["sampling_seed"],
                    "original_lean_message": record.get("lean_message") or "",
                    "proof": record["proof"],
                }
            )
    return candidates


def adjudge_candidates(candidates: list[dict], args: argparse.Namespace, out_path: Path) -> list[dict]:
    results: list[dict] = []
    if args.resume and out_path.exists():
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        results = existing.get("candidates", [])
        done = {(r["checkpoint"], r["theorem_index"], r["sample_index"]) for r in results}
        print(f"resume: {len(results)} candidates already adjudicated", flush=True)
    else:
        done = set()

    for position, candidate in enumerate(candidates, 1):
        key = (candidate["checkpoint"], candidate["theorem_index"], candidate["sample_index"])
        if key in done:
            continue
        attempts: list[dict] = []
        for attempt_number in range(1, args.max_attempts + 1):
            attempts.append(
                run_attempt(
                    candidate["proof"],
                    custom_id=f"{candidate['checkpoint']}-{candidate['theorem_index']}-{candidate['sample_index']}-adj{attempt_number}",
                    timeout=args.timeout,
                )
            )
            kinds = [attempt["kind"] for attempt in attempts]
            if "verified" in kinds:
                break
            if len(kinds) >= 2 and kinds[-1] == kinds[-2] and kinds[-1] in ("timeout", "lean_error"):
                break
            if attempt_number < args.max_attempts:
                time.sleep(args.retry_sleep)
        final_class = classify_candidate(attempts)
        results.append(
            {**{k: v for k, v in candidate.items() if k != "proof"}, "attempts": attempts, "final_class": final_class}
        )
        summary_line = " ".join(f"{a['kind']}({a['duration_s']}s)" for a in attempts)
        print(
            f"[{position}/{len(candidates)}] {candidate['checkpoint']} thm{candidate['theorem_index']}"
            f" s{candidate['sample_index']} -> {final_class} [{summary_line}]",
            flush=True,
        )
        out_path.write_text(
            json.dumps({"artifact_type": "e018d_verifier_error_adjudication", "candidates": results}, indent=2),
            encoding="utf-8",
        )
    return results


def corrected_counts(original: dict[str, dict[int, int]], credits: dict[str, set[tuple[int, int]]]) -> dict[str, dict[int, int]]:
    corrected = {label: dict(counts) for label, counts in original.items()}
    for label, keys in credits.items():
        for theorem_index, _sample_index in keys:
            corrected[label][theorem_index] = corrected[label].get(theorem_index, 0) + 1
    return corrected


def theorem_counts() -> dict[str, dict[int, int]]:
    counts: dict[str, dict[int, int]] = {}
    for label, path in CHECKPOINTS.items():
        artifact = json.loads(path.read_text(encoding="utf-8"))
        values: dict[int, int] = {}
        for record in artifact["records"]:
            index = record["theorem_index"]
            values[index] = values.get(index, 0) + int(record["verified"])
        counts[label] = values
    return counts


def paired_suite(counts: dict[str, dict[int, int]]) -> dict[str, dict]:
    theorem_indices = sorted(counts["step0"])
    suite: dict[str, dict] = {}
    for label in ("step10", "step20", "step30"):
        baseline, treatment = counts["step0"], counts[label]
        deltas = [(treatment[i] - baseline[i]) / SAMPLES_PER_THEOREM for i in theorem_indices]
        suite[f"{label}_vs_step0"] = {
            "bootstrap": paired_bootstrap(deltas),
            "win_tie_loss": win_tie_loss([baseline[i] for i in theorem_indices], [treatment[i] for i in theorem_indices]),
            "solved_indicator": mcnemar_exact(
                [baseline[i] > 0 for i in theorem_indices], [treatment[i] > 0 for i in theorem_indices]
            ),
        }
    return suite


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--probe-timeout", type=float, default=300.0, help="Warm-up probe timeout; cold Mathlib import takes ~80 s.")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument("--limit", type=int, default=0, help="Only the first N candidates (0 = all).")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-probe", action="store_true")
    parser.add_argument("--output", default="experiments/results/e018d_verifier_error_adjudication.json")
    args = parser.parse_args()

    out_path = ROOT / args.output
    if not args.skip_probe:
        probe_started = time.perf_counter()
        probe = run_attempt(_PROBE, "e018d-prewarm-probe", timeout=args.probe_timeout)
        if probe["kind"] != "verified":
            print(f"[ERROR] prewarm probe failed: {probe}", flush=True)
            return 1
        print(f"prewarm probe verified in {time.perf_counter() - probe_started:.2f}s", flush=True)

    candidates = load_verifier_errors()
    if args.limit:
        candidates = candidates[: args.limit]
    print(f"candidates to adjudicate: {len(candidates)}", flush=True)

    results = adjudge_candidates(candidates, args, out_path)

    per_class = Counter(result["final_class"] for result in results)
    per_checkpoint: dict[str, dict] = {}
    credits: dict[str, set[tuple[int, int]]] = {label: set() for label in CHECKPOINTS}
    for result in results:
        bucket = per_checkpoint.setdefault(
            result["checkpoint"],
            {"verifier_errors": 0, "verified_on_recheck": 0, "deterministic_lean_failure": 0,
             "deterministic_timeout_or_resource_exhaustion": 0, "transient_infrastructure_failure": 0,
             "unresolved_verifier_error": 0},
        )
        bucket["verifier_errors"] += 1
        bucket[result["final_class"]] += 1
        if result["final_class"] == "verified_on_recheck":
            credits[result["checkpoint"]].add((result["theorem_index"], result["sample_index"]))

    observed = theorem_counts()
    corrected = corrected_counts(observed, credits)
    optimistic = corrected_counts(
        observed,
        {
            label: {
                (candidate["theorem_index"], candidate["sample_index"])
                for candidate in results
                if candidate["checkpoint"] == label
            }
            for label in CHECKPOINTS
        },
    )

    def delta(counts: dict[str, dict[int, int]], label: str) -> float:
        indices = sorted(counts["step0"])
        return sum(counts[label][i] - counts["step0"][i] for i in indices) / (len(indices) * SAMPLES_PER_THEOREM)

    artifact = {
        "artifact_type": "e018d_verifier_error_adjudication",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {
            "protocol": "single candidate per request; stored proof re-used; no regeneration",
            "container_mem_limit": "40 GiB (docker update, live; no recreate). 24 GiB proved too small for the warm REPL pool and killed it (2026-09-17 08:57 UTC), requiring a re-warm",
            "timeout_s": args.timeout,
            "max_attempts": args.max_attempts,
            "retry_sleep_s": args.retry_sleep,
            "early_stop": "verified, or two identical definitive outcomes (timeout/lean_error)",
        },
        "candidates": results,
        "summary": {"per_class": dict(per_class), "per_checkpoint": per_checkpoint},
        "correction": {
            "observed_counts": {label: dict(values) for label, values in observed.items()},
            "corrected_counts": {label: dict(values) for label, values in corrected.items()},
            "verified_totals": {
                "observed": {label: sum(values.values()) for label, values in observed.items()},
                "corrected": {label: sum(values.values()) for label, values in corrected.items()},
                "optimistic_all_verifier_errors_credited": {label: sum(values.values()) for label, values in optimistic.items()},
            },
            "delta_vs_step0": {
                "observed": {label: delta(observed, label) for label in ("step10", "step20", "step30")},
                "corrected": {label: delta(corrected, label) for label in ("step10", "step20", "step30")},
                "optimistic": {label: delta(optimistic, label) for label in ("step10", "step20", "step30")},
                "pessimistic_equals_observed": True,
            },
            "paired_suite_corrected": paired_suite(corrected),
        },
    }
    out_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps({"per_class": dict(per_class), "per_checkpoint": per_checkpoint}, indent=2), flush=True)
    print(json.dumps(artifact["correction"]["verified_totals"], indent=2), flush=True)
    print(json.dumps(artifact["correction"]["delta_vs_step0"], indent=2), flush=True)
    print(f"Output: {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
