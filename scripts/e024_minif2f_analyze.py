#!/usr/bin/env python3
"""E024 MiniF2F paired analysis: theta0 vs seed1-step60.

Theorem-level pairing only (never candidate-level pseudo-independence):
per-theorem verified counts drive the paired bootstrap, win/tie/loss and the
exact McNemar test on the solved indicator. Pairing guards (statement
alignment, samples-per-theorem equality, frozen seed schedule) must pass
before any statistic is produced. Stored-proof adjudication credits are
optional inputs reported alongside the observed counts; raw artifacts are
never modified.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tinylean_rl.evaluation.p3c_stats import mcnemar_exact, paired_bootstrap, win_tie_loss

ARTIFACT_TYPE = "e024_minif2f_eval"
ANALYSIS_ARTIFACT_TYPE = "e024_minif2f_analysis"
DEFAULT_OUTPUT = "experiments/results/e024_minif2f_analysis.json"


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def load_artifact(path: Path) -> dict[str, Any]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    kind = artifact.get("artifact_type")
    if kind != ARTIFACT_TYPE:
        raise SystemExit(f"{path}: unexpected artifact_type {kind!r}")
    return artifact


def statement_alignment(records: list[dict[str, Any]]) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for record in records:
        index = record["theorem_index"]
        statement = record["statement_id"]
        if index in mapping and mapping[index] != statement:
            raise SystemExit(
                f"inconsistent statement_id for theorem {index}: {mapping[index]!r} vs {statement!r}"
            )
        mapping[index] = statement
    return mapping


def verify_pairing(theta0: dict[str, Any], seed1: dict[str, Any]) -> None:
    samples0 = theta0["settings"]["samples_per_theorem"]
    samples1 = seed1["settings"]["samples_per_theorem"]
    if samples0 != samples1:
        raise SystemExit(
            f"samples-per-theorem mismatch: theta0={samples0} seed1={samples1}"
        )
    if theta0["settings"]["theorems"] != seed1["settings"]["theorems"]:
        raise SystemExit("theorem count mismatch between artifacts")
    left = statement_alignment(theta0["records"])
    right = statement_alignment(seed1["records"])
    if set(left) != set(right):
        raise SystemExit("theorem_index sets differ between artifacts")
    mismatched = [index for index in sorted(left) if left[index] != right[index]]
    if mismatched:
        sample = ", ".join(f"{i}:{left[i]}!={right[i]}" for i in mismatched[:5])
        raise SystemExit(
            f"statement alignment mismatch on {len(mismatched)} theorems (e.g. {sample})"
        )
    seeds_left = {
        (record["theorem_index"], record["sample_index"]): record["sampling_seed"]
        for record in theta0["records"]
    }
    seeds_right = {
        (record["theorem_index"], record["sample_index"]): record["sampling_seed"]
        for record in seed1["records"]
    }
    if set(seeds_left) != set(seeds_right):
        raise SystemExit("(theorem_index, sample_index) coverage differs between artifacts")
    bad = [key for key in sorted(seeds_left) if seeds_left[key] != seeds_right[key]]
    if bad:
        raise SystemExit(
            f"seed schedule mismatch on {len(bad)} candidates (e.g. {bad[0]})"
        )


def counts_by_theorem(records: list[dict[str, Any]]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for record in records:
        index = record["theorem_index"]
        counts[index] = counts.get(index, 0) + int(bool(record["verified"]))
    return counts


def adjudication_credits(path: Path) -> dict[tuple[int, int], int]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    credits: dict[tuple[int, int], int] = {}
    for candidate in artifact.get("candidates", []):
        if candidate.get("final_class") == "verified_on_recheck":
            key = (candidate["theorem_index"], candidate["sample_index"])
            credits[key] = credits.get(key, 0) + 1
    return credits


def apply_credits(
    counts: dict[int, int],
    records: list[dict[str, Any]],
    credits: dict[tuple[int, int], int],
) -> dict[int, int]:
    corrected = dict(counts)
    for record in records:
        key = (record["theorem_index"], record["sample_index"])
        if credits.get(key) and not record["verified"]:
            index = record["theorem_index"]
            corrected[index] = corrected.get(index, 0) + 1
    return corrected


def paired_statistics(
    base_counts: dict[int, int], treat_counts: dict[int, int], samples_per_theorem: int
) -> dict[str, Any]:
    indices = sorted(set(base_counts) | set(treat_counts))
    base = [base_counts.get(index, 0) for index in indices]
    treat = [treat_counts.get(index, 0) for index in indices]
    deltas = [(t - b) / samples_per_theorem for b, t in zip(base, treat)]
    return {
        "theta0_verified": sum(base),
        "seed1_verified": sum(treat),
        "candidate_delta": sum(treat) - sum(base),
        "theorem_level_bootstrap": paired_bootstrap(deltas),
        "win_tie_loss": win_tie_loss(base, treat),
        "mcnemar": mcnemar_exact(
            [value > 0 for value in base], [value > 0 for value in treat]
        ),
    }


def agreement_counts(
    base_counts: dict[int, int], treat_counts: dict[int, int]
) -> dict[str, int]:
    indices = sorted(set(base_counts) | set(treat_counts))
    both = sum(
        1 for i in indices if base_counts.get(i, 0) > 0 and treat_counts.get(i, 0) > 0
    )
    only_base = sum(
        1 for i in indices if base_counts.get(i, 0) > 0 and treat_counts.get(i, 0) == 0
    )
    only_treat = sum(
        1 for i in indices if base_counts.get(i, 0) == 0 and treat_counts.get(i, 0) > 0
    )
    neither = sum(
        1 for i in indices if base_counts.get(i, 0) == 0 and treat_counts.get(i, 0) == 0
    )
    return {
        "both_solved": both,
        "only_theta0_solved": only_base,
        "only_seed1_solved": only_treat,
        "neither_solved": neither,
    }


def model_summary(artifact: dict[str, Any]) -> dict[str, Any]:
    summary = artifact["summary"]
    taxonomy = summary["taxonomy"]
    return {
        "model_label": artifact.get("model_label"),
        "host": artifact.get("host"),
        "git_revision": artifact.get("git_revision"),
        "verified_candidates": summary["verified_candidates"],
        "candidate_verified_rate": summary["candidate_verified_rate"],
        "igr": summary["igr"],
        "pass_at_1": summary["pass_at_1"],
        "pass_at_4": summary["pass_at_4"],
        "theorems_solved_at_least_1": summary["theorems_solved_at_least_1"],
        "theorems_solved_all": summary["theorems_solved_all"],
        "group_rates": summary["group_rates"],
        "truncation_rate": summary["truncation_rate"],
        "verifier_error_rate": summary["verifier_error_rate"],
        "taxonomy": taxonomy,
        "verifier_errors_observed": taxonomy.get("verifier_error", 0),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--theta0", required=True, help="theta0 E024 eval artifact path.")
    parser.add_argument("--seed1", required=True, help="seed1-step60 E024 eval artifact path.")
    parser.add_argument("--theta0-adjudication", default="")
    parser.add_argument("--seed1-adjudication", default="")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    theta0_path = resolve(args.theta0)
    seed1_path = resolve(args.seed1)
    theta0 = load_artifact(theta0_path)
    seed1 = load_artifact(seed1_path)
    verify_pairing(theta0, seed1)
    samples_per_theorem = theta0["settings"]["samples_per_theorem"]

    base_counts = counts_by_theorem(theta0["records"])
    treat_counts = counts_by_theorem(seed1["records"])
    observed = paired_statistics(base_counts, treat_counts, samples_per_theorem)
    observed["theorem_level_agreement"] = agreement_counts(base_counts, treat_counts)

    corrected = None
    if args.theta0_adjudication or args.seed1_adjudication:
        corrected_base = base_counts
        corrected_treat = treat_counts
        credited = {"theta0": 0, "seed1": 0}
        if args.theta0_adjudication:
            credits = adjudication_credits(resolve(args.theta0_adjudication))
            corrected_base = apply_credits(base_counts, theta0["records"], credits)
            credited["theta0"] = sum(credits.values())
        if args.seed1_adjudication:
            credits = adjudication_credits(resolve(args.seed1_adjudication))
            corrected_treat = apply_credits(treat_counts, seed1["records"], credits)
            credited["seed1"] = sum(credits.values())
        corrected = paired_statistics(corrected_base, corrected_treat, samples_per_theorem)
        corrected["credited"] = credited
        corrected["policy"] = (
            "verified_on_recheck credits from stored-proof adjudication; raw artifacts unchanged"
        )

    artifact = {
        "artifact_type": ANALYSIS_ARTIFACT_TYPE,
        "experiment": "E024",
        "manifest": "experiments/manifests/e024_minif2f.yaml",
        "inputs": {
            "theta0": str(theta0_path),
            "seed1": str(seed1_path),
            "theta0_adjudication": args.theta0_adjudication or None,
            "seed1_adjudication": args.seed1_adjudication or None,
        },
        "protocol": {
            "theorems": theta0["settings"]["theorems"],
            "samples_per_theorem": samples_per_theorem,
            "seed_schedule": theta0["settings"]["seed_schedule"],
            "dataset": theta0["dataset"],
        },
        "observed": observed,
        "corrected": corrected,
        "per_model": {"theta0": model_summary(theta0), "seed1_step60": model_summary(seed1)},
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    output_path = resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(artifact, indent=2, ensure_ascii=False))
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
