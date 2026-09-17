#!/usr/bin/env python3
"""P3-C (E018) cross-checkpoint analysis.

Reads the evaluator artifacts for the four checkpoints, recomputes the
theorem-level counts from the raw records, and produces the protocol tables:

* per-checkpoint headline metrics (candidate rate, Pass@1/4/8, all_zero / IGR /
  all_one, truncation, taxonomy);
* paired theorem-level analysis vs the base (mean delta, 95% CI via 10,000
  paired-bootstrap resamples with a fixed seed, win/tie/loss);
* solved-indicator flips with an exact two-sided McNemar test;
* resource accounting (wall time, tokens, verified per GPU-hour).

The pairing unit is the theorem - candidates are never paired.
Writes ``experiments/results/p3c_analysis.json``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tinylean_rl.evaluation.p3c_stats import (
    candidate_metrics,
    mcnemar_exact,
    paired_bootstrap,
    win_tie_loss,
)

DEFAULT_EVALS = {
    "step0": "experiments/results/e018_base.json",
    "step10": "experiments/results/e018_step10.json",
    "step20": "experiments/results/e018_step20.json",
    "step30": "experiments/results/e018_step30.json",
    "step60": "experiments/results/e019_step60.json",
}


def git_revision() -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()


def load_counts(path: Path) -> dict[int, int]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    counts: dict[int, int] = {}
    for record in artifact["records"]:
        index = record["theorem_index"]
        counts[index] = counts.get(index, 0) + int(record["verified"])
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoints",
        default=",".join(f"{label}={path}" for label, path in DEFAULT_EVALS.items()),
        help="Comma-separated label=json-path pairs.",
    )
    parser.add_argument("--samples-per-theorem", type=int, default=8)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260917)
    parser.add_argument("--output", default="experiments/results/p3c_analysis.json")
    args = parser.parse_args()

    entries: dict[str, Path] = {}
    for chunk in args.checkpoints.split(","):
        label, _, path = chunk.partition("=")
        entries[label.strip()] = ROOT / path.strip()

    artifacts: dict[str, dict] = {}
    counts: dict[str, dict[int, int]] = {}
    for label, path in entries.items():
        artifact = json.loads(path.read_text(encoding="utf-8"))
        artifacts[label] = artifact
        counts[label] = load_counts(path)

    theorem_indices = sorted(counts["step0"].keys())
    for label, label_counts in counts.items():
        if sorted(label_counts.keys()) != theorem_indices:
            raise SystemExit(f"[ERROR] {label} does not cover the same theorems as step0")

    per_checkpoint = {
        label: candidate_metrics(artifact["records"], args.samples_per_theorem)
        for label, artifact in artifacts.items()
    }

    paired: dict[str, dict] = {}
    for label in sorted(entry for entry in counts if entry != "step0"):
        if label not in counts:
            continue
        baseline = counts["step0"]
        treatment = counts[label]
        deltas = [
            (treatment[index] - baseline[index]) / args.samples_per_theorem
            for index in theorem_indices
        ]
        bootstrap = paired_bootstrap(
            deltas,
            n_resamples=args.bootstrap_resamples,
            seed=args.bootstrap_seed,
        )
        solved_baseline = [baseline[index] > 0 for index in theorem_indices]
        solved_treatment = [treatment[index] > 0 for index in theorem_indices]
        paired[f"{label}_vs_step0"] = {
            "bootstrap": bootstrap,
            "win_tie_loss": win_tie_loss(
                [baseline[index] for index in theorem_indices],
                [treatment[index] for index in theorem_indices],
            ),
            "solved_indicator": mcnemar_exact(solved_baseline, solved_treatment),
            "per_theorem_deltas": {
                str(index): deltas[position] for position, index in enumerate(theorem_indices)
            },
        }

    resources = {
        label: artifact.get("resources", {})
        for label, artifact in artifacts.items()
    }
    total_seconds = sum(r.get("total_seconds", 0.0) for r in resources.values())
    total_tokens = sum(r.get("total_generated_tokens", 0) for r in resources.values())

    summary = {
        "artifact_type": "p3c_analysis",
        "pairing_unit": "theorem",
        "samples_per_theorem": args.samples_per_theorem,
        "theorems": len(theorem_indices),
        "per_theorem_counts": {
            label: {str(index): value for index, value in sorted(label_counts.items())}
            for label, label_counts in counts.items()
        },
        "per_checkpoint": per_checkpoint,
        "paired_vs_step0": paired,
        "resources": {
            "per_checkpoint": resources,
            "total_seconds": round(total_seconds, 1),
            "total_gpu_hours": round(total_seconds / 3600, 3),
            "total_generated_tokens": total_tokens,
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
    }
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    header = f"{'checkpoint':>10} {'verified':>9} {'cand_rate':>9} {'pass@1':>7} {'pass@4':>7} {'pass@8':>7} {'all0':>6} {'igr':>6} {'all1':>6} {'trunc':>6}"
    print(header)
    for label, metrics in per_checkpoint.items():
        print(
            f"{label:>10} {metrics['verified_candidates']:>4}/{metrics['candidates']:<4} "
            f"{metrics['candidate_verified_rate']:>9.4f} {metrics['pass_at_1']:>7.4f} "
            f"{metrics['pass_at_4']:>7.4f} {metrics['pass_at_8']:>7.4f} "
            f"{metrics['group_rates']['all_zero']:>6.3f} {metrics['igr']:>6.3f} "
            f"{metrics['group_rates']['all_one']:>6.3f} {metrics['truncation_rate']:>6.3f}"
        )
    print()
    for key, analysis in paired.items():
        bootstrap = analysis["bootstrap"]
        wtl = analysis["win_tie_loss"]
        solved = analysis["solved_indicator"]
        print(
            f"{key}: mean_delta={bootstrap['mean_delta']:+.4f} "
            f"CI=[{bootstrap['ci_low']:+.4f},{bootstrap['ci_high']:+.4f}] "
            f"win/tie/loss={wtl['win']}/{wtl['tie']}/{wtl['loss']} "
            f"newly_solved={solved['newly_solved']} newly_lost={solved['newly_lost']} "
            f"mcnemar_p={solved['p_value_two_sided']:.4f}"
        )
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
