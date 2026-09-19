#!/usr/bin/env python
"""M1 multi-seed final-holdout analysis (Phase 5F).

Reads the four final-holdout artifacts (theta0 base + the three seed
replicates' step60) and reports, per seed:
  Delta_seed = theta60_seed - theta0 (theorem-level paired bootstrap,
  win/tie/loss, exact McNemar, solved flips)
plus a descriptive cross-seed summary (sign consistency, effect sizes,
range). With only 2-3 training seeds no seed-level significance testing
is attempted - descriptive consistency only.

Input paths default to the canonical artifact names and can be overridden
explicitly (--theta0/--seed1/--seed2/--seed3), e.g. when the primary host is
harmonized and per-checkpoint file names differ.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tinylean_rl.evaluation.p3c_stats import (
    mcnemar_exact,
    paired_bootstrap,
    win_tie_loss,
)

DEFAULT_EVALS = {
    "theta0": "experiments/results/e023_holdout_base.json",
    "seed1": "experiments/results/e023_holdout_seed1.json",
    "seed2": "experiments/results/e023_holdout_seed2.json",
    "seed3": "experiments/results/e023_holdout_seed3.json",
}


def resolve(path: str) -> Path:
    """Resolve a CLI path relative to the repo root (absolute paths pass through)."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def counts(path: Path) -> tuple[dict[int, int], dict]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    values: dict[int, int] = {}
    for record in artifact["records"]:
        index = record["theorem_index"]
        values[index] = values.get(index, 0) + int(record["verified"])
    return values, artifact


def statement_ids(artifact: dict) -> set[str]:
    return {record["statement_id"] for record in artifact["records"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--theta0", default=DEFAULT_EVALS["theta0"], help="theta0 artifact path (default: %(default)s)"
    )
    parser.add_argument(
        "--seed1", default=DEFAULT_EVALS["seed1"], help="seed1 artifact path (default: %(default)s)"
    )
    parser.add_argument(
        "--seed2", default=DEFAULT_EVALS["seed2"], help="seed2 artifact path (default: %(default)s)"
    )
    parser.add_argument(
        "--seed3", default=DEFAULT_EVALS["seed3"], help="seed3 artifact path (default: %(default)s)"
    )
    parser.add_argument("--output", default="experiments/results/e023_multiseed_analysis.json")
    args = parser.parse_args()

    evals = {
        "theta0": resolve(args.theta0),
        "seed1": resolve(args.seed1),
        "seed2": resolve(args.seed2),
        "seed3": resolve(args.seed3),
    }
    present = {label: path for label, path in evals.items() if path.exists()}
    if "theta0" not in present:
        raise SystemExit("[ERROR] theta0 holdout artifact missing")
    loaded = {label: counts(path) for label, path in present.items()}
    base, base_artifact = loaded["theta0"]
    indices = sorted(base)

    # The normalization factor comes from the artifacts themselves (never from a
    # CLI default that could silently disagree and double all deltas).
    sample_sizes = {label: artifact["settings"]["samples_per_theorem"] for label, (_, artifact) in loaded.items()}
    if len(set(sample_sizes.values())) != 1:
        raise SystemExit(f"[ERROR] samples_per_theorem differs across artifacts: {sample_sizes}")
    samples_per_theorem = next(iter(sample_sizes.values()))

    # Guard against pairing results from different theorem draws (index ranges
    # would still match; only the statement ids prove the same 128 theorems).
    base_ids = statement_ids(base_artifact)
    for label, (_, artifact) in loaded.items():
        if statement_ids(artifact) != base_ids:
            raise SystemExit(f"[ERROR] {label} was evaluated on a different statement set than theta0")

    per_seed: dict[str, dict] = {}
    for label, (treatment, _) in loaded.items():
        if label == "theta0":
            continue
        if sorted(treatment) != indices:
            raise SystemExit(f"[ERROR] {label} does not cover the same holdout theorems")
        deltas = [(treatment[i] - base[i]) / samples_per_theorem for i in indices]
        per_seed[label] = {
            "bootstrap": paired_bootstrap(deltas),
            "win_tie_loss": win_tie_loss([base[i] for i in indices], [treatment[i] for i in indices]),
            "solved_indicator": mcnemar_exact([base[i] > 0 for i in indices], [treatment[i] > 0 for i in indices]),
            "verified_total": sum(treatment.values()),
        }

    deltas = [entry["bootstrap"]["mean_delta"] for entry in per_seed.values()]
    summary = {
        "theta0_verified": sum(base.values()),
        "theorems": len(indices),
        "per_seed": per_seed,
        "cross_seed": {
            "seed_count": len(per_seed),
            "mean_effect": statistics.fmean(deltas) if deltas else None,
            "range": [min(deltas), max(deltas)] if deltas else None,
            "sign_consistency": sum(1 for d in deltas if d > 0) if deltas else 0,
            "note": "descriptive only (2-3 training seeds); no seed-level significance inference",
        },
    }
    artifact = {
        "artifact_type": "e023_multiseed_analysis",
        "samples_per_theorem": samples_per_theorem,
        "sources": {label: display_path(path) for label, path in present.items()},
        **summary,
    }
    output = ROOT / args.output
    output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    print(f"theta0 verified: {summary['theta0_verified']}/{len(indices) * samples_per_theorem}")
    for label, entry in per_seed.items():
        b = entry["bootstrap"]
        m = entry["solved_indicator"]
        print(
            f"{label}: delta={b['mean_delta']:+.4f} CI=[{b['ci_low']:+.4f},{b['ci_high']:+.4f}] "
            f"WTL={entry['win_tie_loss']['win']}/{entry['win_tie_loss']['tie']}/{entry['win_tie_loss']['loss']} "
            f"McNemar p={m['p_value_two_sided']:.3f} (+{m['newly_solved']}/-{m['newly_lost']}) verified={entry['verified_total']}"
        )
    cs = summary["cross_seed"]
    print(f"cross-seed: n={cs['seed_count']} mean={cs['mean_effect']:+.4f} range=[{cs['range'][0]:+.4f},{cs['range'][1]:+.4f}] sign+={cs['sign_consistency']}")
    print(f"Output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
