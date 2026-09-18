#!/usr/bin/env python
"""M1 multi-seed final-holdout analysis (Phase 5F).

Reads the four final-holdout artifacts (theta0 base + the three seed
replicates' step60) and reports, per seed:
  Delta_seed = theta60_seed - theta0 (theorem-level paired bootstrap,
  win/tie/loss, exact McNemar, solved flips)
plus a descriptive cross-seed summary (sign consistency, effect sizes,
range). With only 2-3 training seeds no seed-level significance testing
is attempted - descriptive consistency only.
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
    "theta0": "experiments/results/e022_holdout_base.json",
    "seed1": "experiments/results/e022_holdout_seed1.json",
    "seed2": "experiments/results/e022_holdout_seed2.json",
    "seed3": "experiments/results/e022_holdout_seed3.json",
}


def counts(path: Path) -> dict[int, int]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    values: dict[int, int] = {}
    for record in artifact["records"]:
        index = record["theorem_index"]
        values[index] = values.get(index, 0) + int(record["verified"])
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-theorem", type=int, default=4)
    parser.add_argument("--output", default="experiments/results/e022_multiseed_analysis.json")
    args = parser.parse_args()

    evals = {label: ROOT / path for label, path in DEFAULT_EVALS.items()}
    present = {label: path for label, path in evals.items() if path.exists()}
    if "theta0" not in present:
        raise SystemExit("[ERROR] theta0 holdout artifact missing")
    base = counts(present["theta0"])
    indices = sorted(base)

    per_seed: dict[str, dict] = {}
    for label, path in present.items():
        if label == "theta0":
            continue
        treatment = counts(path)
        if sorted(treatment) != indices:
            raise SystemExit(f"[ERROR] {label} does not cover the same holdout theorems")
        deltas = [(treatment[i] - base[i]) / args.samples_per_theorem for i in indices]
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
        "artifact_type": "e022_multiseed_analysis",
        "samples_per_theorem": args.samples_per_theorem,
        "sources": {label: str(path.relative_to(ROOT)) for label, path in present.items()},
        **summary,
    }
    output = ROOT / args.output
    output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    print(f"theta0 verified: {summary['theta0_verified']}/{len(indices) * args.samples_per_theorem}")
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
