#!/usr/bin/env python
"""Mechanism analysis: temperature x group-size from two n=8 generation runs.

Reads two evaluator artifacts on the same theorem set and seed schedule
(temperature 0.6 and 1.0, both n=8) and derives the four conditions:
  T=0.6 n=4 / T=0.6 n=8 / T=1.0 n=4 / T=1.0 n=8
(n=4 reuses the first four samples of each theorem - no extra generation).

Reports per-condition metrics (verified rate, IGR/Z/O, mean successes per
group, truncation, format failure, response length, wall time, informative
groups per 1M generated tokens) and theorem-level paired comparisons of the
informative-group indicator (newly / lost / still informative / still
degenerate + McNemar exact + win/tie/loss).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tinylean_rl.evaluation.p3c_stats import mcnemar_exact, win_tie_loss


def load(path: Path, n: int) -> dict[int, list[int]]:
    """{theorem_index: [verified flags for the first n samples]} (ordered by sample)."""
    artifact = json.loads(path.read_text(encoding="utf-8"))
    groups: dict[int, dict[int, int]] = defaultdict(dict)
    for record in artifact["records"]:
        if record["sample_index"] < n:
            groups[record["theorem_index"]][record["sample_index"]] = int(record["verified"])
    return {index: [flags[s] for s in sorted(flags)] for index, flags in groups.items()}


def condition_metrics(path: Path, n: int, label: str) -> tuple[dict, dict[int, bool]]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    groups = load(path, n)
    counts = {index: sum(flags) for index, flags in groups.items()}
    values = list(counts.values())
    informative = {index: 0 < c < n for index, c in counts.items()}
    records = [r for r in artifact["records"] if r["sample_index"] < n]
    total_tokens = sum(r["generated_tokens"] for r in records)
    metrics = {
        "label": label,
        "temperature": artifact["settings"]["temperature"],
        "n": n,
        "candidates": len(records),
        "candidate_verified_rate": sum(values) / len(records) if records else 0.0,
        "igr": sum(informative.values()) / len(counts) if counts else 0.0,
        "z": 1 - sum(1 for c in values if c > 0) / len(values) if values else 0.0,
        "o": sum(1 for v in values if v == n) / len(values) if values else 0.0,
        "mean_successes_per_group": statistics.fmean(values) if values else 0.0,
        "truncation": sum(r["truncated"] for r in records) / len(records) if records else 0.0,
        "format_failure": sum(1 for r in records if not r["format_ok"]) / len(records) if records else 0.0,
        "mean_generated_tokens": statistics.fmean(r["generated_tokens"] for r in records) if records else 0.0,
        "wall_seconds": artifact["resources"]["total_seconds"],
        "total_generated_tokens": total_tokens,
        "informative_groups_per_1M_tokens": round(sum(informative.values()) / (total_tokens / 1e6), 3)
        if total_tokens
        else None,
    }
    return metrics, informative


def paired(inf_a: dict[int, bool], inf_b: dict[int, bool], name_a: str, name_b: str) -> dict:
    indices = sorted(set(inf_a) & set(inf_b))
    a = [inf_a[i] for i in indices]
    b = [inf_b[i] for i in indices]
    newly = sum(1 for x, y in zip(a, b) if y and not x)
    lost = sum(1 for x, y in zip(a, b) if x and not y)
    return {
        "comparison": f"{name_b} vs {name_a}",
        "newly_informative": newly,
        "lost_informative": lost,
        "still_informative": sum(1 for x, y in zip(a, b) if x and y),
        "still_degenerate": sum(1 for x, y in zip(a, b) if not x and not y),
        "mcnemar": mcnemar_exact(a, b),
        "win_tie_loss": win_tie_loss([int(x) for x in a], [int(x) for x in b]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t06-artifact", default="experiments/results/e020_mech_t06_n8.json")
    parser.add_argument("--t10-artifact", default="experiments/results/e020_mech_t10_n8.json")
    parser.add_argument("--output", default="experiments/results/e020_mechanism_analysis.json")
    args = parser.parse_args()

    t06 = ROOT / args.t06_artifact
    t10 = ROOT / args.t10_artifact
    conditions: dict[str, dict] = {}
    informative: dict[str, dict[int, bool]] = {}
    for path, tag in ((t06, "t06"), (t10, "t10")):
        for n in (8, 4):
            metrics, inf = condition_metrics(path, n, f"{tag}_n{n}")
            conditions[f"{tag}_n{n}"] = metrics
            informative[f"{tag}_n{n}"] = inf

    comparisons = [
        paired(informative["t06_n4"], informative["t10_n4"], "T=0.6 n=4", "T=1.0 n=4"),
        paired(informative["t06_n8"], informative["t10_n8"], "T=0.6 n=8", "T=1.0 n=8"),
        paired(informative["t06_n4"], informative["t06_n8"], "n=4 @ T=0.6", "n=8 @ T=0.6"),
        paired(informative["t10_n4"], informative["t10_n8"], "n=4 @ T=1.0", "n=8 @ T=1.0"),
    ]

    artifact = {
        "artifact_type": "e020_mechanism_analysis",
        "sources": {"t06": str(t06.relative_to(ROOT)), "t10": str(t10.relative_to(ROOT))},
        "conditions": conditions,
        "paired_informative_groups": comparisons,
        "interpretation_note": (
            "temperature/group size are associated with changes in the probability of obtaining "
            "informative reward groups under the fixed diagnostic protocol; this does not establish "
            "that they improve final RL performance."
        ),
    }
    output = ROOT / args.output
    output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    header = (
        f"{'condition':>9} | {'verified':>8} | {'IGR':>6} | {'Z':>6} | {'O':>6} | "
        f"{'trunc':>6} | {'fmtfail':>7} | {'inf/1Mtok':>9}"
    )
    print(header)
    print("-" * len(header))
    for label, m in conditions.items():
        print(
            f"{label:>9} | {m['candidate_verified_rate']:>8.4f} | {m['igr']:>6.3f} | {m['z']:>6.3f} | "
            f"{m['o']:>6.3f} | {m['truncation']:>6.3f} | {m['format_failure']:>7.3f} | "
            f"{m['informative_groups_per_1M_tokens']:>9.2f}"
        )
    for item in comparisons:
        m = item["mcnemar"]
        print(
            f"{item['comparison']}: newly={item['newly_informative']} lost={item['lost_informative']} "
            f"still_inf={item['still_informative']} still_deg={item['still_degenerate']} McNemar p={m['p_value_two_sided']:.4f}"
        )
    print(f"Output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
