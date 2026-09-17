#!/usr/bin/env python
"""Rollout-dynamics statistics from VERL rollout dumps (E017 / E019).

Reads <dir>/{global_step}.jsonl (one file per training step, written by
trainer.rollout_data_dir), groups the n samples that share a prompt into its
reward group, and reports per-step and per-range statistics:

    IGR (mixed groups) / Z (all-zero) / O (all-one) / score mean / verified rate

P3-B (E017) reference values from the same dumps: steps 1-10 IGR 0.100,
11-20 0.200, 21-30 0.225; Z 0.900 -> 0.800 -> 0.750; score 0.0594 -> 0.1750.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def step_group_stats(path: Path) -> dict:
    groups: dict[str, list[float]] = defaultdict(list)
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue  # the newest step file may be mid-write while training runs
            groups[entry["input"]].append(float(entry["score"]))
    total = len(groups)
    if total == 0:
        return {"prompts": 0, "samples": 0, "igr": 0.0, "z": 0.0, "o": 0.0, "score_mean": 0.0, "verified_rate": 0.0}
    mixed = sum(1 for scores in groups.values() if 0 < sum(1 for s in scores if s > 0) < len(scores))
    all_zero = sum(1 for scores in groups.values() if all(s == 0 for s in scores))
    all_one = sum(1 for scores in groups.values() if all(s > 0 for s in scores))
    flat = [s for scores in groups.values() for s in scores]
    return {
        "prompts": total,
        "samples": len(flat),
        "igr": mixed / total,
        "z": all_zero / total,
        "o": all_one / total,
        "score_mean": statistics.fmean(flat),
        "verified_rate": sum(1 for s in flat if s > 0) / len(flat),
    }


def aggregate(steps: list[int], per_step: dict[int, dict]) -> dict:
    igr = statistics.fmean(per_step[s]["igr"] for s in steps)
    z = statistics.fmean(per_step[s]["z"] for s in steps)
    o = statistics.fmean(per_step[s]["o"] for s in steps)
    score = statistics.fmean(per_step[s]["score_mean"] for s in steps)
    nonzero = sum(1 for s in steps if per_step[s]["igr"] > 0)
    return {"steps": len(steps), "igr": igr, "z": z, "o": o, "score_mean": score, "nonzero_group_steps": nonzero}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="runs/p3b_pilot/rollout_data")
    parser.add_argument("--ranges", default="1-10,11-20,21-30,31-40,41-50,51-60")
    parser.add_argument("--output", default="experiments/results/e019_dynamics.json")
    args = parser.parse_args()

    dump_dir = ROOT / args.dir
    per_step: dict[int, dict] = {}
    for path in sorted(dump_dir.glob("*.jsonl"), key=lambda p: int(p.stem)):
        per_step[int(path.stem)] = step_group_stats(path)
    if not per_step:
        raise SystemExit(f"[ERROR] no rollout dumps under {dump_dir}")

    ranges: dict[str, dict] = {}
    for chunk in args.ranges.split(","):
        start_text, _, end_text = chunk.partition("-")
        start, end = int(start_text), int(end_text)
        steps = [s for s in range(start, end + 1) if s in per_step]
        if steps:
            ranges[f"{start}-{end}"] = aggregate(steps, per_step)
    full = aggregate(sorted(per_step), per_step)

    artifact = {
        "artifact_type": "e019_rollout_dynamics",
        "source_dir": str(dump_dir.relative_to(ROOT)),
        "steps_available": sorted(per_step),
        "per_step": {str(step): per_step[step] for step in sorted(per_step)},
        "ranges": ranges,
        "all_steps": full,
    }
    output = ROOT / args.output
    output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    header = f"{'range':>8} | {'steps':>5} | {'IGR':>6} | {'Z':>6} | {'O':>6} | {'score':>7} | nonzero"
    print(header)
    print("-" * len(header))
    for label, values in ranges.items():
        print(
            f"{label:>8} | {values['steps']:>5} | {values['igr']:>6.3f} | {values['z']:>6.3f} | "
            f"{values['o']:>6.3f} | {values['score_mean']:>7.4f} | {values['nonzero_group_steps']}"
        )
    print(f"{'all':>8} | {full['steps']:>5} | {full['igr']:>6.3f} | {full['z']:>6.3f} | {full['o']:>6.3f} | {full['score_mean']:>7.4f} | {full['nonzero_group_steps']}")
    print(f"Output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
