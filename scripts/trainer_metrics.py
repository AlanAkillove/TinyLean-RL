#!/usr/bin/env python
"""Aggregate VERL trainer per-step metrics into range statistics (E017/E019).

The trainer prints one `step:N - key:value - ...` line per optimizer step.
This script extracts the fields needed by the M1 dynamics table (score mean,
grad_norm, clip ratio, entropy, response length, step time) from one or more
logs (E017: steps 1-30, E019: steps 31-60) and prints per-range aggregates.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FIELDS = {
    "score": "critic/score/mean",
    "grad_norm": "actor/grad_norm",
    "clip_ratio": "response_length/clip_ratio",
    "entropy": "actor/entropy",
    "response_len": "response_length/mean",
    "step_time": "timing_s/step",
    "pg_loss": "actor/pg_loss",
    "ppo_kl": "actor/ppo_kl",
}


def parse_log(path: Path) -> dict[int, dict]:
    per_step: dict[int, dict] = {}
    with path.open(encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            match = re.search(r"step:(\d+) - ", line)
            if not match:
                continue
            step = int(match.group(1))
            if step in per_step:
                continue
            values: dict[str, float] = {}
            for key, field in FIELDS.items():
                field_match = re.search(rf"{re.escape(field)}:(-?[\d.eE+]+)", line)
                if field_match:
                    try:
                        values[key] = float(field_match.group(1))
                    except ValueError:
                        continue
            per_step[step] = values
    return per_step


def aggregate(steps: list[int], per_step: dict[int, dict]) -> dict:
    def mean_of(key: str) -> float | None:
        values = [per_step[s][key] for s in steps if key in per_step[s]]
        return statistics.fmean(values) if values else None

    nonzero_grad = sum(1 for s in steps if per_step[s].get("grad_norm"))
    step_times = [per_step[s]["step_time"] for s in steps if "step_time" in per_step[s]]
    return {
        "steps": len(steps),
        "score_mean": mean_of("score"),
        "grad_norm_mean": mean_of("grad_norm"),
        "nonzero_grad_steps": nonzero_grad,
        "clip_ratio_mean": mean_of("clip_ratio"),
        "entropy_mean": mean_of("entropy"),
        "response_len_mean": mean_of("response_len"),
        "step_time_s_mean": mean_of("step_time"),
        "step_time_s_median": statistics.median(step_times) if step_times else None,
        "step_time_s_max": max(step_times) if step_times else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", default=".cache/p3b_pilot.log,.cache/e019_train.log")
    parser.add_argument("--ranges", default="1-10,11-20,21-30,31-40,41-50,51-60")
    parser.add_argument("--output", default="experiments/results/e019_trainer_metrics.json")
    args = parser.parse_args()

    per_step: dict[int, dict] = {}
    for log_name in args.logs.split(","):
        log_path = ROOT / log_name.strip()
        if not log_path.exists():
            print(f"[warn] missing log: {log_path}")
            continue
        for step, values in parse_log(log_path).items():
            per_step.setdefault(step, values)

    ranges: dict[str, dict] = {}
    for chunk in args.ranges.split(","):
        start_text, _, end_text = chunk.partition("-")
        start, end = int(start_text), int(end_text)
        steps = [s for s in range(start, end + 1) if s in per_step]
        if steps:
            ranges[f"{start}-{end}"] = aggregate(steps, per_step)

    artifact = {
        "artifact_type": "e019_trainer_metrics",
        "sources": args.logs.split(","),
        "steps_available": sorted(per_step),
        "per_step": {str(step): per_step[step] for step in sorted(per_step)},
        "ranges": ranges,
    }
    output = ROOT / args.output
    output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    header = f"{'range':>8} | {'score':>7} | {'grad_nz':>7} | {'clip':>6} | {'entropy':>7} | {'resp_len':>8} | {'step_s':>7}"
    print(header)
    print("-" * len(header))
    for label, values in ranges.items():
        print(
            f"{label:>8} | {values['score_mean']:>7.4f} | {values['nonzero_grad_steps']:>3}/{values['steps']:<3} | "
            f"{values['clip_ratio_mean']:>6.3f} | {values['entropy_mean']:>7.2f} | "
            f"{values['response_len_mean']:>8.1f} | {values['step_time_s_mean']:>7.1f}"
        )
    print(f"Output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
