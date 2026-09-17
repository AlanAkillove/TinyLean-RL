#!/usr/bin/env python3
"""P3-C (E018) checkpoint sanity: parameter-difference evidence.

Loads the base Distill weights and the P3-B checkpoints (global_step_10/20/30),
verifies every key set, and reports:

* per-checkpoint difference vs the base (||theta_t - theta_0|| / ||theta_0||,
  max |diff|, changed keys, plus highlight tensors from several layers);
* consecutive-checkpoint differences (step20 vs step10, step30 vs step20);
* export identity for every exported HF directory (bitwise vs the source .pt).

Output: ``experiments/results/p3c_checkpoint_sanity.json``. Nothing here
modifies weights or trains anything.
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

from tinylean_rl.inference.checkpoint import (
    compare_state_dicts_exact,
    load_checkpoint_state_dict,
    load_safetensors_state_dict,
    state_dict_diff_report,
)


def git_revision() -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="models/weights/kimina_distill_0_6b")
    parser.add_argument("--checkpoint-root", default="runs/p3b_pilot")
    parser.add_argument("--exported-root", default="runs/p3c_models")
    parser.add_argument("--steps", default="10,20,30")
    parser.add_argument("--output", default="experiments/results/p3c_checkpoint_sanity.json")
    args = parser.parse_args()

    base_model = ROOT / args.base_model
    steps = [int(value) for value in args.steps.split(",") if value.strip()]

    print("[1/2] Loading base weights")
    base_state = load_safetensors_state_dict(base_model / "model.safetensors")

    checkpoints: dict[str, dict] = {}
    previous_state = None
    previous_label = None
    for step in [0, *steps]:
        if step == 0:
            label = "step0"
            state = base_state
            entry: dict = {"global_step": 0, "role": "base Kimina-Prover-Distill-0.6B"}
        else:
            label = f"step{step}"
            checkpoint_dir = ROOT / args.checkpoint_root / f"global_step_{step}" / "actor"
            weights_path = checkpoint_dir / "model_world_size_1_rank_0.pt"
            print(f"[2/2] step {step}: diff vs base and export identity")
            state = load_checkpoint_state_dict(weights_path)
            exported = ROOT / args.exported_root / f"step_{step}" / "model.safetensors"
            if not exported.exists():
                raise SystemExit(f"[ERROR] exported weights missing: {exported} (run scripts/p3c_export_checkpoint.py)")
            identity = compare_state_dicts_exact(load_safetensors_state_dict(exported), state)
            entry = {
                "global_step": step,
                "weights_path": str(weights_path),
                "exported_dir": str(exported.parent),
                "diff_vs_base": state_dict_diff_report(state, base_state),
                "export_identity": identity,
            }
        if previous_state is not None:
            entry["diff_vs_previous"] = {
                "previous": previous_label,
                **state_dict_diff_report(state, previous_state)["aggregate"],
            }
        checkpoints[label] = entry
        previous_state = state
        previous_label = label

    for step in steps:
        report = checkpoints[f"step{step}"]
        aggregate = report["diff_vs_base"]["aggregate"]
        print(
            f"  step{step} vs base: rel_l2={aggregate['rel_l2']:.6f} "
            f"max_abs={aggregate['max_abs']:.6f} "
            f"changed_keys={report['diff_vs_base']['changed_keys']}/{report['diff_vs_base']['keys']} "
            f"export_identical={report['export_identity']['identical']}"
        )
        previous = report.get("diff_vs_previous")
        if previous:
            print(
                f"  step{step} vs {previous['previous']}: rel_l2={previous['rel_l2']:.6f} "
                f"max_abs={previous['max_abs']:.6f}"
            )

    summary = {
        "artifact_type": "p3c_checkpoint_sanity",
        "base_model": str(base_model),
        "steps": steps,
        "checkpoints": checkpoints,
        "inequality_evidence": {
            label: {
                "differs_from_base": entry.get("diff_vs_base", {}).get("changed_keys", 0) > 0,
            }
            for label, entry in checkpoints.items()
            if label != "step0"
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
    }
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
