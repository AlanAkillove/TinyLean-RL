#!/usr/bin/env python3
"""Export P3-B VERL checkpoints to HuggingFace inference directories.

The VERL/FSDP ``model_world_size_1_rank_0.pt`` files already use HuggingFace
``Qwen3ForCausalLM`` key names (identity mapping, validated against the base
model); this script

1. verifies the key sets match the base model exactly,
2. copies the checkpoint's own ``huggingface/`` config + tokenizer files,
3. writes ``model.safetensors`` without modifying any tensor, and
4. bitwise-compares the exported weights to the source state dict.

Weights are never modified and no training happens here.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tinylean_rl.inference.checkpoint import (
    compare_state_dicts_exact,
    export_hf_weights,
    load_checkpoint_state_dict,
    load_safetensors_state_dict,
    verify_key_compat,
)

REQUIRED_TOKENIZER_FILES = ("config.json", "tokenizer_config.json", "tokenizer.json")


def export_one(base_model: Path, checkpoint_dir: Path, output_dir: Path) -> dict:
    state_dict_path = checkpoint_dir / "model_world_size_1_rank_0.pt"
    if not state_dict_path.exists():
        raise SystemExit(f"[ERROR] missing checkpoint weights: {state_dict_path}")
    state_dict = load_checkpoint_state_dict(state_dict_path)

    base_weights = base_model / "model.safetensors"
    if not base_weights.exists():
        raise SystemExit(f"[ERROR] missing base weights: {base_weights}")
    base_keys = set(load_safetensors_state_dict(base_weights).keys())
    verify_key_compat(state_dict, base_keys, str(state_dict_path))

    hf_meta = checkpoint_dir / "huggingface"
    missing_meta = [name for name in REQUIRED_TOKENIZER_FILES if not (hf_meta / name).exists()]
    if missing_meta:
        raise SystemExit(f"[ERROR] checkpoint huggingface/ missing: {missing_meta}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for item in sorted(hf_meta.iterdir()):
        if item.is_file():
            shutil.copy2(item, output_dir / item.name)

    export_hf_weights(state_dict, output_dir / "model.safetensors")
    reloaded = load_safetensors_state_dict(output_dir / "model.safetensors")
    identity = compare_state_dicts_exact(reloaded, state_dict)
    if not identity["identical"]:
        raise SystemExit(f"[ERROR] export is not bitwise identical: {identity}")
    return {
        "checkpoint": str(checkpoint_dir),
        "exported_dir": str(output_dir),
        "keys": identity["keys"],
        "identity_identical": True,
        "weights_bytes": (output_dir / "model.safetensors").stat().st_size,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="models/weights/kimina_distill_0_6b")
    parser.add_argument("--checkpoint-root", default="runs/p3b_pilot")
    parser.add_argument("--steps", default="10,20,30")
    parser.add_argument("--output-root", default="runs/p3c_models")
    args = parser.parse_args()

    base_model = ROOT / args.base_model
    checkpoint_root = ROOT / args.checkpoint_root
    output_root = ROOT / args.output_root
    steps = [int(value) for value in args.steps.split(",") if value.strip()]

    report = {
        "artifact_type": "p3c_checkpoint_export",
        "base_model": str(base_model),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "exports": [],
    }
    for step in steps:
        checkpoint_dir = checkpoint_root / f"global_step_{step}" / "actor"
        output_dir = output_root / f"step_{step}"
        print(f"[export] step {step}: {checkpoint_dir} -> {output_dir}")
        entry = export_one(base_model, checkpoint_dir, output_dir)
        entry["global_step"] = step
        report["exports"].append(entry)
        print(f"  keys={entry['keys']} identical=True size={entry['weights_bytes'] / 1e9:.2f} GB")

    report_path = output_root / "export_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
