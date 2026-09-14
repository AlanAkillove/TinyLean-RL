#!/usr/bin/env python3
"""Measure local model generation/formatting before Lean verification is available.

This is a diagnostic artifact, not a benchmark score: every result must still be
checked by Kimina Lean Server before it is treated as a proof success.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tinylean_rl.inference.extract import extract_proof

CASES = [
    ("add_one", "Complete the Lean 4 proof. Return only Lean code.\n\nimport Mathlib\n\nexample : 1 + 1 = 2 := by\n"),
    ("add_two", "Complete the Lean 4 proof. Return only Lean code.\n\nimport Mathlib\n\nexample : 2 + 2 = 4 := by\n"),
    ("zero_add", "Complete the Lean 4 proof. Return only Lean code.\n\nimport Mathlib\n\nexample : 0 + 5 = 5 := by\n"),
    ("true_intro", "Complete the Lean 4 proof. Return only Lean code.\n\nimport Mathlib\n\nexample : True := by\n"),
]

TACTIC_PREFIXES = (
    "by",
    "aesop",
    "apply",
    "constructor",
    "exact",
    "ext",
    "intro",
    "linarith",
    "norm_num",
    "omega",
    "rfl",
    "simp",
    "trivial",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Local model path; defaults to TINYLEAN_MODEL_DIR.")
    parser.add_argument("--model-key", default="kimina_distill_0_6b")
    parser.add_argument("--samples-per-case", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--output", default="experiments/results/p0_generation_diagnostic.json")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    model_dir = args.model or os.getenv("TINYLEAN_MODEL_DIR")
    if not model_dir:
        model_dir = str(ROOT / "models" / "weights" / args.model_key)
    if not Path(model_dir).exists():
        print(f"[ERROR] model directory not found: {model_dir}", file=sys.stderr)
        return 2

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        print(f"[ERROR] inference dependencies unavailable: {exc}", file=sys.stderr)
        return 2

    tokenizer_kwargs = {"trust_remote_code": True}
    load_kwargs = {"trust_remote_code": True}
    if args.offline:
        tokenizer_kwargs["local_files_only"] = True
        load_kwargs["local_files_only"] = True
    if torch.cuda.is_available():
        load_kwargs.update({"dtype": torch.float16, "device_map": "auto"})

    print("[1/3] Loading model")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, **tokenizer_kwargs)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_dir, **load_kwargs)
    device = next(model.parameters()).device
    print(f"[OK] Model loaded on {device}")

    prompts = [prompt for _, prompt in CASES]
    encoded = tokenizer(prompts, return_tensors="pt", padding=True)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    print(f"[2/3] Generating {len(CASES) * args.samples_per_case} candidates")
    started = time.perf_counter()
    with torch.no_grad():
        output = model.generate(
            **encoded,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            num_return_sequences=args.samples_per_case,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - started

    prompt_width = encoded["input_ids"].shape[-1]
    records = []
    for row, (case_id, _) in enumerate(CASES):
        for sample in range(args.samples_per_case):
            index = row * args.samples_per_case + sample
            raw = tokenizer.decode(output[index][prompt_width:], skip_special_tokens=True)
            try:
                extracted = extract_proof(raw)
                extraction_ok = True
            except ValueError:
                extracted = ""
                extraction_ok = False
            first_line = extracted.splitlines()[0].strip() if extracted else ""
            tactic_like = first_line.startswith(TACTIC_PREFIXES)
            records.append(
                {
                    "case_id": case_id,
                    "sample": sample,
                    "raw_output": raw,
                    "extracted": extracted,
                    "extraction_ok": extraction_ok,
                    "raw_tokens": len(tokenizer.encode(raw, add_special_tokens=False)),
                    "contains_markdown_fence": "```" in raw,
                    "tactic_like": tactic_like,
                }
            )

    summary = {
        "artifact_type": "generation_only_diagnostic",
        "warning": "No record is a verified proof until the candidate is submitted to Kimina Lean Server.",
        "model_key": args.model_key,
        "model_dir": str(Path(model_dir).resolve()),
        "samples_per_case": args.samples_per_case,
        "max_new_tokens": args.max_new_tokens,
        "generation_seconds": round(elapsed, 3),
        "generated_candidates": len(records),
        "extraction_successes": sum(record["extraction_ok"] for record in records),
        "tactic_like_candidates": sum(record["tactic_like"] for record in records),
        "markdown_contamination": sum(record["contains_markdown_fence"] for record in records),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "records": records,
    }
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("[3/3] Diagnostic saved")
    print(json.dumps({key: summary[key] for key in summary if key != "records"}, ensure_ascii=False, indent=2))
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
