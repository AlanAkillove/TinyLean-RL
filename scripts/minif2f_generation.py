#!/usr/bin/env python3
"""Generation-only diagnostic on real MiniF2F formal statements.

It deliberately does not report Pass@K. Candidates must be sent to Kimina Lean
Server before any generated proof is counted as correct.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from pyarrow import parquet
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Windows consoles default to GBK, which cannot print Lean goal symbols.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from tinylean_rl.inference.extract import extract_proof

SYSTEM_PROMPT = "You are an expert in mathematics and proving theorems in Lean 4."
USER_TEMPLATE = """Think about and solve the following problems step by step in Lean 4.

# Problem:
{informal_problem}

# Formal Statement:
```lean4
{formal_statement}
```
"""
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
LEAN_START_PREFIXES = ("by ", "by\n", "theorem ", "import ")


def make_prompt(tokenizer, informal: str, formal: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(informal_problem=informal, formal_statement=formal)},
    ]
    try:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    except (AttributeError, ValueError):
        return messages[0]["content"] + "\n\n" + messages[1]["content"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Local model path; defaults to TINYLEAN_MODEL_DIR.")
    parser.add_argument("--model-key", default="kimina_distill_0_6b")
    parser.add_argument("--dataset", default="data/raw/minif2f_hf")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--samples-per-theorem", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output", default="experiments/results/minif2f_generation_cuda.json")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    model_dir = args.model or os.getenv("TINYLEAN_MODEL_DIR")
    if not model_dir:
        model_dir = str(ROOT / "models" / "weights" / args.model_key)
    model_path = Path(model_dir)
    if not model_path.exists():
        print(f"[ERROR] model directory not found: {model_path}", file=sys.stderr)
        return 2

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = ROOT / dataset_path
    if dataset_path.is_dir():
        files = sorted(dataset_path.rglob("*.parquet"))
        if len(files) != 1:
            print(f"[ERROR] expected one parquet file under {dataset_path}, found {len(files)}", file=sys.stderr)
            return 2
        dataset_path = files[0]
    table = parquet.read_table(dataset_path).slice(0, args.limit)
    rows = table.to_pylist()

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

    print("[1/3] Loading model and MiniF2F rows")
    tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    device = next(model.parameters()).device
    print(f"[OK] {len(rows)} theorems, model on {device}")

    records = []
    started = time.perf_counter()
    for start in tqdm(
        range(0, len(rows), args.batch_size),
        desc=args.model_key,
        unit="theorem",
    ):
        batch_rows = rows[start : start + args.batch_size]
        prompts = [make_prompt(tokenizer, row.get("informal_prefix") or "", row["formal_statement"]) for row in batch_rows]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            output = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                num_return_sequences=args.samples_per_theorem,
                pad_token_id=tokenizer.eos_token_id,
            )
        prompt_width = encoded["input_ids"].shape[-1]
        for row_index, row in enumerate(batch_rows):
            for sample in range(args.samples_per_theorem):
                output_index = row_index * args.samples_per_theorem + sample
                raw = tokenizer.decode(output[output_index][prompt_width:], skip_special_tokens=True)
                try:
                    extracted = extract_proof(raw)
                    extraction_ok = bool(extracted)
                except ValueError:
                    extracted = ""
                    extraction_ok = False
                first_line = extracted.splitlines()[0].strip() if extracted else ""
                has_lean4_code_block = bool(re.search(r"```lean4\s*\n.*?```", raw, re.DOTALL))
                likely_lean_candidate = bool(
                    extracted
                    and (has_lean4_code_block or first_line.startswith(LEAN_START_PREFIXES))
                )
                records.append(
                    {
                        "theorem_index": start + row_index,
                        "name": row["name"],
                        "sample": sample,
                        "raw_output": raw,
                        "extracted": extracted,
                        "raw_tokens": len(tokenizer.encode(raw, add_special_tokens=False)),
                        "extraction_ok": extraction_ok,
                        "likely_lean_candidate": likely_lean_candidate,
                        "tactic_like": first_line.startswith(TACTIC_PREFIXES),
                        "contains_markdown_fence": "```" in raw,
                        "has_lean4_code_block": has_lean4_code_block,
                        "has_think_block": "<think>" in raw and "</think>" in raw,
                    }
                )

    elapsed = time.perf_counter() - started
    summary = {
        "artifact_type": "minif2f_generation_only_diagnostic",
        "warning": "No record is a verified proof until submitted to Kimina Lean Server.",
        "model_key": args.model_key,
        "dataset_path": str(dataset_path.resolve()),
        "theorems": len(rows),
        "samples_per_theorem": args.samples_per_theorem,
        "generated_candidates": len(records),
        "max_new_tokens": args.max_new_tokens,
        "generation_seconds": round(elapsed, 3),
        "extraction_successes": sum(record["likely_lean_candidate"] for record in records),
        "tactic_like_candidates": sum(record["tactic_like"] for record in records),
        "markdown_contamination": sum(record["contains_markdown_fence"] for record in records),
        "lean4_code_blocks": sum(record["has_lean4_code_block"] for record in records),
        "complete_think_blocks": sum(record["has_think_block"] for record in records),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "records": records,
    }
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("[3/3] Diagnostic saved")
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2, ensure_ascii=False))
    print(f"Output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
