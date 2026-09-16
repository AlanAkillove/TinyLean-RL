#!/usr/bin/env python3
"""Run a model→parser→Kimina evaluation and compute Pass@K/IGR.

This is the first evaluator intended for P2. It refuses to call a missing
verifier only when ``--dry-run`` is explicitly selected; dry-run outputs are
never labelled as verified results.
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
from typing import Any

import httpx
from pyarrow import parquet
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Windows consoles default to GBK, which cannot print Lean goal symbols.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from tinylean_rl.evaluation.metrics import group_rates, pass_at_k
from tinylean_rl.inference.extract import extract_proof
from tinylean_rl.verifier.kimina import result_lean_status, verify_code, verify_codes

SYSTEM_PROMPT = "You are an expert in mathematics and proving theorems in Lean 4."
USER_TEMPLATE = """Think about and solve the following problems step by step in Lean 4.

# Problem:
{informal_problem}

# Formal Statement:
```lean4
{formal_statement}
```
"""


def find_parquet(path: Path, split: str) -> Path:
    if path.is_file():
        return path
    preferred = path / f"{split}.parquet"
    if preferred.is_file():
        return preferred
    files = sorted(path.rglob("*.parquet"))
    if len(files) != 1:
        raise SystemExit(f"Expected one parquet file under {path}, found {len(files)}")
    return files[0]


def build_prompt(tokenizer, row: dict[str, Any]) -> str:
    if isinstance(row.get("prompt"), list):
        try:
            return tokenizer.apply_chat_template(row["prompt"], add_generation_prompt=True, tokenize=False)
        except (AttributeError, ValueError):
            pass
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(
                informal_problem=row.get("informal_prefix") or row.get("informal_problem") or "",
                formal_statement=row["formal_statement"],
            ),
        },
    ]
    try:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    except (AttributeError, ValueError):
        return messages[0]["content"] + "\n\n" + messages[1]["content"]


def complete_verifier_code(formal_statement: str, extracted: str) -> str | None:
    if not extracted.strip():
        return None
    if extracted.lstrip().startswith(("import ", "theorem ", "lemma ", "example ")):
        return extracted.strip()
    formal = formal_statement.rstrip()
    if re.search(r":=\s*by$", formal):
        body = extracted.strip()
        if body.startswith("by"):
            body = body[2:].lstrip()
        return f"{formal}\n{body}".strip()
    return f"{formal}\nby\n{extracted.strip()}".strip()


def response_items(decoded: dict[str, Any]) -> list[dict[str, Any]]:
    value: Any = decoded
    if isinstance(decoded, dict):
        for key in ("results", "codes", "data", "responses"):
            if key in decoded:
                value = decoded[key]
                break
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def classify_verification(item: dict[str, Any] | None) -> tuple[bool, str]:
    """Map one Kimina result item to ``(verified, verify_status)``.

    ``verified`` follows the official semantics: no REPL error, no ``error``
    severity message and no ``sorry``. ``verify_status`` separates genuine
    proof failures (``lean_error``/``sorry``) from verifier-side trouble
    (``verifier_error``: timeout, REPL/server error, missing response).
    """

    if item is None:
        return False, "verifier_error"
    status = result_lean_status(item)
    if status == "valid":
        return True, "verified"
    if status == "lean_error":
        return False, "lean_error"
    if status == "sorry":
        return False, "sorry"
    return False, "verifier_error"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Local model path; defaults to TINYLEAN_MODEL_DIR.")
    parser.add_argument("--model-key", default="kimina_distill_0_6b")
    parser.add_argument("--dataset", default="data/raw/minif2f_hf")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--samples-per-theorem", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--output", default="experiments/results/p2_evaluation.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    model_dir = args.model or os.getenv("TINYLEAN_MODEL_DIR")
    if not model_dir:
        model_dir = str(ROOT / "models" / "weights" / args.model_key)
    model_path = Path(model_dir)
    if not model_path.exists():
        print(f"[ERROR] model directory not found: {model_path}", file=sys.stderr)
        return 2
    dataset_arg = Path(args.dataset)
    if not dataset_arg.is_absolute():
        dataset_arg = ROOT / dataset_arg
    dataset_path = find_parquet(dataset_arg, args.split)
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

    print(f"[1/4] Loading {args.model_key} and {len(rows)} theorem rows")
    tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    device = next(model.parameters()).device
    print(f"[OK] Model on {device}")

    generated_records: list[dict[str, Any]] = []
    generation_seconds = 0.0
    verification_seconds = 0.0
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_suffix(".partial.json")
    for start in tqdm(
        range(0, len(rows), args.batch_size),
        desc=f"{args.model_key} [{args.split}]",
        unit="theorem",
    ):
        batch_rows = rows[start : start + args.batch_size]
        prompts = [build_prompt(tokenizer, row) for row in batch_rows]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        started = time.perf_counter()
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
        generation_seconds += time.perf_counter() - started
        prompt_width = encoded["input_ids"].shape[-1]

        pending: list[tuple[int, str, str]] = []
        for row_index, row in enumerate(batch_rows):
            theorem_index = start + row_index
            for sample in range(args.samples_per_theorem):
                output_index = row_index * args.samples_per_theorem + sample
                raw = tokenizer.decode(output[output_index][prompt_width:], skip_special_tokens=True)
                try:
                    extracted = extract_proof(raw)
                except ValueError:
                    extracted = ""
                proof = complete_verifier_code(row["formal_statement"], extracted)
                record = {
                    "theorem_index": theorem_index,
                    "name": row.get("name", row.get("statement_id", str(theorem_index))),
                    "sample": sample,
                    "raw_output": raw,
                    "raw_tokens": len(tokenizer.encode(raw, add_special_tokens=False)),
                    "has_lean4_code_block": bool(re.search(r"```lean4\s*\n.*?```", raw, re.DOTALL)),
                    "has_complete_think_block": "<think>" in raw and "</think>" in raw,
                    "format_ok": bool(proof),
                    "verified": False,
                    "verify_status": "not_checked",
                }
                generated_records.append(record)
                if proof and not args.dry_run:
                    pending.append((len(generated_records) - 1, f"{theorem_index}-{sample}", proof))

        if pending:
            verify_started = time.perf_counter()
            proofs = [proof for _, _, proof in pending]
            identifiers = [custom_id for _, custom_id, _ in pending]
            decoded: dict[str, Any] = {}
            for attempt in range(2):
                try:
                    decoded = verify_codes(proofs, custom_ids=identifiers)
                    break
                except httpx.HTTPError as exc:
                    if attempt == 0:
                        # The server briefly drops requests while it rebuilds a
                        # crashed REPL pool; wait before one more attempt.
                        time.sleep(15)
                    else:
                        print(f"  [warn] batch verify failed after retry ({exc}); retrying per candidate")
            items = response_items(decoded)
            if items:
                by_id = {str(item.get("custom_id")): item for item in items}
                for result_index, (record_index, custom_id, _) in enumerate(pending):
                    item = by_id.get(custom_id)
                    if item is None and result_index < len(items):
                        item = items[result_index]
                    valid, status = classify_verification(item)
                    generated_records[record_index]["verified"] = valid
                    generated_records[record_index]["verify_status"] = status
            else:
                # A crashing candidate (e.g. native_decide on huge values) must
                # not invalidate its healthy batch neighbours.
                for record_index, custom_id, proof in pending:
                    try:
                        single = response_items(verify_code(proof, custom_id=custom_id))
                        valid, status = classify_verification(single[0] if single else None)
                        generated_records[record_index]["verified"] = valid
                        generated_records[record_index]["verify_status"] = status
                    except httpx.HTTPError:
                        generated_records[record_index]["verify_status"] = "verifier_error"
                        print(f"  [warn] candidate {custom_id} left unverified (verifier error)")
            verification_seconds += time.perf_counter() - verify_started
        # Persist progress so a late crash cannot discard earlier batches.
        partial_path.write_text(
            json.dumps(
                {
                    "model_key": args.model_key,
                    "processed_theorems": min(start + args.batch_size, len(rows)),
                    "records": generated_records,
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

    counts = []
    theorem_summaries = []
    for theorem_index, row in enumerate(rows):
        theorem_records = [record for record in generated_records if record["theorem_index"] == theorem_index]
        verified_count = sum(record["verified"] for record in theorem_records)
        counts.append(verified_count)
        theorem_summaries.append(
            {
                "theorem_index": theorem_index,
                "name": row.get("name", row.get("statement_id", str(theorem_index))),
                "verified_count": verified_count,
                "format_failures": sum(not record["format_ok"] for record in theorem_records),
                "avg_generated_tokens": sum(record["raw_tokens"] for record in theorem_records) / len(theorem_records),
            }
        )

    rates = group_rates(counts, args.samples_per_theorem)
    lean_statuses = sorted(
        {record["lean_status"] for record in generated_records if record["lean_status"] is not None}
    )
    summary: dict[str, Any] = {
        "artifact_type": "verified_evaluation" if not args.dry_run else "generation_only_dry_run",
        "warning": None if not args.dry_run else "Dry-run records are not verified proofs.",
        "model_key": args.model_key,
        "dataset_path": str(dataset_path.resolve()),
        "theorems": len(rows),
        "samples_per_theorem": args.samples_per_theorem,
        "generated_candidates": len(generated_records),
        "generation_seconds": round(generation_seconds, 3),
        "verification_seconds": round(verification_seconds, 3),
        "format_failures": sum(not record["format_ok"] for record in generated_records),
        "verified_candidates": sum(record["verified"] for record in generated_records),
        "lean_status_counts": {
            status: sum(record["lean_status"] == status for record in generated_records)
            for status in lean_statuses
        },
        "group_rates": rates,
        "pass_at_k": {
            str(k): sum(pass_at_k(count, args.samples_per_theorem, k) for count in counts) / len(counts)
            for k in (1, 8, 32)
            if k <= args.samples_per_theorem
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "theorem_summaries": theorem_summaries,
        "records": generated_records,
    }
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    partial_path.unlink(missing_ok=True)
    print("[4/4] Evaluation saved")
    print(json.dumps({key: value for key, value in summary.items() if key not in {"records", "theorem_summaries"}}, indent=2, ensure_ascii=False))
    print(f"Output: {output_path}")
    if args.dry_run:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
