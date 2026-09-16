#!/usr/bin/env python3
"""Reward profile of the Kimina-Prover-Promptset train distribution.

This is the P2.5 local RL-readiness probe. It samples unique prompts from the
actual P3 training corpus, generates ``K`` candidates per prompt, verifies each
candidate against the local Kimina Lean Server with the official reward
validity rule (no error-severity message **and** no ``sorry``), and writes

* ``experiments/results/p2_5_promptset_profile.json`` - IGR/Z/O group rates,
  truncation / format / verification statistics and prompt token lengths;
* ``experiments/local_rl_batch/`` - cached rollouts (prompt/completion token
  ids, rewards, verifier messages) for the offline GRPO loss rehearsal.

Nothing in this script trains a model. Generation hyper-parameters follow the
E007 evaluation setting (temperature 0.6 / top_p 0.95); the official P3 rollout
uses temperature 1.0, and that difference is only recorded in the metadata.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
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

from tinylean_rl.evaluation.metrics import group_rates
from tinylean_rl.inference.extract import extract_proof
from tinylean_rl.verifier.kimina import verify_code, verify_codes

# Prompt construction must stay byte-identical to the pinned upstream recipe
# (third_party/kimina-prover-rl/recipe/kimina_prover_rl/prepare_data.py).
SYSTEM_PROMPT = "You are an expert in mathematics and proving theorems in Lean 4."
USER_TEMPLATE = """Think about and solve the following problems step by step in Lean 4.

# Problem:
{informal_problem}

# Formal Statement:
```lean4
{formal_statement}
```
"""

REDECLARATION_MARKER = "has already been declared"
DEFAULT_DATASET = "data/raw/kimina_promptset/data/train-00000-of-00001.parquet"


def find_parquet(path: Path) -> Path:
    if path.is_file():
        return path
    files = sorted(path.rglob("*.parquet"))
    if len(files) != 1:
        raise SystemExit(f"Expected one parquet file under {path}, found {len(files)}")
    return files[0]


def build_prompt_text(tokenizer, row: dict[str, Any]) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(
                informal_problem=row.get("natural_language") or "",
                formal_statement=row["formal_statement"],
            ),
        },
    ]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def complete_verifier_code(formal_statement: str, extracted: str) -> str | None:
    """Rebuild the Lean source used for verification.

    The Promptset statements already carry ``import Mathlib``; most of them end
    with a ``:= by sorry`` placeholder, which is stripped before the generated
    proof body is appended (mirroring ``normalize_formal_statement`` upstream).
    """

    if not extracted.strip():
        return None
    if extracted.lstrip().startswith(("import ", "theorem ", "lemma ", "example ")):
        return extracted.strip()
    formal = formal_statement.rstrip()
    formal = re.sub(r":=\s*by\s+sorry\s*$", ":= by", formal)
    if not re.search(r":=\s*by$", formal):
        formal = re.sub(r":=\s*sorry\s*$", ":= by", formal)
        if not re.search(r":=\s*by$", formal):
            formal = f"{formal} := by"
    body = extracted.strip()
    if body.startswith("by"):
        body = body[2:].lstrip()
    return f"{formal}\n{body}".strip()


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


def analyze_item(item: dict[str, Any] | None) -> dict[str, Any]:
    """Apply the official validity rule to one Kimina response item.

    Official rule (kimina_client ``is_valid``): the candidate is a proof only
    when ``response.error`` is empty, no message carries severity ``error`` and
    the ``sorries`` array is empty. Boolean flags from older clients are still
    honoured for compatibility.
    """

    result: dict[str, Any] = {
        "verified": False,
        "has_sorry": False,
        "sorries": 0,
        "error_messages": 0,
        "lean_message": "",
        "redeclaration": False,
        "status": "verifier_error",
    }
    if item is None:
        result["lean_message"] = "no response item for this custom_id"
        return result

    response = item.get("response")
    if isinstance(response, dict):
        if response.get("error"):
            result["lean_message"] = str(response["error"])[:500]
            return result
        sorries = response.get("sorries") or []
        messages = response.get("messages") or []
        errors = [
            str(message.get("data", ""))
            for message in messages
            if str(message.get("severity", "")).lower() == "error"
        ]
        result["sorries"] = len(sorries)
        result["has_sorry"] = bool(sorries)
        result["error_messages"] = len(errors)
        if errors:
            result["status"] = "lean_error"
            result["lean_message"] = errors[0][:500]
        elif sorries:
            result["status"] = "lean_error"
            result["lean_message"] = "declaration uses 'sorry'"
        else:
            result["status"] = "verified"
            result["verified"] = True
        result["redeclaration"] = any(REDECLARATION_MARKER in error for error in errors)
        return result

    for key in ("is_valid", "valid", "verified", "success", "isSuccess"):
        if isinstance(item.get(key), bool):
            result["verified"] = item[key]
            result["status"] = "verified" if item[key] else "lean_error"
            return result
    status = str(item.get("status", item.get("result", ""))).lower()
    result["verified"] = status in {"valid", "verified", "success", "successful", "ok", "pass", "passed"}
    result["status"] = "verified" if result["verified"] else "lean_error"
    return result


def prompt_length_stats(tokenizer, rows: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = []
    for row in rows:
        text = build_prompt_text(tokenizer, row)
        lengths.append(len(tokenizer(text, add_special_tokens=False)["input_ids"]))
    lengths.sort()
    if not lengths:
        return {"count": 0}
    index95 = max(0, int(0.95 * len(lengths)) - 1)
    index99 = max(0, int(0.99 * len(lengths)) - 1)
    return {
        "count": len(lengths),
        "min": lengths[0],
        "median": lengths[len(lengths) // 2],
        "p95": lengths[index95],
        "p99": lengths[index99],
        "max": lengths[-1],
        "mean": round(sum(lengths) / len(lengths), 2),
    }


def git_revision() -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Local model path; defaults to TINYLEAN_MODEL_DIR.")
    parser.add_argument("--model-key", default="kimina_distill_0_6b")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--limit", type=int, default=32, help="Number of unique statements to sample.")
    parser.add_argument("--samples-per-theorem", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verify-timeout", type=float, default=120.0)
    parser.add_argument("--first-verify-timeout", type=float, default=600.0)
    parser.add_argument("--output", default="experiments/results/p2_5_promptset_profile.json")
    parser.add_argument("--batch-dir", default="experiments/local_rl_batch")
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
    dataset_path = find_parquet(dataset_path)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    batch_dir = Path(args.batch_dir)
    if not batch_dir.is_absolute():
        batch_dir = ROOT / batch_dir
    batch_dir.mkdir(parents=True, exist_ok=True)

    rows = parquet.read_table(dataset_path).to_pylist()
    unique_rows: dict[str, dict[str, Any]] = {}
    for row in rows:
        unique_rows.setdefault(row["statement_id"], row)
    unique_list = list(unique_rows.values())
    sampled = random.Random(args.seed).sample(unique_list, min(args.limit, len(unique_list)))
    print(
        f"[1/5] Promptset: {len(rows)} rows, {len(unique_list)} unique statements, "
        f"sampled {len(sampled)} (seed={args.seed})"
    )

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
        load_kwargs.update({"torch_dtype": torch.float16, "device_map": "auto"})

    tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)
    tokenizer.padding_side = "left"

    print(f"[2/5] Tokenizer ready; tokenizing all {len(unique_list)} unique prompts")
    all_prompt_stats = prompt_length_stats(tokenizer, unique_list)
    sampled_prompt_stats = prompt_length_stats(tokenizer, sampled)
    print(f"  all-unique prompt tokens: {json.dumps(all_prompt_stats)}")

    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    device = next(model.parameters()).device
    print(f"[3/5] Model on {device}; generating {len(sampled)}x{args.samples_per_theorem} candidates")

    records: list[dict[str, Any]] = []
    prompt_records: list[dict[str, Any]] = []
    partial_path = output_path.with_suffix(".partial.json")
    generation_seconds = 0.0
    verification_seconds = 0.0
    first_verify = True
    eos_id = tokenizer.eos_token_id

    for start in tqdm(range(0, len(sampled), args.batch_size), desc=args.model_key, unit="theorem"):
        batch_rows = sampled[start : start + args.batch_size]
        prompts = [build_prompt_text(tokenizer, row) for row in batch_rows]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        started = time.perf_counter()
        with torch.no_grad():
            output = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                num_return_sequences=args.samples_per_theorem,
                pad_token_id=eos_id,
            )
        batch_seconds = time.perf_counter() - started
        generation_seconds += batch_seconds
        if device.type == "cuda":
            # The 8 GB card OOMs when the allocator keeps the previous batch's
            # cache pool; return it to the driver before the next generate.
            torch.cuda.empty_cache()
        prompt_width = encoded["input_ids"].shape[-1]
        seconds_per_candidate = batch_seconds / (len(batch_rows) * args.samples_per_theorem)

        pending: list[tuple[int, str, str]] = []
        for row_index, row in enumerate(batch_rows):
            theorem_index = start + row_index
            prompt_ids = encoded["input_ids"][row_index][encoded["attention_mask"][row_index].bool()].tolist()
            prompt_records.append(
                {
                    "theorem_index": theorem_index,
                    "statement_id": row["statement_id"],
                    "name": row.get("name", row["statement_id"]),
                    "source": row.get("source"),
                    "prompt": prompts[row_index],
                    "prompt_ids": prompt_ids,
                    "prompt_tokens": len(prompt_ids),
                    "formal_statement": row["formal_statement"],
                    "natural_language": row.get("natural_language") or "",
                }
            )
            for sample in range(args.samples_per_theorem):
                output_index = row_index * args.samples_per_theorem + sample
                completion_ids = output[output_index][prompt_width:].tolist()
                raw = tokenizer.decode(completion_ids, skip_special_tokens=True)
                try:
                    extracted = extract_proof(raw)
                except ValueError:
                    extracted = ""
                proof = complete_verifier_code(row["formal_statement"], extracted)
                record = {
                    "theorem_index": theorem_index,
                    "statement_id": row["statement_id"],
                    "name": row.get("name", row["statement_id"]),
                    "sample": sample,
                    "raw_output": raw,
                    "completion_ids": completion_ids,
                    "completion_tokens": len(completion_ids),
                    "eos_reached": eos_id in completion_ids,
                    "truncated": eos_id not in completion_ids and len(completion_ids) >= args.max_new_tokens,
                    "has_lean4_code_block": bool(re.search(r"```lean4?\s*\n.*?```", raw, re.DOTALL)),
                    "has_complete_think_block": "<think>" in raw and "</think>" in raw,
                    "format_ok": bool(proof),
                    "proof": proof or "",
                    "generation_time": round(seconds_per_candidate, 3),
                    "verified": False,
                    "has_sorry": False,
                    "sorries": 0,
                    "verify_status": "not_checked",
                    "lean_message": "",
                    "verification_time": None,
                }
                records.append(record)
                if proof:
                    pending.append((len(records) - 1, f"{row['statement_id']}-{sample}", proof))

        if pending:
            verify_started = time.perf_counter()
            timeout = args.first_verify_timeout if first_verify else args.verify_timeout
            proofs = [proof for _, _, proof in pending]
            identifiers = [custom_id for _, custom_id, _ in pending]
            decoded: dict[str, Any] = {}
            for attempt in range(2):
                try:
                    decoded = verify_codes(proofs, custom_ids=identifiers, timeout=timeout)
                    break
                except httpx.HTTPError as exc:
                    if attempt == 0:
                        # The server briefly drops requests while it rebuilds a
                        # crashed REPL pool; wait before one more attempt.
                        time.sleep(15)
                    else:
                        print(f"  [warn] batch verify failed after retry ({exc}); retrying per candidate")
            first_verify = False
            items = response_items(decoded)
            by_id = {str(item.get("custom_id")): item for item in items}
            retry_indices: list[tuple[int, str]] = []
            for result_index, (record_index, custom_id, _) in enumerate(pending):
                item = by_id.get(custom_id)
                if item is None and result_index < len(items) and len(items) == len(pending):
                    item = items[result_index]
                if item is None:
                    # The server occasionally answers 200 while dropping some
                    # results (REPL pool rebuild); fall back to one single
                    # candidate check instead of writing verifier_error.
                    retry_indices.append((result_index, "missing_item"))
                    continue
                analysis = analyze_item(item)
                if analysis["redeclaration"]:
                    # A stale REPL can report an earlier declaration; that is a
                    # verifier-side artifact, not a Lean rejection. Retry once.
                    retry_indices.append((result_index, "redeclaration"))
                    continue
                record = records[record_index]
                record["verified"] = analysis["verified"]
                record["has_sorry"] = analysis["has_sorry"]
                record["sorries"] = analysis["sorries"]
                record["lean_message"] = analysis["lean_message"]
                record["verify_status"] = (
                    "verified" if analysis["verified"] else analysis["status"]
                )
            for result_index, retry_reason in retry_indices:
                record_index, custom_id, proof = pending[result_index]
                record = records[record_index]
                try:
                    single = response_items(
                        verify_code(proof, custom_id=f"{custom_id}-retry", timeout=args.verify_timeout)
                    )
                    analysis = analyze_item(single[0] if single else None)
                except httpx.HTTPError as exc:
                    analysis = analyze_item(None)
                    analysis["lean_message"] = f"verifier error: {exc}"
                record["verified"] = analysis["verified"]
                record["has_sorry"] = analysis["has_sorry"]
                record["sorries"] = analysis["sorries"]
                record["lean_message"] = analysis["lean_message"]
                record["verify_status"] = "verified" if analysis["verified"] else analysis["status"]
                record["single_verify_retry"] = retry_reason
            for record_index, custom_id, _ in pending:
                if records[record_index]["verify_status"] == "not_checked":
                    records[record_index]["verify_status"] = "verifier_error"
                    records[record_index]["lean_message"] = "missing response item"
            elapsed = round(time.perf_counter() - verify_started, 3)
            verification_seconds += elapsed
            per_candidate = elapsed / len(pending)
            for record_index, _, _ in pending:
                records[record_index]["verification_time"] = round(per_candidate, 4)

        partial_path.write_text(
            json.dumps(
                {
                    "model_key": args.model_key,
                    "processed_theorems": min(start + args.batch_size, len(sampled)),
                    "records": records,
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

    print("[4/5] Summarising and exporting the cached rollout batch")
    counts = []
    theorem_summaries = []
    for theorem_index in range(len(sampled)):
        theorem_records = [record for record in records if record["theorem_index"] == theorem_index]
        verified_count = sum(record["verified"] for record in theorem_records)
        counts.append(verified_count)
        theorem_summaries.append(
            {
                "theorem_index": theorem_index,
                "statement_id": sampled[theorem_index]["statement_id"],
                "source": sampled[theorem_index].get("source"),
                "verified_count": verified_count,
                "sorry_count": sum(record["has_sorry"] for record in theorem_records),
                "truncations": sum(record["truncated"] for record in theorem_records),
                "format_failures": sum(not record["format_ok"] for record in theorem_records),
                "verifier_errors": sum(record["verify_status"] == "verifier_error" for record in theorem_records),
                "avg_completion_tokens": round(
                    sum(record["completion_tokens"] for record in theorem_records) / len(theorem_records), 1
                ),
            }
        )
    rates = group_rates(counts, args.samples_per_theorem)
    strict_verified = sum(
        record["verified"] and record["has_lean4_code_block"] and record["has_complete_think_block"]
        for record in records
    )
    summary: dict[str, Any] = {
        "artifact_type": "promptset_reward_profile",
        "validity_rule": "official kimina_client rule: no error-severity message and no sorry",
        "model_key": args.model_key,
        "dataset_path": str(dataset_path.resolve()),
        "dataset_rows": len(rows),
        "dataset_unique_statements": len(unique_list),
        "sampled_statement_ids": [row["statement_id"] for row in sampled],
        "seed": args.seed,
        "theorems": len(sampled),
        "samples_per_theorem": args.samples_per_theorem,
        "generated_candidates": len(records),
        "generation_seconds": round(generation_seconds, 3),
        "verification_seconds": round(verification_seconds, 3),
        "verified_candidates": sum(record["verified"] for record in records),
        "verified_candidates_strict_format": strict_verified,
        "sorry_candidates": sum(record["has_sorry"] for record in records),
        "truncated_candidates": sum(record["truncated"] for record in records),
        "format_failures": sum(not record["format_ok"] for record in records),
        "verifier_errors": sum(record["verify_status"] == "verifier_error" for record in records),
        "single_verify_retries": {
            reason: sum(record.get("single_verify_retry") == reason for record in records)
            for reason in ("missing_item", "redeclaration")
        },
        "group_rates": rates,
        "igr": rates["mixed"],
        "prompt_length_stats_all_unique": all_prompt_stats,
        "prompt_length_stats_sampled": sampled_prompt_stats,
        "generation_settings": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "batch_size": args.batch_size,
            "note": "E007 evaluation setting; official P3 rollout uses temperature 1.0",
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
        "theorem_summaries": theorem_summaries,
        "records": records,
    }
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    partial_path.unlink(missing_ok=True)

    write_jsonl(batch_dir / "prompts.jsonl", prompt_records)
    write_jsonl(
        batch_dir / "rollouts.jsonl",
        [
            {
                "candidate_id": f"{record['statement_id']}-{record['sample']}",
                "statement_id": record["statement_id"],
                "theorem_index": record["theorem_index"],
                "sample": record["sample"],
                "prompt_tokens": len(prompt_records[record["theorem_index"]]["prompt_ids"]),
                "completion_tokens": record["completion_tokens"],
                "completion_ids": record["completion_ids"],
                "completion_text": record["raw_output"],
                "proof_text": record["proof"],
                "truncated": record["truncated"],
                "generation_time": record["generation_time"],
            }
            for record in records
        ],
    )
    write_jsonl(
        batch_dir / "rewards.jsonl",
        [
            {
                "candidate_id": f"{record['statement_id']}-{record['sample']}",
                "statement_id": record["statement_id"],
                "reward": 1.0 if record["verified"] else 0.0,
                "verified": record["verified"],
                "format_ok": record["format_ok"],
                "has_sorry": record["has_sorry"],
                "verify_status": record["verify_status"],
                "lean_message": record["lean_message"],
                "verification_time": record["verification_time"],
            }
            for record in records
        ],
    )
    metadata = {
        key: value
        for key, value in summary.items()
        if key not in {"records", "theorem_summaries"}
    }
    metadata["batch_files"] = {
        "prompts": "prompts.jsonl",
        "rollouts": "rollouts.jsonl",
        "rewards": "rewards.jsonl",
        "note": "generation_time is amortised per candidate within one generate() call",
    }
    (batch_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print("[5/5] Profile saved")
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key not in {"records", "theorem_summaries"}},
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"Output: {output_path}")
    print(f"Cached batch: {batch_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
