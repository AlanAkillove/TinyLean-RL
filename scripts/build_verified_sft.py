#!/usr/bin/env python3
"""Cold-start verified SFT dataset preparation from NuminaMath-LEAN (P2.5 W5).

Prepares (but does not train on) a cold-start SFT corpus:

1. **load** ``data/raw/numinamath_lean`` (104,155 rows, 74.8 MiB);
2. **filter** rows with a non-empty ``formal_statement`` and ``formal_proof``;
3. **dedup** by normalised statement hash (comments stripped, whitespace
   collapsed);
4. **contamination check** against MiniF2F test: normalised statement hash
   and theorem-name match; hits are dropped and reported;
5. **sample verification** of ``--verify-limit`` (default 200) proofs against
   the local Kimina Lean Server, serial + tqdm, official validity rule
   (no error-severity message, no sorry);
6. **token length stats** with the Kimina-Prover-Distill-0.6B tokenizer;
7. **split** a fixed-seed train/val split;
8. **export** ``data/processed/sft_cold_start/{train,val}.parquet`` and
   ``experiments/results/p2_5_sft_prep.json``.

Full verification and training stay in the cloud; this artifact is the
dataset-side prerequisite for the ("Distill works, Base doesn't") branch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

# Windows consoles default to GBK, which cannot print Lean/arrow symbols.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import httpx
import pyarrow as pa
from promptset_rollout_probe import analyze_item, response_items
from pyarrow import parquet
from tqdm import tqdm

from tinylean_rl.verifier.kimina import verify_code

DEFAULT_INPUT = "data/raw/numinamath_lean/data/train-00000-of-00001.parquet"
DEFAULT_MINIF2F = "data/raw/minif2f_hf/data/train-00000-of-00001.parquet"
DEFAULT_OUTPUT_DIR = "data/processed/sft_cold_start"
DEFAULT_OUTPUT = "experiments/results/p2_5_sft_prep.json"

BLOCK_COMMENT = re.compile(r"/-.*?-/", re.DOTALL)
LINE_COMMENT = re.compile(r"--[^\n]*")
WHITESPACE = re.compile(r"\s+")
THEOREM_NAME = re.compile(r"\btheorem\s+([A-Za-z_][A-Za-z0-9_'.]*)")

EXPORT_COLUMNS = [
    "uuid",
    "problem",
    "formal_statement",
    "formal_proof",
    "source",
    "author",
    "statement_hash",
    "theorem_name",
    "verified_sample",
    "verify_status",
]


def git_revision() -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()


def normalise_statement(text: str) -> str:
    text = BLOCK_COMMENT.sub(" ", text)
    text = LINE_COMMENT.sub(" ", text)
    return WHITESPACE.sub(" ", text).strip()


def statement_hash(text: str) -> str:
    return hashlib.sha256(normalise_statement(text).encode("utf-8")).hexdigest()[:16]


def theorem_name(text: str) -> str | None:
    match = THEOREM_NAME.search(text)
    return match.group(1) if match else None


def length_stats(lengths: list[int]) -> dict:
    if not lengths:
        return {"count": 0}
    ordered = sorted(lengths)

    def pick(fraction: float) -> int:
        index = min(len(ordered) - 1, max(0, round(fraction * len(ordered)) - 1))
        return ordered[index]

    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": ordered[len(ordered) // 2],
        "p95": pick(0.95),
        "p99": pick(0.99),
        "max": ordered[-1],
        "mean": round(sum(ordered) / len(ordered), 1),
    }


def token_lengths(tokenizer, texts: list[str], batch_size: int = 256) -> list[int]:
    lengths: list[int] = []
    for start in tqdm(range(0, len(texts), batch_size), desc="tokenize", unit="batch"):
        encoded = tokenizer(texts[start : start + batch_size], add_special_tokens=False)["input_ids"]
        lengths.extend(len(ids) for ids in encoded)
    return lengths


def verify_one(proof: str, custom_id: str, timeout: float) -> dict:
    try:
        decoded = verify_code(proof, custom_id=custom_id, timeout=timeout)
    except httpx.HTTPError as exc:
        # Keep the full analysis shape so callers can rely on every key.
        analysis = analyze_item(None)
        analysis["lean_message"] = f"verifier error: {exc}"
        return analysis
    items = response_items(decoded)
    return analyze_item(items[0] if items else None)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--minif2f", default=DEFAULT_MINIF2F)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--model", help="Local model path; defaults to TINYLEAN_MODEL_DIR.")
    parser.add_argument("--model-key", default="kimina_distill_0_6b")
    parser.add_argument("--verify-limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--verify-timeout", type=float, default=120.0)
    parser.add_argument("--first-verify-timeout", type=float, default=600.0)
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--skip-tokenize", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = ROOT / input_path
    if not input_path.exists():
        print(f"[ERROR] input dataset missing: {input_path}", file=sys.stderr)
        print("  download it with: python3 scripts/download_data.py --dataset-key numinamath_lean", file=sys.stderr)
        return 2

    minif2f_path = Path(args.minif2f)
    if not minif2f_path.is_absolute():
        minif2f_path = ROOT / minif2f_path
    if not minif2f_path.exists():
        print(f"[ERROR] MiniF2F test parquet missing: {minif2f_path}", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[1/8] Loading {input_path}")
    table = parquet.read_table(input_path)
    rows_raw = table.to_pylist()
    print(f"  rows: {len(rows_raw)}")

    print("[2/8] Filtering (formal_statement + formal_proof non-empty)")
    rows = [
        row
        for row in rows_raw
        if (row.get("formal_statement") or "").strip() and (row.get("formal_proof") or "").strip()
    ]
    filtered_out = len(rows_raw) - len(rows)
    print(f"  kept {len(rows)}, filtered out {filtered_out}")

    print("[3/8] Deduplicating by normalised statement hash")
    seen: set[str] = set()
    deduped: list[dict] = []
    for row in rows:
        digest = statement_hash(row["formal_statement"])
        if digest in seen:
            continue
        seen.add(digest)
        record = {
            "uuid": str(row.get("uuid") or ""),
            "problem": str(row.get("problem") or ""),
            "formal_statement": row["formal_statement"],
            "formal_proof": row["formal_proof"],
            "source": str(row.get("source") or ""),
            "author": str(row.get("author") or ""),
            "statement_hash": digest,
            "theorem_name": theorem_name(row["formal_statement"]) or "",
            "verified_sample": None,
            "verify_status": None,
        }
        deduped.append(record)
    duplicates_removed = len(rows) - len(deduped)
    print(f"  unique statements: {len(deduped)}, duplicates removed: {duplicates_removed}")

    print("[4/8] MiniF2F contamination check (statement hash + theorem name)")
    minif2f_rows = parquet.read_table(minif2f_path).to_pylist()
    minif2f_hashes = {
        statement_hash(row["formal_statement"]) for row in minif2f_rows if (row.get("formal_statement") or "").strip()
    }
    minif2f_names = {str(row.get("name")) for row in minif2f_rows if row.get("name")}
    contamination_examples: list[dict] = []
    clean: list[dict] = []
    for record in deduped:
        hash_hit = record["statement_hash"] in minif2f_hashes
        name_hit = bool(record["theorem_name"]) and record["theorem_name"] in minif2f_names
        if hash_hit or name_hit:
            if len(contamination_examples) < 20:
                contamination_examples.append(
                    {
                        "uuid": record["uuid"],
                        "theorem_name": record["theorem_name"],
                        "statement_hash": record["statement_hash"],
                        "hash_hit": hash_hit,
                        "name_hit": name_hit,
                    }
                )
            continue
        clean.append(record)
    print(f"  contaminated rows removed: {len(deduped) - len(clean)}")

    print(f"[5/8] Sampled verification (limit {args.verify_limit})" + (" [skipped]" if args.skip_verify else ""))
    verify_stats = {"requested": 0, "verified": 0, "lean_error": 0, "verifier_error": 0, "sorry": 0, "seconds": 0.0}
    if not args.skip_verify and args.verify_limit > 0 and clean:
        rng = random.Random(args.seed)
        sample = rng.sample(clean, min(args.verify_limit, len(clean)))
        verify_stats["requested"] = len(sample)
        verify_started = time.perf_counter()
        first = True
        for record in tqdm(sample, desc="verify", unit="proof"):
            timeout = args.first_verify_timeout if first else args.verify_timeout
            analysis = None
            for attempt in range(2):
                analysis = verify_one(
                    record["formal_proof"], f"numinamath-{record['uuid']}-{attempt}", timeout
                )
                if analysis["status"] == "verifier_error" and attempt == 0:
                    time.sleep(15)
                    continue
                break
            if analysis["redeclaration"]:
                # Stale REPL artifact, not a Lean rejection; retry once.
                analysis = verify_one(
                    record["formal_proof"], f"numinamath-{record['uuid']}-retry", args.verify_timeout
                )
            first = False
            record["verified_sample"] = bool(analysis["verified"])
            record["verify_status"] = "verified" if analysis["verified"] else analysis["status"]
            if analysis["verified"]:
                verify_stats["verified"] += 1
            else:
                verify_stats[analysis["status"]] = verify_stats.get(analysis["status"], 0) + 1
            if analysis["has_sorry"]:
                verify_stats["sorry"] += 1
        verify_stats["seconds"] = round(time.perf_counter() - verify_started, 2)
        if verify_stats["requested"]:
            verify_stats["verified_ratio"] = round(verify_stats["verified"] / verify_stats["requested"], 4)
        print(f"  verified {verify_stats['verified']}/{verify_stats['requested']} in {verify_stats['seconds']}s")

    print("[6/8] Tokenizer length statistics" + (" [skipped]" if args.skip_tokenize else ""))
    token_stats: dict = {}
    if not args.skip_tokenize and clean:
        from transformers import AutoTokenizer

        model_dir = args.model or os.getenv("TINYLEAN_MODEL_DIR")
        if not model_dir:
            model_dir = str(ROOT / "models" / "weights" / args.model_key)
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        statement_lengths = token_lengths(tokenizer, [record["formal_statement"] for record in clean])
        full_lengths = token_lengths(tokenizer, [record["formal_proof"] for record in clean])
        token_stats = {
            "tokenizer": args.model_key,
            "statement_tokens": length_stats(statement_lengths),
            "full_file_tokens": length_stats(full_lengths),
        }
        print(f"  statement tokens: {token_stats['statement_tokens']}")
        print(f"  full file tokens: {token_stats['full_file_tokens']}")

    print("[7/8] Train/val split")
    rng = random.Random(args.seed)
    shuffled = list(clean)
    rng.shuffle(shuffled)
    val_size = max(1, int(len(shuffled) * args.val_ratio)) if shuffled else 0
    val_rows = shuffled[:val_size]
    train_rows = shuffled[val_size:]
    print(f"  train {len(train_rows)}, val {len(val_rows)} (val_ratio={args.val_ratio}, seed={args.seed})")

    print("[8/8] Exporting parquet and summary")
    for split_name, split_rows in (("train", train_rows), ("val", val_rows)):
        split_table = pa.table({column: [record[column] for record in split_rows] for column in EXPORT_COLUMNS})
        parquet.write_table(split_table, output_dir / f"{split_name}.parquet")
        print(f"  wrote {output_dir / f'{split_name}.parquet'} ({len(split_rows)} rows)")

    summary = {
        "artifact_type": "sft_prep",
        "input_path": str(input_path),
        "rows_total": len(rows_raw),
        "rows_with_statement_and_proof": len(rows),
        "rows_filtered_out": filtered_out,
        "duplicates_removed": duplicates_removed,
        "contamination": {
            "minif2f_path": str(minif2f_path),
            "minif2f_rows": len(minif2f_rows),
            "rows_removed": len(deduped) - len(clean),
            "examples": contamination_examples,
        },
        "verify_sample": verify_stats,
        "token_lengths": token_stats,
        "split": {
            "train": len(train_rows),
            "val": len(val_rows),
            "val_ratio": args.val_ratio,
            "seed": args.seed,
        },
        "outputs": {
            "train": str((output_dir / "train.parquet").relative_to(ROOT)).replace("\\", "/"),
            "val": str((output_dir / "val.parquet").relative_to(ROOT)).replace("\\", "/"),
        },
        "settings": {
            "verify_limit": args.verify_limit,
            "skip_verify": args.skip_verify,
            "skip_tokenize": args.skip_tokenize,
            "verify_timeout": args.verify_timeout,
            "first_verify_timeout": args.first_verify_timeout,
        },
        "seconds_total": round(time.perf_counter() - started, 2),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
    }
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("rows_total", "verify_sample", "split")}, indent=2))
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
