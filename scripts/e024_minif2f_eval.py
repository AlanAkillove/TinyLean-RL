#!/usr/bin/env python3
"""E024 MiniF2F external-benchmark evaluation (theta0 vs seed1-step60).

Formal external evaluator for the E024 pre-registration: the MiniF2F test
split (244 theorems; local AI-MO/minif2f_test mirror), vLLM generation with
the canonical n=8 seed schedule, strict Kimina 2.0.0 verification and the
E023-compatible artifact/taxonomy format.

Engine consistency with E023 (``scripts/p3c_checkpoint_eval.py``): the same
vLLM engine settings, seed schedule, prompt construction, proof extraction,
strict verifier semantics, sub-batch verification, partial persistence, VRAM
recording and taxonomy are reused (shared helpers are imported directly). The
only intended differences are the dataset source, the explicit
``--model``/``--label`` inputs, the added provenance fields (host/GPU/dataset
revision) and the E024 artifact type.

Prompt contract (frozen in ``experiments/manifests/e024_minif2f.yaml``): the
MiniF2F ``informal_prefix`` fills the canonical ``# Problem:`` slot and
``formal_statement`` fills the ``# Formal Statement:`` slot of the pinned
recipe template, then the model chat template is applied. MiniF2F statements
already carry ``import Mathlib`` and end with ``:= by``; the generated proof
body is appended by ``complete_verifier_code``.
"""

from __future__ import annotations

import argparse
import json
import re
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pyarrow import parquet
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

# Windows consoles default to GBK, which cannot print Lean goal symbols.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import httpx
from p3c_checkpoint_eval import (
    SEED_BASE,
    SEED_GROUP_SIZE,
    apply_analysis,
    git_revision,
    gpu_memory_used_mib,
    verify_sub_batches,
)
from promptset_rollout_probe import (
    SYSTEM_PROMPT,
    USER_TEMPLATE,
    analyze_item,
    build_prompt_text,
    complete_verifier_code,
    response_items,
)

from tinylean_rl.evaluation.p3c_stats import candidate_metrics, classify_candidate
from tinylean_rl.inference.extract import extract_proof
from tinylean_rl.verifier.kimina import verify_code

ARTIFACT_TYPE = "e024_minif2f_eval"
EXPERIMENT = "E024"
MINIF2F_REQUIRED_COLUMNS = ("name", "informal_prefix", "formal_statement")
MINIF2F_EXPECTED_TEST_SIZE = 244
LEAN_BLOCK_RE = re.compile(r"```lean4?\s*\n.*?```", re.DOTALL)
LEAN_SERVER_CONTAINER = "tinylean-rl-lean-server"

REQUIRED_ARTIFACT_KEYS = (
    "artifact_type",
    "experiment",
    "host",
    "gpu",
    "git_revision",
    "model_label",
    "model_path",
    "dataset",
    "dataset_revision",
    "split",
    "settings",
    "summary",
    "resources",
    "records",
    "created_at_utc",
)
REQUIRED_RECORD_KEYS = (
    "theorem_index",
    "statement_id",
    "name",
    "sample_index",
    "sampling_seed",
    "raw_output",
    "extracted_proof",
    "proof",
    "format_ok",
    "verified",
    "verify_status",
    "taxonomy",
    "truncated",
    "prompt_tokens",
    "generated_tokens",
)


def find_parquet(path: Path) -> Path:
    if path.is_file():
        return path
    files = sorted(path.rglob("*.parquet"))
    if len(files) != 1:
        raise SystemExit(f"Expected one parquet file under {path}, found {len(files)}")
    return files[0]


def load_minif2f_rows(
    dataset_path: Path, expected_size: int = MINIF2F_EXPECTED_TEST_SIZE
) -> list[dict[str, Any]]:
    """Load the MiniF2F test split with the frozen theorem order.

    ``theorem_index`` is the parquet row order (never shuffled);
    ``statement_id`` is the MiniF2F declaration name.
    """

    path = find_parquet(dataset_path)
    table = parquet.read_table(path)
    missing = [
        column for column in MINIF2F_REQUIRED_COLUMNS if column not in table.column_names
    ]
    if missing:
        raise SystemExit(f"MiniF2F parquet {path} is missing required columns: {missing}")
    rows = table.to_pylist()
    if expected_size and len(rows) != expected_size:
        raise SystemExit(
            f"MiniF2F test size mismatch: expected {expected_size} theorems, "
            f"found {len(rows)} in {path}"
        )
    return [
        {
            "theorem_index": index,
            "statement_id": row["name"],
            "name": row["name"],
            "informal_prefix": row.get("informal_prefix") or "",
            "formal_statement": row["formal_statement"],
        }
        for index, row in enumerate(rows)
    ]


def prompt_fields(row: dict[str, Any]) -> dict[str, str]:
    """Map MiniF2F columns onto the canonical prompt slots.

    The pinned template expects ``natural_language`` and ``formal_statement``;
    MiniF2F provides ``informal_prefix`` and ``formal_statement``. Only the
    informal source column differs from the Promptset evaluator.
    """

    return {
        "natural_language": row.get("informal_prefix") or "",
        "formal_statement": row["formal_statement"],
    }


def seed_for(theorem_index: int, sample_index: int, seed_base: int = SEED_BASE) -> int:
    return seed_base + theorem_index * SEED_GROUP_SIZE + sample_index


def candidate_schedule(
    theorems: list[dict[str, Any]], samples_per_theorem: int, seed_base: int = SEED_BASE
) -> list[dict[str, Any]]:
    """Frozen candidate list: theorem-major order with the canonical seeds."""

    return [
        {
            "theorem_index": theorem["theorem_index"],
            "sample_index": sample_index,
            "sampling_seed": seed_for(theorem["theorem_index"], sample_index, seed_base),
        }
        for theorem in theorems
        for sample_index in range(samples_per_theorem)
    ]


def dataset_revision_of(parquet_path: Path) -> str | None:
    manifest = ROOT / "data" / "manifests" / "resolved.local.json"
    try:
        entries = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    for entry in entries.values():
        local_dir = entry.get("local_dir")
        if not local_dir:
            continue
        resolved_dir = ROOT / local_dir
        if resolved_dir == parquet_path or resolved_dir in parquet_path.parents:
            return entry.get("resolved_revision")
    return None


def gpu_name() -> str | None:
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        ).stdout.strip().splitlines()[0]
        return output or None
    except Exception:  # noqa: BLE001 - provenance is best-effort
        return None


def verifier_image() -> str | None:
    try:
        output = subprocess.run(
            ["docker", "inspect", "--format", "{{.Config.Image}}", LEAN_SERVER_CONTAINER],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        ).stdout.strip()
        return output or None
    except Exception:  # noqa: BLE001 - provenance is best-effort
        return None


def validate_artifact_schema(artifact: dict[str, Any]) -> list[str]:
    """Return the list of missing artifact/record fields (empty = complete)."""

    problems = [
        f"artifact missing key: {key}" for key in REQUIRED_ARTIFACT_KEYS if key not in artifact
    ]
    if artifact.get("artifact_type") != ARTIFACT_TYPE:
        problems.append("unexpected artifact_type: {}".format(artifact.get("artifact_type")))
    if artifact.get("experiment") != EXPERIMENT:
        problems.append("unexpected experiment: {}".format(artifact.get("experiment")))
    for key in ("model_label", "model_path", "dataset", "split"):
        if not artifact.get(key):
            problems.append(f"artifact empty value: {key}")
    records = artifact.get("records") or []
    if not records:
        problems.append("artifact has no records")
    else:
        first = records[0]
        problems.extend(
            f"record missing key: {key}" for key in REQUIRED_RECORD_KEYS if key not in first
        )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Local model directory (HF format).")
    parser.add_argument("--label", required=True, help="Label, e.g. theta0 / seed1_step60.")
    parser.add_argument("--global-step", type=int, default=0, help="Checkpoint training step.")
    parser.add_argument(
        "--dataset", default="data/raw/minif2f_hf", help="MiniF2F parquet file or directory."
    )
    parser.add_argument("--dataset-revision", default="", help="Override revision string.")
    parser.add_argument("--samples-per-theorem", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--chunk-theorems", type=int, default=16)
    parser.add_argument("--seed-base", type=int, default=SEED_BASE)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=5120)
    parser.add_argument("--verify-timeout", type=float, default=120.0)
    parser.add_argument("--first-verify-timeout", type=float, default=600.0)
    parser.add_argument("--verify-batch-size", type=int, default=4)
    parser.add_argument("--verify-workers", type=int, default=4)
    parser.add_argument("--verify-retry-sleep", type=float, default=10.0)
    parser.add_argument(
        "--limit", type=int, default=0, help="Dev only: first N theorems (0 = all)."
    )
    parser.add_argument("--output", default="")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    model_dir = Path(args.model)
    if not model_dir.is_absolute():
        model_dir = ROOT / model_dir
    if not model_dir.exists():
        print(f"[ERROR] model directory not found: {model_dir}", file=sys.stderr)
        return 2

    output_path = (
        Path(args.output)
        if args.output
        else ROOT / f"experiments/results/e024_minif2f_{args.label}.json"
    )
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_suffix(".partial.json")

    dataset_arg = Path(args.dataset)
    if not dataset_arg.is_absolute():
        dataset_arg = ROOT / dataset_arg
    theorems = load_minif2f_rows(dataset_arg)
    full_size = len(theorems)
    if args.limit > 0:
        theorems = theorems[: args.limit]
    parquet_path = find_parquet(dataset_arg)
    revision = args.dataset_revision or dataset_revision_of(parquet_path) or ""

    print(f"[E024] label={args.label} global_step={args.global_step}")
    print(f"[E024] model source: {model_dir}")
    print(f"[E024] dataset: {parquet_path} (revision={revision or 'unknown'})")
    print(
        f"[E024] theorems={len(theorems)}/{full_size} "
        f"samples/theorem={args.samples_per_theorem} temperature={args.temperature} "
        f"top_p={args.top_p} max_new_tokens={args.max_new_tokens}"
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=True, local_files_only=args.offline
    )
    prompts = {
        theorem["theorem_index"]: build_prompt_text(tokenizer, prompt_fields(theorem))
        for theorem in theorems
    }
    theorem_by_index = {theorem["theorem_index"]: theorem for theorem in theorems}

    import torch
    import vllm
    from vllm import LLM, SamplingParams

    print("[E024] initialising the vLLM engine")
    llm = LLM(
        model=str(model_dir),
        tokenizer=str(model_dir),
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=256,
        max_num_batched_tokens=8192,
        disable_log_stats=True,
    )
    torch.cuda.reset_peak_memory_stats()

    records: list[dict[str, Any]] = []
    generation_seconds = 0.0
    verification_seconds = 0.0
    peak_vram_mib = 0
    first_verify = True
    started_all = time.perf_counter()

    chunk_size = max(1, args.chunk_theorems)
    total_chunks = (len(theorems) + chunk_size - 1) // chunk_size
    progress = tqdm(
        range(0, len(theorems), chunk_size),
        desc=f"E024 {args.label}",
        total=total_chunks,
        unit="chunk",
        dynamic_ncols=True,
    )
    for chunk_index, chunk_start in enumerate(progress):
        chunk = theorems[chunk_start : chunk_start + chunk_size]
        schedule = candidate_schedule(chunk, args.samples_per_theorem, args.seed_base)
        chunk_prompts = [prompts[candidate["theorem_index"]] for candidate in schedule]
        chunk_sampling = [
            SamplingParams(
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_new_tokens,
                n=1,
                seed=candidate["sampling_seed"],
            )
            for candidate in schedule
        ]
        progress.set_postfix_str(f"generating {len(chunk_prompts)} candidates")

        generation_started = time.perf_counter()
        outputs = llm.generate(chunk_prompts, chunk_sampling)
        chunk_seconds = time.perf_counter() - generation_started
        generation_seconds += chunk_seconds
        seconds_per_candidate = chunk_seconds / len(chunk_prompts)

        chunk_records: list[dict[str, Any]] = []
        for offset, output in enumerate(outputs):
            candidate = schedule[offset]
            theorem = theorem_by_index[candidate["theorem_index"]]
            completion = output.outputs[0]
            raw = completion.text
            try:
                extracted = extract_proof(raw)
            except ValueError:
                extracted = ""
            proof = complete_verifier_code(theorem["formal_statement"], extracted)
            completion_ids = list(completion.token_ids)
            chunk_records.append(
                {
                    "checkpoint": args.label,
                    "checkpoint_label": args.label,
                    "global_step": args.global_step,
                    "statement_id": theorem["statement_id"],
                    "name": theorem["name"],
                    "theorem_index": theorem["theorem_index"],
                    "sample_index": candidate["sample_index"],
                    "sampling_seed": candidate["sampling_seed"],
                    "prompt_tokens": len(output.prompt_token_ids),
                    "generated_tokens": len(completion_ids),
                    "natural_termination": completion.finish_reason != "length",
                    "truncated": completion.finish_reason == "length",
                    "raw_output": raw,
                    "extracted_proof": extracted,
                    "proof": proof or "",
                    "format_ok": bool(proof),
                    "has_lean4_code_block": bool(LEAN_BLOCK_RE.search(raw)),
                    "has_complete_think_block": "<think>" in raw and "</think>" in raw,
                    "generation_time": round(seconds_per_candidate, 3),
                    "verified": False,
                    "has_sorry": False,
                    "sorries": 0,
                    "verify_status": "not_checked",
                    "reward": 0.0,
                    "lean_message": "",
                    "verification_time": None,
                }
            )

        pending = [
            (index, record)
            for index, record in enumerate(chunk_records)
            if record["format_ok"]
        ]
        if pending:
            verification_started = time.perf_counter()
            timeout = args.first_verify_timeout if first_verify else args.verify_timeout
            custom_ids = [
                "{}-{}".format(record["theorem_index"], record["sample_index"])
                for _, record in pending
            ]
            items = verify_sub_batches(
                pending,
                custom_ids,
                batch_size=max(1, args.verify_batch_size),
                workers=max(1, args.verify_workers),
                timeout=timeout,
                retry_sleep=args.verify_retry_sleep,
            )
            first_verify = False
            by_id = {str(item.get("custom_id")): item for item in items}
            retry_indices: list[tuple[int, str]] = []
            for result_index, (record_index, _record) in enumerate(pending):
                item = by_id.get(custom_ids[result_index])
                if item is None:
                    retry_indices.append((result_index, "missing_item"))
                    continue
                analysis = analyze_item(item)
                if analysis["redeclaration"]:
                    retry_indices.append((result_index, "redeclaration"))
                    continue
                apply_analysis(chunk_records[record_index], analysis)
            for result_index, retry_reason in retry_indices:
                record_index, record = pending[result_index]
                try:
                    single = response_items(
                        verify_code(
                            record["proof"],
                            custom_id=f"{custom_ids[result_index]}-retry",
                            timeout=args.verify_timeout,
                        )
                    )
                    analysis = analyze_item(single[0] if single else None)
                except httpx.HTTPError as exc:
                    analysis = analyze_item(None)
                    analysis["lean_message"] = f"verifier error: {exc}"
                apply_analysis(chunk_records[record_index], analysis)
                chunk_records[record_index]["single_verify_retry"] = retry_reason
            for _, record in pending:
                if record["verify_status"] == "not_checked":
                    record["verify_status"] = "verifier_error"
                    record["lean_message"] = "missing response item"
            elapsed = time.perf_counter() - verification_started
            verification_seconds += elapsed
            per_candidate = elapsed / len(pending)
            for _, record in pending:
                record["verification_time"] = round(per_candidate, 4)

        for record in chunk_records:
            record["taxonomy"] = classify_candidate(
                truncated=record["truncated"],
                format_ok=record["format_ok"],
                verify_status=record["verify_status"],
                lean_message=record["lean_message"],
            )
        records.extend(chunk_records)

        used_mib = gpu_memory_used_mib()
        if used_mib:
            peak_vram_mib = max(peak_vram_mib, used_mib)
        verified_so_far = sum(record["verified"] for record in records)
        progress.write(
            f"[chunk {chunk_index + 1}/{total_chunks}] gen={chunk_seconds:.1f}s "
            f"verified={verified_so_far}/{len(records)} peak_vram={peak_vram_mib} MiB"
        )
        progress.set_postfix_str(
            f"verified {verified_so_far}/{len(records)}, vram {peak_vram_mib} MiB"
        )
        partial_path.write_text(
            json.dumps(
                {"records": records, "processed_theorems": chunk_start + len(chunk)},
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

    total_seconds = time.perf_counter() - started_all
    summary = candidate_metrics(records, args.samples_per_theorem)
    total_tokens = sum(record["generated_tokens"] for record in records)
    resources = {
        "generation_seconds": round(generation_seconds, 3),
        "verification_seconds": round(verification_seconds, 3),
        "total_seconds": round(total_seconds, 3),
        "peak_vram_mib_sampled": peak_vram_mib or None,
        "torch_peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "total_generated_tokens": total_tokens,
        "useful_tokens_per_s": round(total_tokens / generation_seconds, 1)
        if generation_seconds
        else None,
        "verified_per_gpu_hour": round(
            summary.get("verified_candidates", 0) / (total_seconds / 3600), 1
        )
        if total_seconds
        else None,
    }
    artifact = {
        "artifact_type": ARTIFACT_TYPE,
        "experiment": EXPERIMENT,
        "host": socket.gethostname(),
        "gpu": gpu_name(),
        "git_revision": git_revision(),
        "model_label": args.label,
        "model_path": str(model_dir),
        "global_step": args.global_step,
        "engine": "vllm",
        "vllm_version": vllm.__version__,
        "dataset": str(parquet_path.resolve()),
        "dataset_revision": revision,
        "split": "test",
        "prompt_contract": {
            "system_prompt": SYSTEM_PROMPT,
            "user_template": USER_TEMPLATE,
            "informal_source_field": "informal_prefix",
            "formal_source_field": "formal_statement",
        },
        "settings": {
            "theorems": len(theorems),
            "theorems_in_dataset": full_size,
            "limit": args.limit,
            "samples_per_theorem": args.samples_per_theorem,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "chunk_theorems": chunk_size,
            "seed_base": args.seed_base,
            "seed_group_size": SEED_GROUP_SIZE,
            "seed_schedule": "seed_base + theorem_index * 8 + sample_index (canonical n=8 schedule)",
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "verify_timeout_s": args.verify_timeout,
            "first_verify_timeout_s": args.first_verify_timeout,
            "verify_batch_size": args.verify_batch_size,
            "verify_workers": args.verify_workers,
        },
        "verifier": {
            "image": verifier_image(),
            "policy": "strict Kimina validity: no error-severity message and no sorry",
        },
        "summary": summary,
        "resources": resources,
        "records": records,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    output_path.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    partial_path.unlink(missing_ok=True)

    problems = validate_artifact_schema(artifact)
    if problems:
        print(f"[warn] artifact schema issues: {problems}", file=sys.stderr)
    printable = {key: value for key, value in artifact.items() if key != "records"}
    print(json.dumps(printable, indent=2, ensure_ascii=False))
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
