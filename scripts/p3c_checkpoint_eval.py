#!/usr/bin/env python3
"""E018 (P3-C) fixed-set checkpoint evaluation with vLLM.

Evaluates one checkpoint (base Distill or an exported P3-B step) on the sealed
P3-C fixed theorem set with a deterministic per-(theorem, sample) seed schedule,
then verifies every candidate against the local Kimina Lean server.

Engine consistency with the P3-B rollout: vLLM, temperature 1.0 / top_p 1.0,
4096 max new tokens, and the same prompt construction as the rollout probe
(which mirrors the training recipe). The evaluator never modifies weights.

Seed schedule: ``seed_base + theorem_index * 8 + sample_index`` with the
canonical group size 8 (fixed across checkpoints and across preview/full
runs; candidates are NOT treated as paired samples).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import httpx
import torch
from promptset_rollout_probe import (
    analyze_item,
    build_prompt_text,
    complete_verifier_code,
    response_items,
)

from tinylean_rl.evaluation.p3c_stats import candidate_metrics, classify_candidate
from tinylean_rl.inference.extract import extract_proof
from tinylean_rl.verifier.kimina import verify_code, verify_codes

CHECKPOINTS = {
    "base": {"global_step": 0, "model_dir": "models/weights/kimina_distill_0_6b", "label": "step0"},
    "step10": {"global_step": 10, "model_dir": "runs/p3c_models/step_10", "label": "step10"},
    "step20": {"global_step": 20, "model_dir": "runs/p3c_models/step_20", "label": "step20"},
    "step30": {"global_step": 30, "model_dir": "runs/p3c_models/step_30", "label": "step30"},
}
SEED_BASE = 20260917
SEED_GROUP_SIZE = 8
LEAN_BLOCK_RE = re.compile(r"```lean4?\s*\n.*?```", re.DOTALL)


def git_revision() -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()


def _gpu_memory_via_nvidia_smi() -> int | None:
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        ).stdout.strip().splitlines()[0]
        return int(output)
    except Exception:  # noqa: BLE001 - sampling is best-effort
        return None


def gpu_memory_used_mib() -> int | None:
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return int(pynvml.nvmlDeviceGetMemoryInfo(handle).used // 2**20)
    except Exception:  # noqa: BLE001 - fall back to the nvidia-smi CLI
        return _gpu_memory_via_nvidia_smi()


def apply_analysis(record: dict, analysis: dict) -> None:
    record["verified"] = bool(analysis["verified"])
    record["has_sorry"] = bool(analysis["has_sorry"])
    record["sorries"] = int(analysis["sorries"])
    record["lean_message"] = analysis["lean_message"]
    record["verify_status"] = "verified" if analysis["verified"] else analysis["status"]
    record["reward"] = 1.0 if analysis["verified"] else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", choices=sorted(CHECKPOINTS), required=True)
    parser.add_argument("--fixed-set", default="experiments/manifests/p3c_fixed_set.json")
    parser.add_argument("--limit", type=int, default=0, help="First N theorems (0 = all 64).")
    parser.add_argument("--samples-per-theorem", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--chunk-theorems", type=int, default=16)
    parser.add_argument("--seed-base", type=int, default=SEED_BASE)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=5120)
    parser.add_argument("--verify-timeout", type=float, default=120.0)
    parser.add_argument("--first-verify-timeout", type=float, default=600.0)
    parser.add_argument("--output", default="")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    spec = CHECKPOINTS[args.checkpoint]
    model_dir = ROOT / spec["model_dir"]
    if not model_dir.exists():
        print(f"[ERROR] model directory not found: {model_dir}", file=sys.stderr)
        return 2
    output_path = Path(args.output) if args.output else ROOT / f"experiments/results/e018_{args.checkpoint}.json"
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_suffix(".partial.json")

    fixed_set_path = ROOT / args.fixed_set
    fixed_set = json.loads(fixed_set_path.read_text(encoding="utf-8"))
    theorems = fixed_set["theorems"]
    if args.limit > 0:
        theorems = theorems[: args.limit]

    print(f"[E018] checkpoint={args.checkpoint} ({spec['label']}) global_step={spec['global_step']}")
    print(f"[E018] model source: {model_dir}")
    print(f"[E018] fixed set: {fixed_set_path} (selection_seed={fixed_set['selection_seed']})")
    print(
        f"[E018] theorems={len(theorems)} samples/theorem={args.samples_per_theorem} "
        f"temperature={args.temperature} top_p={args.top_p} max_new_tokens={args.max_new_tokens}"
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=True, local_files_only=args.offline
    )
    prompts = {
        theorem["theorem_index"]: build_prompt_text(
            tokenizer,
            {
                "natural_language": theorem["natural_language"],
                "formal_statement": theorem["formal_statement"],
            },
        )
        for theorem in theorems
    }

    import vllm
    from vllm import LLM, SamplingParams

    print("[E018] initialising the vLLM engine")
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

    records: list[dict] = []
    generation_seconds = 0.0
    verification_seconds = 0.0
    peak_vram_mib = 0
    first_verify = True
    started_all = time.perf_counter()

    chunk_size = max(1, args.chunk_theorems)
    total_chunks = (len(theorems) + chunk_size - 1) // chunk_size
    for chunk_index, chunk_start in enumerate(range(0, len(theorems), chunk_size)):
        chunk = theorems[chunk_start : chunk_start + chunk_size]
        chunk_prompts: list[str] = []
        chunk_sampling: list[object] = []
        for theorem in chunk:
            for sample_index in range(args.samples_per_theorem):
                chunk_prompts.append(prompts[theorem["theorem_index"]])
                chunk_sampling.append(
                    SamplingParams(
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_tokens=args.max_new_tokens,
                        n=1,
                        seed=args.seed_base + theorem["theorem_index"] * SEED_GROUP_SIZE + sample_index,
                    )
                )

        generation_started = time.perf_counter()
        outputs = llm.generate(chunk_prompts, chunk_sampling)
        chunk_seconds = time.perf_counter() - generation_started
        generation_seconds += chunk_seconds
        seconds_per_candidate = chunk_seconds / len(chunk_prompts)

        chunk_records: list[dict] = []
        for offset, output in enumerate(outputs):
            theorem = chunk[offset // args.samples_per_theorem]
            sample_index = offset % args.samples_per_theorem
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
                    "checkpoint": args.checkpoint,
                    "checkpoint_label": spec["label"],
                    "global_step": spec["global_step"],
                    "statement_id": theorem["statement_id"],
                    "theorem_index": theorem["theorem_index"],
                    "sample_index": sample_index,
                    "sampling_seed": args.seed_base
                    + theorem["theorem_index"] * SEED_GROUP_SIZE
                    + sample_index,
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
            proofs = [record["proof"] for _, record in pending]
            custom_ids = [
                f"{record['theorem_index']}-{record['sample_index']}" for _, record in pending
            ]
            decoded: dict = {}
            for attempt in range(2):
                try:
                    decoded = verify_codes(proofs, custom_ids=custom_ids, timeout=timeout)
                    break
                except httpx.HTTPError as exc:
                    if attempt == 0:
                        time.sleep(15)
                    else:
                        print(f"  [warn] batch verify failed after retry ({exc}); retrying per candidate")
            first_verify = False
            items = response_items(decoded)
            by_id = {str(item.get("custom_id")): item for item in items}
            retry_indices: list[tuple[int, str]] = []
            for result_index, (record_index, record) in enumerate(pending):
                item = by_id.get(custom_ids[result_index])
                if item is None and result_index < len(items) and len(items) == len(pending):
                    item = items[result_index]
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
                record["single_verify_retry"] = retry_reason
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
        print(
            f"[chunk {chunk_index + 1}/{total_chunks}] gen={chunk_seconds:.1f}s "
            f"verified={verified_so_far}/{len(records)} peak_vram={peak_vram_mib} MiB"
        )
        partial_path.write_text(
            json.dumps({"records": records, "processed_theorems": chunk_start + len(chunk)}, ensure_ascii=False)
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
        "useful_tokens_per_s": round(total_tokens / generation_seconds, 1) if generation_seconds else None,
        "verified_per_gpu_hour": round(
            summary.get("verified_candidates", 0) / (total_seconds / 3600), 1
        )
        if total_seconds
        else None,
    }
    artifact = {
        "artifact_type": "p3c_checkpoint_eval",
        "checkpoint": args.checkpoint,
        "checkpoint_label": spec["label"],
        "global_step": spec["global_step"],
        "model_dir": str(model_dir),
        "engine": "vllm",
        "vllm_version": vllm.__version__,
        "fixed_set": str(fixed_set_path.resolve()),
        "selection_seed": fixed_set["selection_seed"],
        "settings": {
            "theorems": len(theorems),
            "samples_per_theorem": args.samples_per_theorem,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "chunk_theorems": chunk_size,
            "seed_base": args.seed_base,
            "seed_group_size": SEED_GROUP_SIZE,
            "seed_schedule": "seed_base + theorem_index * 8 + sample_index (canonical n=8 schedule)",
        },
        "summary": summary,
        "resources": resources,
        "records": records,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
    }
    output_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    partial_path.unlink(missing_ok=True)

    printable = {key: value for key, value in artifact.items() if key not in {"records"}}
    print(json.dumps(printable, indent=2, ensure_ascii=False))
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
