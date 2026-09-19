#!/usr/bin/env python3
"""V2-E001 (Track B, B1) — budget-semantics audit for theta0.

Checks whether a direct-budget generation G(x, seed, b) is token-identical to
the first b tokens of G(x, seed, 4096) under the canonical single-candidate
path (batch_size=1, num_return_sequences=1, explicit per-candidate seed). This
is the gate that decides whether B2 may use prefix-derived (cheap) budget
labels (see experiments/manifests/v2/V2-E001.yaml).

Stage 1 (canonical single path):
    16 theorems x 2 replicates x [512-det1, 512-det2, 1024, 2048, 3072, 4096]
Stage 2 (batch probes, conditional on stage 1):
    hf-batch: 4 candidates generated together (batch_size=4), budgets
              1024 (x2 determinism), 4096
    vllm:     4 candidates, n=1 per-request seed, budgets 1024 (x2), 4096

Raw token ids are appended to experiments/results/v2_b1_rollouts.jsonl
(gitignored); resume-safe: jobs already present are skipped. Analysis lives
in scripts/v2_b1_analyze.py — neither script ever compares decoded strings
instead of token ids.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

# Windows consoles default to GBK, which cannot print Lean goal symbols.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from promptset_rollout_probe import build_prompt_text
from tqdm import tqdm

DEFAULT_THEOREMS = "experiments/manifests/v2/v2_e001_theorems.json"
DEFAULT_MODEL = "models/weights/kimina_distill_0_6b"
DEFAULT_OUTPUT = "experiments/results/v2_b1_rollouts.jsonl"

SEED_BASE = 20260919
SEED_GROUP_SIZE = 8
REPLICATES = (0, 1)
STAGE1_RUNS = (
    ("det1", 512),
    ("det2", 512),
    ("b1024", 1024),
    ("b2048", 2048),
    ("b3072", 3072),
    ("full4096", 4096),
)
STAGE2_RANKS = (0, 1, 2, 3)
STAGE2_REPLICATE = 0
HF_BATCH_RUNS = (("batch1024a", 1024), ("batch1024b", 1024), ("batch4096", 4096))
VLLM_RUNS = (("v1024a", 1024), ("v1024b", 1024), ("v4096", 4096))


def git_revision() -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()


def load_theorems(path: Path) -> list[dict[str, Any]]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("artifact_type") != "v2_e001_theorem_list":
        raise SystemExit(f"unexpected theorem list artifact: {path}")
    return artifact["theorems"]


def make_prompt(tokenizer, theorem: dict[str, Any]) -> str:
    return build_prompt_text(
        tokenizer,
        {
            "natural_language": theorem["natural_language"],
            "formal_statement": theorem["formal_statement"],
        },
    )


def job_key(stage: str, engine: str, rank: int, replicate: int, tag: str) -> str:
    return f"{stage}|{engine}|{rank}|{replicate}|{tag}"


def load_done_keys(output_path: Path) -> set[str]:
    done: set[str] = set()
    if not output_path.exists():
        return done
    with output_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            done.add(
                job_key(
                    str(record["stage"]),
                    str(record["engine"]),
                    int(record["theorem_rank"]),
                    int(record["replicate"]),
                    str(record["run_tag"]),
                )
            )
    return done


def append_record(output_path: Path, record: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def build_record(
    *,
    stage: str,
    engine: str,
    theorem: dict[str, Any],
    replicate: int,
    gen_seed: int,
    budget: int,
    run_tag: str,
    prompt_tokens: int,
    token_ids: list[int],
    runtime_seconds: float,
    eos_token_id: int | None,
) -> dict[str, Any]:
    hit_eos = bool(token_ids) and token_ids[-1] == eos_token_id
    return {
        "stage": stage,
        "engine": engine,
        "experiment": "V2-E001",
        "track": "B",
        "theorem_rank": theorem["rank"],
        "statement_id": theorem["statement_id"],
        "statement_sha256": theorem["statement_sha256"],
        "replicate": replicate,
        "gen_seed": gen_seed,
        "budget": budget,
        "run_tag": run_tag,
        "prompt_tokens": prompt_tokens,
        "n_generated": len(token_ids),
        "hit_eos": hit_eos,
        "hit_budget": (not hit_eos) and len(token_ids) == budget,
        "token_ids": token_ids,
        "runtime_seconds": round(runtime_seconds, 4),
        "git_revision": git_revision(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def run_stage1(args: argparse.Namespace) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    theorems = load_theorems(Path(args.theorems))
    model_dir = Path(args.model)
    if not model_dir.is_absolute():
        model_dir = ROOT / model_dir
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path

    print(f"[V2-E001 stage1] model: {model_dir}")
    print(f"[V2-E001 stage1] theorems: {len(theorems)} x replicates {REPLICATES} x runs {[tag for tag, _ in STAGE1_RUNS]}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"[V2-E001 stage1] model loaded on {device}")

    prompts = {theorem["rank"]: make_prompt(tokenizer, theorem) for theorem in theorems}
    {
        rank: int(tokenizer(prompt, return_tensors="pt")["input_ids"].shape[-1])
        for rank, prompt in prompts.items()
    }

    done = load_done_keys(output_path)
    jobs = []
    for theorem in theorems:
        for replicate in REPLICATES:
            for tag, budget in STAGE1_RUNS:
                key = job_key("1", "hf", theorem["rank"], replicate, tag)
                if key not in done:
                    jobs.append((theorem, replicate, tag, budget))
    print(f"[V2-E001 stage1] pending jobs: {len(jobs)} (already done: {len(done)})")

    progress = tqdm(jobs, desc="V2-E001 stage1", unit="gen", dynamic_ncols=True)
    for theorem, replicate, tag, budget in progress:
        rank = theorem["rank"]
        seed = SEED_BASE + rank * SEED_GROUP_SIZE + replicate
        encoded = tokenizer(prompts[rank], return_tensors="pt").to(device)
        set_seed(seed)
        started = time.perf_counter()
        with torch.no_grad():
            output = model.generate(
                **encoded,
                max_new_tokens=budget,
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
            )
        runtime = time.perf_counter() - started
        prompt_width = int(encoded["input_ids"].shape[-1])
        token_ids = output[0][prompt_width:].tolist()
        record = build_record(
            stage="1",
            engine="hf",
            theorem=theorem,
            replicate=replicate,
            gen_seed=seed,
            budget=budget,
            run_tag=tag,
            prompt_tokens=prompt_width,
            token_ids=token_ids,
            runtime_seconds=runtime,
            eos_token_id=tokenizer.eos_token_id,
        )
        append_record(output_path, record)
        progress.set_postfix_str(
            f"r{rank} rep{replicate} {tag} n={len(token_ids)} {runtime:.1f}s"
        )
    print(f"[V2-E001 stage1] complete; output: {output_path}")
    return 0


def _batch_record(
    *,
    stage: str,
    engine: str,
    run_tag: str,
    budget: int,
    batch_seed: int,
    items: list[dict[str, Any]],
    runtime_seconds: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "stage": stage,
        "engine": engine,
        "experiment": "V2-E001",
        "track": "B",
        "theorem_rank": -1,
        "replicate": -1,
        "gen_seed": batch_seed,
        "budget": budget,
        "run_tag": run_tag,
        "batch_size": len(items),
        "items": items,
        "runtime_seconds": round(runtime_seconds, 4),
        "git_revision": git_revision(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        record.update(extra)
    return record


def run_stage2_hf(args: argparse.Namespace) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    theorems = load_theorems(Path(args.theorems))
    selected = [t for t in theorems if t["rank"] in STAGE2_RANKS]
    model_dir = Path(args.model)
    if not model_dir.is_absolute():
        model_dir = ROOT / model_dir
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    device = next(model.parameters()).device

    prompts = [make_prompt(tokenizer, theorem) for theorem in selected]
    batch_seed = SEED_BASE + STAGE2_RANKS[0] * SEED_GROUP_SIZE + STAGE2_REPLICATE

    done = load_done_keys(output_path)
    pending = [
        (tag, budget)
        for tag, budget in HF_BATCH_RUNS
        if job_key("2", "hf-batch", -1, -1, tag) not in done
    ]
    print(f"[V2-E001 stage2-hf] batch_size={len(prompts)} batch_seed={batch_seed} pending={[tag for tag, _ in pending]}")

    progress = tqdm(pending, desc="V2-E001 stage2-hf", unit="batch", dynamic_ncols=True)
    for tag, budget in progress:
        encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        set_seed(batch_seed)
        started = time.perf_counter()
        with torch.no_grad():
            output = model.generate(
                **encoded,
                max_new_tokens=budget,
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
            )
        runtime = time.perf_counter() - started
        prompt_width = int(encoded["input_ids"].shape[-1])
        items = []
        for index, theorem in enumerate(selected):
            token_ids = output[index][prompt_width:].tolist()
            items.append(
                {
                    "theorem_rank": theorem["rank"],
                    "statement_id": theorem["statement_id"],
                    "replicate": STAGE2_REPLICATE,
                    "n_generated": len(token_ids),
                    "hit_eos": bool(token_ids) and token_ids[-1] == tokenizer.eos_token_id,
                    "token_ids": token_ids,
                }
            )
        append_record(
            output_path,
            _batch_record(
                stage="2",
                engine="hf-batch",
                run_tag=tag,
                budget=budget,
                batch_seed=batch_seed,
                items=items,
                runtime_seconds=runtime,
                extra={"vendor": "transformers"},
            ),
        )
        progress.set_postfix_str(f"{tag} {runtime:.1f}s")
    print(f"[V2-E001 stage2-hf] complete; output: {output_path}")
    return 0


def run_stage2_vllm(args: argparse.Namespace) -> int:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    theorems = load_theorems(Path(args.theorems))
    selected = [t for t in theorems if t["rank"] in STAGE2_RANKS]
    model_dir = Path(args.model)
    if not model_dir.is_absolute():
        model_dir = ROOT / model_dir
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
    prompts = [make_prompt(tokenizer, theorem) for theorem in selected]

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

    done = load_done_keys(output_path)
    pending = [
        (tag, budget)
        for tag, budget in VLLM_RUNS
        if job_key("2", "vllm", -1, -1, tag) not in done
    ]
    print(f"[V2-E001 stage2-vllm] pending={[tag for tag, _ in pending]}")

    progress = tqdm(pending, desc="V2-E001 stage2-vllm", unit="batch", dynamic_ncols=True)
    for tag, budget in progress:
        sampling = [
            SamplingParams(
                temperature=1.0,
                top_p=1.0,
                max_tokens=budget,
                n=1,
                seed=SEED_BASE + theorem["rank"] * SEED_GROUP_SIZE + STAGE2_REPLICATE,
            )
            for theorem in selected
        ]
        started = time.perf_counter()
        outputs = llm.generate(prompts, sampling)
        runtime = time.perf_counter() - started
        items = []
        for index, theorem in enumerate(selected):
            completion = outputs[index].outputs[0]
            token_ids = list(completion.token_ids)
            items.append(
                {
                    "theorem_rank": theorem["rank"],
                    "statement_id": theorem["statement_id"],
                    "replicate": STAGE2_REPLICATE,
                    "n_generated": len(token_ids),
                    "hit_eos": completion.finish_reason != "length",
                    "token_ids": token_ids,
                }
            )
        append_record(
            output_path,
            _batch_record(
                stage="2",
                engine="vllm",
                run_tag=tag,
                budget=budget,
                batch_seed=SEED_BASE,
                items=items,
                runtime_seconds=runtime,
                extra={"vendor": "vllm"},
            ),
        )
        progress.set_postfix_str(f"{tag} {runtime:.1f}s")
    print(f"[V2-E001 stage2-vllm] complete; output: {output_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("1", "2hf", "2vllm"), required=True)
    parser.add_argument("--theorems", default=DEFAULT_THEOREMS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--max-model-len", type=int, default=5120)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = parser.parse_args()
    if args.stage == "1":
        return run_stage1(args)
    if args.stage == "2hf":
        return run_stage2_hf(args)
    return run_stage2_vllm(args)


if __name__ == "__main__":
    raise SystemExit(main())
