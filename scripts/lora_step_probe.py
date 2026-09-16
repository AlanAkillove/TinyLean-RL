#!/usr/bin/env python3
"""Single-step LoRA training memory probe on Kimina-Prover-Distill-0.6B (P2.5 W3).

Answers one question: can a GRPO *policy update* (LoRA + backward +
optimizer.step) fit into locally available GPU memory?  For every
``(lora_rank, seq_len)`` combination the script

1. loads the 0.6B checkpoint in fp16 and wraps q/k/v/o with LoRA,
2. enables gradient checkpointing,
3. picks the longest cached rollout that fits ``seq_len`` and runs
   forward -> ported GRPO loss (:mod:`tinylean_rl.rl.grpo`) -> backward ->
   ``optimizer.step()`` with micro-batch 1,
4. records allocated/reserved/peak VRAM and forward/backward/step seconds.

An OOM is a perfectly valid outcome and is recorded as such (``status``),
so the artifact answers the feasibility question either way.  Results go to
``experiments/results/p2_5_lora_step_probe.json``.

Reference configuration for context: the pinned VERL recipe trains the
0.6B model with full-parameter FSDP on 8 GPUs; this probe bounds the
*single-device* footprint of the update step only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
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

from grpo_loss_rehearsal import git_revision, response_log_probs

from tinylean_rl.rl.batch import load_cached_rollout_batch
from tinylean_rl.rl.grpo import (
    PINNED_VERL_COMMIT,
    compute_grpo_outcome_advantage,
    compute_policy_loss_vanilla,
)

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


def choose_probe_sample(prompts, rollouts, seq_len):
    """Longest (prompt + truncated completion) rollout that fits ``seq_len``."""

    best = None
    for row, rollout in enumerate(rollouts):
        prompt = prompts[rollout["theorem_index"]]["prompt_ids"]
        completion = rollout["completion_ids"]
        prompt_len = len(prompt)
        if prompt_len >= seq_len:
            continue
        response_len = min(len(completion), seq_len - prompt_len)
        total = prompt_len + response_len
        if response_len <= 0:
            continue
        if best is None or total > best["total"]:
            best = {
                "row": row,
                "prompt": prompt,
                "completion": completion[:response_len],
                "prompt_len": prompt_len,
                "response_len": response_len,
                "total": total,
            }
    return best


def build_probe_tensors(sample, device):
    import torch

    input_ids = torch.tensor(
        [sample["prompt"] + sample["completion"]], dtype=torch.long, device=device
    )
    response_mask = torch.zeros(1, input_ids.shape[1], device=device)
    response_mask[0, sample["prompt_len"] :] = 1.0
    return input_ids, response_mask


def run_combination(
    model,
    optimizer,
    prompts,
    rollouts,
    advantages,
    lora_rank,
    seq_len,
    clip_args,
    device,
):
    """One forward/backward/step attempt; returns a JSON-ready dict."""

    import torch

    result = {
        "lora_rank": lora_rank,
        "seq_len_target": seq_len,
        "status": "ok",
    }
    sample = choose_probe_sample(prompts, rollouts, seq_len)
    if sample is None:
        result["status"] = "skipped"
        result["reason"] = "no cached rollout fits the requested seq_len"
        return result

    result.update(
        {
            "rollout_index": sample["row"],
            "prompt_tokens": sample["prompt_len"],
            "response_tokens": sample["response_len"],
            "actual_seq_len": sample["total"],
        }
    )

    input_ids, response_mask = build_probe_tensors(sample, device)
    prompt_lengths = [sample["prompt_len"]]
    response_lengths = [sample["response_len"]]
    advantage_row = advantages[sample["row"]].to(device)

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    result["vram_before"] = {
        "free_mb": round(free_bytes / 2**20, 1),
        "total_mb": round(total_bytes / 2**20, 1),
    }

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model.zero_grad(set_to_none=True)

    try:
        with torch.no_grad():
            old_log_probs = response_log_probs(model, input_ids, prompt_lengths, response_lengths)

        forward_started = time.perf_counter()
        log_probs = response_log_probs(model, input_ids, prompt_lengths, response_lengths)
        torch.cuda.synchronize()
        forward_seconds = time.perf_counter() - forward_started

        loss, clipfrac, ppo_kl, clipfrac_lower = compute_policy_loss_vanilla(
            old_log_probs,
            log_probs,
            advantage_row,
            response_mask,
            clip_ratio_low=clip_args["low"],
            clip_ratio_high=clip_args["high"],
            clip_ratio_c=clip_args["c"],
            loss_agg_mode=clip_args["loss_agg_mode"],
        )

        backward_started = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        backward_seconds = time.perf_counter() - backward_started

        grad_squared = 0.0
        trainable = 0
        with_grad = 0
        for parameter in model.parameters():
            if not parameter.requires_grad:
                continue
            trainable += 1
            if parameter.grad is not None:
                with_grad += 1
                grad_squared += parameter.grad.detach().float().norm().item() ** 2

        step_started = time.perf_counter()
        optimizer.step()
        torch.cuda.synchronize()
        optimizer_seconds = time.perf_counter() - step_started
        optimizer.zero_grad(set_to_none=True)

        result.update(
            {
                "loss": round(loss.item(), 8),
                "pg_clipfrac": round(clipfrac.item(), 6),
                "ppo_kl": round(ppo_kl.item(), 6),
                "pg_clipfrac_lower": round(clipfrac_lower.item(), 6),
                "grad_norm": round(math.sqrt(grad_squared), 6),
                "finite_grad_norm": math.isfinite(grad_squared),
                "trainable_parameter_tensors": trainable,
                "tensors_with_grad": with_grad,
                "vram": {
                    "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
                    "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 2**20, 1),
                },
                "seconds": {
                    "forward": round(forward_seconds, 4),
                    "backward": round(backward_seconds, 4),
                    "optimizer_step": round(optimizer_seconds, 4),
                },
            }
        )
    except RuntimeError as exc:
        message = str(exc)
        if "out of memory" in message.lower():
            result["status"] = "oom"
            result["error"] = message.split("\n")[0]
            result["vram"] = {
                "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
                "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 2**20, 1),
            }
        else:
            raise
        finally_free, _ = torch.cuda.mem_get_info()
        result["vram_after_oom"] = {"free_mb": round(finally_free / 2**20, 1)}
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

    del input_ids, response_mask, advantage_row
    torch.cuda.empty_cache()
    return result


def load_lora_model(model_path, lora_rank, dtype, gradient_checkpointing, device):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, dtype=dtype)
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=2 * lora_rank,
        lora_dropout=0.0,
        target_modules=TARGET_MODULES,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.to(device)
    return model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Local model path; defaults to TINYLEAN_MODEL_DIR.")
    parser.add_argument("--model-key", default="kimina_distill_0_6b")
    parser.add_argument("--batch-dir", default="experiments/local_rl_batch")
    parser.add_argument("--output", default="experiments/results/p2_5_lora_step_probe.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--lora-ranks", default="16,32", help="Comma-separated LoRA ranks.")
    parser.add_argument("--seq-lens", default="1024,2048", help="Comma-separated sequence lengths.")
    parser.add_argument("--loss-agg-mode", default="seq-mean-token-sum-norm")
    parser.add_argument("--clip-ratio-low", type=float, default=0.2)
    parser.add_argument("--clip-ratio-high", type=float, default=0.3)
    parser.add_argument("--clip-ratio-c", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        help="Disable gradient checkpointing (measures the un-checkpointed footprint).",
    )
    args = parser.parse_args()

    started = time.perf_counter()

    model_dir = args.model or os.getenv("TINYLEAN_MODEL_DIR")
    if not model_dir:
        model_dir = str(ROOT / "models" / "weights" / args.model_key)
    model_path = Path(model_dir)
    if not model_path.exists():
        print(f"[ERROR] model directory not found: {model_path}", file=sys.stderr)
        return 2

    batch_dir = Path(args.batch_dir)
    if not batch_dir.is_absolute():
        batch_dir = ROOT / batch_dir
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        from peft import __version__ as peft_version
    except ImportError as exc:
        print(f"[ERROR] probe dependencies unavailable ({exc}); run `uv sync --extra probe`.", file=sys.stderr)
        return 2

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[ERROR] CUDA device requested but not available.", file=sys.stderr)
        return 2

    lora_ranks = [int(value) for value in args.lora_ranks.split(",") if value.strip()]
    seq_lens = [int(value) for value in args.seq_lens.split(",") if value.strip()]
    gradient_checkpointing = not args.no_gradient_checkpointing

    print(f"[1/4] Loading cached rollout batch from {batch_dir}")
    prompts, rollouts, reward_records, metadata = load_cached_rollout_batch(batch_dir)
    print(f"  {len(prompts)} theorems, {len(rollouts)} cached candidates")

    # Group-normalised advantages are computed once on the full cached batch so
    # the probe steps on realistic GRPO gradients.
    reward_values = [float(record["reward"]) for record in reward_records]
    token_rewards = torch.zeros(len(rollouts), 1)
    token_rewards[:, 0] = torch.tensor(reward_values)
    response_mask_full = torch.ones(len(rollouts), 1)
    uids = [prompts[rollout["theorem_index"]]["statement_id"] for rollout in rollouts]
    advantages, _ = compute_grpo_outcome_advantage(
        token_rewards, response_mask_full, uids, norm_adv_by_std_in_grpo=False
    )

    clip_args = {
        "low": args.clip_ratio_low,
        "high": args.clip_ratio_high,
        "c": args.clip_ratio_c,
        "loss_agg_mode": args.loss_agg_mode,
    }

    combinations = []
    for lora_rank in lora_ranks:
        print(f"[2/4] LoRA r={lora_rank}: loading fp16 model + gradient checkpointing={gradient_checkpointing}")
        load_started = time.perf_counter()
        model = load_lora_model(
            model_path, lora_rank, getattr(torch, args.dtype), gradient_checkpointing, args.device
        )
        load_seconds = time.perf_counter() - load_started
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=args.learning_rate
        )

        for seq_len in seq_lens:
            print(f"[3/4] r={lora_rank} seq_len<={seq_len}: forward -> loss -> backward -> step")
            result = run_combination(
                model,
                optimizer,
                prompts,
                rollouts,
                advantages,
                lora_rank,
                seq_len,
                clip_args,
                args.device,
            )
            result["model_load_seconds"] = round(load_seconds, 2)
            result["trainable_parameters"] = trainable_params
            result["trainable_fraction"] = round(trainable_params / total_params, 8)
            result["peak_vram_seen_mb"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)
            combinations.append(result)
            print(f"  status={result['status']} {json.dumps(result.get('vram', {}))}")

        del optimizer, model
        torch.cuda.empty_cache()

    ok_combinations = [item for item in combinations if item["status"] == "ok"]
    max_peak_reserved = max((item["vram"]["peak_reserved_mb"] for item in ok_combinations), default=None)
    total_vram = round(torch.cuda.get_device_properties(0).total_memory / 2**20, 1) if torch.cuda.is_available() else None

    summary = {
        "artifact_type": "lora_step_probe",
        "pinned_verl_commit": PINNED_VERL_COMMIT,
        "batch_dir": str(batch_dir),
        "model_key": args.model_key,
        "model_path": str(model_path),
        "device": args.device,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "total_vram_mb": total_vram,
        "dtype": args.dtype,
        "gradient_checkpointing": gradient_checkpointing,
        "micro_batch_size": 1,
        "lora": {
            "target_modules": TARGET_MODULES,
            "lora_alpha": "2 * rank",
            "lora_dropout": 0.0,
            "optimizer": "AdamW",
            "learning_rate": args.learning_rate,
        },
        "grpo": {
            "loss_agg_mode": args.loss_agg_mode,
            "clip_ratio_low": args.clip_ratio_low,
            "clip_ratio_high": args.clip_ratio_high,
            "clip_ratio_c": args.clip_ratio_c,
            "norm_adv_by_std_in_grpo": False,
        },
        "combinations": combinations,
        "conclusion": {
            "all_combinations_ok": len(ok_combinations) == len(combinations) and len(combinations) > 0,
            "max_peak_reserved_mb": max_peak_reserved,
            "peak_headroom_mb": round(total_vram - max_peak_reserved, 1)
            if total_vram is not None and max_peak_reserved is not None
            else None,
        },
        "batch_excerpt": {
            key: metadata.get(key)
            for key in ("sampled_statement_ids", "theorems", "samples_per_theorem", "created_at_utc")
        },
        "peft_version": peft_version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
    }
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("[4/4] Probe summary")
    for item in combinations:
        vram = item.get("vram", {})
        print(
            f"  r={item['lora_rank']} seq<={item['seq_len_target']} "
            f"actual={item.get('actual_seq_len')} status={item['status']} "
            f"peak_reserved={vram.get('peak_reserved_mb')}MB"
        )
    print(json.dumps(summary["conclusion"], indent=2))
    print(f"Output: {output_path}")
    print(f"Total seconds: {time.perf_counter() - started:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
