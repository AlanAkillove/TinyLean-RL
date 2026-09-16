#!/usr/bin/env python3
"""Offline GRPO loss rehearsal on the cached rollout batch (P2.5 W2).

Loads ``experiments/local_rl_batch/`` (written by ``promptset_rollout_probe.py``),
assembles VERL-style tensors, runs the Kimina-Prover-Distill-0.6B forward pass
to obtain per-token log-probs, computes the ported GRPO/DrGRPO advantage and
policy loss (``src/tinylean_rl/rl/grpo.py``), and back-propagates.

With a single no-update pass the ratio is exactly 1 by construction (old and
current log-probs come from the same weights), so the expected baseline is
``loss = -mean(advantage)`` with ``clipfrac = 0``; what this rehearsal proves
is that the cached rollout -> advantage -> loss -> backward chain runs
end-to-end and that the gradient norm is finite.  Results go to
``experiments/results/p2_5_grpo_rehearsal.json``.  CPU-runnable (0.6B fp32).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Windows consoles default to GBK, which cannot print Lean/arrow symbols.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from tinylean_rl.rl.batch import build_training_batch, load_cached_rollout_batch
from tinylean_rl.rl.grpo import (
    PINNED_VERL_COMMIT,
    VALID_LOSS_AGG_MODES,
    compute_grpo_outcome_advantage,
    compute_policy_loss_vanilla,
    masked_mean,
)


def git_revision() -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()


def response_log_probs(model, input_ids, prompt_lengths, response_lengths):
    """Per-token log-probs at the response positions of ``input_ids``.

    Returns a ``(bs, width)`` tensor with values in the response span (same
    layout as ``response_mask``) so it can be fed straight into the loss.
    Only the response positions are projected through ``lm_head`` to keep the
    fp32 logits footprint at ``sum(response_lengths) x vocab``.
    """

    import torch

    bs, width = input_ids.shape
    # ``PeftModel.model`` resolves to the wrapped tuner's inner CausalLM, so
    # unwrap to the base checkpoint first: ``.model``/``.lm_head`` then mean
    # the transformer body and the LM head again.  LoRA layers stay in place
    # inside the body and keep receiving gradients either way.
    inner = model.get_base_model() if hasattr(model, "get_base_model") else model
    base = getattr(inner, "model", None)
    lm_head = getattr(inner, "lm_head", None)
    if base is None or lm_head is None:
        raise SystemExit("Model exposes neither .model nor .lm_head; cannot slice responses.")

    hidden = base(input_ids).last_hidden_state  # (bs, width, d)
    pieces = []
    targets = []
    spans = []
    for row in range(bs):
        prompt_len = prompt_lengths[row]
        response_len = response_lengths[row]
        if response_len == 0:
            spans.append((prompt_len, 0))
            continue
        # log p(token at position t) depends on the hidden state at t-1.
        pieces.append(hidden[row, prompt_len - 1 : prompt_len + response_len - 1, :])
        targets.append(input_ids[row, prompt_len : prompt_len + response_len])
        spans.append((prompt_len, response_len))

    out = torch.zeros(bs, width, device=input_ids.device, dtype=torch.float32)
    if not pieces:
        return out

    segment_hidden = torch.cat(pieces, dim=0)
    logits = lm_head(segment_hidden)
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    flat_targets = torch.cat(targets, dim=0)
    token_log_probs = log_probs.gather(-1, flat_targets.unsqueeze(-1)).squeeze(-1)

    offset = 0
    for row, (prompt_len, response_len) in enumerate(spans):
        if response_len == 0:
            continue
        out[row, prompt_len : prompt_len + response_len] = token_log_probs[
            offset : offset + response_len
        ]
        offset += response_len
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Local model path; defaults to TINYLEAN_MODEL_DIR.")
    parser.add_argument("--model-key", default="kimina_distill_0_6b")
    parser.add_argument("--batch-dir", default="experiments/local_rl_batch")
    parser.add_argument("--output", default="experiments/results/p2_5_grpo_rehearsal.json")
    parser.add_argument("--device", default="cpu", help="cpu (default) or cuda.")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument(
        "--max-theorems",
        type=int,
        default=0,
        help="Keep only the rollouts of the first N theorems (0 = all); bounds CPU runs.",
    )
    parser.add_argument(
        "--theorem-indices",
        help="Comma-separated theorem_index values to keep (overrides --max-theorems), "
        "e.g. '17' or '15,16,17,18'; use for mixed-reward group rehearsals.",
    )
    parser.add_argument(
        "--max-response-length",
        type=int,
        default=1024,
        help="Truncate each completion (0 = keep full 4096-token rollouts).",
    )
    parser.add_argument("--loss-agg-mode", choices=VALID_LOSS_AGG_MODES, default="seq-mean-token-sum-norm")
    parser.add_argument("--clip-ratio-low", type=float, default=0.2)
    parser.add_argument("--clip-ratio-high", type=float, default=0.3)
    parser.add_argument("--clip-ratio-c", type=float, default=3.0)
    parser.add_argument("--norm-adv-by-std", action="store_true", help="GRPO-style std scaling (default: DrGRPO off).")
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
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        print(f"[ERROR] inference dependencies unavailable: {exc}", file=sys.stderr)
        return 2

    print(f"[1/5] Loading cached rollout batch from {batch_dir}")
    prompts, rollouts, rewards, metadata = load_cached_rollout_batch(batch_dir)
    if args.theorem_indices:
        wanted = {int(token) for token in args.theorem_indices.split(",") if token.strip()}
        keep = [index for index, rollout in enumerate(rollouts) if rollout["theorem_index"] in wanted]
        rollouts = [rollouts[index] for index in keep]
        rewards = [rewards[index] for index in keep]
        print(f"  --theorem-indices {sorted(wanted)}: kept {len(rollouts)} candidates")
    elif args.max_theorems > 0:
        keep = [index for index, rollout in enumerate(rollouts) if rollout["theorem_index"] < args.max_theorems]
        rollouts = [rollouts[index] for index in keep]
        rewards = [rewards[index] for index in keep]
        print(f"  --max-theorems {args.max_theorems}: kept {len(rollouts)} candidates")
    max_response_length = args.max_response_length or None

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    pad_token_id = tokenizer.eos_token_id or 0
    batch = build_training_batch(
        prompts, rollouts, rewards, pad_token_id=pad_token_id, max_response_length=max_response_length
    )
    input_ids = batch["input_ids"]
    response_mask = batch["response_mask"]
    token_level_rewards = batch["token_level_rewards"]
    uids = batch["uids"]
    bsz, width = input_ids.shape
    total_response_tokens = int(response_mask.sum().item())
    print(
        f"  {len(prompts)} theorems, {bsz} candidates, padded width {width}, "
        f"{total_response_tokens} response tokens"
    )

    print(f"[2/5] Loading {args.model_key} on {args.device} ({args.dtype})")
    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, dtype=getattr(torch, args.dtype)
    )
    model.to(args.device)
    model.eval()
    load_seconds = time.perf_counter() - load_started

    prompt_lengths = [len(prompts[row["theorem_index"]]["prompt_ids"]) for row in rollouts]
    response_lengths = batch["response_lengths"]

    print("[3/5] Forward pass (no_grad) for old log-probs")
    old_log_probs = torch.zeros(bsz, width)
    forward_started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, bsz, args.micro_batch_size):
            stop = min(start + args.micro_batch_size, bsz)
            old_log_probs[start:stop] = response_log_probs(
                model,
                input_ids[start:stop].to(args.device),
                prompt_lengths[start:stop],
                response_lengths[start:stop],
            ).cpu()
    old_logprob_seconds = time.perf_counter() - forward_started

    print("[4/5] GRPO advantage, then loss and backward")
    advantages, _ = compute_grpo_outcome_advantage(
        token_level_rewards,
        response_mask,
        uids,
        norm_adv_by_std_in_grpo=args.norm_adv_by_std,
    )
    valid = response_mask.bool()
    advantage_values = advantages[valid]
    advantage_all_zero = bool((advantage_values == 0).all())

    device_inputs = input_ids.to(args.device)
    device_mask = response_mask.to(args.device)
    device_advantages = advantages.to(args.device)
    device_old_log_probs = old_log_probs.to(args.device)

    model.zero_grad(set_to_none=True)
    loss_totals = []
    loss_seconds = 0.0
    backward_seconds = 0.0
    clipfrac_weighted = 0.0
    kl_weighted = 0.0
    clipfrac_lower_weighted = 0.0
    ratio_weighted = 0.0
    ratio_max = 0.0
    microbatch_losses = []
    for start in range(0, bsz, args.micro_batch_size):
        stop = min(start + args.micro_batch_size, bsz)
        forward_started = time.perf_counter()
        log_probs = response_log_probs(
            model,
            device_inputs[start:stop],
            prompt_lengths[start:stop],
            response_lengths[start:stop],
        )
        loss_seconds += time.perf_counter() - forward_started

        micro_mask = device_mask[start:stop]
        token_count = float(micro_mask.sum().item())
        loss, clipfrac, ppo_kl, clipfrac_lower = compute_policy_loss_vanilla(
            device_old_log_probs[start:stop],
            log_probs,
            device_advantages[start:stop],
            micro_mask,
            clip_ratio_low=args.clip_ratio_low,
            clip_ratio_high=args.clip_ratio_high,
            clip_ratio_c=args.clip_ratio_c,
            loss_agg_mode=args.loss_agg_mode,
        )

        backward_started = time.perf_counter()
        loss.backward()
        backward_seconds += time.perf_counter() - backward_started

        with torch.no_grad():
            ratio = torch.exp(torch.clamp(log_probs - device_old_log_probs[start:stop], -20.0, 20.0))
            if token_count > 0:
                ratio_weighted += masked_mean(ratio, micro_mask).item() * token_count
                ratio_max = max(ratio_max, float(ratio[micro_mask.bool()].max().item()))
                clipfrac_weighted += clipfrac.item() * token_count
                kl_weighted += ppo_kl.item() * token_count
                clipfrac_lower_weighted += clipfrac_lower.item() * token_count
        loss_totals.append(loss.item())
        microbatch_losses.append(round(loss.item(), 8))
        print(f"  micro-batch [{start}:{stop}] loss={loss.item():.6f} clipfrac={clipfrac.item():.4f}")

    total_loss = sum(loss_totals)
    grad_squared = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            grad_squared += parameter.grad.detach().norm().item() ** 2
    grad_norm = math.sqrt(grad_squared)

    summary = {
        "artifact_type": "grpo_loss_rehearsal",
        "pinned_verl_commit": PINNED_VERL_COMMIT,
        "batch_dir": str(batch_dir),
        "model_key": args.model_key,
        "model_path": str(model_path),
        "device": args.device,
        "dtype": args.dtype,
        "batch": {
            "theorems": len({rollout["theorem_index"] for rollout in rollouts}),
            "max_theorems_filter": args.max_theorems,
            "candidates": bsz,
            "padded_width": width,
            "response_tokens": total_response_tokens,
            "max_response_length": max_response_length,
            "note": "completions truncated for CPU memory" if max_response_length else "full rollouts",
        },
        "grpo": {
            "norm_adv_by_std_in_grpo": args.norm_adv_by_std,
            "clip_ratio_low": args.clip_ratio_low,
            "clip_ratio_high": args.clip_ratio_high,
            "clip_ratio_c": args.clip_ratio_c,
            "loss_agg_mode": args.loss_agg_mode,
            "use_kl_loss": False,
            "entropy_coeff": 0.0,
            "micro_batch_size": args.micro_batch_size,
        },
        "advantage": {
            "mean": round(advantage_values.mean().item(), 6) if advantage_values.numel() else None,
            "min": round(advantage_values.min().item(), 6) if advantage_values.numel() else None,
            "max": round(advantage_values.max().item(), 6) if advantage_values.numel() else None,
            "all_zero": advantage_all_zero,
        },
        "loss": {
            "total": round(total_loss, 8),
            "negative_mean_advantage_reference": round(-advantage_values.mean().item(), 6)
            if advantage_values.numel()
            else None,
            "ratio_mean": round(ratio_weighted / total_response_tokens, 8) if total_response_tokens else None,
            "ratio_max": round(ratio_max, 8),
            "pg_clipfrac": round(clipfrac_weighted / total_response_tokens, 8) if total_response_tokens else None,
            "pg_clipfrac_lower": round(clipfrac_lower_weighted / total_response_tokens, 8)
            if total_response_tokens
            else None,
            "ppo_kl": round(kl_weighted / total_response_tokens, 8) if total_response_tokens else None,
            "grad_norm": round(grad_norm, 6),
            "finite_grad_norm": math.isfinite(grad_norm),
            "per_microbatch": microbatch_losses,
        },
        "seconds": {
            "load_model": round(load_seconds, 2),
            "old_logprobs": round(old_logprob_seconds, 2),
            "loss_forward": round(loss_seconds, 2),
            "backward": round(backward_seconds, 2),
            "total": round(time.perf_counter() - started, 2),
        },
        "metadata_excerpt": {
            key: metadata.get(key)
            for key in ("sampled_statement_ids", "theorems", "samples_per_theorem", "created_at_utc")
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
    }
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("[5/5] Rehearsal summary")
    print(json.dumps({key: summary[key] for key in ("batch", "advantage", "loss", "seconds")}, indent=2))
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
