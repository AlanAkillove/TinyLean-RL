#!/usr/bin/env python3
"""V4-P001 §F/§G/§H audit — chat template, rendered arm hashes, prompt lengths, diagnostic budget.

Read-only, no generation, no verifier, no GPU, and nothing from the V4 formal sample (which does not
exist). It answers four questions with measurements:

1. does the frozen Kimina/Qwen3 chat template render a multi-turn user/assistant revision prompt
   stably, and does an invented tool role appear anywhere? (it must not be used)
2. what are the exact rendered prompt templates and hashes of Arms A, B and C?
3. what is the prompt-length distribution of the pool, and what is the worst-case context demand
   once a failed proof and a normalized diagnostic are added?
4. how often would the frozen diagnostic normalizer truncate at the 512-token budget?

Output: experiments/manifests/v4/v4_p001_prompt_audit.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tinylean_rl.evaluation.v4_diagnostics import (
    DIAGNOSTIC_TOKEN_BUDGET,
    NORMALIZATION_VERSION,
    bound_diagnostic,
    normalize_diagnostic,
)
from tinylean_rl.evaluation.v4_prompts import (
    CORRECTION_REQUEST,
    RENDERER_VERSION,
    arm_suffix_invariant,
    canonical_messages,
    render_arm,
    sha256_text,
)
from tinylean_rl.inference.extract import extract_proof

OUT = "experiments/manifests/v4/v4_p001_prompt_audit.json"
POOL = "experiments/manifests/v4/v4_p001_pool.json"
TRAIN_PARQUET = "data/processed/p3_promptset/prompt_sets/AI-MO/Kimina-Prover-Promptset/train.parquet"
TOKENIZER = "models/weights/kimina_distill_0_6b"
V1_DUMP_GLOBS = ("runs/p3b_pilot/rollout_data/*.jsonl", "runs/m1_seed2/rollout_data/*.jsonl", "runs/m1_seed3/rollout_data/*.jsonl")
R001_RAW = "runs/v3_r001_rollout_archive_from_fly122/attempt2_complete_20260925T0839Z/v3_r001_raw_rollout.jsonl"
E023_BASE = "experiments/results/e023_holdout_base.json"
MAX_RESPONSE_TOKENS = 4096


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentiles(values: list[int]) -> dict:
    if not values:
        return {}
    ordered = sorted(values)

    def q(fraction: float) -> int:
        return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]

    return {
        "n": len(ordered),
        "min": ordered[0],
        "median": q(0.5),
        "p90": q(0.9),
        "p95": q(0.95),
        "p99": q(0.99),
        "max": ordered[-1],
        "mean": round(sum(ordered) / len(ordered), 1),
    }


def load_failed_proofs() -> tuple[list[dict], list[str]]:
    """Extracted failed proofs and raw diagnostics from ALREADY-CONSUMED corpora only."""

    proofs: list[dict] = []
    diagnostics: list[str] = []
    for pattern in V1_DUMP_GLOBS:
        for path in sorted(ROOT.glob(pattern)):
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                pred = str(row.get("pred") or "")
                feedback = str(row.get("tool_feedback") or "")
                if pred.strip() and feedback.startswith("# Error"):
                    proofs.append({"origin": f"v1:{path.parent.parent.name}/{path.name}", "text": pred})
                if feedback.startswith("# Error"):
                    diagnostics.append(feedback)
    if (ROOT / R001_RAW).exists():
        for line in (ROOT / R001_RAW).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("verify_status") != "lean_error":
                continue
            try:
                extracted = extract_proof(str(row.get("completion_text") or ""))
            except ValueError:
                extracted = ""
            if extracted.strip():
                proofs.append({"origin": "v3_r001/attempt2", "text": extracted})
            if row.get("lean_message"):
                diagnostics.append(str(row["lean_message"]))
    if (ROOT / E023_BASE).exists():
        for record in json.loads((ROOT / E023_BASE).read_text())["records"]:
            if record.get("lean_message"):
                diagnostics.append(str(record["lean_message"]))
    return proofs, diagnostics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=OUT)
    args = parser.parse_args()

    import pandas as pd
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / TOKENIZER))
    pool = json.loads((ROOT / POOL).read_text())
    members = pool["members"]

    template = tokenizer.chat_template or ""
    audit: dict = {
        "artifact_type": "v4_p001_prompt_audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": subprocess.run(["hostname"], capture_output=True, text=True, check=False).stdout.strip(),
        "git_revision": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip(),
        "note": "read-only over already-consumed artifacts; no generation, no verifier, no GPU, no V4 formal data",
        "renderer_version": RENDERER_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
        "correction_request": CORRECTION_REQUEST,
    }

    # --- 1. chat template ---------------------------------------------------------------------
    probe_messages = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "U2"},
    ]
    single_turn = [{"role": "system", "content": "S"}, {"role": "user", "content": "U1"}]
    probe_rendered = tokenizer.apply_chat_template(probe_messages, add_generation_prompt=True, tokenize=False)
    audit["chat_template"] = {
        "name": type(tokenizer).__name__,
        "sha256": sha256_text(template),
        "length_chars": len(template),
        "supports_multi_turn_user_assistant": "message['role'] == 'assistant'" in template or "'assistant'" in template,
        "renders_mid_conversation_assistant_plain": "<|im_start|>assistant\nA1<|im_end|>" in probe_rendered,
        "generation_prompt_suffix": "<|im_start|>assistant\n" if probe_rendered.endswith("<|im_start|>assistant\n") else "",
        "rendered_probe": probe_rendered,
        "rendered_probe_sha256": sha256_text(probe_rendered),
        "single_turn_rendered": tokenizer.apply_chat_template(single_turn, add_generation_prompt=True, tokenize=False),
        "tool_role_supported": "tool" in template,
        "tool_role_used_by_renderer": False,
        "thinking_control": "enable_thinking" in template,
    }

    # --- 2. frozen example: real theorem + real failed proof + real diagnostic ----------------
    example_member = members[0]
    df = pd.read_parquet(ROOT / TRAIN_PARQUET)
    prompts = {
        str(sid): p.tolist() if hasattr(p, "tolist") else list(p)
        for sid, p in zip(df["statement_id"].astype(str), df["prompt"].tolist(), strict=True)
    }
    base_messages = canonical_messages(prompts[example_member["statement_id"]])
    proofs, diagnostics = load_failed_proofs()
    example_proof = proofs[0]["text"]
    example_diagnostic, truncated_example = bound_diagnostic(
        normalize_diagnostic(diagnostics[0]), tokenizer
    )
    rendered = {
        arm: render_arm(
            tokenizer,
            base_messages,
            arm=arm,
            failed_proof=example_proof if arm != "A_FRESH_RETRY" else "",
            diagnostic=example_diagnostic if arm == "C_VERIFIER_REPAIR" else "",
        )
        for arm in ("A_FRESH_RETRY", "B_SELF_REVISION", "C_VERIFIER_REPAIR")
    }
    audit["frozen_example"] = {
        "theorem_statement_id": example_member["statement_id"],
        "theorem_name": example_member["name"],
        "theorem_screening_rank": example_member["screening_rank"],
        "failed_proof_origin": proofs[0]["origin"],
        "failed_proof_tokens": len(tokenizer.encode(example_proof, add_special_tokens=False)),
        "failed_proof_sha256": sha256_text(example_proof),
        "diagnostic_source": "first observed Lean diagnostic of the consumed audit corpus",
        "diagnostic_normalized": example_diagnostic,
        "diagnostic_tokens": len(tokenizer.encode(example_diagnostic, add_special_tokens=False)),
        "diagnostic_truncated": truncated_example,
        "prompt_hashes": {arm: sha256_text(text) for arm, text in rendered.items()},
        "prompt_tokens": {arm: len(tokenizer.encode(text, add_special_tokens=False)) for arm, text in rendered.items()},
        "rendered_prompts": rendered,
        "arm_c_equals_arm_b_plus_diagnostic_after_removal": (
            rendered["C_VERIFIER_REPAIR"].replace(f"\n\n{example_diagnostic}", "") == rendered["B_SELF_REVISION"]
        ),
        "arm_b_c_suffix_invariant": arm_suffix_invariant(
            base_messages, failed_proof=example_proof, diagnostic=example_diagnostic
        ),
        "arm_a_has_no_previous_attempt_reference": (
            "previous" not in rendered["A_FRESH_RETRY"].lower()
            and "rejected" not in rendered["A_FRESH_RETRY"].lower()
        ),
    }
    audit["frozen_example"]["arm_b_c_rendered_difference"] = rendered["C_VERIFIER_REPAIR"][len(rendered["B_SELF_REVISION"]):]

    # --- 3. prompt length distributions -------------------------------------------------------
    arm_a_tokens, arm_c_tokens, theorem_tokens = [], [], []
    for member in members:
        messages = canonical_messages(prompts[member["statement_id"]])
        text_a = render_arm(tokenizer, messages, arm="A_FRESH_RETRY")
        text_c = render_arm(
            tokenizer, messages, arm="C_VERIFIER_REPAIR", failed_proof=example_proof, diagnostic=example_diagnostic
        )
        arm_a_tokens.append(len(tokenizer.encode(text_a, add_special_tokens=False)))
        arm_c_tokens.append(len(tokenizer.encode(text_c, add_special_tokens=False)))
        theorem_tokens.append(member["prompt_tokens"])
    assert arm_a_tokens, "empty pool"
    audit["prompt_lengths"] = {
        "theorem_prompt_tokens_recorded_in_pool": percentiles(theorem_tokens),
        "arm_a_rendered_tokens": percentiles(arm_a_tokens),
        "arm_c_rendered_tokens_with_example_failed_proof": percentiles(arm_c_tokens),
        "renderer_matches_pool_prompt_tokens": all(
            abs(a - b) <= 1 for a, b in zip(arm_a_tokens, theorem_tokens, strict=True)
        ),
    }

    # --- 4. failed-proof and diagnostic budgets ----------------------------------------------
    proof_tokens = [len(tokenizer.encode(p["text"], add_special_tokens=False)) for p in proofs]
    normalized, raw_tokens, norm_tokens, truncated = [], [], [], 0
    for raw in diagnostics:
        norm = normalize_diagnostic(raw)
        if not norm:
            continue
        bounded, was_truncated = bound_diagnostic(norm, tokenizer)
        truncated += int(was_truncated)
        normalized.append(bounded)
        raw_tokens.append(len(tokenizer.encode(raw, add_special_tokens=False)))
        norm_tokens.append(len(tokenizer.encode(bounded, add_special_tokens=False)))
    audit["failed_proof_tokens"] = {
        **percentiles(proof_tokens),
        "corpora": dict(Counter(p["origin"].split("/")[0] for p in proofs)),
        "note": "extracted failed proofs of already-consumed theta0 roll-outs; the second-stage context demand proxy",
    }
    audit["diagnostic_normalization"] = {
        "raw_tokens": percentiles(raw_tokens),
        "normalized_tokens": percentiles(norm_tokens),
        "budget": DIAGNOSTIC_TOKEN_BUDGET,
        "truncation_rate": round(truncated / len(normalized), 4) if normalized else None,
        "truncated": truncated,
        "n": len(normalized),
        "information_preserved": "only paths/ids/timestamps/server wrapper lines are removed; message, goal state, types, identifiers and positions are byte-preserved",
    }
    worst_case = {
        "max_theorem_prompt_tokens": max(arm_a_tokens),
        "max_failed_proof_tokens_observed": max(proof_tokens) if proof_tokens else 0,
        "diagnostic_budget": DIAGNOSTIC_TOKEN_BUDGET,
        "max_response_tokens": MAX_RESPONSE_TOKENS,
    }
    worst_case["worst_case_context_tokens"] = (
        worst_case["max_theorem_prompt_tokens"]
        + worst_case["max_failed_proof_tokens_observed"]
        + worst_case["diagnostic_budget"]
        + worst_case["max_response_tokens"]
    )
    audit["context_requirement"] = worst_case

    destination = ROOT / args.out
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(audit, indent=2) + "\n")
    print("arm hashes:", json.dumps(audit["frozen_example"]["prompt_hashes"], indent=1))
    print("arm tokens:", json.dumps(audit["frozen_example"]["prompt_tokens"]))
    print("template multi-turn ok:", audit["chat_template"]["renders_mid_conversation_assistant_plain"],
          "| tool role supported:", audit["chat_template"]["tool_role_supported"])
    print("diagnostic:", json.dumps(audit["diagnostic_normalization"], indent=1))
    print("context:", json.dumps(audit["context_requirement"], indent=1))
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
