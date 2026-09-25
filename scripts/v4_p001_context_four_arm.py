#!/usr/bin/env python3
"""V4-P001 Amendment A §10 — four-arm context feasibility audit (NONFORMAL, no generation, no GPU).

Read-only over already-consumed material. It answers three questions:

1. what is the rendered prompt-token distribution of each of the four arms over the frozen pool
   (max, p95, p99), including the mismatched-diagnostic arm D?
2. does the worst-case prompt of every arm fit the frozen engine context together with a full
   4096-token response: ``max_prompt + 4096 <= max_model_len``?
3. does the audit reproduce, byte for byte, the worst-case Arm C prompt that the §I context smoke
   already ran on the GPU (so the measured 10 GB memory behaviour is evidence about *this* design)?

The worst-case construction is the frozen one: the longest theorem prompt of the pool, the longest
failed proof of the consumed corpora, and a normalized diagnostic bounded at the frozen 512-token
budget. Arm D is measured with a *different* diagnostic than Arm C -- the whole point of the arm is
that the diagnostic belongs to another theorem -- and because both diagnostics come from the same
bounded stream, the arm-D ceiling is structurally the arm-C ceiling.

Output: experiments/manifests/v4/v4_p001_context_four_arm.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "scripts", ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from v4_p001_prompt_audit import percentiles

from tinylean_rl.evaluation.v4_diagnostics import (
    DIAGNOSTIC_TOKEN_BUDGET,
    NORMALIZATION_VERSION,
    bound_diagnostic,
    normalize_diagnostic,
)
from tinylean_rl.evaluation.v4_prompts import (
    ARMS,
    DIAGNOSTIC_ARMS,
    RENDERER_VERSION,
    canonical_messages,
    diagnostic_arm_invariants,
    render_arm,
    sha256_text,
)
from tinylean_rl.inference.extract import extract_proof

OUT = "experiments/manifests/v4/v4_p001_context_four_arm.json"
POOL = "experiments/manifests/v4/v4_p001_pool.json"
TRAIN_PARQUET = "data/processed/p3_promptset/prompt_sets/AI-MO/Kimina-Prover-Promptset/train.parquet"
TOKENIZER = "models/weights/kimina_distill_0_6b"
SMOKE = "experiments/manifests/v4/v4_p001_context_smoke.json"
SMOKE_PROMPT = "experiments/manifests/v4/v4_p001_smoke_prompt.json"
V1_DUMP_GLOBS = (
    "runs/p3b_pilot/rollout_data/*.jsonl",
    "runs/m1_seed2/rollout_data/*.jsonl",
    "runs/m1_seed3/rollout_data/*.jsonl",
)
R001_RAW = "runs/v3_r001_rollout_archive_from_fly122/attempt2_complete_20260925T0839Z/v3_r001_raw_rollout.jsonl"
MAX_RESPONSE_TOKENS = 4096
FROZEN_MAX_MODEL_LEN = 10240


def load_smoke_streams() -> tuple[list[tuple[str, str]], list[str]]:
    """The §I smoke streams (V1 roll-out dumps + the V3-R001 archive), in a fixed encounter order.

    Deliberately the same two streams as ``scripts/v4_p001_smoke_prompt.py``: the audit has to
    reproduce that prompt byte for byte for the GPU memory evidence to be about this design, and a
    differing stream would silently change the worst-case diagnostic. Ordering is deterministic
    (sorted globs, file order), so the ``max`` tie-breaks below are reproducible.
    """

    proofs: list[tuple[str, str]] = []
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
                    proofs.append((f"v1:{path.parent.parent.name}/{path.name}", pred))
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
                proofs.append(("v3_r001/attempt2", extracted))
            if row.get("lean_message"):
                diagnostics.append(str(row["lean_message"]))
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
    proofs, diagnostics = load_smoke_streams()
    if not proofs or not diagnostics:
        raise SystemExit("no consumed failed proofs/diagnostics found")

    # --- worst case, same selection rule as the §I smoke prompt (length only) --------------------
    member = max(members, key=lambda row: row["prompt_tokens"])
    longest_proof_origin, longest_proof = max(
        proofs, key=lambda item: len(tokenizer.encode(item[1], add_special_tokens=False))
    )
    bounded = [bound_diagnostic(normalize_diagnostic(text), tokenizer) for text in diagnostics]
    own_diagnostic, own_truncated = max(
        bounded, key=lambda item: len(tokenizer.encode(item[0], add_special_tokens=False))
    )
    donor_diagnostic = max(
        (text for text, _ in bounded if text != own_diagnostic),
        key=lambda text: len(tokenizer.encode(text, add_special_tokens=False)),
    )
    diagnostic_tokens = {
        "own": len(tokenizer.encode(own_diagnostic, add_special_tokens=False)),
        "donor": len(tokenizer.encode(donor_diagnostic, add_special_tokens=False)),
    }

    df = pd.read_parquet(ROOT / TRAIN_PARQUET)
    prompts = {
        str(sid): (p.tolist() if hasattr(p, "tolist") else list(p))
        for sid, p in zip(df["statement_id"].astype(str), df["prompt"].tolist(), strict=True)
    }
    base_messages = canonical_messages(prompts[member["statement_id"]])

    def render(arm: str, proof: str, diagnostic: str) -> str:
        return render_arm(
            tokenizer,
            base_messages,
            arm=arm,
            failed_proof=proof if arm != "A_FRESH_RETRY" else "",
            diagnostic=diagnostic if arm in DIAGNOSTIC_ARMS else "",
        )

    worst_case_text = {
        "A_FRESH_RETRY": render("A_FRESH_RETRY", "", ""),
        "B_SELF_REVISION": render("B_SELF_REVISION", longest_proof, ""),
        "C_VERIFIER_REPAIR": render("C_VERIFIER_REPAIR", longest_proof, own_diagnostic),
        "D_MISMATCHED_DIAGNOSTIC": render("D_MISMATCHED_DIAGNOSTIC", longest_proof, donor_diagnostic),
    }
    worst_case_tokens = {
        arm: len(tokenizer.encode(text, add_special_tokens=False)) for arm, text in worst_case_text.items()
    }

    smoke = json.loads((ROOT / SMOKE).read_text())
    max_model_len = int(smoke["engine"]["max_model_len"])
    if max_model_len != FROZEN_MAX_MODEL_LEN:
        raise SystemExit(f"frozen max_model_len mismatch: smoke says {max_model_len}")

    context_rows = {}
    for arm in ARMS:
        prompt_tokens = worst_case_tokens[arm]
        total = prompt_tokens + MAX_RESPONSE_TOKENS
        context_rows[arm] = {
            "worst_case_prompt_tokens": prompt_tokens,
            "max_response_tokens": MAX_RESPONSE_TOKENS,
            "worst_case_total_tokens": total,
            "max_model_len": max_model_len,
            "headroom_tokens": max_model_len - total,
            "fits": total <= max_model_len,
        }

    invariants = diagnostic_arm_invariants(
        base_messages,
        failed_proof=longest_proof,
        own_diagnostic=own_diagnostic,
        donor_diagnostic=donor_diagnostic,
    )
    smoke_prompt = json.loads((ROOT / SMOKE_PROMPT).read_text())
    structural = {
        **invariants,
        "worst_case_c_reproduces_smoke_prompt_sha256": (
            sha256_text(worst_case_text["C_VERIFIER_REPAIR"]) == smoke["prompt"]["prompt_sha256"]
        ),
        "arm_d_has_the_arm_c_message_count": (
            worst_case_text["D_MISMATCHED_DIAGNOSTIC"].count("<|im_start|>")
            == worst_case_text["C_VERIFIER_REPAIR"].count("<|im_start|>")
        ),
        "worst_case_diagnostic_tokens_within_budget": max(diagnostic_tokens.values())
        <= DIAGNOSTIC_TOKEN_BUDGET,
    }

    # --- pool-wide distributions (reference proof/diagnostic, same pairing as the prompt audit) ---
    reference_proof = proofs[0][1]
    reference_diagnostic = bounded[0][0]
    pool_tokens: dict[str, list[int]] = {arm: [] for arm in ARMS}
    for row in members:
        messages = canonical_messages(prompts[row["statement_id"]])
        for arm in ARMS:
            text = render_arm(
                tokenizer,
                messages,
                arm=arm,
                failed_proof=reference_proof if arm != "A_FRESH_RETRY" else "",
                diagnostic=reference_diagnostic if arm in DIAGNOSTIC_ARMS else "",
            )
            pool_tokens[arm].append(len(tokenizer.encode(text, add_special_tokens=False)))
    distribution = {arm: percentiles(values) for arm, values in pool_tokens.items()}

    checks = {
        "all_arms_fit_with_full_response": all(row["fits"] for row in context_rows.values()),
        "worst_case_c_is_the_longest_arm": worst_case_tokens["C_VERIFIER_REPAIR"]
        >= max(worst_case_tokens.values()),
        "arm_a_worst_case_equals_pool_max_theorem": worst_case_tokens["A_FRESH_RETRY"] == member["prompt_tokens"],
        "arm_d_worst_case_is_bounded_by_b_plus_the_diagnostic_budget": worst_case_tokens[
            "D_MISMATCHED_DIAGNOSTIC"
        ]
        <= worst_case_tokens["B_SELF_REVISION"] + DIAGNOSTIC_TOKEN_BUDGET,
        "worst_case_c_reproduces_smoke_prompt_sha256": structural[
            "worst_case_c_reproduces_smoke_prompt_sha256"
        ],
        "arm_d_differs_from_c_only_in_the_diagnostic": invariants["d_equals_c_with_own_replaced_by_donor"],
        "no_self_diagnostic_in_the_audited_pairing": own_diagnostic != donor_diagnostic,
        "smoke_prompt_used_the_frozen_arm_c_and_renderer": (
            smoke_prompt["arm"] == "C_VERIFIER_REPAIR" and smoke_prompt["renderer_version"] == RENDERER_VERSION
        ),
    }
    failing = [name for name, ok in checks.items() if not ok]
    if failing:
        raise SystemExit(f"four-arm context audit failed: {failing}")

    artifact = {
        "artifact_type": "v4_p001_context_four_arm",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": subprocess.run(["hostname"], capture_output=True, text=True, check=False).stdout.strip(),
        "git_revision": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip(),
        "note": "NONFORMAL context feasibility audit, CPU only; no generation, no verifier, no V4 formal theorem",
        "renderer_version": RENDERER_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
        "max_response_tokens": MAX_RESPONSE_TOKENS,
        "max_model_len": max_model_len,
        "selection_rule": "longest by token count in each stream; no outcome, source, family or error label is read",
        "worst_case_inputs": {
            "theorem": {
                "statement_id": member["statement_id"],
                "name": member["name"],
                "screening_rank": member["screening_rank"],
                "tier": member["tier"],
                "theorem_prompt_tokens": member["prompt_tokens"],
            },
            "failed_proof": {
                "origin": longest_proof_origin,
                "tokens": len(tokenizer.encode(longest_proof, add_special_tokens=False)),
                "sha256": sha256_text(longest_proof),
            },
            "own_diagnostic_tokens": diagnostic_tokens["own"],
            "donor_diagnostic_tokens": diagnostic_tokens["donor"],
            "own_diagnostic_truncated_at_budget": own_truncated,
            "diagnostic_stream": "the §I smoke streams (V1 roll-out dumps + V3-R001 archive); the audit corpus additionally holds E023 diagnostics, which carry no proof and are excluded here so the reproduction check is exact",
            "diagnostic_budget": DIAGNOSTIC_TOKEN_BUDGET,
        },
        "worst_case_by_arm": context_rows,
        "worst_case_prompt_sha256": {arm: sha256_text(text) for arm, text in worst_case_text.items()},
        "pool_prompt_tokens": distribution,
        "structural_checks": structural,
        "checks": checks,
        "smoke_evidence": {
            "artifact": SMOKE,
            "engine": smoke["engine"],
            "peak_vram_mib_nvidia_smi": smoke["memory"]["peak_vram_mib_nvidia_smi"],
            "measured_prompt_tokens": smoke["prompt"]["prompt_tokens_measured_here"],
            "measured_total_tokens": smoke["prompt"]["prompt_tokens_measured_here"] + MAX_RESPONSE_TOKENS,
            "measured_headroom_tokens": max_model_len
            - (smoke["prompt"]["prompt_tokens_measured_here"] + MAX_RESPONSE_TOKENS),
            "PASS": smoke.get("PASS"),
            "why_it_still_covers_the_four_arm_design": (
                "the smoke ran the byte-identical worst-case arm-C prompt reproduced above; arm D uses the "
                "same message structure and a diagnostic bounded by the same budget, so it cannot exceed the "
                "arm-C ceiling on this pool. No engine parameter, tokenizer or model revision changes."
            ),
        },
        "pool_size": len(members),
        "pool_hash": pool["pool_hash"],
    }
    destination = ROOT / args.out
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact, indent=2) + "\n")
    print("worst-case prompt tokens:", json.dumps(worst_case_tokens))
    print("worst-case totals:", json.dumps({a: r["worst_case_total_tokens"] for a, r in context_rows.items()}))
    print("headroom tokens:", json.dumps({a: r["headroom_tokens"] for a, r in context_rows.items()}))
    print("pool max/p95/p99:", json.dumps({a: {"max": d["max"], "p95": d["p95"], "p99": d["p99"]} for a, d in distribution.items()}))
    print("checks:", json.dumps(checks, indent=1))
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
