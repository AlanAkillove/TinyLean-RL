#!/usr/bin/env python3
"""V4-P001 §I — prepare the worst-case Arm C prompt for the context smoke (CPU, no generation).

The prompt is built from ALREADY-CONSUMED material only: a pool theorem (development families, never
the V4 formal sample, which does not exist), the longest failed proof observed in the consumed
corpora, and the longest normalized diagnostic at the frozen budget. Selection is by token length
alone -- no outcome, source or repairability label is read.

Output: experiments/manifests/v4/v4_p001_smoke_prompt.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
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
    RENDERER_VERSION,
    canonical_messages,
    render_arm,
    sha256_text,
)
from tinylean_rl.inference.extract import extract_proof

OUT = "experiments/manifests/v4/v4_p001_smoke_prompt.json"
POOL = "experiments/manifests/v4/v4_p001_pool.json"
TRAIN_PARQUET = "data/processed/p3_promptset/prompt_sets/AI-MO/Kimina-Prover-Promptset/train.parquet"
TOKENIZER = "models/weights/kimina_distill_0_6b"
V1_DUMP_GLOBS = ("runs/p3b_pilot/rollout_data/*.jsonl", "runs/m1_seed2/rollout_data/*.jsonl", "runs/m1_seed3/rollout_data/*.jsonl")
R001_RAW = "runs/v3_r001_rollout_archive_from_fly122/attempt2_complete_20260925T0839Z/v3_r001_raw_rollout.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=OUT)
    args = parser.parse_args()

    import pandas as pd
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / TOKENIZER))
    pool = json.loads((ROOT / POOL).read_text())

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
    if not proofs or not diagnostics:
        raise SystemExit("no consumed failed proofs/diagnostics found for the smoke prompt")

    longest_proof_origin, longest_proof = max(
        proofs, key=lambda item: len(tokenizer.encode(item[1], add_special_tokens=False))
    )
    normalized = [bound_diagnostic(normalize_diagnostic(text), tokenizer) for text in diagnostics]
    longest_diagnostic, longest_diagnostic_truncated = max(
        normalized, key=lambda item: len(tokenizer.encode(item[0], add_special_tokens=False))
    )

    member = max(pool["members"], key=lambda row: row["prompt_tokens"])
    prompts = {
        str(sid): (p.tolist() if hasattr(p, "tolist") else list(p))
        for sid, p in zip(
            pd.read_parquet(ROOT / TRAIN_PARQUET)["statement_id"].astype(str),
            pd.read_parquet(ROOT / TRAIN_PARQUET)["prompt"].tolist(),
            strict=True,
        )
    }
    base_messages = canonical_messages(prompts[member["statement_id"]])
    prompt_text = render_arm(
        tokenizer,
        base_messages,
        arm="C_VERIFIER_REPAIR",
        failed_proof=longest_proof,
        diagnostic=longest_diagnostic,
    )
    prompt_tokens = len(tokenizer.encode(prompt_text, add_special_tokens=False))
    artifact = {
        "artifact_type": "v4_p001_smoke_prompt",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": "worst-case Arm C prompt for the §I context smoke; built from consumed material only",
        "renderer_version": RENDERER_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
        "arm": "C_VERIFIER_REPAIR",
        "theorem": {
            "statement_id": member["statement_id"],
            "component_id": member["component_id"],
            "name": member["name"],
            "screening_rank": member["screening_rank"],
            "tier": member["tier"],
            "theorem_prompt_tokens": member["prompt_tokens"],
        },
        "failed_proof": {
            "origin": longest_proof_origin,
            "tokens": len(tokenizer.encode(longest_proof, add_special_tokens=False)),
            "sha256": sha256_text(longest_proof),
            "text": longest_proof,
        },
        "diagnostic": {
            "tokens": len(tokenizer.encode(longest_diagnostic, add_special_tokens=False)),
            "truncated_at_budget": longest_diagnostic_truncated,
            "budget": DIAGNOSTIC_TOKEN_BUDGET,
            "text": longest_diagnostic,
            "snapshot_taken_from": "the longest normalized diagnostic of the consumed audit corpus",
        },
        "prompt_tokens": prompt_tokens,
        "prompt_sha256": sha256_text(prompt_text),
        "prompt_text": prompt_text,
        "selection_rule": "longest by token count in each stream; no outcome or source label is read",
        "git_revision": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip(),
    }
    destination = ROOT / args.out
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"prompt tokens: {prompt_tokens} (theorem {member['prompt_tokens']}, proof {artifact['failed_proof']['tokens']}, diagnostic {artifact['diagnostic']['tokens']})")
    print(f"prompt_sha256: {artifact['prompt_sha256']}")
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
