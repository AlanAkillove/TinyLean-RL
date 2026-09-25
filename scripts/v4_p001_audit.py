#!/usr/bin/env python3
"""V4-P001 §C/§F/§H audit — historical failure prevalence, taxonomy coverage, prompt lengths.

Everything here is read-only over ALREADY-CONSUMED artifacts; no generation, no verifier call, no
GPU work, and nothing in the V4 formal sample (which does not exist yet). Three questions are
answered with measurements rather than assumptions:

1. How often did theta0 actually produce a *primary* (semantic) failure on first attempt, and what
   is the syntax / format / resource split? Corpora: the V1 training dumps (the Tier-1 source
   population, including the step-1 dumps that are literally theta0 rollouts), E023's theta0
   evaluation of the sealed holdout, and V3-R001 attempt-2's theta0 rollouts.
2. Does the frozen taxonomy cover the observed diagnostics, and how often is a message ambiguous
   across primary rules?
3. What is the prompt-length distribution, and how often would the diagnostic normalization have to
   truncate at the frozen 512-token budget?

Output: experiments/manifests/v4/v4_p001_audit.json
"""

from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tinylean_rl.evaluation.v4_taxonomy import (
    FORMAT_NO_CODE,
    INFRA,
    OTHER_SEMANTIC,
    PRIMARY_CATEGORIES,
    category_counts,
    classify_first_attempt,
    matched_primary_rules,
)

OUT = "experiments/manifests/v4/v4_p001_audit.json"
POOL = "experiments/manifests/v4/v4_p001_pool.json"
V1_DUMP_GLOBS = (
    "runs/p3b_pilot/rollout_data/*.jsonl",
    "runs/m1_seed2/rollout_data/*.jsonl",
    "runs/m1_seed3/rollout_data/*.jsonl",
)
R001_RAW = "runs/v3_r001_rollout_archive_from_fly122/attempt2_complete_20260925T0839Z/v3_r001_raw_rollout.jsonl"
E023_BASE = "experiments/results/e023_holdout_base.json"
E024_PARTIAL = "experiments/results/e024_minif2f_theta0.partial.json"
DIAGNOSTIC_TOKEN_BUDGET = 512


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def first_error_block(tool_feedback: str) -> str:
    """The first ``# Error N:`` / ``Error message:`` block of a V1 tool feedback string."""

    text = tool_feedback or ""
    if not text.startswith("# Error"):
        return ""
    head, _, _ = text.partition("# Error 2:")
    body = head.split("Error message:", 1)[-1]
    return body.strip()


def classify_v1_dump(rows: list[dict]) -> list[dict]:
    """V1 dumps: ``format_error`` names the formatting failure; ``tool_feedback`` carries the
    first Lean diagnostic and ``score`` the verifier outcome."""

    out = []
    for row in rows:
        score = float(row.get("score") or 0.0)
        feedback = str(row.get("tool_feedback") or "")
        fmt_err = str(row.get("format_error") or "")
        extracted = str(row.get("pred") or "")
        if score >= 1.0:
            category = "verified"
        elif fmt_err and fmt_err != "No error.":
            category = FORMAT_NO_CODE
        elif feedback.startswith("# System Error"):
            category = INFRA
        elif feedback.startswith("# Error"):
            category = classify_first_attempt(
                verify_status="lean_error",
                lean_message=first_error_block(feedback),
                truncated=False,
                format_ok=True,
                has_lean_block=True,
                extracted=extracted,
            )
        elif feedback.startswith("filtered proof."):
            category = FORMAT_NO_CODE
        else:
            category = OTHER_SEMANTIC
        out.append({"category": category, "message": first_error_block(feedback), "score": score})
    return out


def classify_e023(records: list[dict]) -> list[dict]:
    out = []
    for r in records:
        extracted = str(r.get("extracted_proof") or r.get("proof") or "")
        category = classify_first_attempt(
            verify_status="verified" if r.get("verified") else str(r.get("verify_status") or "lean_error"),
            lean_message=str(r.get("lean_message") or ""),
            truncated=bool(r.get("truncated")),
            format_ok=bool(r.get("format_ok")),
            has_lean_block=bool(r.get("has_lean4_code_block")),
            extracted=extracted,
        )
        out.append({"category": category, "message": str(r.get("lean_message") or ""), "score": 1.0 if r.get("verified") else 0.0})
    return out


def classify_r001(rows: list[dict]) -> tuple[list[dict], int]:
    """V3-R001 stored the raw completion but not the extracted body, so the proof text is rebuilt
    with the project's own frozen extractor over the *recorded* text: deterministic re-extraction,
    no generation and no verifier call. The reconstruction is validated against the recorded
    ``extracted_proof_present`` flag; the mismatch count is reported, never silently accepted."""

    from tinylean_rl.inference.extract import _FENCED_LEAN, extract_proof

    out, mismatches = [], 0
    for r in rows:
        completion = str(r.get("completion_text") or "")
        try:
            extracted = extract_proof(completion)
        except ValueError:
            extracted = ""
        if bool(extracted) != bool(r.get("extracted_proof_present")):
            mismatches += 1
        out.append(
            {
                "category": classify_first_attempt(
                    verify_status=("verified" if r.get("verified") else str(r.get("verify_status") or "")),
                    lean_message=str(r.get("lean_message") or ""),
                    truncated=bool(r.get("truncated")),
                    format_ok=bool(extracted),
                    has_lean_block=bool(_FENCED_LEAN.search(completion)),
                    extracted=extracted,
                ),
                "message": str(r.get("lean_message") or ""),
                "score": float(r.get("score") or 0.0),
                "format_ok": bool(extracted),
                "truncated": bool(r.get("truncated")),
                "has_fence": bool(_FENCED_LEAN.search(completion)),
                "status": str(r.get("verify_status") or ""),
            }
        )
    return out, mismatches


def summarise(name: str, rows: list[dict], extra: dict | None = None) -> dict:
    counts = category_counts([r["category"] for r in rows])
    n = len(rows)
    primary = sum(counts[c] for c in PRIMARY_CATEGORIES)
    ambiguous = sum(
        1
        for r in rows
        if len(matched_primary_rules(verify_status="lean_error", lean_message=r["message"])) > 1
    )
    unmatched = collections.Counter(
        r["message"].strip().splitlines()[0][:110]
        for r in rows
        if r["category"] == OTHER_SEMANTIC and r["message"].strip()
    )
    block = {
        "candidates": n,
        "counts": counts,
        "primary_rate": round(primary / n, 4) if n else None,
        "syntax_rate": round(counts["syntax_parser"] / n, 4) if n else None,
        "format_rate": round(counts[FORMAT_NO_CODE] / n, 4) if n else None,
        "timeout_resource_rate": round(counts["timeout_resource"] / n, 4) if n else None,
        "infra_rate": round(counts[INFRA] / n, 4) if n else None,
        "verified_rate": round(counts["verified"] / n, 4) if n else None,
        "ambiguous_primary_fraction": round(ambiguous / n, 4) if n else None,
        "other_semantic_top_messages": dict(unmatched.most_common(10)),
    }
    if extra:
        block.update(extra)
    return {name: block}


def diagnostic_token_stats(corpus: list[str]) -> dict:
    """Token-budget feasibility of the diagnostic normalizer over observed diagnostics."""

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(ROOT / "models/weights/kimina_distill_0_6b"))
    lengths = [len(tok.encode(text)) for text in corpus if text]
    if not lengths:
        return {"n": 0}
    lengths.sort()

    def q(fraction: float) -> int:
        return lengths[min(len(lengths) - 1, int(fraction * len(lengths)))]

    return {
        "n": len(lengths),
        "min": lengths[0],
        "median": q(0.5),
        "p90": q(0.9),
        "p95": q(0.95),
        "max": lengths[-1],
        "budget": DIAGNOSTIC_TOKEN_BUDGET,
        "over_budget_fraction": round(sum(1 for v in lengths if v > DIAGNOSTIC_TOKEN_BUDGET) / len(lengths), 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=OUT)
    args = parser.parse_args()

    audit: dict = {
        "artifact_type": "v4_p001_historical_audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": subprocess.run(["hostname"], capture_output=True, text=True, check=False).stdout.strip(),
        "git_revision": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip(),
        "note": "read-only over already-consumed artifacts; no generation, no verifier, no GPU, no V4 formal data",
    }

    pool = json.loads((ROOT / POOL).read_text())
    pool_sources = collections.Counter(m["source"] for m in pool["members"])
    audit["development_pool"] = {
        "components": len(pool["members"]),
        "sources": dict(sorted(pool_sources.items())),
        "pool_hash": pool["pool_hash"],
        "order_hash": order_hash if (order_hash := pool.get("order_hash")) else None,
    }

    # --- corpus 1: V1 training dumps (Tier-1 source population) -------------------------------
    v1_rows, v1_step1 = [], []
    corpus_messages: list[str] = []
    for pattern in V1_DUMP_GLOBS:
        for path in sorted(glob.glob(str(ROOT / pattern))):
            for line in (ROOT / path).read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                v1_rows.append(row)
                if path.endswith("/1.jsonl"):
                    v1_step1.append(row)
    audit.update(summarise("v1_training_population", classify_v1_dump(v1_rows)))
    audit.update(
        summarise(
            "v1_step1_theta0_proxy",
            classify_v1_dump(v1_step1),
            {"note": "the step-1 dump of each seed is the on-policy theta0 rollout that started training; closest available proxy for theta0 on Tier-1 families"},
        )
    )
    for row in v1_rows:
        block = first_error_block(str(row.get("tool_feedback") or ""))
        if block:
            corpus_messages.append(block)

    # --- corpus 2: E023 sealed holdout, theta0 -------------------------------------------------
    e023 = json.loads((ROOT / E023_BASE).read_text())
    audit.update(summarise("e023_holdout_theta0", classify_e023(e023["records"])))
    for r in e023["records"]:
        msg = str(r.get("lean_message") or "")
        if msg:
            corpus_messages.append(msg)

    # --- corpus 3: V3-R001 attempt-2, theta0 --------------------------------------------------
    if (ROOT / R001_RAW).exists():
        rows = [json.loads(line) for line in (ROOT / R001_RAW).read_text().splitlines() if line.strip()]
        r001_rows, r001_mismatch = classify_r001(rows)
        audit.update(
            summarise(
                "v3_r001_attempt2_theta0",
                r001_rows,
                {
                    "note": "the extracted proof body is rebuilt from the recorded completion with the frozen extractor",
                    "re_extraction_mismatches_vs_recorded_extracted_proof_present": r001_mismatch,
                    "format_failures_with_fenced_block": sum(
                        1 for r in r001_rows if r["category"] == FORMAT_NO_CODE and r["has_fence"]
                    ),
                    "format_failures_without_fenced_block": sum(
                        1 for r in r001_rows if r["category"] == FORMAT_NO_CODE and not r["has_fence"]
                    ),
                },
            )
        )
        for r in rows:
            msg = str(r.get("lean_message") or "")
            if msg:
                corpus_messages.append(msg)
    else:
        audit["v3_r001_attempt2_theta0"] = {"error": "raw artifact not present on this host"}

    # --- corpus 4: E024 MiniF2F partial, theta0 (if present) ----------------------------------
    if (ROOT / E024_PARTIAL).exists():
        e024 = json.loads((ROOT / E024_PARTIAL).read_text())
        records = e024.get("records") if isinstance(e024, dict) else e024
        if isinstance(records, list):
            audit.update(
                summarise(
                    "e024_minif2f_theta0_partial",
                    classify_e023(records),
                    {"note": "PAUSED run; partial artifact; recorded for taxonomy coverage only, never as a result"},
                )
            )
    else:
        audit["e024_minif2f_theta0_partial"] = {"error": "partial artifact lives on fly122 only"}

    # --- diagnostic token budget --------------------------------------------------------------
    audit["diagnostic_tokens"] = diagnostic_token_stats(corpus_messages)

    # --- screening projection -----------------------------------------------------------------
    # Rates are kept per corpus: the Tier-1 pool is the V1 training population, for which the only
    # theta0 measurement is the step-1 dump; E023 is an independent theta0 measurement on holdout
    # families. Both are reported, and the frozen budget is compared against its own rate floor.
    rates = {
        "v1_step1_theta0_proxy__tier1_families": audit["v1_step1_theta0_proxy"].get("primary_rate"),
        "e023_holdout_theta0__independent": audit["e023_holdout_theta0"].get("primary_rate"),
        "v1_training_population__all_steps__not_theta0": audit["v1_training_population"].get("primary_rate"),
    }
    theta0_rates = [rates["v1_step1_theta0_proxy__tier1_families"], rates["e023_holdout_theta0__independent"]]
    observed = [r for r in theta0_rates if r]
    pool_tiers = pool.get("capacity", {}).get("pool_components_by_tier", {})
    budget = pool.get("screening_budget", {})
    floor = budget.get("primary_rate_floor")
    max_screens = budget.get("max_screens")
    capacity = len(pool["members"])
    projection = {
        "primary_rate_estimates": rates,
        "theta0_estimate_central": round(sum(observed) / len(observed), 4) if observed else None,
        "planning_rate_floor": floor,
        "pool_capacity": {"tier1": pool_tiers.get("tier1"), "tier2": pool_tiers.get("tier2"), "total": capacity},
        "screens_needed_for_128": {str(rate): int((128 / rate) + 0.9999) for rate in (0.20, 0.25, 0.30, 0.3125)},
        "frozen_max_screens": max_screens,
        "expected_primary_failures_at_frozen_budget": {str(rate): round(rate * max_screens, 1) for rate in (0.20, 0.25, 0.30, 0.3125)},
        "expected_primary_failures_in_tier1_alone": {str(rate): round(rate * (pool_tiers.get("tier1") or 0), 1) for rate in (0.20, 0.25, 0.30, 0.3125)},
        "tier2_needed": bool(observed and min(observed) * (pool_tiers.get("tier1") or 0) < 128),
        "capacity_is_sufficient": bool(max_screens and capacity >= max_screens and floor and (128 / floor) <= max_screens),
    }
    audit["screening_projection"] = projection

    destination = ROOT / args.out
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(audit, indent=2) + "\n")
    for name in ("v1_training_population", "v1_step1_theta0_proxy", "e023_holdout_theta0", "v3_r001_attempt2_theta0", "e024_minif2f_theta0_partial"):
        block = audit.get(name, {})
        if "counts" in block:
            print(f"{name:34s} n={block['candidates']:5d} primary={block['primary_rate']} syntax={block['syntax_rate']} format={block['format_rate']} verified={block['verified_rate']}")
    print("diagnostic tokens:", json.dumps(audit["diagnostic_tokens"]))
    print("projection:", json.dumps(projection, indent=1))
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
