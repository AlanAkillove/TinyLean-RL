#!/usr/bin/env python3
"""V4-P001 owner §10 -- freeze the Stage-1 screening data before any D mapping or plan.

Read-only over the screening raw artifact. It writes one compact record:

    experiments/manifests/v4/V4-P001_stage1_freeze.json

with the counts, the hashes, the timestamps and the structural checks the owner asked for, plus
the §17 descriptive distributions of the frozen cohort (source, error category, prompt / failed
proof / diagnostic token counts). No generation, no verifier call, no artifact of the run is
modified. The raw generations stay outside git; this record is the commit-sized summary that pins
them by sha256.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "scripts", ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import v4_p001_spec as S

from tinylean_rl.evaluation.v4_taxonomy import PRIMARY_CATEGORIES, classify_first_attempt

DEFAULT_RAW = "runs/v4_p001/rollout/v4_p001_screening_raw.jsonl"
DEFAULT_SUMMARY = "runs/v4_p001/rollout/v4_p001_screening_summary.json"
DEFAULT_RECOVERY_LOG = "runs/v4_p001/rollout/v4_p001_recovery_log.jsonl"
DEFAULT_EVENTS = "runs/v4_p001/rollout/v4_p001_verifier_events.jsonl"
DEFAULT_OUT = "experiments/manifests/v4/V4-P001_stage1_freeze.json"


def distribution(values: list[int]) -> dict:
    ordered = sorted(values)

    def q(fraction: float) -> int:
        return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]

    return {"n": len(ordered), "min": ordered[0], "p50": q(0.5), "p95": q(0.95), "p99": q(0.99),
            "max": ordered[-1], "mean": round(sum(ordered) / len(ordered), 1)}


def load_rows(raw_path: Path) -> list[dict]:
    rows = []
    for row in S.rows_jsonl(raw_path):
        S.validate_screening_row(row)
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default=DEFAULT_RAW)
    parser.add_argument("--summary", default=DEFAULT_SUMMARY)
    parser.add_argument("--recovery-log", default=DEFAULT_RECOVERY_LOG)
    parser.add_argument("--events", default=DEFAULT_EVENTS)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--peak-vram-mib", type=int, default=0,
                        help="peak VRAM observed by the external sampler during the run, MiB")
    args = parser.parse_args()

    raw_path = ROOT / args.raw
    summary_path = ROOT / args.summary
    rows = load_rows(raw_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    frozen = S.load_frozen(verify_files=True)

    import v4_p001_rollout as R
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(ROOT / S.MODEL["dir"], trust_remote_code=True,
                                             local_files_only=True)
    surface = R.load_surface(tokenizer)
    plan = R.build_screening_plan(frozen, surface)
    plan_by_rank = {entry["screening_rank"]: entry for entry in plan}

    status_counts = {name: 0 for name in S.SCREENING_STATUSES}
    for row in rows:
        status_counts[row["screening_status"]] += 1
    cohort = [row for row in rows if row["formal_rank"] is not None]
    beyond = [row for row in rows if row["formal_rank"] is None]
    primaries = [row for row in rows if row["screening_status"] == S.PRIMARY_SEMANTIC_FAILURE]
    primaries_beyond = [row for row in beyond
                        if row["screening_status"] == S.PRIMARY_SEMANTIC_FAILURE]
    primary_128 = max(row["screening_rank"] for row in cohort)

    checks = {
        "no_duplicate_screening_rank":
            len({row["screening_rank"] for row in rows}) == len(rows),
        "screening_ranks_are_contiguous_from_one":
            [row["screening_rank"] for row in rows] == list(range(1, len(rows) + 1)),
        "every_row_is_within_the_frozen_plan":
            len(rows) <= S.MAX_SCREENING and all(row["screening_rank"] in plan_by_rank for row in rows),
        "seeds_are_first_stage_seed_of_the_rank":
            all(row["seed"] == S.first_stage_seed(row["screening_rank"])
                == plan_by_rank[row["screening_rank"]]["seed"] for row in rows),
        "seeds_are_unique":
            len({row["seed"] for row in rows}) == len(rows),
        "model_hash_and_revision_match_the_frozen_theta0":
            all(row["model_sha256"] == S.MODEL["weights_sha256"]
                and row["model_revision"] == S.MODEL["revision"] for row in rows),
        "prompt_hashes_and_token_counts_match_the_pinned_parquet":
            all(row["prompt_sha256"] == surface[row["statement_id"]]["prompt_sha256"]
                and row["prompt_token_count"] == surface[row["statement_id"]]["prompt_token_count"]
                == plan_by_rank[row["screening_rank"]]["prompt_tokens"] for row in rows),
        "cohort_is_the_first_128_primary_failures_in_frozen_order":
            [row["formal_rank"] for row in cohort] == list(range(1, S.N_PRIMARY + 1))
            and all(row["screening_status"] == S.PRIMARY_SEMANTIC_FAILURE for row in cohort)
            and [row["screening_rank"] for row in primaries][:S.N_PRIMARY]
            == sorted(row["screening_rank"] for row in cohort)
            and len(primaries) == len(cohort) + len(primaries_beyond),
        "no_primary_was_reclassified_by_hand":
            all(classify_first_attempt(
                verify_status=row["verify_status"], lean_message=row["lean_message"],
                truncated=bool(row["truncated"]), format_ok=bool(row["format_ok"]),
                has_lean_block=bool(row["has_lean_block"]),
                extracted=row["extracted_proof"]) == row["error_category"] for row in rows),
        "primary_eligibility_is_the_frozen_predicate":
            all(bool(row["primary_eligible"]) == (
                row["error_category"] in PRIMARY_CATEGORIES
                and bool(str(row["extracted_proof"]).strip())
                and bool(str(row["diagnostic_text"]).strip())
                and bool(row["context_fits"])) for row in rows
                if row["screening_status"] == S.PRIMARY_SEMANTIC_FAILURE),
        "cohort_carries_attributable_proofs_and_diagnostics":
            all(str(row["extracted_proof"]).strip() and str(row["diagnostic_text"]).strip()
                for row in cohort),
        "sealed_reserve_and_other_exclusions_untouched":
            all(value == 0 for key, value in frozen.pool["checks"].items()
                if key.endswith("_touched")),
        "candidate_and_verifier_budget_respected":
            len(rows) <= S.MAX_SCREENING
            and int(summary["candidates_generated"]) == len(rows)
            and summary["cohort_complete"] is True,
    }
    failing = sorted(name for name, ok in checks.items() if not ok)

    sources = {}
    categories = {}
    for row in cohort:
        sources[str(row["source"])] = sources.get(str(row["source"]), 0) + 1
        categories[str(row["error_category"])] = categories.get(str(row["error_category"]), 0) + 1

    recovery_log = ROOT / args.recovery_log
    events = ROOT / args.events
    infra = summary.get("verifier_infrastructure") or {}
    record = {
        "artifact_type": "V4-P001_stage1_freeze",
        "experiment_id": S.EXPERIMENT_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "validated_commit": S.collect_env()["git_revision"],
        "raw_artifact": {
            "path": args.raw,
            "sha256": S.sha256_file(raw_path),
            "bytes": raw_path.stat().st_size,
            "rows": len(rows),
            "run_id": summary.get("run_id"),
            "created_at_min": min(row["created_at"] for row in rows),
            "created_at_max": max(row["created_at"] for row in rows),
        },
        "screening": {
            "screens_used": len(rows),
            "screens_remaining_in_budget": S.MAX_SCREENING - len(rows),
            "status_counts": status_counts,
            "cohort_size": len(cohort),
            "primary_128_reached_at_screening_rank": primary_128,
            "primaries_after_the_cohort_closed": [
                {"screening_rank": row["screening_rank"], "statement_id": row["statement_id"],
                 "screening_status": row["screening_status"],
                 "note": "finalized in the same generation chunk as primary #128; not part of the "
                         "cohort and never a backup pool"}
                for row in primaries_beyond],
            "generation_seconds": summary.get("generation_seconds"),
        },
        "primary_cohort": {
            "N": len(cohort),
            "sources": sources,
            "error_categories": categories,
            "failed_proof_tokens": distribution([row["extracted_proof_tokens"] for row in cohort]),
            "diagnostic_tokens": distribution([row["diagnostic_tokens"] for row in cohort]),
            "prompt_tokens": distribution([row["prompt_token_count"] for row in cohort]),
        },
        "hashes": {
            "screening_order_hash": frozen.pool["order_hash"],
            "pool_hash": frozen.pool["pool_hash"],
            "screening_seed_hash": frozen.seeds["screening_seed_hash"],
            "theta0_weights_sha256": S.sha256_file(
                ROOT / S.MODEL["dir"] / S.MODEL["weights_filename"]),
            "script_versions": summary.get("script_versions"),
            "summary_sha256": S.sha256_file(summary_path),
            "recovery_log_sha256": S.sha256_file(recovery_log) if recovery_log.exists() else None,
            "verifier_events_sha256": S.sha256_file(events) if events.exists() else None,
            "frozen_settings_sha256": summary.get("frozen_settings_sha256"),
            "sealed_reserve_touch_counts": {key: value for key, value in
                                            frozen.pool["checks"].items() if key.endswith("_touched")},
        },
        "infrastructure": {
            "recoveries_attempted": infra.get("recoveries_attempted"),
            "recoveries_succeeded": infra.get("recoveries_succeeded"),
            "recoveries_allowed": infra.get("recoveries_allowed"),
            "recoveries_by_theorem": infra.get("recoveries_by_theorem"),
            "dedicated_instance": infra.get("dedicated_instance"),
            "verifier_health": summary.get("verifier_health"),
            "vllm_version": summary.get("vllm_version"),
            "peak_vram_mib_nvidia_smi_observed": args.peak_vram_mib,
            "torch_peak_allocated_gb_reported": summary.get("torch_peak_allocated_gb"),
            "torch_counter_note": ("the runner resets the torch peak counter after the engine is "
                                   "built, so it reads ~0 when vLLM allocates its pools at load "
                                   "time; the nvidia-smi observation is the peak of record"),
        },
        "structural_checks": checks,
        "n_checks": len(checks),
        "n_failed": len(failing),
        "status": "FROZEN" if not failing else "FAILED",
        "failing_checks": failing,
        "note": ("Stage-1 raw freeze, owner §10. The raw generations stay in runs/ (gitignored) and "
                 "are pinned by sha256 here. No D mapping and no second-stage prompt generation has "
                 "run at the time this record was written."),
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    S.write_json_atomic(out, record)
    print(json.dumps({"out": str(out), "status": record["status"], "n_checks": record["n_checks"],
                      "n_failed": record["n_failed"], "screens": len(rows),
                      "primaries": status_counts[S.PRIMARY_SEMANTIC_FAILURE],
                      "primary_128_at_rank": primary_128,
                      "raw_sha256": record["raw_artifact"]["sha256"]}, indent=2))
    return 0 if not failing else 1


if __name__ == "__main__":
    raise SystemExit(main())
