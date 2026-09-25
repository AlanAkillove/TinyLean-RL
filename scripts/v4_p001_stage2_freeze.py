#!/usr/bin/env python3
"""V4-P001 owner §13 -- freeze the Stage-2 raw artifact before any scientific analysis.

Read-only over the second-stage raw artifact. It writes one compact record:

    experiments/manifests/v4/V4-P001_stage2_freeze.json

with the counts, the hashes, the timings, the per-arm infrastructure census and the structural
checks the owner listed for the boundary between "execution ended" and "analysis begins". It
computes no endpoint: no success rate per arm, no C-A/C-B/C-D, no test statistic. Those belong to
the single canonical analyzer run.
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
from v4_p001_verifier_plan import CEILING_ROUNDING, V4_MAX_ARMS_PER_THEOREM

from tinylean_rl.evaluation.v4_seeds import second_stage_seed

DEFAULT_RAW = "runs/v4_p001/rollout/v4_p001_second_stage_raw.jsonl"
DEFAULT_SUMMARY = "runs/v4_p001/rollout/v4_p001_run_summary.json"
DEFAULT_RECOVERY_LOG = "runs/v4_p001/rollout/v4_p001_recovery_log.jsonl"
DEFAULT_EVENTS = "runs/v4_p001/rollout/v4_p001_verifier_events.jsonl"
DEFAULT_COHORT = "experiments/manifests/v4/V4-P001_primary_cohort.json"
DEFAULT_DERANGEMENT = "experiments/manifests/v4/V4-P001_diagnostic_derangement.json"
DEFAULT_PLAN = "experiments/manifests/v4/V4-P001_second_stage_plan.json"
DEFAULT_GPU_SAMPLES = "runs/v4_p001/second_gpu_samples.csv"
DEFAULT_OUT = "experiments/manifests/v4/V4-P001_stage2_freeze.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default=DEFAULT_RAW)
    parser.add_argument("--summary", default=DEFAULT_SUMMARY)
    parser.add_argument("--recovery-log", default=DEFAULT_RECOVERY_LOG)
    parser.add_argument("--events", default=DEFAULT_EVENTS)
    parser.add_argument("--cohort", default=DEFAULT_COHORT)
    parser.add_argument("--derangement", default=DEFAULT_DERANGEMENT)
    parser.add_argument("--plan", default=DEFAULT_PLAN)
    parser.add_argument("--gpu-samples", default=DEFAULT_GPU_SAMPLES)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    raw_path = ROOT / args.raw
    summary_path = ROOT / args.summary
    recovery_path = ROOT / args.recovery_log
    events_path = ROOT / args.events
    cohort = load_json(ROOT / args.cohort)
    derangement = load_json(ROOT / args.derangement)
    plan = load_json(ROOT / args.plan)
    summary = load_json(summary_path)
    frozen = S.load_frozen(verify_files=True)

    rows = []
    for row in S.rows_jsonl(raw_path):
        S.validate_repair_row(row)
        rows.append(row)
    groups: dict[int, dict[str, dict]] = {}
    for row in rows:
        groups.setdefault(int(row["formal_rank"]), {})[str(row["arm"])] = row

    plan_by_rank = {int(t["formal_rank"]): t for t in plan["theorems"]}
    cohort_by_statement = {m["statement_id"]: m for m in cohort["members"]}
    cohort_statements = set(cohort_by_statement)
    beyond_statements = {row["statement_id"] for row in cohort["screened_after_the_cohort_closed"]}
    donor_by_statement = {entry["recipient"]: entry["donor"]
                          for entry in derangement["derangement"]["mapping"]}
    donor_diagnostic = {entry["recipient"]: entry["donor_diagnostic_sha256"]
                        for entry in derangement["derangement"]["mapping"]}
    own_diagnostic = {member["statement_id"]: member["diagnostic_sha256"]
                      for member in cohort["members"]}

    arm_counts = {arm: 0 for arm in S.ARM_ORDER}
    censored_by_arm = {arm: 0 for arm in S.ARM_ORDER}
    infra_status_by_arm = {arm: 0 for arm in S.ARM_ORDER}
    for row in rows:
        arm_counts[row["arm"]] += 1
        if row["censored"]:
            censored_by_arm[row["arm"]] += 1
        elif row["verify_status"] in S.INFRA_STATUSES:
            infra_status_by_arm[row["arm"]] += 1

    complete_ranks = []
    for rank in range(1, S.N_PRIMARY + 1):
        plan_row = plan_by_rank[rank]
        by_arm = groups.get(rank, {})
        ordered = [by_arm.get(arm) for arm in plan_row["arm_order"]]
        if len(by_arm) == S.N_ARMS and all(ordered):
            complete_ranks.append(rank)

    def expected_prompt(rank: int, arm: str) -> dict:
        return plan_by_rank[rank]["arms"][arm]

    checks = {
        "raw_rows_conform_to_the_frozen_repair_schema":
            bool(rows) and all(row["schema_version"] == S.REPAIR_SCHEMA_VERSION for row in rows),
        "every_row_is_a_cohort_rank_with_the_frozen_statement":
            all(int(row["formal_rank"]) in plan_by_rank
                and row["statement_id"] == plan_by_rank[int(row["formal_rank"])]["statement_id"]
                and row["statement_id"] in cohort_statements
                and int(row["screening_rank"]) == plan_by_rank[int(row["formal_rank"])]["screening_rank"]
                for row in rows),
        "no_stage1_extra_or_out_of_cohort_theorem_present":
            not (beyond_statements & {row["statement_id"] for row in rows})
            and {int(row["formal_rank"]) for row in rows} <= set(range(1, S.N_PRIMARY + 1)),
        "arm_order_and_positions_match_the_frozen_schedule":
            all(sorted(by_arm, key=lambda arm: by_arm[arm]["position"])
                == list(plan_by_rank[rank]["arm_order"])
                and sorted(by_arm[arm]["position"] for arm in by_arm) == [1, 2, 3, 4]
                for rank, by_arm in groups.items()),
        "prompt_and_context_hashes_match_the_frozen_plan":
            all(row["prompt_sha256"] == expected_prompt(int(row["formal_rank"]), row["arm"])["prompt_sha256"]
                and row["prompt_token_count"]
                == expected_prompt(int(row["formal_rank"]), row["arm"])["prompt_tokens"]
                and row["context_tokens_with_response"]
                == expected_prompt(int(row["formal_rank"]), row["arm"])["context_tokens_with_response"]
                for row in rows),
        "seeds_match_the_paired_seed_plan":
            all(row["seed"] == plan_by_rank[int(row["formal_rank"])]["seed"]
                == second_stage_seed(int(row["formal_rank"])) for row in rows),
        "model_hash_and_revision_match_the_frozen_theta0":
            all(row["model_sha256"] == S.MODEL["weights_sha256"]
                and row["model_revision"] == S.MODEL["revision"] for row in rows),
        "diagnostic_donor_mapping_is_exact":
            all(
                (row["diagnostic_sha256"] == own_diagnostic[row["statement_id"]]
                 if row["arm"] == "C_VERIFIER_REPAIR" else
                 row["diagnostic_sha256"] == donor_diagnostic[row["statement_id"]]
                 if row["arm"] == "D_MISMATCHED_DIAGNOSTIC" else
                 row["diagnostic_sha256"] == "")
                for row in rows)
            and all(donor_by_statement[s] != s for s in donor_by_statement),
        "renderer_and_normalization_versions_are_frozen":
            all(row["renderer_version"] == plan["renderer_version"]
                and row["normalization_version"] == plan["normalization_version"] for row in rows),
        "quadruplets_are_complete_or_explicitly_infra_missing":
            len(complete_ranks) == S.N_PRIMARY
            and all(row["censored"] is True and row["verify_status"] in S.INFRA_STATUSES
                    and row["score"] is None and not row["completion_text"]
                    for row in rows if row["censored"]),
        "arm_candidate_counts_match_the_plan":
            all(arm_counts[arm] == S.N_PRIMARY for arm in S.ARM_ORDER)
            and len(rows) <= S.SECOND_STAGE_CANDIDATES,
        "recovery_budget_respected":
            all(entry["theorem_recovery_index"] <= V4_MAX_ARMS_PER_THEOREM
                for entry in (json.loads(line) for line in
                              recovery_path.read_text(encoding="utf-8").splitlines() if line.strip()))
            and int(summary.get("verifier_infrastructure", {}).get("recoveries_attempted", 0))
            <= CEILING_ROUNDING,
        "sealed_reserve_and_v3_sample_untouched":
            all(value == 0 for key, value in frozen.pool["checks"].items()
                if key.endswith("_touched")),
    }
    failing = sorted(name for name, ok in checks.items() if not ok)

    created = [row["created_at"] for row in rows]
    start, end = (min(created), max(created)) if created else (None, None)
    recovery_entries = [json.loads(line) for line in
                        recovery_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    stage2_recoveries = [entry for entry in recovery_entries
                         if start is not None and entry["at"] >= start]
    peak_vram = None
    samples_path = ROOT / args.gpu_samples
    if samples_path.exists():
        values = []
        for line in samples_path.read_text(encoding="utf-8").splitlines()[1:]:
            parts = line.split(",")
            if len(parts) >= 3 and parts[2].strip().isdigit():
                values.append(int(parts[2]))
        peak_vram = max(values) if values else None
    verification_seconds = round(sum(float(row["verification_time"] or 0.0) for row in rows
                                     if not row["censored"]), 1)

    record = {
        "artifact_type": "V4-P001_stage2_freeze",
        "experiment_id": S.EXPERIMENT_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "validated_commit": S.collect_env()["git_revision"],
        "raw_artifact": {
            "path": args.raw,
            "sha256": S.sha256_file(raw_path) if raw_path.exists() else None,
            "bytes": raw_path.stat().st_size if raw_path.exists() else None,
            "rows": len(rows),
            "run_id": summary.get("run_id"),
            "created_at_min": start,
            "created_at_max": end,
            "resume_compaction": summary.get("resume_compaction"),
        },
        "candidates": {
            "expected": S.SECOND_STAGE_CANDIDATES,
            "observed_rows": len(rows),
            "generated_this_launch": summary.get("candidates_generated"),
            "censored_rows": summary.get("candidates_censored"),
            "arm_counts": arm_counts,
            "infrastructure_missing_by_arm": {
                arm: {"explicitly_censored": censored_by_arm[arm],
                      "generated_but_infra_status": infra_status_by_arm[arm],
                      "total_missing": censored_by_arm[arm] + infra_status_by_arm[arm]}
                for arm in S.ARM_ORDER
            },
        },
        "theorems": {
            "planned": S.N_PRIMARY,
            "structurally_complete": len(complete_ranks),
            "ranks_on_disk": sorted(groups),
            "ranks_not_on_disk": [rank for rank in range(1, S.N_PRIMARY + 1) if rank not in groups],
        },
        "recovery": {
            "stage2_attempts": len(stage2_recoveries),
            "stage2_failures": sum(1 for entry in stage2_recoveries if not entry.get("ok")),
            "stage2_ranks": sorted({entry["theorem_rank"] for entry in stage2_recoveries}),
            "journal_total_entries": len(recovery_entries),
            "journal_sha256": S.sha256_file(recovery_path) if recovery_path.exists() else None,
            "verifier_events_sha256": S.sha256_file(events_path) if events_path.exists() else None,
        },
        "timings": {
            "generation_seconds": summary.get("generation_seconds"),
            "verification_seconds_summed_from_rows": verification_seconds,
            "peak_vram_mib_gpu_sampler": peak_vram,
            "peak_vram_samples_file": args.gpu_samples,
            "torch_peak_allocated_gb_reported": summary.get("torch_peak_allocated_gb"),
            "torch_counter_note": ("the runner resets the torch peak counter after the engine is "
                                   "built, so it reads ~0 when vLLM allocates its pools at load "
                                   "time; the nvidia-smi sampler is the peak of record"),
        },
        "hashes": {
            "plan_sha256": plan["plan_sha256"],
            "arm_schedule_hash": plan["arm_schedule_hash"],
            "second_stage_seed_hash": plan["second_stage_seed_hash"],
            "derangement_mapping_sha256": plan["derangement_mapping_sha256"],
            "cohort_content_sha256": cohort["content_sha256"],
            "summary_sha256": S.sha256_file(summary_path) if summary_path.exists() else None,
            "frozen_settings_sha256": summary.get("frozen_settings_sha256"),
            "boundary_validation": summary.get("boundary_validation"),
            "script_versions": summary.get("script_versions"),
            "verifier_infrastructure": summary.get("verifier_infrastructure"),
        },
        "structural_checks": checks,
        "n_checks": len(checks),
        "n_failed": len(failing),
        "status": "FROZEN" if not failing else "FAILED",
        "failing_checks": failing,
        "note": ("Stage-2 raw freeze, owner §13. No endpoint was computed here: success rates per "
                 "arm, C-A/C-B/C-D and every test statistic belong to the single canonical analyzer "
                 "run that follows the archive step."),
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    S.write_json_atomic(out, record)
    print(json.dumps({"out": str(out), "status": record["status"], "n_checks": record["n_checks"],
                      "n_failed": record["n_failed"], "rows": len(rows),
                      "theorems_complete": len(complete_ranks), "arm_counts": arm_counts,
                      "missing_by_arm": record["candidates"]["infrastructure_missing_by_arm"],
                      "stage2_recoveries": len(stage2_recoveries),
                      "start": start, "end": end,
                      "raw_sha256": record["raw_artifact"]["sha256"]}, indent=2))
    return 0 if not failing else 1


if __name__ == "__main__":
    raise SystemExit(main())
