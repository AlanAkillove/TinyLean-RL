#!/usr/bin/env python3
"""V4-P001 analyzer -- score the frozen four-arm design against the frozen second-stage raw artifact.

Owner §13 (Amendment A §14) fixes the order of this file, and the order is load-bearing:

    provenance -> complete quadruplets -> N guard -> differential-censoring guard
      -> C-A -> C-B -> C-D specificity -> classification -> secondary mechanisms

The analyzer reads the frozen design (`v4_p001_spec.load_frozen`), the three Stage-1/Stage-2
boundary artifacts, the boundary validation report, the screening raw artifact and the second-stage
raw artifact. It never generates, never verifies, never re-selects the cohort, never lowers N and
never re-derives a threshold: every number comes from `tinylean_rl.evaluation.v4_stats` -- the same
functions the power artifact was computed with -- and every classification comes from
`v4_stats.decide_outcome`, whose guard order this file merely *calls*.

What is deliberately absent: **no AUPRC and no predictive metric of any kind** (owner §L). This is a
paired repair probe, not a ranking study. There is also no efficiency claim: verifier recovery
counts are reported as data-completeness facts, never as a throughput result.

Infrastructure-censored arms are *missing*, never failures. The infra-as-failure reading of
Amendment A §5 is computed and reported separately (`sensitivity_infra_as_failure`) and never
redefines the primary endpoint or the classification.

Exit codes: 0 when the analysis was written (or check-only passed), 3 when a fail-closed provenance
check failed and nothing was written.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "scripts", ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import v4_p001_spec as S

from tinylean_rl.evaluation.v4_diagnostics import NORMALIZATION_VERSION
from tinylean_rl.evaluation.v4_prompts import RENDERER_VERSION
from tinylean_rl.evaluation.v4_seeds import second_stage_seed
from tinylean_rl.evaluation.v4_stats import (
    ALPHA,
    BOOTSTRAP_LEVEL,
    BOOTSTRAP_REPS,
    BOOTSTRAP_SEED,
    CENSORING_RANGE_MAX,
    DATA_GUARD_MIN,
    DELTA_CA_MIN,
    DELTA_CB_MIN,
    Gate,
    Mechanism,
    Outcome,
    censoring_range,
    decide_outcome,
    mechanism_label,
    paired_gate,
    paired_gate_infra_as_failure,
)
from tinylean_rl.evaluation.v4_taxonomy import (
    PRIMARY_CATEGORIES,
    TRANSITION_CLASSES,
    transition_class,
)

RESULTS_DEFAULT = "experiments/manifests/v4/V4-P001_results.json"
OUT_DIR_DEFAULT = "runs/v4_p001/rollout"
SCRIPT_VERSION_FILES = ("scripts/v4_p001_analyze.py", "scripts/v4_p001_spec.py",
                        "scripts/v4_p001_rollout.py")

#: Owner §15: the error-transition table is reported separately for these three arms.
TRANSITION_ARMS = ("B_SELF_REVISION", "C_VERIFIER_REPAIR", "D_MISMATCHED_DIAGNOSTIC")
NOT_EVALUATED_OUTCOMES = (Outcome.D_INCONCLUSIVE_BY_DATA,
                          Outcome.D_INCONCLUSIVE_BY_DIFFERENTIAL_CENSORING)
REQUIRED_ARTIFACTS = ("second_stage_raw", "summary", "screening_raw", "cohort", "derangement",
                      "plan", "boundary_validation")

#: How the boundary validation report names the three artifacts it hashed.
BOUNDARY_ARTIFACT_NAMES = {"cohort": "primary_cohort", "derangement": "diagnostic_derangement",
                           "plan": "second_stage_plan"}

#: Used only when there is not a single complete quadruplet: `decide_outcome` must still be the
#: function that decides, and the N guard fires on `n_complete` before it ever looks at a gate.
_ZERO_GATE = Gate(delta=0.0, mcnemar_p=1.0, ci_lower=0.0, n_favor=0, n_against=0)


# --- loading -------------------------------------------------------------------------------------

def artifact_paths(out_dir: Path) -> dict[str, Path]:
    return {
        "second_stage_raw": out_dir / S.SECOND_STAGE_RAW_BASENAME,
        "summary": out_dir / S.SUMMARY_BASENAME,
        "screening_raw": out_dir / S.SCREENING_RAW_BASENAME,
        "cohort": out_dir / S.PRIMARY_COHORT_BASENAME,
        "derangement": out_dir / S.DERANGEMENT_BASENAME,
        "plan": out_dir / S.SECOND_STAGE_PLAN_BASENAME,
        "boundary_validation": out_dir / S.BOUNDARY_VALIDATION_BASENAME,
        "recovery_log": out_dir / S.RECOVERY_LOG_BASENAME,
        "verifier_events": out_dir / S.VERIFIER_EVENTS_BASENAME,
    }


def load_inputs(out_dir: Path) -> tuple[dict, dict]:
    """Read what the analyzer is allowed to read. A missing required artifact is fatal."""
    paths = artifact_paths(out_dir)
    missing = [f"{name} ({paths[name]})" for name in REQUIRED_ARTIFACTS if not paths[name].exists()]
    if missing:
        raise S.FrozenViolation(
            "cannot analyze: required artifact(s) absent -- " + ", ".join(missing)
            + ". The formal second stage writes the second-stage raw artifact and the run summary; "
              "an aborted or never-launched run has no analyzable outcome.")
    payloads = {name: json.loads(paths[name].read_text(encoding="utf-8"))
                for name in ("summary", "cohort", "derangement", "plan", "boundary_validation")}
    return paths, payloads


def read_repair_groups(path: Path) -> dict:
    return S.read_grouped(path, S.validate_repair_row)


# --- 1. provenance -------------------------------------------------------------------------------

def _diagnostic_source_statement(plan_row: dict, arm: str) -> str:
    """Whose diagnostic an arm's prompt carries: the theorem's own, except for Arm D's donor."""
    return (plan_row["donor_statement_id"] if arm == "D_MISMATCHED_DIAGNOSTIC"
            else plan_row["statement_id"])


def check_provenance(*, frozen: S.Frozen, paths: dict, payloads: dict, groups: dict,
                     screening_rows: list[dict]) -> tuple[list[dict], list[str]]:
    """Prove the raw artifact is the rollout this frozen design predicted, before any statistic.

    Every check is a recomputation against the committed artifacts, never a comparison of two
    recorded flags. Fail-closed: a single failing check aborts the analysis.
    """
    checks: list[dict] = []
    problems: list[str] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "pass": bool(ok), "detail": detail})
        if not ok:
            problems.append(f"{name}: {detail}")

    summary, cohort, derangement, plan, boundary = (
        payloads["summary"], payloads["cohort"], payloads["derangement"], payloads["plan"],
        payloads["boundary_validation"])
    rows = [row for group in groups.values() for row in group]

    check("boundary_validation_passed",
          boundary.get("status") == "PASS" and boundary.get("n_failed") == 0,
          f"the Stage-1/Stage-2 boundary report is {boundary.get('status')!r} "
          f"({boundary.get('n_failed')} failed check(s)); the second stage must not have run")
    check("boundary_recorded_zero_candidates_generated",
          all(payloads[name].get("second_stage_candidates_generated") == 0
              for name in ("cohort", "derangement", "plan")),
          "a boundary artifact records generated candidates, so the boundary was not a boundary")
    check("boundary_report_hashes_the_artifacts",
          all(boundary.get("artifacts", {}).get(BOUNDARY_ARTIFACT_NAMES[name], {}).get("sha256")
              == S.sha256_file(paths[name])
              and boundary.get("artifacts", {}).get(BOUNDARY_ARTIFACT_NAMES[name], {}).get(
                  "content_sha256") == payloads[name].get("content_sha256")
              for name in ("cohort", "derangement", "plan")),
          "the boundary report does not hash the artifact files on disk")
    check("screening_raw_matches_the_boundary",
          boundary.get("screening_raw", {}).get("sha256") == S.sha256_file(paths["screening_raw"])
          and cohort.get("screening_raw_sha256") == S.sha256_file(paths["screening_raw"]),
          "the screening raw artifact on disk is not the one the boundary was frozen from")
    check("boundary_hash_chain",
          cohort.get("content_sha256")
          and plan.get("cohort_content_sha256") == cohort["content_sha256"]
          and derangement.get("cohort_content_sha256") == cohort["content_sha256"]
          and plan.get("derangement_mapping_sha256") == derangement.get("mapping_sha256")
          and derangement.get("mapping_sha256") == derangement.get("derangement", {}).get(
              "mapping_sha256"),
          "the three boundary artifacts do not reference each other's content hashes")
    check("cohort_is_the_frozen_nominal",
          cohort.get("cohort_size") == S.N_PRIMARY
          and len(cohort.get("members", [])) == S.N_PRIMARY
          and [m.get("formal_rank") for m in cohort.get("members", [])]
          == list(range(1, S.N_PRIMARY + 1)),
          f"cohort_size={cohort.get('cohort_size')!r}; the nominal cohort is {S.N_PRIMARY}")
    check("summary_is_this_run",
          summary.get("experiment_id") == S.EXPERIMENT_ID
          and summary.get("schema_version") == S.REPAIR_SCHEMA_VERSION
          and summary.get("prereg_commit") == S.PREREG_COMMIT
          and summary.get("frozen_settings_sha256") == S.sha(S.FROZEN_SETTINGS)
          and summary.get("plan_sha256") == plan.get("plan_sha256")
          and summary.get("arm_prompt_plan_sha256") == plan.get("arm_prompt_plan_sha256")
          and summary.get("arm_schedule_hash") == plan.get("arm_schedule_hash")
          and summary.get("second_stage_seed_hash") == plan.get("second_stage_seed_hash")
          and summary.get("derangement_mapping_sha256") == derangement.get("mapping_sha256")
          and summary.get("boundary_validation", {}).get("status") == "PASS"
          and summary.get("n_theorems") == S.N_PRIMARY
          and summary.get("n_candidates") == S.SECOND_STAGE_CANDIDATES,
          "the run summary's frozen-settings / plan / schedule / seed / derangement hashes are not "
          "the committed ones, so the raw artifact cannot be attributed to this design")
    check("frozen_design_checks_all_pass",
          all(entry["pass"] for entry in frozen.checks) and len(frozen.checks) >= 40,
          f"{len(frozen.checks)} frozen-design checks, "
          f"{sum(1 for e in frozen.checks if not e['pass'])} failed")
    pool_checks = frozen.pool["checks"]
    touched = {key: value for key, value in pool_checks.items() if key.endswith("_touched")}
    off_pool = sorted({row["statement_id"] for row in rows} - set(frozen.member_by_statement))
    check("sealed_reserve_untouched",
          all(value == 0 for value in touched.values()) and not off_pool,
          f"pool touch counts {touched}; {len(off_pool)} candidate row(s) outside the frozen pool")

    check("rows_carry_the_frozen_settings",
          all(row["experiment_id"] == S.EXPERIMENT_ID
              and row["model_sha256"] == S.MODEL["weights_sha256"]
              and row["renderer_version"] == RENDERER_VERSION
              and row["normalization_version"] == NORMALIZATION_VERSION
              for row in rows),
          "a candidate row was produced by another model, experiment or renderer/normalizer")

    cohort_by_rank = {m["formal_rank"]: m for m in cohort["members"]}
    plan_by_rank = {t["formal_rank"]: t for t in plan["theorems"]}
    screening_by_statement = {row["statement_id"]: row for row in screening_rows}
    missing_screening = sorted({m["statement_id"] for m in cohort["members"]}
                              - set(screening_by_statement))
    check("screening_artifact_covers_the_cohort",
          not missing_screening,
          f"{len(missing_screening)} cohort theorem(s) have no screening row, so their first-attempt "
          f"diagnostic cannot be re-read (first: {missing_screening[:3]})")
    bad_rank = sorted({row["formal_rank"] for row in rows} - set(range(1, S.N_PRIMARY + 1)))
    check("every_candidate_is_a_cohort_rank",
          not bad_rank and len(groups) == S.N_PRIMARY,
          f"unknown formal_rank(s) {bad_rank[:5]}; {len(groups)} rank group(s) in the artifact")
    off_cohort = sorted({(row["formal_rank"], row["statement_id"]) for row in rows
                         if row["statement_id"] != cohort_by_rank.get(row["formal_rank"], {}).get(
                             "statement_id")
                         or row["error_category"] != cohort_by_rank.get(
                             row["formal_rank"], {}).get("error_category")})
    check("every_candidate_is_the_frozen_theorem",
          not off_cohort,
          f"{len(off_cohort)} candidate row(s) do not match the frozen cohort's statement_id / "
          f"error_category at their rank (first: {off_cohort[:3]})")
    bad_seed = sorted({row["formal_rank"] for row in rows
                       if row["seed"] != second_stage_seed(row["formal_rank"])
                       or row["seed"] != plan_by_rank[row["formal_rank"]]["seed"]})
    check("seeds_are_the_frozen_stream",
          not bad_seed,
          f"{len(bad_seed)} formal rank(s) do not carry second_stage_seed(rank) (first: "
          f"{bad_seed[:5]})")
    check("screening_seeds_were_not_reused",
          not ({row["seed"] for row in rows}
               & {row["seed"] for row in screening_rows}),
          "a second-stage candidate carries a first-stage screening seed")
    bad_position = sorted({(row["formal_rank"], row["arm"]) for row in rows
                           if row["position"]
                           != list(plan_by_rank[row["formal_rank"]]["arm_order"]).index(row["arm"])})
    check("execution_order_is_the_frozen_balanced_schedule",
          not bad_position,
          f"{len(bad_position)} candidate row(s) are not at their frozen schedule position "
          f"(first: {bad_position[:3]})")
    bad_prompt = sorted({(row["formal_rank"], row["arm"]) for row in rows
                         if row["prompt_sha256"]
                         != plan_by_rank[row["formal_rank"]]["arms"][row["arm"]]["prompt_sha256"]
                         or row["diagnostic_sha256"]
                         != plan_by_rank[row["formal_rank"]]["arms"][row["arm"]]["diagnostic_sha256"]
                         or row["diagnostic_source"]
                         != plan_by_rank[row["formal_rank"]]["arms"][row["arm"]]["diagnostic_source"]
                         or row["failed_proof_sha256"]
                         != plan_by_rank[row["formal_rank"]]["failed_proof_sha256"]})
    check("prompts_are_the_frozen_prompts",
          not bad_prompt,
          f"{len(bad_prompt)} candidate row(s) do not reproduce the frozen prompt / diagnostic / "
          f"failed-proof hash (first: {bad_prompt[:3]})")
    diagnostic_mismatch = sorted({(row["formal_rank"], row["arm"]) for row in rows
                                  if row["diagnostic_sha256"]
                                  and row["diagnostic_sha256"] != S.sha256_text(str(
                                      screening_by_statement[
                                          _diagnostic_source_statement(plan_by_rank[row["formal_rank"]],
                                                                       row["arm"])
                                      ]["diagnostic_text"]))})
    check("arm_d_carries_the_derangement_donor",
          not diagnostic_mismatch,
          f"{len(diagnostic_mismatch)} row(s) do not carry the diagnostic of their assigned theorem "
          f"(first: {diagnostic_mismatch[:3]})")
    return checks, problems


# --- 2. complete quadruplets ---------------------------------------------------------------------

def build_quadruplets(*, groups: dict, cohort: dict, screening_by_statement: dict) -> list[dict]:
    """One record per frozen formal rank: the four arm outcomes, True / False / None.

    ``None`` is an arm the infrastructure censored (never attempted, or attempted and unresolved).
    It is missing data: it makes the quadruplet incomplete, and it is never read as a failure.
    """
    cohort_by_rank = {m["formal_rank"]: m for m in cohort["members"]}
    theorems: list[dict] = []
    for rank in range(1, S.N_PRIMARY + 1):
        member = cohort_by_rank[rank]
        rows = groups.get(rank, [])
        by_arm = {row["arm"]: row for row in rows}
        arms = sorted(by_arm)
        if arms != sorted(S.ARM_ORDER) or len(rows) != S.N_ARMS:
            raise S.FrozenViolation(
                f"formal_rank {rank} carries {arms} instead of one row per frozen arm. A partially "
                "written group means the second stage did not finish; the analyzer refuses to "
                "interpret an unfinished rollout (re-run it with --resume instead).")
        screening_row = screening_by_statement[member["statement_id"]]
        entry = {
            "formal_rank": rank, "statement_id": member["statement_id"],
            "component_id": member["component_id"], "source": screening_row["source"],
            "error_category": member["error_category"],
            "arms": {}, "transitions": {},
        }
        for arm in S.ARM_ORDER:
            row = by_arm[arm]
            score = row["score"]
            label = None if score is None else bool(score)
            entry["arms"][arm] = {
                "success": label, "censored": bool(row["censored"]),
                "verify_status": row["verify_status"],
                "censored_reason": row["censored_reason"] or None,
                "generated_tokens": row["generated_tokens"],
                "generation_time": row["generation_time"],
                "truncated": bool(row["truncated"]),
                "format_ok": bool(row["format_ok"]),
            }
            entry["transitions"][arm] = transition_class(member["error_category"],
                                                        row["error_category_second"])
        entry["n_censored"] = sum(1 for arm in S.ARM_ORDER
                                  if entry["arms"][arm]["success"] is None)
        entry["complete"] = entry["n_censored"] == 0
        theorems.append(entry)
    return theorems


def paired_vectors(theorems: list[dict], first: str, second: str) -> tuple[list[bool], list[bool]]:
    """The two arms' labels over the *complete* quadruplets, in frozen rank order."""
    complete = [t for t in theorems if t["complete"]]
    return ([bool(t["arms"][first]["success"]) for t in complete],
            [bool(t["arms"][second]["success"]) for t in complete])


def full_vectors(theorems: list[dict], first: str, second: str) -> tuple[list, list]:
    """Both arms' labels over all 128 ranks, with ``None`` for a censored arm (sensitivity input)."""
    return ([t["arms"][first]["success"] for t in theorems],
            [t["arms"][second]["success"] for t in theorems])


# --- 3. reporting helpers ------------------------------------------------------------------------

def _gate_payload(gate: Gate, *, threshold: float | None, evaluated: bool, computed: bool,
                  why: str | None, rule: str) -> dict:
    """A paired comparison as the report states it.

    When a guard ended the study the numbers are still the artifact's own, but the *gate* is not
    evaluated: `passes` is None and no threshold verdict is attached, so a printed effect size can
    never be read as a classification. When not even one quadruplet exists, `computed` is False and
    the numbers are withheld rather than fabricated.
    """
    return {
        "rule": rule, "threshold": threshold,
        "n_pairs": gate.n_favor + gate.n_against if computed else None,
        "delta": round(gate.delta, 6) if computed else None,
        "delta_pp": round(100.0 * gate.delta, 2) if computed else None,
        "mcnemar_p_one_sided_exact": gate.mcnemar_p if computed else None,
        "ci_lower": gate.ci_lower if computed else None,
        "n_favor": gate.n_favor if computed else None,
        "n_against": gate.n_against if computed else None,
        "passes": (gate.delta >= (threshold if threshold is not None else 0.0)
                   and gate.mcnemar_p <= ALPHA and gate.ci_lower > 0) if evaluated and computed
                  else None,
        "evaluated": bool(evaluated and computed),
        "not_evaluated_because": None if (evaluated and computed) else why,
    }


def _condition_table(gate: Gate, threshold: float | None) -> dict:
    """The exact condition booleans, so the classification is auditable by hand."""
    return {
        "delta_at_or_above_threshold": bool(gate.delta >= (threshold if threshold is not None
                                                           else 0.0)),
        "mcnemar_p_at_or_below_alpha": bool(gate.mcnemar_p <= ALPHA),
        "ci_lower_above_zero": bool(gate.ci_lower > 0),
    }


def _recompute_classification(conditions_ca: dict, conditions_cb: dict) -> Outcome:
    """The condition table, turned back into a classification. A cross-check, not a second decider.

    `analyze()` compares this against `v4_stats.decide_outcome` and fails the run if the two ever
    disagree -- that is the whole point of computing it here.
    """
    ca = all(conditions_ca.values())
    cb = all(conditions_cb.values())
    if ca and cb:
        return Outcome.A_VERIFIER_SPECIFIC_GO
    if ca:
        return Outcome.B_SELF_REVISION_ONLY
    return Outcome.C_NO_REPAIR_GAIN


def _distribution(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    ordered = sorted(values)

    def q(fraction: float) -> float:
        return round(ordered[min(len(ordered) - 1, int(fraction * len(ordered)))], 3)

    return {"n": len(ordered), "min": round(ordered[0], 3), "median": q(0.5), "p90": q(0.9),
            "max": round(ordered[-1], 3), "mean": round(sum(ordered) / len(ordered), 3)}


def transition_report(theorems: list[dict]) -> dict:
    """Owner §15: the frozen label mapping, reported separately for B, C and D."""
    report: dict = {"labels": list(TRANSITION_CLASSES), "by_arm": {}}
    for arm in TRANSITION_ARMS:
        counts = Counter(t["transitions"][arm] for t in theorems)
        report["by_arm"][arm] = {
            "n_theorems": len(theorems),
            "counts": {label: counts.get(label, 0) for label in TRANSITION_CLASSES},
            "same_error_persistence_rate": round(
                counts.get("SAME_ERROR_CATEGORY", 0) / len(theorems), 4),
            "success_rate": round(counts.get("SUCCESS", 0) / len(theorems), 4),
            "different_error_rate": round(
                counts.get("DIFFERENT_ERROR_CATEGORY", 0) / len(theorems), 4),
            "infra_missing_rate": round(counts.get("INFRA_MISSING", 0) / len(theorems), 4),
        }
    report["by_arm_and_original_category"] = {
        arm: {
            category: {
                "n": len([t for t in theorems if t["error_category"] == category]),
                "counts": {
                    label: len([t for t in theorems if t["error_category"] == category
                                and t["transitions"][arm] == label])
                    for label in TRANSITION_CLASSES},
            }
            for category in PRIMARY_CATEGORIES
        }
        for arm in TRANSITION_ARMS
    }
    return report


def success_matrix(theorems: list[dict]) -> dict:
    """The arm-to-arm second-attempt concordance matrix over complete quadruplets (descriptive)."""
    complete = [t for t in theorems if t["complete"]]
    return {
        arm: {other: sum(1 for t in complete
                         if t["arms"][arm]["success"] and t["arms"][other]["success"])
              for other in S.ARM_ORDER}
        for arm in S.ARM_ORDER
    }


def secondary_report(theorems: list[dict]) -> dict:
    """Preregistration §14 descriptives. Every figure carries its denominator; none is a gate."""
    complete = [t for t in theorems if t["complete"]]
    return {
        "note": ("descriptive only (preregistration §14): nothing here may modify the GO gate, and "
                 "no efficiency or throughput claim is made from the verifier counts"),
        "success_by_arm_over_complete_quadruplets": {
            arm: {"n_pairs": len(complete),
                  "n_success": sum(1 for t in complete if t["arms"][arm]["success"]),
                  "rate": round(sum(1 for t in complete if t["arms"][arm]["success"])
                                / len(complete), 4) if complete else None}
            for arm in S.ARM_ORDER},
        "success_matrix_complete_quadruplets": success_matrix(theorems),
        "by_error_category": {
            category: {
                arm: {"n": len([t for t in complete if t["error_category"] == category]),
                      "n_success": len([t for t in complete if t["error_category"] == category
                                        and t["arms"][arm]["success"]])}
                for arm in S.ARM_ORDER}
            for category in PRIMARY_CATEGORIES},
        "by_source": {
            source: {
                "n": len([t for t in complete if t["source"] == source]),
                "n_success_by_arm": {arm: len([t for t in complete if t["source"] == source
                                               and t["arms"][arm]["success"]])
                                     for arm in S.ARM_ORDER},
            }
            for source in sorted({t["source"] for t in complete})},
        "generated_tokens": {
            arm: _distribution([float(t["arms"][arm]["generated_tokens"]) for t in complete
                                if t["arms"][arm]["generated_tokens"] is not None])
            for arm in S.ARM_ORDER},
        "generation_seconds_total": {
            arm: round(sum(float(t["arms"][arm]["generation_time"] or 0.0) for t in theorems), 2)
            for arm in S.ARM_ORDER},
        "truncated": {arm: sum(1 for t in theorems if t["arms"][arm]["truncated"])
                      for arm in S.ARM_ORDER},
        "format_extraction_failed": {arm: sum(1 for t in theorems
                                              if not t["arms"][arm]["format_ok"])
                                     for arm in S.ARM_ORDER},
    }


def infrastructure_report(paths: dict) -> dict:
    """Data-completeness facts about the verifier. Never a throughput or efficiency claim."""
    events = sum(1 for _ in S.rows_jsonl(paths["verifier_events"]))
    recoveries = sum(1 for _ in S.rows_jsonl(paths["recovery_log"]))
    return {
        "verifier_events_artifact": {"path": str(paths["verifier_events"]), "rows": events},
        "recovery_log_artifact": {"path": str(paths["recovery_log"]), "rows": recoveries},
        "no_efficiency_claim": ("the frozen plan forbids any throughput or efficiency claim from "
                                "these counts; they exist so a reader can audit data completeness"),
    }


# --- 4. the analysis -----------------------------------------------------------------------------

def not_evaluated(why: str) -> dict:
    return {"status": "NOT_EVALUATED", "why": why}


def analyze(*, frozen: S.Frozen, paths: dict, payloads: dict, groups: dict,
            screening_rows: list[dict], env: dict) -> dict:
    """The whole analyzer, in the owner's fixed order. Returns the results artifact."""
    checks, problems = check_provenance(frozen=frozen, paths=paths, payloads=payloads, groups=groups,
                                        screening_rows=screening_rows)
    provenance = {"checks": checks, "n_checks": len(checks),
                  "n_failed": sum(1 for c in checks if not c["pass"]),
                  "status": "PASS" if not problems else "FAIL"}
    if problems:
        raise S.FrozenViolation("raw second-stage provenance failed: " + "; ".join(problems))

    cohort, plan, derangement = payloads["cohort"], payloads["plan"], payloads["derangement"]
    summary = payloads["summary"]
    theorems = build_quadruplets(groups=groups, cohort=cohort,
                                 screening_by_statement={r["statement_id"]: r
                                                         for r in screening_rows})
    complete = [t for t in theorems if t["complete"]]
    n_complete = len(complete)

    # --- guard 1: the data guard (owner §6, Amendment A §4). Checked BEFORE any gate.
    data_guard = {
        "rule": (f"n_complete >= {DATA_GUARD_MIN}: the primary analyzable unit is a valid A/B/C/D "
                 "quadruplet, and the minimum is the frozen one (never lowered after seeing data)"),
        "n_nominal": S.N_PRIMARY, "n_complete": n_complete,
        "n_incomplete": S.N_PRIMARY - n_complete,
        "minimum_required": DATA_GUARD_MIN, "pass": bool(n_complete >= DATA_GUARD_MIN),
    }
    incomplete = [t for t in theorems if not t["complete"]]
    incomplete_reasons = Counter(
        (arm, t["arms"][arm]["censored_reason"] or t["arms"][arm]["verify_status"])
        for t in incomplete for arm in S.ARM_ORDER if t["arms"][arm]["success"] is None)

    # --- guard 2: differential censoring (Amendment A §3). r_arm over the nominal 128.
    rates = {arm: sum(1 for t in theorems if t["arms"][arm]["success"] is None) / S.N_PRIMARY
             for arm in S.ARM_ORDER}
    range_value = censoring_range(rates)
    censoring_guard = {
        "rule": ("r_arm = unresolved infrastructure-censored candidates / 128 per arm; "
                 f"censoring_range = max(r) - min(r) <= {CENSORING_RANGE_MAX}"),
        "rates": {arm: round(rate, 6) for arm, rate in rates.items()},
        "range": round(range_value, 6), "maximum_allowed": CENSORING_RANGE_MAX,
        "pass": bool(range_value <= CENSORING_RANGE_MAX),
        "infra_is_missing_data_never_a_failure": True,
    }

    # --- the paired comparisons. A gate is only *evaluated* when both guards passed.
    computed = n_complete > 0
    if computed:
        a_complete, c_complete = paired_vectors(theorems, "A_FRESH_RETRY", "C_VERIFIER_REPAIR")
        b_complete, _ = paired_vectors(theorems, "B_SELF_REVISION", "C_VERIFIER_REPAIR")
        d_complete, _ = paired_vectors(theorems, "D_MISMATCHED_DIAGNOSTIC", "C_VERIFIER_REPAIR")
        gate_ca = paired_gate(a_complete, c_complete)
        gate_cb = paired_gate(b_complete, c_complete)
        gate_cd = paired_gate(d_complete, c_complete)
        gate_ba = paired_gate(a_complete, b_complete)
    else:
        gate_ca = gate_cb = gate_cd = gate_ba = _ZERO_GATE

    outcome = decide_outcome(n_complete, rates, gate_ca, gate_cb)
    mechanism = mechanism_label(outcome, gate_cd)
    evaluated = outcome not in NOT_EVALUATED_OUTCOMES
    if not data_guard["pass"]:
        why = (f"data guard failed before any gate: n_complete={n_complete} "
               f"(needs {DATA_GUARD_MIN})")
    elif not censoring_guard["pass"]:
        why = (f"differential-censoring guard failed before any gate: range={range_value:.4f} "
               f"(allows {CENSORING_RANGE_MAX}); rates={censoring_guard['rates']}")
    elif outcome is Outcome.A_VERIFIER_SPECIFIC_GO:
        why = (f"C-A and C-B each satisfy delta >= threshold, exact one-sided McNemar p <= {ALPHA} "
               "and bootstrap CI lower bound > 0")
    elif outcome is Outcome.B_SELF_REVISION_ONLY:
        why = "C-A satisfies its three conditions and C-B does not"
    else:
        why = "C-A does not satisfy its three conditions, so no repair gain over a fresh retry"

    conditions_ca = _condition_table(gate_ca, DELTA_CA_MIN)
    conditions_cb = _condition_table(gate_cb, DELTA_CB_MIN)
    recomputed = _recompute_classification(conditions_ca, conditions_cb) if evaluated else None

    # --- sensitivity (Amendment A §3): infra read as failure, reported separately, never primary.
    sensitivity = {}
    for name, first in (("C_minus_A", "A_FRESH_RETRY"), ("C_minus_B", "B_SELF_REVISION"),
                        ("C_minus_D", "D_MISMATCHED_DIAGNOSTIC")):
        x_all, c_all = full_vectors(theorems, first, "C_VERIFIER_REPAIR")
        gate, pairs_completed = paired_gate_infra_as_failure(x_all, c_all)
        sensitivity[name] = {
            "delta_pp": round(100.0 * gate.delta, 2), "mcnemar_p_one_sided_exact": gate.mcnemar_p,
            "n_favor": gate.n_favor, "n_against": gate.n_against,
            "pairs_completed_with_a_censored_arm": pairs_completed,
        }
    sensitivity["rule"] = (
        "Amendment A §3 sensitivity: every infrastructure-censored arm outcome is read as a failure, "
        "over all 128 frozen ranks. It is reported separately and never redefines the primary "
        "endpoint, the guards or the classification.")

    result = {
        "artifact_type": "v4_p001_results",
        "schema_version": "v4-p001-results-1",
        "status": ("COMPUTED -- the analysis of a frozen raw artifact by frozen code; it decides "
                   "nothing beyond the preregistered taxonomy"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": env["hostname"], "git_revision": env["git_revision"],
        "experiment_id": S.EXPERIMENT_ID,
        "question": ("Given a Lean-level failed proof from frozen Kimina-Distill-0.6B theta0, does "
                     "the exact verifier diagnostic improve second-attempt success beyond a fresh "
                     "retry (A), beyond self-revision without the diagnostic (B), and is any gain "
                     "specific to this theorem's own diagnostic (D)?"),
        "prohibited_readings": [
            "not a training result: no RL, SFT, checkpoint update or controller work is authorized",
            "no AUPRC and no predictive endpoint exists in this study (owner §L)",
            "no throughput or efficiency claim may be derived from the verifier recovery counts",
            "a D_INCONCLUSIVE_* classification is not a NO-GO: it means the design could not answer",
            ("DIAGNOSTIC_NONSPECIFIC forbids the claim that the model understood the "
             "theorem-specific diagnostic; the permitted claim is that supplying verifier "
             "diagnostic text improved repair (Amendment A §6.3)"),
            "not evidence about the sealed 93-component V3-FINAL-HOLDOUT reserve",
        ],
        "frozen": {
            "prereg_commit": S.PREREG_COMMIT,
            "prereg_doc": S.PREREG_DOC, "amendment_doc": S.AMENDMENT_DOC,
            "pool_hash": frozen.pool["pool_hash"], "order_hash": frozen.pool["order_hash"],
            "screening_seed_hash": frozen.seeds["screening_seed_hash"],
            "paired_seed_hash": frozen.seeds["paired_seed_hash"],
            "arm_schedule_hash": payloads["plan"]["arm_schedule_hash"],
            "derangement_version": derangement.get("version"),
            "derangement_mapping_sha256": derangement.get("mapping_sha256"),
            "plan_sha256": plan.get("plan_sha256"),
            "arm_prompt_plan_sha256": plan.get("arm_prompt_plan_sha256"),
            "renderer_version": RENDERER_VERSION,
            "normalization_version": NORMALIZATION_VERSION,
            "theta0_weights_sha256": S.MODEL["weights_sha256"],
            "theta0_revision": S.MODEL["revision"],
            "thresholds": {"delta_ca_min": DELTA_CA_MIN, "delta_cb_min": DELTA_CB_MIN,
                           "alpha": ALPHA, "bootstrap_reps": BOOTSTRAP_REPS,
                           "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_level": BOOTSTRAP_LEVEL,
                           "data_guard_min": DATA_GUARD_MIN,
                           "censoring_range_max": CENSORING_RANGE_MAX},
            "statistics": ("one-sided exact paired McNemar on the discordant pairs + paired theorem "
                           "percentile bootstrap CI; comparison(A, B) = mean(B - A)"),
            "frozen_settings_sha256": S.sha(S.FROZEN_SETTINGS),
            "script_versions": S.script_version_hash(*SCRIPT_VERSION_FILES),
            "raw_artifact_sha256": S.sha256_file(paths["second_stage_raw"]),
            "screening_raw_sha256": S.sha256_file(paths["screening_raw"]),
            "boundary_artifacts": {name: payloads[name].get("content_sha256")
                                   for name in ("cohort", "derangement", "plan")},
            "run_id": summary.get("run_id"),
        },
        "analysis_tree_clean": env["status_porcelain"] == "",
        "provenance": provenance,
        "guard_order": ["provenance", "n_complete_guard", "differential_censoring_guard",
                        "primary_gates", "specificity_analysis", "mechanism_and_descriptive"],
        "data_guard_outcome_D_first": data_guard,
        "censoring_guard": censoring_guard,
        "arms": list(S.ARM_ORDER),
        "cohort": {
            "n_nominal": S.N_PRIMARY, "n_complete_quadruplets": n_complete,
            "n_incomplete": S.N_PRIMARY - n_complete,
            "complete_fraction": round(n_complete / S.N_PRIMARY, 4),
            "incomplete_by_arm_and_reason": {f"{arm}|{reason}": count
                                             for (arm, reason), count in
                                             sorted(incomplete_reasons.items())},
            "error_category_counts": dict(Counter(t["error_category"] for t in theorems)),
        },
        "primary_gates": {
            "C_minus_A": {**_gate_payload(
                gate_ca, threshold=DELTA_CA_MIN, evaluated=evaluated, computed=computed,
                why=why, rule="mean(C - A) over complete quadruplets, delta >= 0.08, exact one-sided "
                              "McNemar p <= 0.05, bootstrap CI lower > 0"),
                "conditions": conditions_ca},
            "C_minus_B": {**_gate_payload(
                gate_cb, threshold=DELTA_CB_MIN, evaluated=evaluated, computed=computed,
                why=why, rule="mean(C - B) over complete quadruplets, delta >= 0.05, exact one-sided "
                              "McNemar p <= 0.05, bootstrap CI lower > 0"),
                "conditions": conditions_cb},
        },
        "specificity_analysis": {
            **_gate_payload(gate_cd, threshold=None, evaluated=evaluated, computed=computed,
                            why=why,
                            rule=("mean(C - D): directional rule with no effect-size threshold -- "
                                  "DIAGNOSTIC_SPECIFIC iff delta > 0, exact one-sided McNemar "
                                  "p <= 0.05 and CI lower > 0 (Amendment A §6.3)")),
            "n_byte_identical_donor_diagnostics": plan.get("n_byte_identical_donor_diagnostics"),
            "donor_is_a_different_theorem_for_every_rank": plan.get(
                "donor_is_a_different_theorem_for_every_rank"),
            "interpretation_limit": (
                "at N = 128 the C-D rule detects a genuine specificity effect of roughly +9 to +10 pp "
                "at realistic discordance (Amendment A §7.5); a null here is weakly informative "
                "about specificity and must be reported as such"),
        },
        "descriptive_B_minus_A": {
            **_gate_payload(gate_ba, threshold=None, evaluated=False, computed=computed, why=None,
                            rule="descriptive only (preregistration §11): B - A carries no gate"),
            "conditions": _condition_table(gate_ba, None) if computed else None,
        },
        "outcome": {
            "classification": outcome.value, "mechanism_label": mechanism.value,
            "why": why,
            "taxonomy": [o.value for o in Outcome],
            "mechanism_labels": [m.value for m in Mechanism],
            "classification_matches_the_condition_table": (
                None if recomputed is None else recomputed is outcome),
            "the_taxonomy_was_not_evaluated_when_a_guard_failed": not evaluated,
            "mechanism_note": ("the mechanism label is appended to, never merged with, the "
                               "classification: A_VERIFIER_SPECIFIC_GO never requires C-D evidence"),
        },
        "sensitivity_infra_as_failure": sensitivity,
        "transitions": transition_report(theorems),
        "secondary": secondary_report(theorems),
        "infrastructure": infrastructure_report(paths),
        "per_theorem": [
            {"formal_rank": t["formal_rank"], "statement_id": t["statement_id"],
             "component_id": t["component_id"], "source": t["source"],
             "error_category": t["error_category"], "complete": t["complete"],
             "arms": {arm: t["arms"][arm]["success"] for arm in S.ARM_ORDER},
             "transitions": dict(t["transitions"]),
             "censored_reason": {arm: t["arms"][arm]["censored_reason"] for arm in S.ARM_ORDER
                                 if t["arms"][arm]["success"] is None}}
            for t in theorems],
    }
    if result["outcome"]["classification_matches_the_condition_table"] is False:
        raise S.FrozenViolation(
            "the condition table recomputation disagrees with v4_stats.decide_outcome; the "
            "classification would not be auditable by hand, so no result is written")
    return result


# --- 5. CLI --------------------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="V4-P001 analyzer: guards first, then C-A / C-B / C-D, mechanically.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--out-dir", default=OUT_DIR_DEFAULT, help="artifact directory")
    parser.add_argument("--out", default=RESULTS_DEFAULT, help="where the results artifact goes")
    parser.add_argument("--check-only", action="store_true",
                        help="run every provenance check and report the classification; write nothing")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    try:
        paths, payloads = load_inputs(out_dir)
        frozen = S.load_frozen(verify_files=True)
        groups = read_repair_groups(paths["second_stage_raw"])
        # the screening rows are validated too: the analyzer re-reads the first-attempt diagnostics
        # from them, so they must satisfy the frozen schema before a number is computed
        screening_rows = [row for group in
                          S.read_grouped(paths["screening_raw"], S.validate_screening_row).values()
                          for row in group]
        result = analyze(frozen=frozen, paths=paths, payloads=payloads, groups=groups,
                         screening_rows=screening_rows, env=S.collect_env())
    except S.FrozenViolation as exc:
        print(f"[ABORT BEFORE ANALYSIS] {exc}", file=sys.stderr)
        return 3
    if not result["analysis_tree_clean"]:
        print("[WARNING] the working tree was not clean when this analysis ran; the recorded "
              "artifact hashes still identify every input", file=sys.stderr)

    outcome = result["outcome"]
    print(json.dumps({
        "classification": outcome["classification"],
        "mechanism_label": outcome["mechanism_label"],
        "why": outcome["why"],
        "data_guard": {"n_complete": result["data_guard_outcome_D_first"]["n_complete"],
                       "pass": result["data_guard_outcome_D_first"]["pass"]},
        "censoring_guard": {"range": result["censoring_guard"]["range"],
                            "pass": result["censoring_guard"]["pass"]},
        "C_minus_A": {k: result["primary_gates"]["C_minus_A"][k]
                      for k in ("delta_pp", "mcnemar_p_one_sided_exact", "ci_lower", "n_favor",
                                "n_against", "passes")},
        "C_minus_B": {k: result["primary_gates"]["C_minus_B"][k]
                      for k in ("delta_pp", "mcnemar_p_one_sided_exact", "ci_lower", "n_favor",
                                "n_against", "passes")},
        "C_minus_D": {k: result["specificity_analysis"][k]
                      for k in ("delta_pp", "mcnemar_p_one_sided_exact", "ci_lower", "passes")},
        "transitions": {arm: result["transitions"]["by_arm"][arm]["counts"]
                        for arm in TRANSITION_ARMS},
        "provenance": {"n_checks": result["provenance"]["n_checks"],
                       "n_failed": result["provenance"]["n_failed"]},
    }, indent=2, ensure_ascii=False))
    if args.check_only:
        print("[check-only] nothing written")
        return 0
    out = Path(args.out)
    S.write_json_atomic(out if out.is_absolute() else ROOT / out, result)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
