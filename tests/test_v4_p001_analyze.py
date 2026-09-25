"""V4-P001 analyzer fixtures (owner §18; Amendment A §16).

Every outcome and every guard gets a fixture, and the mechanical order of Amendment A §4/§14 is
itself under test: the guards decide before the gates, and a guard that fails leaves the gate
`passes` flags unevaluated.

The fixtures fabricate the *outcomes* (which the formal run would produce on fly122) but not the
design: the cohort statements, error categories, seeds, arm order, positions, prompt hashes and the
derangement mapping are all built to be self-consistent, and the frozen design is the committed one
(`v4_p001_spec.load_frozen`), so a fixture cannot pass a check that a real artifact would fail on a
technicality. The analyzer's own provenance stage runs on every fixture: these tests fail closed if
a fixture is not a legal artifact.
"""

from __future__ import annotations

import functools
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import v4_p001_analyze as A
import v4_p001_spec as S

from tinylean_rl.evaluation.v4_seeds import second_stage_seed
from tinylean_rl.evaluation.v4_stats import (
    DELTA_CA_MIN,
    DELTA_CB_MIN,
    Mechanism,
    Outcome,
)

CATEGORIES = ("unsolved_goals", "elaboration_type_mismatch", "unknown_identifier", "tactic_failure",
              "typeclass_synthesis", "other_semantic_lean_failure")


@functools.lru_cache(maxsize=1)
def frozen() -> S.Frozen:
    return S.load_frozen(verify_files=True)


def screening_row(member: dict, category: str) -> dict:
    diagnostic = f"{category}\ncase h\n⊢ 1 + 1 = 2"
    proof = "by\n  simp [Nat.add_comm]"
    return {
        "schema_version": S.SCREENING_SCHEMA_VERSION, "run_id": "fixture",
        "experiment_id": S.EXPERIMENT_ID,
        "screening_rank": member["screening_rank"], "formal_rank": member["screening_rank"],
        "statement_id": member["statement_id"], "component_id": member["component_id"],
        "name": member["name"], "source": member["source"], "tier": member["tier"],
        "classes": member["classes"], "prompt_tokens": member["prompt_tokens"],
        "prompt_sha256": S.sha256_text(f"prompt|{member['statement_id']}"),
        "prompt_token_count": member["prompt_tokens"],
        "seed": S.first_stage_seed(member["screening_rank"]),
        "model_sha256": S.MODEL["weights_sha256"], "model_revision": S.MODEL["revision"],
        "completion_text": f"{proof}\ntrailing prose", "completion_sha256": S.sha256_text(proof),
        "generated_tokens": 40, "truncated": False, "format_ok": True, "has_lean_block": True,
        "extracted_proof": proof, "extracted_proof_sha256": S.sha256_text(proof),
        "extracted_proof_present": True, "extracted_proof_tokens": 8,
        "diagnostic_text": diagnostic, "diagnostic_sha256": S.sha256_text(diagnostic),
        "diagnostic_tokens": len(diagnostic.split()), "diagnostic_source": "verify_item",
        "truncated_diagnostic": False, "context_fits": True,
        "context_tokens_with_response": member["prompt_tokens"] + S.MAX_RESPONSE_TOKENS,
        "verified": False, "score": 0, "acc": 0, "verify_status": "lean_error",
        "lean_message": diagnostic, "error_category": category,
        "screening_status": S.PRIMARY_SEMANTIC_FAILURE, "primary_eligible": True,
        "generation_time": 1.0, "verification_time": None,
        "host": "fixture", "gpu": "", "created_at": "2026-09-25T00:00:00+00:00",
    }


def repair_row(*, plan_row: dict, screening: dict, arm: str, position: int, success: bool | None,
               censored: bool, rank_succeeds: bool) -> dict:
    arm_plan = plan_row["arms"][arm]
    row = {
        "schema_version": S.REPAIR_SCHEMA_VERSION, "run_id": "fixture",
        "experiment_id": S.EXPERIMENT_ID,
        "formal_rank": plan_row["formal_rank"], "screening_rank": plan_row["screening_rank"],
        "statement_id": plan_row["statement_id"], "component_id": screening["component_id"],
        "name": screening["name"], "source": screening["source"],
        "error_category": plan_row["error_category"], "arm": arm, "position": position,
        "seed": plan_row["seed"],
        "prompt_sha256": arm_plan["prompt_sha256"], "prompt_token_count": arm_plan["prompt_tokens"],
        "context_tokens_with_response": arm_plan["context_tokens_with_response"],
        "renderer_version": "v4-prompt-1", "normalization_version": "v4-diag-1",
        "diagnostic_sha256": arm_plan["diagnostic_sha256"],
        "diagnostic_source": arm_plan["diagnostic_source"],
        "failed_proof_sha256": plan_row["failed_proof_sha256"],
        "model_sha256": S.MODEL["weights_sha256"], "model_revision": S.MODEL["revision"],
        "completion_text": "by\n  simp", "completion_sha256": S.sha256_text("by\n  simp"),
        "generated_tokens": 11, "truncated": False, "format_ok": True, "has_lean_block": True,
        "extracted_proof_present": True,
        "verified": False, "score": 0, "acc": 0, "verify_status": "lean_error",
        "lean_message": "unsolved goals" if success is False else "",
        "error_category_second": plan_row["error_category"],
        "success": False, "censored": False, "censored_reason": "",
        "generation_time": 0.4, "verification_time": 0.1,
        "host": "fixture", "gpu": "", "created_at": "2026-09-25T00:00:00+00:00",
    }
    if censored:
        row.update({"completion_text": "", "completion_sha256": S.sha256_text(""),
                    "generated_tokens": 0, "format_ok": False, "has_lean_block": False,
                    "extracted_proof_present": False, "verified": False, "score": None, "acc": None,
                    "verify_status": "unresolved_infra_error", "lean_message": "",
                    "error_category_second": None, "success": False, "censored": True,
                    "censored_reason": "per_theorem_recovery_exposure",
                    "generation_time": None, "verification_time": None})
    elif success:
        row.update({"verified": True, "score": 1, "acc": 1, "verify_status": "verified",
                    "error_category_second": "verified", "success": True})
    return row


def build_case(out_dir: Path, *, succeed_through: dict, censor: dict | None = None,
               tamper=None) -> dict:
    """Write a self-consistent boundary + second-stage artifact set for one outcome pattern."""
    design = frozen()
    members = design.screening_theorems[:S.N_PRIMARY]
    censor = censor or {}
    out_dir.mkdir(parents=True, exist_ok=True)

    screening_rows = [screening_row(m, CATEGORIES[i % len(CATEGORIES)])
                      for i, m in enumerate(members)]
    for row in screening_rows:
        S.validate_screening_row(row)
    screening_path = out_dir / S.SCREENING_RAW_BASENAME
    screening_path.unlink(missing_ok=True)
    S.append_rows_durable(screening_path, screening_rows)
    by_statement = {row["statement_id"]: row for row in screening_rows}

    donors = {row["statement_id"]: members[(i + 1) % len(members)]["statement_id"]
              for i, row in enumerate(members)}
    theorems = []
    for index, member in enumerate(members):
        rank = index + 1
        row = by_statement[member["statement_id"]]
        donor_id = donors[member["statement_id"]]
        donor = by_statement[donor_id]
        assert donor_id != member["statement_id"]
        arms = {}
        for arm in S.ARM_ORDER:
            diagnostic = (str(row["diagnostic_text"]) if arm == "C_VERIFIER_REPAIR" else
                          (str(donor["diagnostic_text"]) if arm == "D_MISMATCHED_DIAGNOSTIC" else ""))
            arms[arm] = {
                "prompt_sha256": S.sha256_text(f"{member['statement_id']}|{arm}"),
                "prompt_tokens": 800 + (10 if diagnostic else 0),
                "context_tokens_with_response": 800 + (10 if diagnostic else 0)
                + S.MAX_RESPONSE_TOKENS,
                "diagnostic_sha256": S.sha256_text(diagnostic) if diagnostic else "",
                "diagnostic_source": ("verify_item" if arm == "C_VERIFIER_REPAIR" else
                                      ("derangement_donor" if arm == "D_MISMATCHED_DIAGNOSTIC"
                                       else "none")),
            }
        theorems.append({
            "formal_rank": rank, "screening_rank": member["screening_rank"],
            "statement_id": member["statement_id"], "error_category": row["error_category"],
            "seed": second_stage_seed(rank), "arm_order": list(S.arm_order(rank)),
            "failed_proof_sha256": row["extracted_proof_sha256"],
            "donor_statement_id": donor_id,
            "arms": arms,
        })

    mapping = [{"recipient": t["statement_id"], "donor": t["donor_statement_id"]}
               for t in theorems]
    mapping_sha256 = S.sha({"version": "v4-derange-1", "mapping": mapping})
    derangement = {
        "artifact_type": "v4_p001_diagnostic_derangement", "schema_version": "fixture",
        "second_stage_candidates_generated": 0, "cohort_content_sha256": "PENDING",
        "version": "v4-derange-1", "mapping_sha256": mapping_sha256,
        "derangement": {"version": "v4-derange-1", "mapping": mapping,
                        "mapping_sha256": mapping_sha256},
    }
    cohort = {
        "artifact_type": "v4_p001_primary_cohort", "schema_version": "fixture",
        "second_stage_candidates_generated": 0, "cohort_size": S.N_PRIMARY,
        "screening_raw_sha256": S.sha256_file(screening_path),
        "members": [{"formal_rank": t["formal_rank"], "statement_id": t["statement_id"],
                     "component_id": members[t["formal_rank"] - 1]["component_id"],
                     "error_category": t["error_category"]} for t in theorems],
    }
    plan = {
        "artifact_type": "v4_p001_second_stage_plan", "schema_version": "fixture",
        "second_stage_candidates_generated": 0, "theorems": theorems,
        "n_theorems": S.N_PRIMARY, "n_candidates": S.SECOND_STAGE_CANDIDATES,
        "plan_sha256": S.sha(theorems),
        "arm_prompt_plan_sha256": S.sha([{arm: t["arms"][arm]["prompt_sha256"]
                                          for arm in S.ARM_ORDER} for t in theorems]),
        "arm_schedule_hash": S.schedule_hash(), "second_stage_seed_hash": S.PAIRED_SEED_HASH,
        "derangement_mapping_sha256": mapping_sha256,
        "donor_is_a_different_theorem_for_every_rank": True,
        "n_byte_identical_donor_diagnostics": 0,
    }
    # the three boundary artifacts chain to each other by content hash (sealed like the runner's)
    def seal(payload: dict) -> dict:
        return {**payload, "content_sha256": S.sha({k: v for k, v in payload.items()
                                                     if k != "content_sha256"})}

    cohort, derangement, plan = seal(cohort), seal(derangement), seal(plan)
    plan["cohort_content_sha256"] = cohort["content_sha256"]
    plan["derangement_mapping_sha256"] = derangement["mapping_sha256"]
    derangement["cohort_content_sha256"] = cohort["content_sha256"]
    plan = seal({k: v for k, v in plan.items() if k != "content_sha256"})
    if tamper is not None:
        tamper(plan, cohort, derangement)

    paths = {"cohort": out_dir / S.PRIMARY_COHORT_BASENAME,
             "derangement": out_dir / S.DERANGEMENT_BASENAME,
             "plan": out_dir / S.SECOND_STAGE_PLAN_BASENAME}
    for name, payload in (("cohort", cohort), ("derangement", derangement), ("plan", plan)):
        S.write_json_atomic(paths[name], payload)
    boundary = {
        "artifact_type": "v4_p001_boundary_validation", "status": "PASS", "n_failed": 0,
        "n_checks": 29, "second_stage_candidates_generated": 0,
        "artifacts": {A.BOUNDARY_ARTIFACT_NAMES[name]: {
            "path": str(paths[name]), "sha256": S.sha256_file(paths[name]),
            "content_sha256": payload["content_sha256"]}
            for name, payload in (("cohort", cohort), ("derangement", derangement), ("plan", plan))},
        "screening_raw": {"path": str(screening_path), "sha256": S.sha256_file(screening_path),
                          "rows": len(screening_rows)},
    }
    S.write_json_atomic(out_dir / S.BOUNDARY_VALIDATION_BASENAME, boundary)

    rows = []
    for theorem in theorems:
        rank = theorem["formal_rank"]
        for position, arm in enumerate(theorem["arm_order"]):
            censored = rank in censor.get(arm, set())
            success = None if censored else rank <= succeed_through.get(arm, 0)
            rows.append(repair_row(plan_row=theorem, screening=by_statement[theorem["statement_id"]],
                                   arm=arm, position=position, success=success, censored=censored,
                                   rank_succeeds=bool(success)))
    raw_path = out_dir / S.SECOND_STAGE_RAW_BASENAME
    raw_path.unlink(missing_ok=True)
    for start in range(0, len(rows), S.N_ARMS):
        S.append_rows_durable(raw_path, rows[start:start + S.N_ARMS])
    summary = {
        "experiment_id": S.EXPERIMENT_ID, "schema_version": S.REPAIR_SCHEMA_VERSION,
        "run_id": "fixture", "stage": "second",
        "frozen_settings": S.FROZEN_SETTINGS, "frozen_settings_sha256": S.sha(S.FROZEN_SETTINGS),
        "script_versions": S.script_version_hash("scripts/v4_p001_rollout.py"),
        "prereg_commit": S.PREREG_COMMIT, "git": {"hostname": "fixture"},
        "boundary_validation": {"status": "PASS", "n_checks": 29, "n_failed": 0,
                                "artifacts": boundary["artifacts"],
                                "screening_raw": boundary["screening_raw"]},
        "plan_sha256": plan["plan_sha256"],
        "arm_prompt_plan_sha256": plan["arm_prompt_plan_sha256"],
        "arm_schedule_hash": plan["arm_schedule_hash"],
        "second_stage_seed_hash": plan["second_stage_seed_hash"],
        "derangement_mapping_sha256": plan["derangement_mapping_sha256"],
        "n_theorems": S.N_PRIMARY, "n_candidates": S.SECOND_STAGE_CANDIDATES, "mode": "FIXTURE",
    }
    S.write_json_atomic(out_dir / S.SUMMARY_BASENAME, summary)
    return {"out_dir": out_dir, "plan": plan, "cohort": cohort, "derangement": derangement,
            "boundary": boundary, "screening_rows": screening_rows}


def analyze_case(tmp_path: Path, case: dict) -> dict:
    paths, payloads = A.load_inputs(case["out_dir"])
    groups = A.read_repair_groups(paths["second_stage_raw"])
    screening_rows = [row for group in
                      S.read_grouped(paths["screening_raw"], S.validate_screening_row).values()
                      for row in group]
    return A.analyze(frozen=frozen(), paths=paths, payloads=payloads, groups=groups,
                     screening_rows=screening_rows,
                     env={"hostname": "fixture", "git_revision": "0" * 40,
                          "status_porcelain": ""})


# --- the six outcome fixtures (owner §18) ---------------------------------------------------------

def test_go_with_a_diagnostic_specific_mechanism(tmp_path: Path) -> None:
    case = build_case(tmp_path, succeed_through={"A_FRESH_RETRY": 60, "B_SELF_REVISION": 90,
                                                "C_VERIFIER_REPAIR": 128,
                                                "D_MISMATCHED_DIAGNOSTIC": 95})
    result = analyze_case(tmp_path, case)
    assert result["provenance"]["status"] == "PASS"
    assert result["outcome"]["classification"] == Outcome.A_VERIFIER_SPECIFIC_GO.value
    assert result["outcome"]["mechanism_label"] == Mechanism.DIAGNOSTIC_SPECIFIC.value
    ca, cb = result["primary_gates"]["C_minus_A"], result["primary_gates"]["C_minus_B"]
    assert (ca["delta_pp"], cb["delta_pp"]) == (53.12, 29.69)
    assert ca["passes"] and cb["passes"]
    assert ca["n_favor"] == 68 and ca["n_against"] == 0
    assert ca["delta_pp"] >= 100 * DELTA_CA_MIN and cb["delta_pp"] >= 100 * DELTA_CB_MIN
    cd = result["specificity_analysis"]
    assert cd["delta_pp"] == 25.78 and cd["passes"] is True
    assert result["outcome"]["classification_matches_the_condition_table"] is True
    assert result["cohort"]["n_complete_quadruplets"] == 128


def test_go_with_a_nonspecific_mechanism(tmp_path: Path) -> None:
    # C beats A and B, but D matches C: the gain is not attributable to the theorem's own diagnostic
    case = build_case(tmp_path, succeed_through={"A_FRESH_RETRY": 40, "B_SELF_REVISION": 60,
                                                "C_VERIFIER_REPAIR": 80,
                                                "D_MISMATCHED_DIAGNOSTIC": 80})
    result = analyze_case(tmp_path, case)
    assert result["outcome"]["classification"] == Outcome.A_VERIFIER_SPECIFIC_GO.value
    assert result["outcome"]["mechanism_label"] == Mechanism.DIAGNOSTIC_NONSPECIFIC.value
    assert result["specificity_analysis"]["delta_pp"] == 0.0
    assert result["specificity_analysis"]["passes"] is False
    assert any("theorem-specific diagnostic" in reading for reading in result["prohibited_readings"])


def test_self_revision_only(tmp_path: Path) -> None:
    case = build_case(tmp_path, succeed_through={"A_FRESH_RETRY": 40, "B_SELF_REVISION": 122,
                                                "C_VERIFIER_REPAIR": 128})
    result = analyze_case(tmp_path, case)
    assert result["outcome"]["classification"] == Outcome.B_SELF_REVISION_ONLY.value
    assert result["primary_gates"]["C_minus_A"]["passes"] is True
    assert result["primary_gates"]["C_minus_B"]["passes"] is False
    # +4.7 pp is below the frozen +5 pp threshold, and only that condition fails
    cb = result["primary_gates"]["C_minus_B"]
    assert cb["delta_pp"] == 4.69 and cb["conditions"]["mcnemar_p_at_or_below_alpha"] is True
    assert cb["conditions"]["delta_at_or_above_threshold"] is False


def test_no_repair_gain(tmp_path: Path) -> None:
    case = build_case(tmp_path, succeed_through={"A_FRESH_RETRY": 100, "B_SELF_REVISION": 100,
                                                "C_VERIFIER_REPAIR": 100})
    result = analyze_case(tmp_path, case)
    assert result["outcome"]["classification"] == Outcome.C_NO_REPAIR_GAIN.value
    assert result["primary_gates"]["C_minus_A"]["passes"] is False
    assert result["outcome"]["mechanism_label"] in {Mechanism.DIAGNOSTIC_NONSPECIFIC.value,
                                                    Mechanism.DIAGNOSTIC_SPECIFIC.value}


def test_the_ci_condition_binds_the_gate(tmp_path: Path) -> None:
    """The Amendment A §7.7 configuration: delta and McNemar pass, the bootstrap CI does not.

    The amendment's simulated dataset (27 favourable, 15 adverse discordant pairs) is reproduced
    here: delta = 42/128 - 30/128 = 0.09375 and p = 0.0442 are order-independent, so they match the
    amendment exactly. The percentile bootstrap is drawn over theorem positions, so its realized
    lower bound is not: the amendment's simulation realized -0.0078, and this fixture's ordering
    realizes exactly 0.0, because 86 of the 128 paired differences are 0. Both fail the frozen
    ``ci_lower > 0`` condition, which is the whole point of the cross-check.
    """
    case = build_case(tmp_path, succeed_through={"A_FRESH_RETRY": 20, "B_SELF_REVISION": 20,
                                                "C_VERIFIER_REPAIR": 20})
    # rewrite the raw artifact with the exact discordant pattern
    out_dir = case["out_dir"]
    plan = case["plan"]
    screening = {row["statement_id"]: row for row in case["screening_rows"]}
    rows = []
    for theorem in plan["theorems"]:
        rank = theorem["formal_rank"]
        rank_a = rank <= 20                               # 5 joint + 15 adverse discordant
        rank_c = rank <= 5 or 20 < rank <= 47             # 5 joint + 27 favourable discordant
        for position, arm in enumerate(theorem["arm_order"]):
            success = {"A_FRESH_RETRY": rank_a, "C_VERIFIER_REPAIR": rank_c,
                       "B_SELF_REVISION": rank_a, "D_MISMATCHED_DIAGNOSTIC": rank_a}[arm]
            rows.append(repair_row(plan_row=theorem,
                                   screening=screening[theorem["statement_id"]], arm=arm,
                                   position=position, success=success, censored=False,
                                   rank_succeeds=success))
    raw_path = out_dir / S.SECOND_STAGE_RAW_BASENAME
    raw_path.unlink()
    for start in range(0, len(rows), S.N_ARMS):
        S.append_rows_durable(raw_path, rows[start:start + S.N_ARMS])
    result = analyze_case(tmp_path, case)
    ca = result["primary_gates"]["C_minus_A"]
    assert ca["delta_pp"] == 9.38
    assert ca["n_favor"] == 27 and ca["n_against"] == 15
    assert round(ca["mcnemar_p_one_sided_exact"], 4) == 0.0442
    assert ca["ci_lower"] <= 0 and ca["passes"] is False
    # +9.38 pp clears the frozen +8 pp floor and p = 0.0442 clears alpha: only the CI condition fails
    assert ca["conditions"] == {"delta_at_or_above_threshold": True,
                                "mcnemar_p_at_or_below_alpha": True,
                                "ci_lower_above_zero": False}
    assert result["outcome"]["classification"] == Outcome.C_NO_REPAIR_GAIN.value


# --- the two guards (owner §18) ------------------------------------------------------------------

def test_data_guard_ends_the_study_before_any_gate(tmp_path: Path) -> None:
    case = build_case(tmp_path, succeed_through={"A_FRESH_RETRY": 128, "B_SELF_REVISION": 128,
                                                "C_VERIFIER_REPAIR": 128,
                                                "D_MISMATCHED_DIAGNOSTIC": 128},
                      censor={arm: set(range(1, 36)) for arm in S.ARM_ORDER})
    result = analyze_case(tmp_path, case)
    guard = result["data_guard_outcome_D_first"]
    assert guard["n_complete"] == 93 and guard["pass"] is False
    assert result["outcome"]["classification"] == Outcome.D_INCONCLUSIVE_BY_DATA.value
    assert result["outcome"]["mechanism_label"] == Mechanism.NOT_EVALUABLE.value
    # the taxonomy is not evaluated: no gate carries a verdict, and the guards are the reason
    for gate in ("C_minus_A", "C_minus_B"):
        assert result["primary_gates"][gate]["passes"] is None
        assert result["primary_gates"][gate]["evaluated"] is False
        assert "data guard" in result["primary_gates"][gate]["not_evaluated_because"]
    assert result["outcome"]["classification_matches_the_condition_table"] is None


def test_differential_censoring_guard_ends_the_study_before_any_gate(tmp_path: Path) -> None:
    # perfect effects in the complete quadruplets, and still no GO: 10/128 = 7.8 % > 5 %
    case = build_case(tmp_path, succeed_through={"A_FRESH_RETRY": 0, "B_SELF_REVISION": 0,
                                                "C_VERIFIER_REPAIR": 128},
                      censor={"A_FRESH_RETRY": set(range(1, 11))})
    result = analyze_case(tmp_path, case)
    assert result["data_guard_outcome_D_first"]["pass"] is True
    guard = result["censoring_guard"]
    assert guard["rates"]["A_FRESH_RETRY"] == 10 / 128
    assert guard["range"] > 0.05 and guard["pass"] is False
    assert result["outcome"]["classification"] == \
        Outcome.D_INCONCLUSIVE_BY_DIFFERENTIAL_CENSORING.value
    assert result["outcome"]["mechanism_label"] == Mechanism.NOT_EVALUABLE.value
    assert result["outcome"]["the_taxonomy_was_not_evaluated_when_a_guard_failed"] is True
    # infra stays missing data in the primary reading: 118 complete quadruplets, not 128 failures
    assert result["cohort"]["n_complete_quadruplets"] == 118
    assert result["primary_gates"]["C_minus_A"]["passes"] is None


def test_censoring_inside_the_frozen_maximum_does_not_end_the_study(tmp_path: Path) -> None:
    """6/128 = 4.69 pp of A censored and 3/128 of B is inside the frozen 0.05, and the taxonomy is
    then evaluated normally. The exact 0.05 boundary is checked in tests/test_v4_p001.py, where a
    range of 0.05 is constructed directly: 0.05 * 128 is not an integer, so no 128-theorem fixture
    can land on it."""
    case = build_case(tmp_path, succeed_through={"C_VERIFIER_REPAIR": 128,
                                                "D_MISMATCHED_DIAGNOSTIC": 128},
                      censor={"A_FRESH_RETRY": set(range(1, 7)),
                              "B_SELF_REVISION": set(range(1, 4))})
    result = analyze_case(tmp_path, case)
    assert result["censoring_guard"]["range"] == round(6 / 128, 6)
    assert result["censoring_guard"]["pass"] is True
    assert result["cohort"]["n_complete_quadruplets"] == 122
    assert result["outcome"]["classification"] == Outcome.A_VERIFIER_SPECIFIC_GO.value


# --- sensitivity, transitions and the guard order -------------------------------------------------

def test_infra_as_failure_sensitivity_is_separate_and_never_primary(tmp_path: Path) -> None:
    case = build_case(tmp_path, succeed_through={"A_FRESH_RETRY": 0, "B_SELF_REVISION": 0,
                                                "C_VERIFIER_REPAIR": 128},
                      censor={"A_FRESH_RETRY": set(range(1, 11))})
    result = analyze_case(tmp_path, case)
    sensitivity = result["sensitivity_infra_as_failure"]
    assert sensitivity["C_minus_A"]["pairs_completed_with_a_censored_arm"] == 10
    # primary: those 10 theorems are missing and are excluded from the comparison
    assert result["cohort"]["n_complete_quadruplets"] == 118
    assert result["outcome"]["classification"] == \
        Outcome.D_INCONCLUSIVE_BY_DIFFERENTIAL_CENSORING.value
    assert json.dumps(sensitivity).count("never redefines the primary endpoint") == 1


def test_transitions_are_reported_for_b_c_and_d_only(tmp_path: Path) -> None:
    case = build_case(tmp_path, succeed_through={"A_FRESH_RETRY": 128, "B_SELF_REVISION": 0,
                                                "C_VERIFIER_REPAIR": 64})
    result = analyze_case(tmp_path, case)
    assert sorted(result["transitions"]["by_arm"]) == sorted(A.TRANSITION_ARMS)
    assert "A_FRESH_RETRY" not in result["transitions"]["by_arm"]
    c = result["transitions"]["by_arm"]["C_VERIFIER_REPAIR"]["counts"]
    assert c["SUCCESS"] == 64
    assert sum(c.values()) == S.N_PRIMARY
    # the frozen taxonomy: a primary failure that fails again in a primary category is the same error
    assert c["SAME_ERROR_CATEGORY"] == 64       # the fixture repeats the same category on failure
    assert c["DIFFERENT_ERROR_CATEGORY"] == 0
    infra = result["transitions"]["by_arm"]["B_SELF_REVISION"]["counts"]["INFRA_MISSING"]
    assert infra == 0
    assert len(result["transitions"]["by_arm_and_original_category"]["C_VERIFIER_REPAIR"]) == 6


def test_descriptive_b_minus_a_is_reported_without_a_gate(tmp_path: Path) -> None:
    case = build_case(tmp_path, succeed_through={"A_FRESH_RETRY": 60, "B_SELF_REVISION": 90,
                                                "C_VERIFIER_REPAIR": 128,
                                                "D_MISMATCHED_DIAGNOSTIC": 95})
    result = analyze_case(tmp_path, case)
    descriptive = result["descriptive_B_minus_A"]
    assert descriptive["delta_pp"] == 23.44 and descriptive["threshold"] is None
    assert descriptive["evaluated"] is False and descriptive["passes"] is None


def test_guard_order_is_the_frozen_one(tmp_path: Path) -> None:
    case = build_case(tmp_path, succeed_through={arm: 128 for arm in S.ARM_ORDER})
    result = analyze_case(tmp_path, case)
    assert result["guard_order"] == ["provenance", "n_complete_guard",
                                     "differential_censoring_guard", "primary_gates",
                                     "specificity_analysis", "mechanism_and_descriptive"]
    assert result["provenance"]["status"] == "PASS"


def _tokens(node) -> str:
    """Every key and every string in a result tree, so the grep below cannot miss a nested metric."""
    if isinstance(node, dict):
        return " ".join(f"{key} {_tokens(value)}" for key, value in node.items())
    if isinstance(node, list):
        return " ".join(_tokens(item) for item in node)
    return node if isinstance(node, str) else ""


def test_the_analyzer_has_no_predictive_endpoint(tmp_path: Path) -> None:
    """Owner §L: no AUPRC and no predictive metric of any kind. The only place the ban may be named
    is the prohibited_readings list, which exists to state it."""
    case = build_case(tmp_path, succeed_through={arm: 64 for arm in S.ARM_ORDER})
    result = analyze_case(tmp_path, case)
    blob = _tokens({key: value for key, value in result.items()
                    if key != "prohibited_readings"}).lower()
    assert "auprc" in " ".join(result["prohibited_readings"]).lower()
    for forbidden in ("auprc", "average_precision", "auroc", "roc_auc", "brier", "enrichment",
                      "precision_recall", "predictive_score"):
        assert forbidden not in blob
    assert "auprc" not in _tokens(result["provenance"]).lower()


# --- the boundary is a boundary ------------------------------------------------------------------

def test_a_second_stage_candidate_at_the_boundary_is_fatal(tmp_path: Path) -> None:
    case = build_case(tmp_path, succeed_through={arm: 128 for arm in S.ARM_ORDER},
                      tamper=lambda plan, cohort, derangement: plan.update(
                          {"second_stage_candidates_generated": 1}))
    out_dir = case["out_dir"]
    plan_path = out_dir / S.SECOND_STAGE_PLAN_BASENAME
    payload = json.loads(plan_path.read_text())
    payload["second_stage_candidates_generated"] = 1
    # the tamper also invalidates the file hash, so the analyzer refuses on either ground
    S.write_json_atomic(plan_path, payload)
    with pytest.raises(S.FrozenViolation):
        analyze_case(tmp_path, case)


def test_a_raw_row_outside_the_frozen_plan_is_fatal(tmp_path: Path) -> None:
    case = build_case(tmp_path, succeed_through={arm: 128 for arm in S.ARM_ORDER})
    raw_path = case["out_dir"] / S.SECOND_STAGE_RAW_BASENAME
    rows = [json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()]
    rows[0]["seed"] = 1
    S.write_json_atomic(raw_path, {})          # placeholder to keep the path a file
    raw_path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    with pytest.raises(S.FrozenViolation, match="seed"):
        analyze_case(tmp_path, case)


def test_an_unfinished_group_is_fatal(tmp_path: Path) -> None:
    case = build_case(tmp_path, succeed_through={arm: 128 for arm in S.ARM_ORDER})
    raw_path = case["out_dir"] / S.SECOND_STAGE_RAW_BASENAME
    rows = [json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()]
    raw_path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows[:-1]))
    with pytest.raises(S.FrozenViolation, match="unfinished"):
        analyze_case(tmp_path, case)


def test_derangement_mapping_disagreement_is_fatal(tmp_path: Path) -> None:
    case = build_case(tmp_path, succeed_through={arm: 128 for arm in S.ARM_ORDER})
    der_path = case["out_dir"] / S.DERANGEMENT_BASENAME
    payload = json.loads(der_path.read_text())
    payload["derangement"]["mapping_sha256"] = "0" * 64
    S.write_json_atomic(der_path, payload)
    with pytest.raises(S.FrozenViolation):
        analyze_case(tmp_path, case)


# --- the CLI -------------------------------------------------------------------------------------

def test_check_only_writes_nothing(tmp_path: Path) -> None:
    case = build_case(tmp_path, succeed_through={arm: 128 for arm in S.ARM_ORDER})
    results = out_dir = case["out_dir"]
    assert A.main(["--out-dir", str(out_dir), "--check-only"]) == 0
    assert not (results / "V4-P001_results.json").exists()


def test_a_missing_raw_artifact_is_fatal(tmp_path: Path) -> None:
    (tmp_path / "v4_p001_rollout").mkdir(parents=True)
    assert A.main(["--out-dir", str(tmp_path / "v4_p001_rollout"), "--check-only"]) == 3


def test_the_synthetic_case_fixture_is_itself_consistent(tmp_path: Path) -> None:
    """Guard against a fixture that only 'passes' because the analyzer never looked at it."""
    case = build_case(tmp_path, succeed_through={arm: 128 for arm in S.ARM_ORDER})
    theorems = case["plan"]["theorems"]
    assert len(theorems) == S.N_PRIMARY
    assert all(sorted(t["arm_order"]) == sorted(S.ARM_ORDER) for t in theorems)
    assert len({tuple(t["arm_order"]) for t in theorems}) >= 1
    positions = Counter((arm, position) for t in theorems
                        for position, arm in enumerate(t["arm_order"]))
    assert set(positions.values()) == {S.N_PRIMARY // S.N_ARMS}
    rows = [json.loads(line) for line
            in (case["out_dir"] / S.SECOND_STAGE_RAW_BASENAME).read_text().splitlines() if line]
    assert len(rows) == S.SECOND_STAGE_CANDIDATES + 0
    assert set(Counter((r["formal_rank"], r["arm"]) for r in rows).values()) == {1}
    assert len(case["screening_rows"]) == S.N_PRIMARY
    result = analyze_case(tmp_path, case)
    assert result["provenance"]["status"] == "PASS" and result["provenance"]["n_failed"] == 0
