"""V4-P001 runner — the Stage-1/Stage-2 freeze boundary (owner §12; Amendment A §13 and §16).

These fixtures fabricate the *screening outcome* — the one input only a formal run on fly122 can
produce — and nothing else. The pool, the screening order, the seeds, the arm schedule, the prompts,
the tokenizer and the model are the committed frozen design, and the runner's own `freeze` and
`validate` stages run unmodified on top of it, so a check that fails here would fail in the formal
run too. The property under test is the owner's §12: the process that discovers the cohort stops
*before* second-stage generation, and the boundary validation refuses a boundary that already
contains one.
"""

from __future__ import annotations

import functools
import json
import shutil
import sys
import unittest.mock
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import v4_p001_rollout as R
import v4_p001_spec as S

from tinylean_rl.evaluation.v4_seeds import first_stage_seed

#: A clean checkout on a node that is not the formal GPU host: freeze and validate must accept it
#: (only the generating stages require fly122), and the frozen design must be verified from disk.
CLEAN_ENV = {
    "hostname": "fly90-test", "ips": "10.3.25.90", "git_revision": "0" * 40, "branch": S.BRANCH,
    "status_porcelain": "", "gpu_names": "", "gpu_compute_apps": "", "prereg_is_ancestor": True,
}

#: One plausible normalized diagnostic per primary error category; the runner reads only the
#: category, the diagnostic text, its hash, its token count and the fit of the Arm-C context.
CATEGORIES = (
    ("unsolved_goals", "unsolved goals\n\nn m : Nat\n⊢ n + m = m + n"),
    ("elaboration_type_mismatch", "type mismatch\n  hf : ∀ x, x ∈ ∅\n  ⊢ ∃ x, x ∈ ∅"),
    ("unknown_identifier", "unknown identifier 'foo'"),
    ("tactic_failure", "tactic 'omega' failed"),
    ("typeclass_synthesis", "failed to synthesize\n  OfNat α 0"),
    ("other_semantic_lean_failure", "cannot find a contradiction for the given hypotheses"),
)


@functools.lru_cache(maxsize=1)
def design() -> tuple[S.Frozen, dict]:
    """The frozen design and the canonical prompt surface, loaded once per session."""
    frozen = S.load_frozen(verify_files=True)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(ROOT / S.MODEL["dir"], trust_remote_code=True,
                                              local_files_only=True)
    return frozen, R.load_surface(tokenizer)


def screening_row(member: dict, surface_row: dict, index: int) -> dict:
    category, diagnostic = CATEGORIES[index % len(CATEGORIES)]
    proof = f"by\n  -- attempt {index}\n  simp [Nat.add_comm, Nat.add_assoc]"
    completion = f"{proof}\ntrailing prose"
    row = {
        "schema_version": S.SCREENING_SCHEMA_VERSION, "run_id": "test",
        "experiment_id": S.EXPERIMENT_ID,
        "screening_rank": member["screening_rank"], "formal_rank": index + 1,
        "statement_id": member["statement_id"], "component_id": member["component_id"],
        "name": member["name"], "source": member["source"], "tier": member["tier"],
        "classes": member["classes"], "prompt_tokens": member["prompt_tokens"],
        "prompt_sha256": surface_row["prompt_sha256"],
        "prompt_token_count": surface_row["prompt_token_count"],
        "seed": first_stage_seed(member["screening_rank"]),
        "model_sha256": S.MODEL["weights_sha256"], "model_revision": S.MODEL["revision"],
        "completion_text": completion, "completion_sha256": S.sha256_text(completion),
        "generated_tokens": 40, "truncated": False, "format_ok": True, "has_lean_block": True,
        "extracted_proof": proof, "extracted_proof_sha256": S.sha256_text(proof),
        "extracted_proof_present": True, "extracted_proof_tokens": len(proof.split()),
        "diagnostic_text": diagnostic, "diagnostic_sha256": S.sha256_text(diagnostic),
        "diagnostic_tokens": len(diagnostic.split()), "diagnostic_source": "verify_item",
        "truncated_diagnostic": False, "context_fits": True,
        "context_tokens_with_response": member["prompt_tokens"] + S.MAX_RESPONSE_TOKENS,
        "verified": False, "score": 0, "acc": 0, "verify_status": "lean_error",
        "lean_message": diagnostic, "error_category": category,
        "screening_status": S.PRIMARY_SEMANTIC_FAILURE, "primary_eligible": True,
        "generation_time": 1.0, "verification_time": None, "host": "test", "gpu": "",
        "created_at": "2026-09-25T00:00:00+00:00",
    }
    S.validate_screening_row(row)
    return row


def write_screening(out_dir: Path) -> int:
    """The 128 primary semantic failures the formal screen would have to produce, in frozen order."""
    frozen, surface = design()
    rows = [screening_row(member, surface[member["statement_id"]], index)
            for index, member in enumerate(frozen.screening_theorems[:S.N_PRIMARY])]
    S.append_rows_durable(out_dir / S.SCREENING_RAW_BASENAME, rows)
    return len(rows)


@pytest.fixture(scope="module")
def boundary_template(tmp_path_factory) -> Path:
    """A written, sealed Stage-1/Stage-2 boundary produced by the runner itself, once per module.

    The freeze stage loads the tokenizer, reads the pinned parquet and renders all 512 prompts, so
    it is built once and copied per test rather than re-run twelve times.
    """
    out_dir = tmp_path_factory.mktemp("boundary") / "rollout"
    out_dir.mkdir()
    with unittest.mock.patch.object(R.S, "collect_env", lambda: dict(CLEAN_ENV)):
        assert write_screening(out_dir) == S.N_PRIMARY
        assert R.main(["--stage", "freeze", "--out-dir", str(out_dir)]) == 0
    return out_dir


@pytest.fixture
def boundary(tmp_path: Path, boundary_template: Path) -> Path:
    out_dir = tmp_path / "rollout"
    shutil.copytree(boundary_template, out_dir)
    return out_dir


def validate(out_dir: Path, *extra: str) -> int:
    with unittest.mock.patch.object(R.S, "collect_env", lambda: dict(CLEAN_ENV)):
        return R.main(["--stage", "validate", *extra, "--out-dir", str(out_dir)])


def boundary_report(out_dir: Path) -> dict:
    return json.loads((out_dir / S.BOUNDARY_VALIDATION_BASENAME).read_text(encoding="utf-8"))


def checks(report: dict) -> dict:
    return {entry["check"]: entry["pass"] for entry in report["checks"]}


# --- the interlock and the freeze stage -----------------------------------------------------------

def test_a_formal_stage_is_refused_without_the_owner_launch_flag(tmp_path: Path) -> None:
    with unittest.mock.patch.object(R.S, "collect_env", lambda: dict(CLEAN_ENV)):
        for stage in ("screening", "freeze", "second"):
            assert R.main(["--stage", stage, "--out-dir", str(tmp_path / stage)]) == 3
    assert list(tmp_path.iterdir()) == []


def test_the_freeze_stage_writes_three_sealed_artifacts_and_stops(boundary: Path) -> None:
    written = {"primary_cohort": S.PRIMARY_COHORT_BASENAME,
               "derangement": S.DERANGEMENT_BASENAME, "plan": S.SECOND_STAGE_PLAN_BASENAME}
    for name, basename in written.items():
        payload = json.loads((boundary / basename).read_text(encoding="utf-8"))
        sealed = {k: v for k, v in payload.items() if k != "content_sha256"}
        assert payload["content_sha256"] == S.sha(sealed), f"{name} is not sealed"
        # the boundary itself: no second-stage candidate exists when it is written (owner §12)
        assert payload["second_stage_candidates_generated"] == 0
    # the process that discovered the cohort wrote no second-stage artifact of any kind
    for absent in (S.SECOND_STAGE_RAW_BASENAME, S.SUMMARY_BASENAME):
        assert not (boundary / absent).exists()
    # a second freeze without --resume refuses to overwrite a sealed boundary
    with unittest.mock.patch.object(R.S, "collect_env", lambda: dict(CLEAN_ENV)):
        assert R.main(["--stage", "freeze", "--out-dir", str(boundary)]) == 3


def test_the_freeze_stage_recomputes_the_frozen_seed_and_schedule_hashes(boundary: Path) -> None:
    plan = json.loads((boundary / S.SECOND_STAGE_PLAN_BASENAME).read_text(encoding="utf-8"))
    frozen, _surface = design()
    assert plan["second_stage_seed_hash"] == S.PAIRED_SEED_HASH == frozen.seeds["paired_seed_hash"]
    assert plan["arm_schedule_hash"] == frozen.schedule["schedule_hash"]
    theorems = plan["theorems"]
    assert [t["formal_rank"] for t in theorems] == list(range(1, S.N_PRIMARY + 1))
    assert [t["seed"] for t in theorems] == frozen.seeds["paired_seeds_by_rank"]
    assert [list(t["arm_order"]) for t in theorems] == [row["order"]
                                                       for row in frozen.schedule["schedule"]]


# --- the boundary validation (owner §12) ----------------------------------------------------------

def test_the_boundary_validation_passes_on_a_fresh_boundary(boundary: Path) -> None:
    assert validate(boundary) == 0
    report = boundary_report(boundary)
    assert report["status"] == "PASS" and report["n_failed"] == 0
    assert report["second_stage_candidates_generated"] == 0
    result = checks(report)
    # every §12 ingredient: cohort membership, error taxonomy, prompt fit, the derangement, the
    # seed set, the arm schedule and the sealed reserve
    for required in ("boundary_artifacts_present", "self_hash::primary_cohort",
                     "self_hash::diagnostic_derangement", "self_hash::second_stage_plan",
                     "second_stage_candidates_generated_is_zero", "screening_raw_hash_matches",
                     "cohort_is_exactly_128", "cohort_membership_matches_the_artifact",
                     "cohort_categories_are_primary", "derangement_recomputes",
                     "derangement_no_self_diagnostic", "second_stage_plan_recomputes",
                     "arm_a_is_the_screening_prompt", "arm_d_uses_the_derangement_donor",
                     "context_fits_all_arms", "second_stage_seeds_are_the_frozen_stream",
                     "second_stage_seeds_are_unique_and_disjoint_from_screening",
                     "arm_order_is_the_frozen_schedule", "arm_schedule_is_position_balanced",
                     "plan_candidate_count", "hash_chain", "sealed_reserve_untouched"):
        assert result.get(required) is True, f"{required}: {result.get(required)}"


def test_the_boundary_validation_fails_if_a_second_stage_candidate_exists(boundary: Path) -> None:
    """The owner's §12 row of the fixture table, on the *validation command*, not on a reader."""
    plan_path = boundary / S.SECOND_STAGE_PLAN_BASENAME
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    payload["second_stage_candidates_generated"] = 1
    # re-seal, so the single failing check is the boundary claim itself and not a broken hash
    payload["content_sha256"] = S.sha({k: v for k, v in payload.items()
                                       if k != "content_sha256"})
    S.write_json_atomic(plan_path, payload)

    assert validate(boundary) == 3
    report = boundary_report(boundary)
    assert report["status"] == "FAIL"
    failed = [entry["check"] for entry in report["checks"] if not entry["pass"]]
    assert failed == ["second_stage_candidates_generated_is_zero"]


def test_the_boundary_validation_reports_a_missing_boundary_without_inventing_one(
        tmp_path: Path) -> None:
    out_dir = tmp_path / "empty"
    out_dir.mkdir()
    assert validate(out_dir) == 3
    report = boundary_report(out_dir)
    assert report["status"] == "FAIL"
    # it refuses to generate anything, and says what to run instead
    assert report["second_stage_candidates_generated"] == 0
    assert checks(report) == {"boundary_artifacts_present": False}
    assert "run --stage screening" in report["note"]
    assert not (out_dir / S.SECOND_STAGE_RAW_BASENAME).exists()


def test_a_dry_run_boundary_validation_writes_nothing(boundary: Path) -> None:
    report_path = boundary / S.BOUNDARY_VALIDATION_BASENAME
    assert validate(boundary) == 0
    assert report_path.exists()
    report_path.unlink()
    assert validate(boundary, "--dry-run") == 0
    assert not report_path.exists()


@pytest.mark.parametrize("tamper", ["seed", "arm_order", "prompt", "donor"])
def test_a_tampered_plan_is_caught_by_the_boundary_validation(tamper: str,
                                                             boundary: Path) -> None:
    plan_path = boundary / S.SECOND_STAGE_PLAN_BASENAME
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    theorem = payload["theorems"][0]
    if tamper == "seed":
        theorem["seed"] += 1
    elif tamper == "arm_order":
        theorem["arm_order"] = list(reversed(theorem["arm_order"]))
    elif tamper == "prompt":
        theorem["arms"][S.ARM_ORDER[0]]["prompt_sha256"] = "0" * 64
    else:
        theorem["donor_statement_id"] = theorem["statement_id"]
    # re-seal the payload *and* its own digest of the plan: the validation must recompute the plan
    # from the frozen design rather than trust the digests the artifact carries
    payload["plan_sha256"] = S.sha(payload["theorems"])
    payload["content_sha256"] = S.sha({k: v for k, v in payload.items()
                                       if k != "content_sha256"})
    S.write_json_atomic(plan_path, payload)

    assert validate(boundary) == 3
    report = boundary_report(boundary)
    assert report["status"] == "FAIL"
    assert checks(report)["second_stage_plan_recomputes"] is False
