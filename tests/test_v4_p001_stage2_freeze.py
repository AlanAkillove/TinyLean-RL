"""V4-P001 owner §11 Commit B -- the read-only Stage-2 structural freeze (owner §13).

`scripts/v4_p001_stage2_freeze.py` runs between "second-stage execution ended" and "analysis
begins": it reads the raw artifact and its journals, checks every structural claim the frozen design
makes about them, and writes one record. It computes no endpoint — no success rate per arm, no
C-A/C-B/C-D — which is asserted here as a property of the record's shape.

These tests drive it with a *synthetic* second-stage artifact built on the frozen fixture boundary
(imported from `tests/test_v4_p001_rollout.py`): the runner's own row builder writes 512 explicitly
censored candidates for the 128 frozen theorems, plus a summary and an empty recovery journal. No
formal run is needed, no model is loaded and no verifier is contacted.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import unittest.mock
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "scripts", Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import test_v4_p001_rollout as T
import v4_p001_rollout as R
import v4_p001_spec as S

FREEZE_SCRIPT = ROOT / "scripts/v4_p001_stage2_freeze.py"


@pytest.fixture(scope="module")
def sealed_boundary(tmp_path_factory) -> Path:
    """The frozen fixture boundary, built once: synthetic screening rows, the real freeze stage."""
    out_dir = tmp_path_factory.mktemp("stage2_freeze") / "rollout"
    out_dir.mkdir()
    with unittest.mock.patch.object(R.S, "collect_env", lambda: dict(T.CLEAN_ENV)):
        assert T.write_screening(out_dir) == S.N_PRIMARY
        assert R.main(["--stage", "freeze", "--out-dir", str(out_dir)]) == 0
    return out_dir


@pytest.fixture
def boundary(tmp_path: Path, sealed_boundary: Path) -> Path:
    out_dir = tmp_path / "rollout"
    shutil.copytree(sealed_boundary, out_dir)
    return out_dir

#: The record's whole public shape. §13 pins the tool to structure and provenance only, so a key
#: that is not on this list is a scope violation, and a missing key is a lost fact.
RECORD_KEYS = {
    "artifact_type", "experiment_id", "created_at_utc", "validated_commit", "raw_artifact",
    "candidates", "theorems", "recovery", "timings", "hashes", "structural_checks", "n_checks",
    "n_failed", "status", "failing_checks", "note",
}


def write_synthetic_second_stage(out_dir: Path, *, tamper: str | None = None) -> int:
    """The 512 candidates a complete execution would leave, as legal censored rows."""
    plan = T.plan_artifact(out_dir)
    rows_by_statement = T.screening_by_statement(out_dir)
    env = T.formal_env()
    rows = [R.censored_repair_row(plan_row=theorem,
                                  screening_row=rows_by_statement[theorem["statement_id"]],
                                  arm=arm, position=position, env=env,
                                  run_id="stage2-freeze-fixture",
                                  reason="written by the Stage-2 freeze fixture")
            for theorem in plan["theorems"]
            for position, arm in enumerate(theorem["arm_order"], start=1)]
    if tamper == "prompt_sha256":
        rows[3]["prompt_sha256"] = "0" * 64
    S.append_rows_durable(out_dir / S.SECOND_STAGE_RAW_BASENAME, rows)
    summary = {
        "experiment_id": S.EXPERIMENT_ID, "stage": "second", "mode": "FORMAL",
        "run_id": "20260925T000000Z",
        "frozen_settings_sha256": S.sha(S.FROZEN_SETTINGS),
        "candidates_generated": 0, "candidates_censored": len(rows),
        "generation_seconds": 0.0, "torch_peak_allocated_gb": 0.0,
        "resume_compaction": {"present": False},
        "verifier_infrastructure": {"recoveries_attempted": 0, "recoveries_succeeded": 0},
    }
    S.write_json_atomic(out_dir / S.SUMMARY_BASENAME, summary)
    (out_dir / S.RECOVERY_LOG_BASENAME).write_text("", encoding="utf-8")
    return len(rows)


def run_freeze(out_dir: Path, out_path: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(FREEZE_SCRIPT),
         "--raw", str(out_dir / S.SECOND_STAGE_RAW_BASENAME),
         "--summary", str(out_dir / S.SUMMARY_BASENAME),
         "--recovery-log", str(out_dir / S.RECOVERY_LOG_BASENAME),
         "--events", str(out_dir / S.VERIFIER_EVENTS_BASENAME),
         "--cohort", str(out_dir / S.PRIMARY_COHORT_BASENAME),
         "--derangement", str(out_dir / S.DERANGEMENT_BASENAME),
         "--plan", str(out_dir / S.SECOND_STAGE_PLAN_BASENAME),
         "--gpu-samples", str(out_dir / "absent_gpu_samples.csv"),
         "--out", str(out_path)],
        cwd=ROOT, capture_output=True, text=True, check=False)


def freeze_record(out_dir: Path, tmp_path: Path) -> tuple[subprocess.CompletedProcess, dict]:
    out_path = tmp_path / "V4-P001_stage2_freeze.json"
    proc = run_freeze(out_dir, out_path)
    return proc, json.loads(out_path.read_text(encoding="utf-8"))


def test_the_stage_two_freeze_records_structure_without_computing_an_endpoint(
        boundary: Path, tmp_path: Path) -> None:
    assert write_synthetic_second_stage(boundary) == S.SECOND_STAGE_CANDIDATES
    proc, record = freeze_record(boundary, tmp_path)

    assert proc.returncode == 0, proc.stderr
    assert record["status"] == "FROZEN" and record["n_failed"] == 0
    assert record["failing_checks"] == [] and record["n_checks"] == len(record["structural_checks"])
    assert set(record) == RECORD_KEYS
    # §13: structure and provenance only -- the analyzer owns every endpoint
    assert "analyzer" in record["note"] and "endpoint" in record["note"]
    for forbidden in ("C_A", "C_B", "C_D", "success_rate", "p_value", "test_statistic"):
        assert forbidden not in json.dumps(record)

    # the raw artifact: 512 candidates, 128 structurally complete quadruplets, the frozen plan bound
    assert record["raw_artifact"]["rows"] == S.SECOND_STAGE_CANDIDATES
    assert record["raw_artifact"]["sha256"] == S.sha256_file(
        boundary / S.SECOND_STAGE_RAW_BASENAME)
    assert record["theorems"]["structurally_complete"] == S.N_PRIMARY
    assert record["theorems"]["ranks_not_on_disk"] == []
    assert record["candidates"]["arm_counts"] == {arm: S.N_PRIMARY for arm in S.ARM_ORDER}
    assert record["candidates"]["infrastructure_missing_by_arm"][S.ARM_ORDER[0]][
        "explicitly_censored"] == S.N_PRIMARY

    # the frozen identities the record is bound to
    plan = T.plan_artifact(boundary)
    cohort = json.loads((boundary / S.PRIMARY_COHORT_BASENAME).read_text(encoding="utf-8"))
    assert record["hashes"]["plan_sha256"] == plan["plan_sha256"]
    assert record["hashes"]["arm_schedule_hash"] == plan["arm_schedule_hash"]
    assert record["hashes"]["second_stage_seed_hash"] == plan["second_stage_seed_hash"]
    assert record["hashes"]["cohort_content_sha256"] == cohort["content_sha256"]
    assert record["hashes"]["frozen_settings_sha256"] == S.sha(S.FROZEN_SETTINGS)
    assert record["recovery"]["stage2_attempts"] == 0
    assert record["recovery"]["journal_total_entries"] == 0


def test_the_stage_two_freeze_refuses_a_candidate_whose_frozen_prompt_moved(
        boundary: Path, tmp_path: Path) -> None:
    write_synthetic_second_stage(boundary, tamper="prompt_sha256")
    proc, record = freeze_record(boundary, tmp_path)

    assert proc.returncode == 1, proc.stdout
    assert record["status"] == "FAILED"
    assert record["failing_checks"] == ["prompt_and_context_hashes_match_the_frozen_plan"]
    assert record["n_checks"] > 1 and record["n_failed"] == 1
    assert record["raw_artifact"]["rows"] == S.SECOND_STAGE_CANDIDATES
