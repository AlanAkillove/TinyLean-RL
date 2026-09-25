"""V4-P001 runner — the Stage-1/Stage-2 freeze boundary (owner §12; Amendment A §13 and §16).

These fixtures fabricate the *screening outcome* — the one input only a formal run on fly122 can
produce — and nothing else. The pool, the screening order, the seeds, the arm schedule, the prompts,
the tokenizer and the model are the committed frozen design, and the runner's own `freeze` and
`validate` stages run unmodified on top of it, so a check that fails here would fail in the formal
run too. The property under test is the owner's §12: the process that discovers the cohort stops
*before* second-stage generation, and the boundary validation refuses a boundary that already
contains one.

The last section is Amendment B (owner pre-outcome execution-code amendment, Stage-2 launch-0
structural abort): the formal statement of a candidate is read from the pinned parquet surface, never
from the screening-row schema, the identity guard fails closed, every row access is audited against
its own schema, and one synthetic theorem runs all four arms through the *real* Stage-2 path with a
fake model and a fake verifier (0 real generations, 0 formal verifier calls).
"""

from __future__ import annotations

import ast
import contextlib
import functools
import json
import shutil
import sys
import types
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
from tinylean_rl.inference.extract import extract_proof
from tinylean_rl.verifier.policy import Classified, VerifyOutcome

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


# --- Amendment B: the Stage-2 candidate assembly (owner §2-§7) ------------------------------------

#: A fly122-shaped environment, so the *formal* Stage-2 path — the one that crashed on launch-0 —
#: passes the runner's own `S.check_environment` gate. Everything past the gate is a fake; the gate
#: is not.
FORMAL_ENV = dict(CLEAN_ENV, hostname="fly122", ips="10.3.25.122",
                  gpu_names="NVIDIA GeForce RTX 3080")


def formal_env() -> dict:
    return dict(FORMAL_ENV, model_sha256=S.MODEL["weights_sha256"],
                model_revision=S.MODEL["revision"])


def plan_artifact(out_dir: Path) -> dict:
    return json.loads((out_dir / S.SECOND_STAGE_PLAN_BASENAME).read_text(encoding="utf-8"))


def screening_by_statement(out_dir: Path) -> dict:
    return {row["statement_id"]: row
            for row in R.read_screening(out_dir / S.SCREENING_RAW_BASENAME)}


def rank_theorem(out_dir: Path, rank: int) -> dict:
    return next(t for t in plan_artifact(out_dir)["theorems"] if t["formal_rank"] == rank)


def synthetic_completion(prompt: str) -> str:
    """A deterministic stand-in for a model completion, tied to the prompt it answers."""
    return (f"```lean\nby\n  -- synthetic proof of {S.sha256_text(prompt)[:16]}\n"
            "  simp [Nat.add_comm]\n```\ntrailing prose that is not Lean")


class FakeCompletion:
    """The generation surface of one candidate: exactly the attributes `repair_one` reads."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.token_ids = list(range(7, 41))
        self.finish_reason = "stop"


class FakeLLM:
    """A vLLM stand-in that echoes its prompt into the completion: no weights, no GPU, no model."""

    def __init__(self, **_kwargs) -> None:
        self.calls: list[list[str]] = []
        self.params: list[list[dict]] = []

    def generate(self, prompts, params):
        prompts, params = list(prompts), list(params)
        self.calls.append(prompts)
        self.params.append(params)
        return [unittest.mock.Mock(outputs=[FakeCompletion(synthetic_completion(prompt))])
                for prompt in prompts]


class FakeSession:
    """The verifier surface of these fixtures: it records, it never contacts anything.

    The real policy is never constructed (`R.make_session` is patched), so a fixture run makes 0
    formal verifier calls by construction. `verified` names the arms whose synthetic verdict is
    `verified`; every other arm is a conclusive Lean error, which keeps the C′ recovery sequence — a
    real container restart — out of the fixture.
    """

    def __init__(self, *, verified: tuple[str, ...] = ()) -> None:
        self.verified = set(verified)
        self.calls: list[tuple[str, str]] = []
        self.events: list = []

    def verify(self, proofs, custom_ids):
        out = []
        for proof, custom_id in zip(proofs, custom_ids):
            self.calls.append((custom_id, proof))
            arm = custom_id.split("-", 1)[1]
            out.append(Classified(VerifyOutcome.VERIFIED, "proof verified")
                       if arm in self.verified else
                       Classified(VerifyOutcome.LEAN_ERROR, f"{arm}: unsolved goals"))
        return out


class FakeRecovery:
    """The C′ recovery surface, inert: any recovery in these fixtures would be a bug."""

    endpoint = S.VERIFIER_INFRA["endpoint"]
    container = "fixture"
    image = "fixture"
    max_recoveries = S.VERIFIER_INFRA["max_recoveries_per_run"]
    max_per_theorem = S.VERIFIER_INFRA["max_recoveries_per_theorem"]

    def __init__(self, out_dir: Path) -> None:
        self.log_path = Path(out_dir) / S.RECOVERY_LOG_BASENAME
        self.n_attempted = 0
        self.n_succeeded = 0
        self.recoveries_by_theorem: dict = {}
        self.events: list = []
        self.identity_calls = 0
        self.recover_calls: list = []

    def verify_identity(self) -> dict:
        self.identity_calls += 1
        return {"container_id": self.container, "image": self.image, "published": self.endpoint,
                "env": {"LEAN_SERVER_MAX_REPLS": "1"}}

    def theorem_exhausted(self, rank: int) -> bool:
        return False

    def recover(self, trigger: dict, rank: int) -> dict:
        self.recover_calls.append((trigger, rank))
        return {"ok": False, "reason": "these fixtures must never need a recovery"}


def fake_torch() -> types.ModuleType:
    module = types.ModuleType("torch")
    module.cuda = types.SimpleNamespace(reset_peak_memory_stats=lambda: None,
                                        max_memory_allocated=lambda: 6 * 2**30)
    return module


def fake_vllm(llm: FakeLLM) -> types.ModuleType:
    module = types.ModuleType("vllm")
    module.__version__ = "fixture"
    module.LLM = lambda **_kwargs: llm
    module.SamplingParams = lambda **kwargs: dict(kwargs)
    return module


@contextlib.contextmanager
def second_stage_fakes(*, out_dir: Path, env: dict, llm: FakeLLM | None = None,
                       session: FakeSession | None = None, recovery: FakeRecovery | None = None):
    """The fake GPU and the fake verifier around the *real* Stage-2 code path.

    `transformers` and the real `torch` must already be imported (every caller reaches this through
    `design()`): a fake `torch` entering `sys.modules` before transformers' package discovery breaks
    it. The assertion makes that coupling explicit instead of accidental.
    """
    assert "transformers" in sys.modules and "torch" in sys.modules
    llm = llm if llm is not None else FakeLLM()
    session = session if session is not None else FakeSession()
    recovery = recovery if recovery is not None else FakeRecovery(out_dir)
    with contextlib.ExitStack() as stack:
        stack.enter_context(unittest.mock.patch.dict(
            sys.modules, {"torch": fake_torch(), "vllm": fake_vllm(llm)}))
        stack.enter_context(unittest.mock.patch.object(R.S, "collect_env", lambda: dict(env)))
        stack.enter_context(unittest.mock.patch.object(R, "make_session", lambda: session))
        stack.enter_context(unittest.mock.patch.object(R, "VerifierRecovery",
                                                       lambda _out_dir: recovery))
        stack.enter_context(unittest.mock.patch.object(R, "check_verifier", lambda _session: {
            "health_status": 200, "endpoint": S.VERIFIER_INFRA["endpoint"], "health_url": "fixture",
            "canary": {"verified": True, "status": "verified", "seconds": 0.01,
                       "uses_formal_theorem": False}}))
        # the frozen policy must never be reached: this mock is the tripwire (call_count == 0)
        real_policy = stack.enter_context(
            unittest.mock.patch.object(R.VerificationSession, "verify", autospec=True))
        yield {"llm": llm, "session": session, "recovery": recovery, "real_policy": real_policy}


def test_repair_one_sends_exactly_the_assembled_source_of_the_pinned_statement(
        boundary: Path) -> None:
    """Owner §5 R1: a known formal statement, a synthetic completion, a recording verifier."""
    _frozen, surface = design()
    theorem = rank_theorem(boundary, 7)
    screening_row = screening_by_statement(boundary)[theorem["statement_id"]]
    formal = surface[theorem["statement_id"]]["formal_statement"]
    arm = theorem["arm_order"][0]
    completion = FakeCompletion(synthetic_completion("R1 does not need the frozen arm prompt"))
    session = FakeSession(verified=(arm,))

    repaired, abort_reason, exhausted = R.repair_one(
        None, session, FakeRecovery(boundary), R.ResultItemRecorder(), plan_row=theorem,
        screening_row=screening_row, formal_statement=formal, arm=arm, position=1,
        completion=completion, generation_seconds=1.25, env=formal_env(), run_id="fixture")

    assert (abort_reason, exhausted) == (None, False)
    assert session.calls == [(f"rank{theorem['formal_rank']}-{arm}",
                              R.complete_verifier_code(formal, extract_proof(completion.text)))]
    assert repaired["verify_status"] == VerifyOutcome.VERIFIED.value
    assert repaired["score"] == 1 and repaired["success"] is True
    assert repaired["format_ok"] is True and repaired["has_lean_block"] is True
    assert repaired["extracted_proof_present"] is True
    S.validate_repair_row(repaired)


def test_a_frozen_screening_row_carries_no_formal_statement_and_repair_one_needs_none(
        boundary: Path) -> None:
    """Owner §5 R2: nothing may read a formal statement off the screening-row schema."""
    _frozen, surface = design()
    theorem = rank_theorem(boundary, 11)
    row = screening_by_statement(boundary)[theorem["statement_id"]]
    assert set(row) == set(S.SCREENING_FIELDS)
    assert "formal_statement" not in row
    formal = surface[theorem["statement_id"]]["formal_statement"]
    arm = theorem["arm_order"][0]
    completion = FakeCompletion(synthetic_completion(f"R2 {theorem['statement_id']}"))
    session = FakeSession()

    # a screening row that carries a formal statement anyway must change nothing: it is never read
    poisoned = dict(row, formal_statement="theorem backdoor : False := by sorry")
    for candidate_row in (row, poisoned):
        repaired, _abort, _exhausted = R.repair_one(
            None, session, FakeRecovery(boundary), R.ResultItemRecorder(), plan_row=theorem,
            screening_row=candidate_row, formal_statement=formal, arm=arm, position=1,
            completion=completion, generation_seconds=0.5, env=formal_env(), run_id="fixture")
        S.validate_repair_row(repaired)

    expected = (f"rank{theorem['formal_rank']}-{arm}",
                R.complete_verifier_code(formal, extract_proof(completion.text)))
    assert session.calls == [expected, expected]


def test_the_formal_statement_guard_fails_closed_on_every_identity_mismatch(
        boundary: Path) -> None:
    """Owner §3/§5 R3: missing, empty or unbound statements raise before anything is assembled."""
    _frozen, surface = design()
    theorem = rank_theorem(boundary, S.N_PRIMARY)
    sid = theorem["statement_id"]
    row = surface[sid]
    assert R.frozen_formal_statement(surface, theorem) == row["formal_statement"]
    assert row["prompt_sha256"] == theorem["arms"][S.ARM_ORDER[0]]["prompt_sha256"]

    broken_surfaces = {
        "not in the pinned surface": {k: v for k, v in surface.items() if k != sid},
        "empty statement": {**surface, sid: {**row, "formal_statement": ""}},
        "whitespace-only statement": {**surface, sid: {**row, "formal_statement": " \n "}},
        "a different first-attempt prompt": {**surface, sid: {**row, "prompt_sha256": "0" * 64}},
    }
    for poisoned in broken_surfaces.values():
        with pytest.raises(S.FrozenViolation):
            R.frozen_formal_statement(poisoned, theorem)
    broken_plans = {"no arm rows": {**theorem, "arms": {}},
                    "arm A without a prompt hash": {
                        **theorem, "arms": {S.ARM_ORDER[0]: {"prompt_tokens": 1}}}}
    for poisoned in broken_plans.values():
        with pytest.raises(S.FrozenViolation):
            R.frozen_formal_statement(surface, poisoned)


def test_a_missing_pinned_formal_statement_fails_closed_before_any_verification(
        boundary: Path, capsys) -> None:
    """Owner §3 end to end: the run stops before a single candidate of the cohort is generated."""
    _frozen, surface = design()
    theorem = rank_theorem(boundary, S.N_PRIMARY)
    poisoned = {sid: dict(row) for sid, row in surface.items()}
    poisoned[theorem["statement_id"]]["formal_statement"] = ""
    session = FakeSession()

    with unittest.mock.patch.object(R, "load_surface", lambda _tokenizer: poisoned), \
            unittest.mock.patch.object(R, "make_session", lambda: session), \
            second_stage_fakes(out_dir=boundary, env=formal_env()) as fakes:
        assert R.main(["--stage", "second", "--i-have-owner-launch-authorization",
                       "--out-dir", str(boundary)]) == 3

    stderr = capsys.readouterr().err
    assert "ABORT BEFORE GENERATION" in stderr and "Amendment B §3" in stderr
    assert session.calls == [] and fakes["real_policy"].call_count == 0
    assert not (boundary / S.SECOND_STAGE_RAW_BASENAME).exists()
    assert not (boundary / S.SUMMARY_BASENAME).exists()


def test_the_second_stage_dry_run_validates_the_whole_candidate_assembly(boundary: Path,
                                                                        capsys) -> None:
    """Owner §7: the dry run reaches every schema and plumbing check the formal path needs.

    One shared implementation performs the validation (`validate_candidate_assembly`), so the dry run
    cannot diverge from the formal run: this asserts the 128-theorem evidence it must produce.
    """
    assert not (boundary / S.SECOND_STAGE_RAW_BASENAME).exists()
    with unittest.mock.patch.object(R.S, "collect_env", lambda: dict(CLEAN_ENV)):
        assert R.main(["--stage", "second", "--dry-run", "--out-dir", str(boundary)]) == 0
    out = capsys.readouterr().out

    marker = f"[{S.EXPERIMENT_ID}] DRY RUN COMPLETE"
    lines = out[:out.index(marker)].splitlines()
    start = max((i for i, line in enumerate(lines) if line.strip() == "{"), default=0)
    standalone = json.loads("\n".join(lines[start:]))
    assembly, counts = standalone["candidate_assembly"], standalone["candidate_assembly"]["counts"]

    assert assembly["theorems"] == S.N_PRIMARY and assembly["n_failed"] == 0
    for name in ("surface_statements_available", "formal_statements_nonempty",
                 "formal_statements_bound_to_the_frozen_plan", "failed_proofs_available",
                 "own_diagnostics_available", "donor_diagnostics_available",
                 "screening_identity_fields_available", "assembly_probes_passed",
                 "paired_seeds_valid", "arm_orders_valid", "arm_plan_rows_complete"):
        assert counts[name] == S.N_PRIMARY, f"{name}: {counts[name]}"
    assert counts["arm_prompt_hashes_matched"] == S.SECOND_STAGE_CANDIDATES
    assert assembly["generations"] == 0 and assembly["formal_verifier_calls"] == 0
    assert standalone["theorems_to_run"] == list(range(1, S.N_PRIMARY + 1))
    assert f"theorems_planned={S.N_PRIMARY}" in out
    assert f"candidates_planned={S.SECOND_STAGE_CANDIDATES}" in out
    assert f"surface_formal_statements={S.N_PRIMARY}" in out
    assert "candidate_assembly_validated=True" in out
    # a dry run writes no raw artifact and no summary
    for absent in (S.SECOND_STAGE_RAW_BASENAME, S.SUMMARY_BASENAME):
        assert not (boundary / absent).exists()


def test_all_four_arms_of_one_theorem_run_through_the_real_second_stage(boundary: Path) -> None:
    """Owner §6: the real Stage-2 path past the launch-0 crash, with 0 generations and 0 verifications.

    Ranks 1..127 are pre-written as legal censored quadruplets, so `--resume` leaves exactly one
    theorem to execute: plan lookup -> pinned surface -> frozen arm prompt -> completion ->
    extraction -> formal-statement plumbing -> source assembly -> (fake) policy call -> candidate
    row. The model and the verifier are fakes, so the run makes 0 real model generations and 0
    formal verifier calls, and the tripwire on the frozen policy stays at zero.
    """
    _frozen, surface = design()
    plan = plan_artifact(boundary)
    rows_by_statement = screening_by_statement(boundary)
    target, theorem = S.N_PRIMARY, rank_theorem(boundary, S.N_PRIMARY)
    env = formal_env()

    raw_path = boundary / S.SECOND_STAGE_RAW_BASENAME
    for earlier in plan["theorems"]:
        if earlier["formal_rank"] == target:
            break
        S.append_rows_durable(raw_path, [
            R.censored_repair_row(plan_row=earlier,
                                  screening_row=rows_by_statement[earlier["statement_id"]],
                                  arm=earlier["arm_order"][position - 1], position=position,
                                  env=env, run_id="amendment-b-fixture",
                                  reason="pre-written by the Amendment B fixture")
            for position in range(1, S.N_ARMS + 1)])

    verified_arm = theorem["arm_order"][1]
    llm, session, recovery = FakeLLM(), FakeSession(verified=(verified_arm,)), FakeRecovery(boundary)
    with second_stage_fakes(out_dir=boundary, env=env, llm=llm, session=session,
                            recovery=recovery) as fakes:
        assert R.main(["--stage", "second", "--resume", "--i-have-owner-launch-authorization",
                       "--out-dir", str(boundary)]) == 0

    # the generation: one frozen arm prompt per position, one synthetic completion each, the frozen
    # sampling parameters and this theorem's paired seed -- and no model anywhere
    assert [len(call) for call in llm.calls] == [1] * S.N_ARMS
    prompts = [call[0] for call in llm.calls]
    assert [S.sha256_text(prompt) for prompt in prompts] == [
        theorem["arms"][arm]["prompt_sha256"] for arm in theorem["arm_order"]]
    assert [params[0] for params in llm.params] == [
        {"temperature": S.TEMPERATURE, "top_p": S.TOP_P, "max_tokens": S.MAX_RESPONSE_TOKENS,
         "n": 1, "seed": theorem["seed"]} for _arm in theorem["arm_order"]]
    assert fakes["real_policy"].call_count == 0

    # the verification: the (fake) verifier received exactly the assembled source of each candidate,
    # built from the pinned formal statement and the body extracted from that arm's completion
    formal = surface[theorem["statement_id"]]["formal_statement"]
    assert session.calls == [
        (f"rank{target}-{arm}",
         R.complete_verifier_code(formal, extract_proof(synthetic_completion(prompt))))
        for prompt, arm in zip(prompts, theorem["arm_order"])]

    # the rows: a complete quadruplet for the target theorem, in the frozen arm order
    rows = list(S.rows_jsonl(raw_path))
    assert len(rows) == S.SECOND_STAGE_CANDIDATES
    assert [row["arm"] for row in rows[-S.N_ARMS:]] == list(theorem["arm_order"])
    by_arm = {row["arm"]: row for row in rows[-S.N_ARMS:]}
    for arm, row in by_arm.items():
        assert row["statement_id"] == theorem["statement_id"] and row["seed"] == theorem["seed"]
        assert row["failed_proof_sha256"] == rows_by_statement[theorem["statement_id"]][
            "extracted_proof_sha256"]
        assert row["format_ok"] is True and row["extracted_proof_present"] is True
        if arm == verified_arm:
            assert row["verify_status"] == VerifyOutcome.VERIFIED.value
            assert row["score"] == 1 and row["success"] is True
        else:
            assert row["verify_status"] == VerifyOutcome.LEAN_ERROR.value
            assert row["score"] == 0 and row["success"] is False
        S.validate_repair_row(row)

    summary = json.loads((boundary / S.SUMMARY_BASENAME).read_text(encoding="utf-8"))
    assert summary["mode"] == "FORMAL"
    assert summary["frozen_settings_sha256"] == S.sha(S.FROZEN_SETTINGS)
    assert summary["candidates_generated"] == S.N_ARMS
    assert summary["theorems_complete_on_disk"] == S.N_PRIMARY
    assert summary["candidate_assembly"]["theorems"] == S.N_PRIMARY
    assert summary["candidate_assembly"]["n_failed"] == 0
    # the C′ recovery sequence stayed out of the fixture: no restart, no canary, no recovery
    assert recovery.identity_calls == 1 and recovery.recover_calls == []
    assert recovery.n_attempted == 0 and recovery.n_succeeded == 0


# --- Amendment B: the schema audit (owner §4) and the single implementation (owner §7) -------------

def subscript_keys(path: Path, names: set[str]) -> tuple[dict[str, set[str]], dict[str, list[int]]]:
    """Every literal key a variable is subscripted with in a module, per variable name.

    Integer subscripts (``plan[0]``, ``plan[-1]``) are indices rather than schema keys and are
    ignored. The second return value holds the subscripts whose key is not a literal at all: an audit
    like this one cannot see through those, so the caller asserts there are none for its names.
    """
    found: dict[str, set[str]] = {name: set() for name in names}
    dynamic: dict[str, list[int]] = {}
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Subscript) or not isinstance(node.value, ast.Name):
            continue
        if node.value.id not in names:
            continue
        key = node.slice
        is_index = ((isinstance(key, ast.Constant) and isinstance(key.value, int))
                    or (isinstance(key, ast.UnaryOp) and isinstance(key.op, ast.USub)
                        and isinstance(key.operand, ast.Constant)))
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            found[node.value.id].add(key.value)
        elif not is_index:
            dynamic.setdefault(node.value.id, []).append(node.lineno)
    return found, dynamic


def call_sites(path: Path, callees: set[str]) -> dict[str, set[str]]:
    """The enclosing function of every call to *callees* — the 'no second copy' machine check."""
    found: dict[str, set[str]] = {callee: set() for callee in callees}
    for func in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(func, ast.FunctionDef):
            continue
        for node in ast.walk(func):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id in callees):
                found[node.func.id].add(func.name)
    return found


def test_every_stage_two_row_access_is_drawn_from_its_own_schema(boundary: Path) -> None:
    """Owner §4: the audit the launch-0 KeyError demands, not just a fix of that one key.

    `screening_row[...]` must stay inside the frozen screening schema (which has no formal
    statement), `plan_row[...]`/`theorem[...]`/`arm_plan[...]` inside the frozen plan schema,
    `surface_row[...]` inside the pinned surface schema, and the polymorphic loop variables inside
    the union of the schemas they can hold. The runner has no `cohort_row` binding: the cohort schema
    *is* the screening-row schema (its members are screening rows) and is audited as such.
    """
    frozen, surface = design()
    plan = plan_artifact(boundary)["theorems"]
    screening_rows = R.read_screening(boundary / S.SCREENING_RAW_BASENAME)
    cohort_entry_keys = set(R.cohort_entries(R.cohort_rows(screening_rows))[0])
    derangement = json.loads((boundary / S.DERANGEMENT_BASENAME)
                             .read_text(encoding="utf-8"))["derangement"]
    plan_payload_keys = set(plan_artifact(boundary))
    schemas = {
        "screening_row": set(S.SCREENING_FIELDS),
        "plan_row": set(plan[0]),
        "theorem": set(plan[0]),
        "t": set(plan[0]),
        "arm_plan": set(plan[0]["arms"][S.ARM_ORDER[0]]),
        "surface_row": set(surface[plan[0]["statement_id"]]),
        "previous": set(surface[plan[0]["statement_id"]]),
        "member": set(frozen.screening_theorems[0]),
        "entry_schedule": set(frozen.seeds["screening_schedule"][0]),
        "e": set(R.build_screening_plan(frozen, surface)[0]),
        # `entry` is polymorphic too: a screening-plan entry in the screening loop, a surface row in
        # `load_surface`, and a boundary-check record in the validation report.
        "entry": (set(R.build_screening_plan(frozen, surface)[0]) | set(surface[plan[0]["statement_id"]])
                  | cohort_entry_keys | {"check", "pass", "detail"}),
        "plan": plan_payload_keys,
        "plan_artifact": plan_payload_keys,
        # `row` is polymorphic by design: screening row, cohort entry, derangement mapping row, pool
        # member or repair row, depending on the stage. The union is the bound the audit can enforce.
        "row": (set(S.SCREENING_FIELDS) | set(S.REPAIR_FIELDS) | cohort_entry_keys
                | set(derangement["mapping"][0]) | set(frozen.screening_theorems[0])),
    }
    found, dynamic = subscript_keys(ROOT / "scripts/v4_p001_rollout.py", set(schemas))
    assert dynamic == {}, f"a non-literal subscript key cannot be audited: {dynamic}"
    assert set(found) == set(schemas), "the audited variables changed name; update this audit"
    assert all(found[name] for name in schemas), "an audited variable vanished from the runner"
    for name in sorted(found):
        outside = found[name] - schemas[name]
        assert not outside, f"{name}[...] reads {sorted(outside)}, which is not in its schema"

    # the launch-0 defect itself: the screening schema carries no formal statement ...
    assert "formal_statement" not in S.SCREENING_FIELDS
    assert "formal_statement" not in found["screening_row"]
    # ... and every one of the 128 frozen theorems has a pinned surface row that carries one
    bound = [theorem for theorem in plan
             if theorem["statement_id"] in surface
             and set(surface[theorem["statement_id"]]) == set(surface[plan[0]["statement_id"]])
             and str(surface[theorem["statement_id"]]["formal_statement"]).strip()]
    assert len(bound) == S.N_PRIMARY


def test_the_candidate_source_has_a_single_implementation() -> None:
    """Owner §7: the dry run and the formal run may not implement separate assembly copies."""
    sites = call_sites(ROOT / "scripts/v4_p001_rollout.py",
                       {"complete_verifier_code", "candidate_verifier_source",
                        "frozen_formal_statement", "validate_candidate_assembly"})
    assert sites["complete_verifier_code"] == {"screen_one", "candidate_verifier_source"}
    assert sites["candidate_verifier_source"] == {"repair_one", "validate_candidate_assembly"}
    assert sites["frozen_formal_statement"] == {"stage_second"}
    assert sites["validate_candidate_assembly"] == {"stage_second"}
