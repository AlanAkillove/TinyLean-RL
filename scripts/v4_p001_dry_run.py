#!/usr/bin/env python3
"""V4-P001 owner §19 -- the execution-package dry run: everything, with zero formal generation.

This script is the machine transcript the owner asked for. It exercises the whole package on the
formal node, at one committed revision, and writes down what it observed:

    pool              the frozen pool, its order hash, the sealed-reserve exclusion
    screen order      the 640-theorem screening plan rebuilt from the pinned parquet, every prompt
                      re-tokenized to its frozen length, every seed the frozen formula
    model             the pinned weights file, hashed
    verifier          the dedicated instance, /health, the nonformal canary, the frozen recovery plan
    context           the four-arm worst-case audit and its arithmetic against the frozen window
    runner            --stage screening --dry-run; the owner-authorization interlock on both formal
                      stages; a *synthetic* screening artifact frozen and validated by the real
                      `--stage freeze` / `--stage validate` code paths; --stage second --dry-run,
                      which since Amendment B also validates the whole candidate-assembly path
                      (pinned formal statements, source assembly) for all 128 theorems
    analyzer          `v4_p001_analyze.py --check-only` on the synthetic boundary (proves the
                      analyzer runs end to end on this host and writes nothing), plus its
                      fail-closed refusal when the artifacts are absent
    artifact locations the committed design artifacts, and the raw directory that must stay empty
    recovery policy   the frozen ceilings and semantics, and the V3 archive they were projected from

The synthetic part is a *fixture*: it lives under /tmp (never under `runs/`), every row it writes
carries the marker ``SYNTHETIC-DRY-RUN``, and it is deleted before this script exits. Its numbers
are not results and are never presented as such -- they exist only to prove the CPU half of the
pipeline (freeze -> validate -> analyze) executes on this machine at this commit. No model is
loaded, no candidate is generated, no formal theorem is ever sent to the verifier: the only
verifier event of this whole run is the canary.

Output: experiments/manifests/v4/V4-P001_execution_package_dry_run.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "scripts", ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import v4_p001_spec as S

DEFAULT_OUT = "experiments/manifests/v4/V4-P001_execution_package_dry_run.json"
DEFAULT_SCRATCH = "/tmp/v4_p001_dry_run"
FORMAL_HOST_IP = "10.3.25.122"
SYNTHETIC_MARK = "SYNTHETIC-DRY-RUN"
SYNTHETIC_HOST = "v4-p001-dry-run-synthetic"
SYNTHETIC_CATEGORIES = ("unsolved_goals", "elaboration_type_mismatch", "unknown_identifier",
                        "tactic_failure", "typeclass_synthesis", "other_semantic_lean_failure")
BOUNDARY_BASENAMES = (S.PRIMARY_COHORT_BASENAME, S.DERANGEMENT_BASENAME,
                      S.SECOND_STAGE_PLAN_BASENAME)
RAW_BASENAMES = (S.SCREENING_RAW_BASENAME, S.SECOND_STAGE_RAW_BASENAME, S.SUMMARY_BASENAME,
                 S.BOUNDARY_VALIDATION_BASENAME, S.RECOVERY_LOG_BASENAME,
                 S.VERIFIER_EVENTS_BASENAME)


def tail(text: str, n: int = 12) -> str:
    lines = [line for line in (text or "").splitlines() if line.strip()]
    return "\n".join(lines[-n:])


def run(cmd: list, *, timeout: float = 7200.0, keep_stdout: bool = False) -> dict:
    """Run a command from the repo root, capturing what a reviewer needs to reproduce it."""
    started = time.perf_counter()
    proc = subprocess.run([str(c) for c in cmd], cwd=ROOT, capture_output=True, text=True,
                          timeout=timeout, check=False)
    out = {"command": " ".join(str(c) for c in cmd), "exit_code": proc.returncode,
           "seconds": round(time.perf_counter() - started, 2),
           "stdout_tail": tail(proc.stdout), "stderr_tail": tail(proc.stderr, 6)}
    if keep_stdout:
        out["stdout"] = proc.stdout
    return out


class Transcript:
    """One artifact, one check list. A failing check is recorded and the exit code says so."""

    def __init__(self, *, informal_host: bool) -> None:
        self.sections: dict = {}
        self.checks: list = []
        self.informal_host = bool(informal_host)

    def section(self, name: str) -> dict:
        self.sections[name] = {}
        return self.sections[name]

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.checks.append({"check": name, "pass": bool(ok), "detail": detail})
        return bool(ok)

    @property
    def n_failed(self) -> int:
        return sum(1 for entry in self.checks if not entry["pass"])


# --- section 1: the frozen design and the pool ----------------------------------------------------

def section_frozen(t: Transcript) -> object:
    section = t.section("frozen_design")
    try:
        frozen = S.load_frozen(verify_files=True)
    except S.FrozenViolation as exc:
        t.check("frozen_design_loads", False, str(exc))
        raise
    t.check("frozen_design_loads", True,
            f"{len(frozen.checks)} frozen-artifact checks passed")
    section.update({
        "n_frozen_checks": len(frozen.checks),
        "n_frozen_checks_failed": sum(1 for e in frozen.checks if not e["pass"]),
        "file_hashes": frozen.file_hashes,
        "registry_pins": frozen.registry_pins,
        "frozen_settings": S.FROZEN_SETTINGS,
        "frozen_settings_sha256": S.sha(S.FROZEN_SETTINGS),
        "prereg_commit": S.PREREG_COMMIT,
        "amendment_doc": S.AMENDMENT_DOC,
        "authorization_note": ("the design is frozen and complete; it authorizes nothing. The two "
                               "formal stages require the owner's written launch authorization."),
    })
    return frozen


def section_pool(t: Transcript, frozen) -> None:
    section = t.section("pool")
    pool = frozen.pool
    members = frozen.members
    by_tier: dict = {}
    by_source: dict = {}
    for member in members:
        by_tier[str(member["tier"])] = by_tier.get(str(member["tier"]), 0) + 1
        by_source[str(member["source"])] = by_source.get(str(member["source"]), 0) + 1
    touched = {key: value for key, value in pool["checks"].items() if key.endswith("_touched")}
    section.update({
        "members": len(members),
        "unique_components": len({m["component_id"] for m in members}),
        "unique_statements": len({m["statement_id"] for m in members}),
        "pool_hash": pool["pool_hash"],
        "order_hash": pool["order_hash"],
        "base_seed": pool["base_seed"],
        "screening_order_rule": pool["screening_order_rule"],
        "tier_rule": pool["tier_rule"],
        "members_by_tier": by_tier,
        "members_by_source": by_source,
        "screening_budget": pool["screening_budget"],
        "worst_case_formal_generations": S.WORST_CASE_FORMAL_GENERATIONS,
        "sealed_reserve_touch_counts": touched,
        "checks": pool["checks"],
    })
    t.check("pool_hash_recomputes", S.pool_hash_recompute(pool, members) == pool["pool_hash"],
            f"{pool['pool_hash'][:16]}...")
    t.check("order_hash_recomputes", S.order_hash_recompute(pool, members) == pool["order_hash"],
            f"{pool['order_hash'][:16]}...")
    t.check("sealed_reserve_untouched", all(value == 0 for value in touched.values()),
            f"touch counts {touched}")
    t.check("screening_budget_is_the_frozen_one",
            pool["screening_budget"]["max_screens"] == S.MAX_SCREENING
            and pool["screening_budget"]["n_primary_required"] == S.N_PRIMARY,
            str(pool["screening_budget"]))
    t.check("worst_case_budget_is_1152",
            S.WORST_CASE_FORMAL_GENERATIONS == S.MAX_SCREENING + S.SECOND_STAGE_CANDIDATES
            == 1152,
            f"{S.MAX_SCREENING} + {S.SECOND_STAGE_CANDIDATES} = {S.WORST_CASE_FORMAL_GENERATIONS}")


# --- section 2: the screening order, on the real tokenizer ----------------------------------------

def section_screen_order(t: Transcript, frozen) -> tuple:
    section = t.section("screen_order")
    import v4_p001_rollout as R
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(ROOT / S.MODEL["dir"], trust_remote_code=True,
                                             local_files_only=True)
    surface = R.load_surface(tokenizer)
    plan = R.build_screening_plan(frozen, surface)
    seeds = frozen.seeds["screening_schedule"]
    section.update({
        "theorems": len(plan),
        "unique_statement_ids": len({entry["statement_id"] for entry in plan}),
        "first": plan[0],
        "last": plan[-1],
        "prompt_token_equality": ("every rendered first-attempt prompt re-tokenizes to the pool's "
                                  "frozen prompt_tokens (checked inside build_screening_plan)"),
        "prompt_tokens_max": max(entry["prompt_tokens"] for entry in plan),
        "seed_formula": "first_stage_seed(rank) = sha256(f'v4-p001-screening|{rank}') truncated",
        "screening_seed_hash": frozen.seeds["screening_seed_hash"],
        "screening_seed_hash_recomputed": S.sha([row["seed"] for row in seeds]),
        "screening_seeds_unique": len({row["seed"] for row in seeds}) == len(seeds),
        "schedule_covers_the_frozen_order": [row["statement_id"] for row in seeds]
        == [m["statement_id"] for m in frozen.screening_theorems],
        "tokenizer": type(tokenizer).__name__,
        "tokenizer_vocab_size": int(tokenizer.vocab_size),
    })
    t.check("screening_plan_is_640", len(plan) == S.MAX_SCREENING, f"{len(plan)} theorems")
    t.check("screening_plan_renders_on_this_host", True,
            "640 prompts rendered and re-tokenized; every count equals the frozen pool's")
    t.check("screening_seed_hash_recomputes",
            section["screening_seed_hash_recomputed"] == frozen.seeds["screening_seed_hash"]
            == S.SCREENING_SEED_HASH,
            f"{section['screening_seed_hash_recomputed'][:16]}...")
    t.check("screening_seeds_are_unique", section["screening_seeds_unique"],
            f"{len(seeds)} seeds, {len({row['seed'] for row in seeds})} unique")
    return surface, plan


# --- section 3: the model --------------------------------------------------------------------------

def section_model(t: Transcript) -> None:
    section = t.section("model")
    path = ROOT / S.MODEL["dir"]
    weights = path / S.MODEL["weights_filename"]
    section.update({"dir": S.MODEL["dir"], "repo_id": S.MODEL["repo_id"],
                    "revision": S.MODEL["revision"], "weights_filename": S.MODEL["weights_filename"],
                    "weights_sha256_frozen": S.MODEL["weights_sha256"]})
    t.check("model_dir_exists", path.is_dir(), str(path))
    t.check("weights_file_exists", weights.is_file(), str(weights))
    if weights.is_file():
        digest = S.sha256_file(weights)
        section["weights_sha256_recomputed_on_this_host"] = digest
        section["weights_bytes"] = weights.stat().st_size
        t.check("weights_sha256_matches_the_frozen_pin", digest == S.MODEL["weights_sha256"],
                f"{digest[:16]}...")
    else:
        t.check("weights_sha256_matches_the_frozen_pin", False, "weights file absent")


# --- section 4: the verifier and the recovery policy -----------------------------------------------

def section_verifier(t: Transcript, frozen, scratch: Path) -> None:
    import v4_p001_rollout as R

    from tinylean_rl.verifier.policy import CANARY_PROOF

    section = t.section("verifier")
    recovery = R.VerifierRecovery(scratch / "verifier_probe")
    identity_error = None
    try:
        instance = recovery.describe_instance()
        identity = recovery.verify_identity()
    except Exception as exc:  # noqa: BLE001 - a probe must report, never abort the transcript
        instance, identity = {"error": f"{type(exc).__name__}: {exc}"}, None
        identity_error = f"{type(exc).__name__}: {exc}"
    session = R.make_session()
    health = R.check_verifier(session)
    canary = health.get("canary", {})
    section.update({
        "instance": {key: (identity or instance).get(key) for key in (
            "present", "running", "image", "container_id", "env", "published", "error")},
        "identity_error": identity_error,
        "health_url": health.get("health_url"),
        "health_status": health.get("health_status"),
        "canary": canary,
        "canary_proof_sha256": S.sha256_text(CANARY_PROOF),
        "session_events": [{"kind": e.kind, "detail": e.detail} for e in session.events],
        "session_events_note": "the only verifier event of the dry run is the canary",
        "verifier_formal_calls": 0,
        "policy": S.VERIFIER["policy"],
        "endpoint": S.VERIFIER_INFRA["endpoint"],
        "container": S.VERIFIER_INFRA["container"],
        "max_repls": S.VERIFIER_INFRA["max_repls"],
    })
    t.check("dedicated_instance_identity", identity_error is None, identity_error or "match")
    t.check("verifier_health_is_200", health.get("health_status") == 200,
            f"status={health.get('health_status')!r}")
    t.check("canary_verifies_and_uses_no_formal_theorem",
            bool(canary.get("verified")) and canary.get("uses_formal_theorem") is False,
            f"canary={canary.get('status')!r} in {canary.get('seconds')}s")
    t.check("no_formal_verifier_call", all(e["kind"] == "canary" for e in section["session_events"]),
            f"{len(section['session_events'])} event(s)")


def section_recovery_policy(t: Transcript, frozen) -> None:
    section = t.section("recovery_policy")
    plan = frozen.verifier_plan
    frozen_plan = plan["frozen"]
    section.update({
        "frozen": frozen_plan,
        "max_recoveries_per_run_matches_the_code": (
            int(S.VERIFIER_INFRA["max_recoveries_per_run"]) == frozen_plan[
                "max_recoveries_per_run"] == 192),
        "max_recoveries_per_theorem_matches_the_code": (
            int(S.VERIFIER_INFRA["max_recoveries_per_theorem"]) == frozen_plan[
                "max_recoveries_per_theorem"] == 3),
        "projection": plan["projection"],
        "observed_basis": {
            "source": plan["observed"]["source"],
            "n_executions": len(plan["observed"]["executions"]),
            "note": ("the ceilings were projected from the V3-R001 attempt-2 archive, read-only; "
                     "the archive is evidence about infrastructure, never about this design"),
        },
        "not_claimed": plan["not_claimed"],
        "what_the_runner_does_on_a_trigger": (
            "restart -> /health -> nonformal cold canary, before the next candidate; a recovery is "
            "not a candidate attempt and moves no label; at the per-theorem ceiling the remaining "
            "arm candidates of that theorem are marked INFRA_CENSORED (missing, never a failure) "
            "and the run continues; at the per-run ceiling the run fails closed with an aborted, "
            "resumable summary"),
        "infra_is_never_a_proof_failure": (
            "an unresolved candidate is written with score=None and an infrastructure status; the "
            "sensitivity reading that completes it as a failure belongs to the analyzer and never "
            "redefines the primary endpoint"),
    })
    t.check("recovery_ceilings_match_the_frozen_plan",
            section["max_recoveries_per_run_matches_the_code"]
            and section["max_recoveries_per_theorem_matches_the_code"],
            f"run={frozen_plan['max_recoveries_per_run']} "
            f"theorem={frozen_plan['max_recoveries_per_theorem']}")
    t.check("recovery_is_not_a_candidate_retry",
            frozen_plan["recovery_is_not_a_candidate_retry"] is True, "")
    t.check("no_efficiency_claim",
            any("no verifier throughput" in item for item in plan["not_claimed"]),
            "the frozen plan forbids throughput/latency/capacity claims")


# --- section 5: the four-arm context ---------------------------------------------------------------

def section_context(t: Transcript, frozen, surface: dict) -> None:
    section = t.section("context")
    artifact = frozen.context
    worst = artifact["worst_case_by_arm"]
    section.update({
        "artifact": S.CONTEXT_FOUR_ARM,
        "max_response_tokens": artifact["max_response_tokens"],
        "max_model_len": artifact["max_model_len"],
        "worst_case_by_arm": {arm: {"prompt_tokens": row["worst_case_prompt_tokens"],
                                    "total_tokens": row["worst_case_total_tokens"],
                                    "headroom_tokens": row["headroom_tokens"],
                                    "fits": row["fits"]} for arm, row in worst.items()},
        "pool_prompt_tokens": {arm: {k: row[k] for k in ("max", "p95", "p99")}
                               for arm, row in artifact["pool_prompt_tokens"].items()},
        "checks": artifact["checks"],
        "smoke_evidence": artifact["smoke_evidence"],
    })
    live_pool_max = max(surface[m["statement_id"]]["prompt_token_count"]
                        for m in frozen.members if m["statement_id"] in surface)
    section["live_recomputation"] = {
        "arm_a_worst_case_rendered_here": live_pool_max,
        "artifact_arm_a_worst_case": worst["A_FRESH_RETRY"]["worst_case_prompt_tokens"],
        "pool_members_not_in_the_pinned_parquet": sum(
            1 for m in frozen.members if m["statement_id"] not in surface),
        "note": ("Arm A's prompt *is* the first-attempt screening prompt, so re-rendering the pool "
                 "above gives its worst case directly. The worst cases of the other three arms are "
                 "the committed audit's; they are reproducible with "
                 "`scripts/v4_p001_context_four_arm.py`, which reads the consumed failure corpora."),
    }
    t.check("four_arm_context_checks_all_pass", all(artifact["checks"].values()),
            str([k for k, v in artifact["checks"].items() if not v]))
    t.check("every_arm_fits_with_a_full_response",
            all(row["fits"] for row in worst.values())
            and all(row["worst_case_total_tokens"] <= artifact["max_model_len"]
                    for row in worst.values()),
            f"max total {max(row['worst_case_total_tokens'] for row in worst.values())} "
            f"vs window {artifact['max_model_len']}")
    t.check("frozen_window_matches_the_code",
            artifact["max_model_len"] == S.MAX_MODEL_LEN
            and artifact["max_response_tokens"] == S.MAX_RESPONSE_TOKENS,
            f"{artifact['max_model_len']} / {artifact['max_response_tokens']}")
    t.check("arm_a_worst_case_recomputes_from_the_pinned_parquet",
            live_pool_max == worst["A_FRESH_RETRY"]["worst_case_prompt_tokens"],
            f"re-rendered {live_pool_max} vs artifact "
            f"{worst['A_FRESH_RETRY']['worst_case_prompt_tokens']}")


# --- section 6: the runner -------------------------------------------------------------------------

def synthetic_screening_rows(*, surface: dict, plan: list) -> list[dict]:
    """A legal screening artifact whose *outcomes* are fabricated and whose inputs are real.

    Everything the boundary validation recomputes -- statement ids, prompt hashes, token counts,
    seeds, component/name/source/tier/classes -- is taken from the rebuilt frozen plan. Only the
    verdicts, the completions and the diagnostics are invented, and every row says so.
    """
    rows = []
    proof = "by\n  simp [Nat.add_comm]"
    for index, entry in enumerate(plan[:S.N_PRIMARY]):
        category = SYNTHETIC_CATEGORIES[index % len(SYNTHETIC_CATEGORIES)]
        diagnostic = f"{category}\ncase h{index + 1}\n⊢ 1 + {index + 1} = {index + 2}"
        rows.append({
            "schema_version": S.SCREENING_SCHEMA_VERSION, "run_id": SYNTHETIC_MARK,
            "experiment_id": S.EXPERIMENT_ID,
            "screening_rank": entry["screening_rank"], "formal_rank": entry["screening_rank"],
            "statement_id": entry["statement_id"], "component_id": entry["component_id"],
            "name": entry["name"], "source": entry["source"], "tier": entry["tier"],
            "classes": entry["classes"], "prompt_tokens": entry["prompt_tokens"],
            "prompt_sha256": surface[entry["statement_id"]]["prompt_sha256"],
            "prompt_token_count": surface[entry["statement_id"]]["prompt_token_count"],
            "seed": entry["seed"],
            "model_sha256": S.MODEL["weights_sha256"], "model_revision": S.MODEL["revision"],
            "completion_text": f"{proof}\ntrailing prose", "completion_sha256": S.sha256_text(proof),
            "generated_tokens": 40, "truncated": False, "format_ok": True, "has_lean_block": True,
            "extracted_proof": proof, "extracted_proof_sha256": S.sha256_text(proof),
            "extracted_proof_present": True, "extracted_proof_tokens": 8,
            "diagnostic_text": diagnostic, "diagnostic_sha256": S.sha256_text(diagnostic),
            "diagnostic_tokens": len(diagnostic.split()), "diagnostic_source": "verify_item",
            "truncated_diagnostic": False, "context_fits": True,
            "context_tokens_with_response": entry["prompt_tokens"] + S.MAX_RESPONSE_TOKENS,
            "verified": False, "score": 0, "acc": 0, "verify_status": "lean_error",
            "lean_message": diagnostic, "error_category": category,
            "screening_status": S.PRIMARY_SEMANTIC_FAILURE, "primary_eligible": True,
            "generation_time": 1.0, "verification_time": None,
            "host": SYNTHETIC_HOST, "gpu": "", "created_at": "2026-09-25T00:00:00+00:00",
        })
    return rows


def write_synthetic_screening(out_dir: Path, rows: list) -> Path:
    for row in rows:
        S.validate_screening_row(row)
    path = out_dir / S.SCREENING_RAW_BASENAME
    S.append_rows_durable(path, rows)
    return path


def synthetic_second_stage_rows(*, plan_artifact: dict, screening_by_statement: dict) -> list[dict]:
    """One row per arm candidate of every formal rank: a legal artifact with a made-up outcome.

    The construction is the §18 fixture pattern `C survives on every theorem, A/B/D never do`, so
    the analyzer's classification on it is a *demonstration of the machinery*, not a finding.
    """
    rows = []
    for theorem in plan_artifact["theorems"]:
        row = screening_by_statement[theorem["statement_id"]]
        for position, arm in enumerate(theorem["arm_order"]):
            success = arm == "C_VERIFIER_REPAIR"
            arm_plan = theorem["arms"][arm]
            rows.append({
                "schema_version": S.REPAIR_SCHEMA_VERSION, "run_id": SYNTHETIC_MARK,
                "experiment_id": S.EXPERIMENT_ID,
                "formal_rank": theorem["formal_rank"], "screening_rank": row["screening_rank"],
                "statement_id": theorem["statement_id"], "component_id": row["component_id"],
                "name": row["name"], "source": row["source"],
                "error_category": theorem["error_category"], "arm": arm, "position": position,
                "seed": theorem["seed"],
                "prompt_sha256": arm_plan["prompt_sha256"],
                "prompt_token_count": arm_plan["prompt_tokens"],
                "context_tokens_with_response": arm_plan["context_tokens_with_response"],
                "renderer_version": "v4-prompt-1", "normalization_version": "v4-diag-1",
                "diagnostic_sha256": arm_plan["diagnostic_sha256"],
                "diagnostic_source": arm_plan["diagnostic_source"],
                "failed_proof_sha256": theorem["failed_proof_sha256"],
                "model_sha256": S.MODEL["weights_sha256"], "model_revision": S.MODEL["revision"],
                "completion_text": "by\n  simp", "completion_sha256": S.sha256_text("by\n  simp"),
                "generated_tokens": 11, "truncated": False, "format_ok": True,
                "has_lean_block": True, "extracted_proof_present": True,
                "verified": success, "score": 1 if success else 0, "acc": 1 if success else 0,
                "verify_status": "verified" if success else "lean_error",
                "lean_message": "" if success else "unsolved goals",
                "error_category_second": "verified" if success else theorem["error_category"],
                "success": success, "censored": False, "censored_reason": "",
                "generation_time": 0.4, "verification_time": 0.1,
                "host": SYNTHETIC_HOST, "gpu": "", "created_at": "2026-09-25T00:00:00+00:00",
            })
    return rows


def write_synthetic_second_stage(out_dir: Path, rows: list) -> Path:
    for row in rows:
        S.validate_repair_row(row)
    path = out_dir / S.SECOND_STAGE_RAW_BASENAME
    for start in range(0, len(rows), S.N_ARMS):
        S.append_rows_durable(path, rows[start:start + S.N_ARMS])
    return path


def write_synthetic_summary(out_dir: Path, boundary_report: dict, plan_artifact: dict) -> Path:
    payload = {
        "experiment_id": S.EXPERIMENT_ID, "schema_version": S.REPAIR_SCHEMA_VERSION,
        "run_id": SYNTHETIC_MARK, "stage": "second",
        "frozen_settings": S.FROZEN_SETTINGS,
        "frozen_settings_sha256": S.sha(S.FROZEN_SETTINGS),
        "script_versions": S.script_version_hash("scripts/v4_p001_rollout.py"),
        "prereg_commit": S.PREREG_COMMIT, "git": {"hostname": SYNTHETIC_HOST},
        "boundary_validation": {"status": boundary_report["status"],
                                "n_checks": boundary_report["n_checks"],
                                "n_failed": boundary_report["n_failed"],
                                "artifacts": boundary_report["artifacts"],
                                "screening_raw": boundary_report["screening_raw"]},
        "plan_sha256": plan_artifact["plan_sha256"],
        "arm_prompt_plan_sha256": plan_artifact["arm_prompt_plan_sha256"],
        "arm_schedule_hash": plan_artifact["arm_schedule_hash"],
        "second_stage_seed_hash": plan_artifact["second_stage_seed_hash"],
        "derangement_mapping_sha256": plan_artifact["derangement_mapping_sha256"],
        "n_theorems": S.N_PRIMARY, "n_candidates": S.SECOND_STAGE_CANDIDATES,
        "mode": SYNTHETIC_MARK,
    }
    path = out_dir / S.SUMMARY_BASENAME
    S.write_json_atomic(path, payload)
    return path


def section_runner(t: Transcript, scratch: Path, surface: dict, plan: list) -> dict:
    section = t.section("runner")
    script = ROOT / "scripts/v4_p001_rollout.py"
    empty = scratch / "empty"
    empty.mkdir(parents=True, exist_ok=True)
    synthetic = scratch / "synthetic"
    synthetic.mkdir(parents=True, exist_ok=True)

    dry = run([sys.executable, script, "--stage", "screening", "--dry-run", "--out-dir", empty])
    section["screening_dry_run"] = dry
    t.check("screening_dry_run_exits_zero", dry["exit_code"] == 0, f"exit={dry['exit_code']}")
    t.check("screening_dry_run_wrote_nothing", not any(empty.iterdir()),
            f"{len(list(empty.iterdir()))} file(s) in the dry-run out-dir")

    deny_screening = run([sys.executable, script, "--stage", "screening", "--out-dir", empty])
    deny_second = run([sys.executable, script, "--stage", "second", "--out-dir", empty])
    section["launch_interlock"] = {"screening": deny_screening, "second": deny_second}
    t.check("screening_stage_is_owner_gated",
            deny_screening["exit_code"] == 3 and "--i-have-owner-launch-authorization" in
            deny_screening["stderr_tail"],
            f"exit={deny_screening['exit_code']}")
    t.check("second_stage_is_owner_gated",
            deny_second["exit_code"] == 3
            and "--i-have-owner-launch-authorization" in deny_second["stderr_tail"],
            f"exit={deny_second['exit_code']}")

    freeze_empty = run([sys.executable, script, "--stage", "freeze", "--out-dir", empty])
    section["freeze_without_a_cohort"] = freeze_empty
    t.check("freeze_refuses_without_a_screening_artifact",
            freeze_empty["exit_code"] == 3 and "does not exist" in freeze_empty["stderr_tail"],
            f"exit={freeze_empty['exit_code']}")

    # the synthetic chain: a fabricated *cohort* through the real freeze/validate/second code
    rows = synthetic_screening_rows(surface=surface, plan=plan)
    write_synthetic_screening(synthetic, rows)
    freeze = run([sys.executable, script, "--stage", "freeze", "--out-dir", synthetic])
    section["synthetic_freeze"] = freeze
    section["synthetic_screening_rows"] = len(rows)
    section["synthetic_marker"] = SYNTHETIC_MARK
    t.check("synthetic_freeze_exits_zero", freeze["exit_code"] == 0, f"exit={freeze['exit_code']}")
    sealed = {}
    for name in BOUNDARY_BASENAMES:
        path = synthetic / name
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            sealed[name] = {"sha256": S.sha256_file(path),
                            "second_stage_candidates_generated":
                                payload.get("second_stage_candidates_generated")}
    section["synthetic_boundary_artifacts"] = sealed
    t.check("freeze_wrote_the_three_sealed_artifacts", len(sealed) == 3, str(sorted(sealed)))
    t.check("boundary_artifacts_record_zero_candidates_generated",
            len(sealed) == 3 and all(v["second_stage_candidates_generated"] == 0
                                     for v in sealed.values()),
            str({k: v["second_stage_candidates_generated"] for k, v in sealed.items()}))
    t.check("freeze_wrote_no_second_stage_raw",
            not (synthetic / S.SECOND_STAGE_RAW_BASENAME).exists(), "no second-stage candidate")

    validate = run([sys.executable, script, "--stage", "validate", "--out-dir", synthetic])
    section["synthetic_validate"] = validate
    report_path = synthetic / S.BOUNDARY_VALIDATION_BASENAME
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    section["synthetic_boundary_report"] = {
        "status": report.get("status"), "n_checks": report.get("n_checks"),
        "n_failed": report.get("n_failed"),
        "checks": [entry["check"] for entry in report.get("checks", [])],
        "second_stage_candidates_generated": report.get("second_stage_candidates_generated"),
    }
    t.check("validate_exits_zero", validate["exit_code"] == 0, f"exit={validate['exit_code']}")
    t.check("boundary_report_is_PASS_with_zero_failed",
            report.get("status") == "PASS" and report.get("n_failed") == 0
            and report.get("n_checks", 0) >= 20,
            f"{report.get('n_checks')} checks, {report.get('n_failed')} failed")

    second_dry = run([sys.executable, script, "--stage", "second", "--dry-run", "--out-dir",
                      synthetic], keep_stdout=True)
    dry_stdout = second_dry.pop("stdout", "")
    section["synthetic_second_dry_run"] = second_dry
    marker = f"[{S.EXPERIMENT_ID}] DRY RUN COMPLETE"
    head = dry_stdout[:dry_stdout.index(marker)] if marker in dry_stdout else ""
    lines = head.splitlines()
    start = max((i for i, line in enumerate(lines) if line.strip() == "{"), default=len(lines))
    try:
        standalone = json.loads("\n".join(lines[start:]))
    except ValueError:
        standalone = {}
    todo = standalone.get("theorems_to_run") or []
    assembly = standalone.get("candidate_assembly") or {}
    counts = assembly.get("counts") or {}
    section["synthetic_second_dry_run_standalone"] = {
        "mode": standalone.get("mode"), "n_candidates": standalone.get("n_candidates"),
        "theorems_to_run_count": len(todo),
        "theorems_to_run_is_the_whole_cohort": todo == list(range(1, S.N_PRIMARY + 1)),
        "candidate_assembly_n_failed": assembly.get("n_failed"),
    }
    section["synthetic_second_dry_run_assembly"] = {
        "theorems": assembly.get("theorems"), "n_failed": assembly.get("n_failed"),
        "counts": counts, "generations": assembly.get("generations"),
        "formal_verifier_calls": assembly.get("formal_verifier_calls"),
        "note": assembly.get("note"),
    }
    t.check("second_stage_dry_run_revalidates_and_generates_nothing",
            second_dry["exit_code"] == 0
            and "candidates_generated=0" in dry_stdout
            and len(todo) == S.N_PRIMARY,
            f"exit={second_dry['exit_code']}, theorems_to_run={len(todo)}")
    # Amendment B §3/§7: the dry run must validate the *whole* candidate-assembly path for the
    # frozen cohort -- surface statements, failed proofs, diagnostics, arm plan rows, rendered arm
    # prompts, paired seeds, arm order and the verifier-source assembly itself -- because that is
    # exactly the path the launch-0 crash was in and the original dry run never reached.
    t.check("second_stage_dry_run_validates_the_candidate_assembly",
            assembly.get("theorems") == S.N_PRIMARY and assembly.get("n_failed") == 0
            and counts.get("surface_statements_available") == S.N_PRIMARY
            and counts.get("formal_statements_nonempty") == S.N_PRIMARY
            and counts.get("formal_statements_bound_to_the_frozen_plan") == S.N_PRIMARY
            and counts.get("failed_proofs_available") == S.N_PRIMARY
            and counts.get("own_diagnostics_available") == S.N_PRIMARY
            and counts.get("donor_diagnostics_available") == S.N_PRIMARY
            and counts.get("arm_prompt_hashes_matched") == S.SECOND_STAGE_CANDIDATES
            and counts.get("assembly_probes_passed") == S.N_PRIMARY,
            f"n_failed={assembly.get('n_failed')} counts={counts}")
    t.check("candidate_assembly_used_no_model_and_no_verifier",
            assembly.get("generations") == 0 and assembly.get("formal_verifier_calls") == 0
            and f"surface_formal_statements={S.N_PRIMARY}" in dry_stdout
            and "candidate_assembly_validated=True" in dry_stdout,
            f"generations={assembly.get('generations')} "
            f"verifier_calls={assembly.get('formal_verifier_calls')}")
    t.check("second_stage_dry_run_wrote_no_raw",
            not (synthetic / S.SECOND_STAGE_RAW_BASENAME).exists(), "no second-stage raw")
    return {"synthetic_dir": synthetic, "report": report,
            "plan_artifact": json.loads((synthetic / S.SECOND_STAGE_PLAN_BASENAME)
                                        .read_text(encoding="utf-8")),
            "screening_by_statement": {row["statement_id"]: row for row in rows}}


def section_analyzer(t: Transcript, scratch: Path, synthetic: dict) -> None:
    section = t.section("analyzer")
    script = ROOT / "scripts/v4_p001_analyze.py"
    out_dir = synthetic["synthetic_dir"]
    rows = synthetic_second_stage_rows(plan_artifact=synthetic["plan_artifact"],
                                       screening_by_statement=synthetic["screening_by_statement"])
    write_synthetic_second_stage(out_dir, rows)
    write_synthetic_summary(out_dir, synthetic["report"], synthetic["plan_artifact"])
    section["synthetic_candidate_rows"] = len(rows)

    check = run([sys.executable, script, "--out-dir", out_dir, "--check-only"], keep_stdout=True)
    stdout = check.pop("stdout", "")
    section["check_only"] = check
    parsed = {}
    if check["exit_code"] == 0 and "[check-only]" in stdout:
        try:
            parsed = json.loads(stdout[:stdout.index("[check-only]")])
        except (ValueError, json.JSONDecodeError):
            parsed = {}
    section["synthetic_classification"] = parsed
    section["classification_note"] = (
        "the classification below is what the analyzer computes on a *fabricated* outcome pattern "
        "(Arm C succeeds on all 128 ranks, A/B/D never do). It proves the analyzer executes the "
        "mechanical order on this host and writes nothing; it is not a result, and no formal "
        "outcome exists.")
    t.check("analyzer_check_only_runs_on_this_host", check["exit_code"] == 0,
            f"exit={check['exit_code']}")
    t.check("analyzer_reported_a_classification",
            parsed.get("classification") == "A_VERIFIER_SPECIFIC_GO"
            and parsed.get("provenance", {}).get("n_failed") == 0,
            str(parsed.get("classification")))
    t.check("analyzer_guards_ran_before_the_gates",
            parsed.get("data_guard", {}).get("n_complete") == S.N_PRIMARY
            and parsed.get("censoring_guard", {}).get("pass") is True,
            f"n_complete={parsed.get('data_guard', {}).get('n_complete')}")
    t.check("analyzer_wrote_nothing_in_check_only_mode",
            "[check-only] nothing written" in check["stdout_tail"]
            and sorted(p.name for p in out_dir.iterdir()) == sorted([
                S.SCREENING_RAW_BASENAME, S.SECOND_STAGE_RAW_BASENAME, S.SUMMARY_BASENAME,
                S.PRIMARY_COHORT_BASENAME, S.DERANGEMENT_BASENAME, S.SECOND_STAGE_PLAN_BASENAME,
                S.BOUNDARY_VALIDATION_BASENAME]),
            f"{len(list(out_dir.iterdir()))} file(s)")

    absent = run([sys.executable, script, "--out-dir", scratch / "empty", "--check-only"])
    section["refuses_when_artifacts_are_absent"] = absent
    t.check("analyzer_fails_closed_without_artifacts",
            absent["exit_code"] == 3 and "required artifact(s) absent" in absent["stderr_tail"],
            f"exit={absent['exit_code']}")


# --- section 7: artifact locations -----------------------------------------------------------------

def section_artifacts(t: Transcript, scratch: Path) -> None:
    section = t.section("artifact_locations")
    committed = [S.POOL, S.SEEDS, S.SCHEDULE, S.PROMPT_AUDIT, S.PROMPT_PROVENANCE,
                 S.CONTEXT_SMOKE, S.CONTEXT_FOUR_ARM, S.VERIFIER_PLAN, S.POWER, S.REGISTRY]
    section["committed_design_artifacts"] = {
        rel: {"sha256": S.sha256_file(ROOT / rel), "exists": (ROOT / rel).exists()}
        for rel in committed}
    section["raw_artifact_basenames"] = list(RAW_BASENAMES)
    section["raw_directory"] = "runs/v4_p001/rollout (gitignored)"
    raw_dir = ROOT / "runs/v4_p001/rollout"
    section["raw_directory_exists"] = raw_dir.exists()
    ignore = subprocess.run(["git", "-C", str(ROOT), "check-ignore", "-v",
                             "runs/v4_p001/rollout/" + S.SCREENING_RAW_BASENAME],
                            capture_output=True, text=True, check=False)
    section["gitignore_rule"] = ignore.stdout.strip()
    section["scratch_dir"] = str(scratch)
    section["scratch_note"] = ("every synthetic artifact of this run lives under the scratch dir "
                               "and is removed before this script exits; nothing is written under "
                               "runs/")
    t.check("committed_design_artifacts_present",
            all(entry["exists"] for entry in section["committed_design_artifacts"].values()),
            str([rel for rel, entry in section["committed_design_artifacts"].items()
                 if not entry["exists"]]))
    t.check("raw_artifact_directory_does_not_exist_yet", not raw_dir.exists(), str(raw_dir))
    t.check("raw_artifacts_are_gitignored", ignore.returncode == 0 and ignore.stdout.strip() != "",
            section["gitignore_rule"])


# --- section 8: the standing gates on the formal node ----------------------------------------------

def section_gates(t: Transcript, *, run_suite: bool) -> None:
    section = t.section("gates_on_this_host_at_this_commit")
    if run_suite:
        suite = run([sys.executable, "-m", "pytest", "tests/"], timeout=10800.0)
        section["pytest"] = suite
        summary_line = (suite["stdout_tail"] or "").splitlines()[-1] if suite["stdout_tail"] else ""
        section["pytest_summary"] = summary_line
        t.check("full_suite_is_green", suite["exit_code"] == 0, summary_line)
    else:
        section["pytest"] = {"skipped": True,
                             "note": "skipped by --skip-suite; the formal transcript runs it"}
    ruff = run([sys.executable, "-m", "ruff", "check", "src", "scripts", "tests"])
    section["ruff"] = ruff
    t.check("ruff_is_clean", ruff["exit_code"] == 0, ruff["stdout_tail"].splitlines()[-1]
            if ruff["stdout_tail"] else "")


# --- the transcript --------------------------------------------------------------------------------

def build_transcript(*, run_suite: bool, allow_informal: bool, allow_dirty: bool) -> dict:
    env = S.collect_env()
    informal = FORMAL_HOST_IP not in env.get("ips", "")
    t = Transcript(informal_host=informal)
    scratch = Path(f"{DEFAULT_SCRATCH}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}")
    started = time.perf_counter()

    def cleanliness(name: str, porcelain: str) -> None:
        if allow_dirty:
            t.section("cleanliness_waiver")[name] = {"porcelain": porcelain,
                                                    "note": "recorded, not enforced (--allow-dirty)"
                                                    }
            return
        t.check(name, porcelain == "", f"porcelain={porcelain[:200]!r}")

    cleanliness("working_tree_is_clean_before_the_dry_run", env["status_porcelain"])
    if not allow_informal:
        t.check("dry_run_ran_on_the_formal_node", not informal,
                f"ips={env['ips']!r}; fly122 is {FORMAL_HOST_IP}")
    stage: dict = {}
    crashed: str | None = None
    try:
        frozen = section_frozen(t)
        section_pool(t, frozen)
        surface, plan = section_screen_order(t, frozen)
        section_model(t)
        section_verifier(t, frozen, scratch)
        section_recovery_policy(t, frozen)
        section_context(t, frozen, surface)
        stage = section_runner(t, scratch, surface, plan)
        section_analyzer(t, scratch, stage)
        section_artifacts(t, scratch)
        section_gates(t, run_suite=run_suite)
    except S.FrozenViolation as exc:
        crashed = f"FrozenViolation: {exc}"
    except Exception as exc:  # noqa: BLE001 - the transcript records a crash instead of hiding it
        crashed = f"{type(exc).__name__}: {exc}"
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    env_after = S.collect_env()
    cleanliness("working_tree_is_clean_after_the_dry_run", env_after["status_porcelain"])
    t.check("dry_run_completed", crashed is None, crashed or "every section ran to the end")
    t.check("scratch_directory_removed", not scratch.exists(), str(scratch))
    t.check("raw_artifact_directory_still_absent",
            not (ROOT / "runs/v4_p001/rollout").exists(), "runs/v4_p001/rollout")

    host = {key: env[key] for key in ("hostname", "ips", "git_revision", "branch",
                                      "status_porcelain", "gpu_names", "gpu_compute_apps",
                                      "prereg_is_ancestor")}
    host["role"] = ("the formal node" if not informal else
                    "NOT the formal node: this transcript is a rehearsal and must be re-run on "
                    "fly122 before it means anything")
    script_hashes = S.script_version_hash(
        "scripts/v4_p001_rollout.py", "scripts/v4_p001_analyze.py", "scripts/v4_p001_spec.py",
        "scripts/v4_p001_dry_run.py", "scripts/v4_p001_context_four_arm.py",
        "scripts/v4_p001_prompt_provenance.py", "scripts/v4_p001_arm_schedule.py",
        "tests/test_v4_p001.py", "tests/test_v4_p001_analyze.py", "tests/test_v4_p001_rollout.py")
    transcript = {
        "artifact_type": "V4-P001_execution_package_dry_run",
        "experiment_id": S.EXPERIMENT_ID,
        "what_this_is": (
            "The owner-§19 no-generation dry run of the V4-P001 execution package: pool, screen "
            "order, model, verifier, context, runner, analyzer, artifact locations and recovery "
            "policy, exercised on the formal node at one committed revision. It is not a result: no "
            "model was loaded, no candidate was generated, no formal theorem was sent to the "
            "verifier (the only event is the canary), and no raw artifact was written anywhere. The "
            "freeze/validate/analyze middle section runs on a *synthetic* cohort that is explicitly "
            "marked and deleted before the script exits."),
        "status": "NOT_LAUNCHED",
        "informal_host": informal,
        "formal_candidates_generated": 0,
        "formal_verifier_calls": 0,
        "validated_commit": {
            "sha": env["git_revision"],
            "short": env["git_revision"][:9],
            "subject": subprocess.run(["git", "-C", str(ROOT), "log", "-1", "--pretty=%s"],
                                      capture_output=True, text=True, check=False).stdout.strip(),
            "branch": env["branch"],
        },
        "host": host,
        "interpreter": {"executable": sys.executable, "version": sys.version.split()[0]},
        "duration_seconds": round(time.perf_counter() - started, 1),
        "when_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_design": t.sections.get("frozen_design", {}),
        "pool": t.sections.get("pool", {}),
        "screen_order": t.sections.get("screen_order", {}),
        "model": t.sections.get("model", {}),
        "verifier": t.sections.get("verifier", {}),
        "recovery_policy": t.sections.get("recovery_policy", {}),
        "context": t.sections.get("context", {}),
        "runner": t.sections.get("runner", {}),
        "analyzer": t.sections.get("analyzer", {}),
        "artifact_locations": t.sections.get("artifact_locations", {}),
        "gates_on_this_host_at_this_commit": t.sections.get("gates_on_this_host_at_this_commit", {}),
        "script_hashes": script_hashes,
        "checks": t.checks,
        "n_checks": len(t.checks),
        "n_failed": t.n_failed,
        "status_of_this_transcript": "PASS" if t.n_failed == 0 else "FAIL",
        "cleanliness_waiver": t.sections.get("cleanliness_waiver", {}),
        "how_to_reproduce": [
            "git checkout <the validated_commit.sha above>",
            ".venv/bin/python scripts/v4_p001_dry_run.py    # on the formal node; writes this file",
            ".venv/bin/python -m pytest tests/              # expect the suite green",
            ".venv/bin/python -m ruff check src scripts tests",
            (".venv/bin/python scripts/v4_p001_context_four_arm.py    # re-derives the four-arm "
             "context audit from the consumed corpora"),
        ],
        "owner_constraints_observed": [
            "No V4 formal theorem generation: 0 candidates generated, 0 formal verifier calls",
            "the formal stages were refused without --i-have-owner-launch-authorization",
            ("the synthetic cohort lives under /tmp, is marked SYNTHETIC-DRY-RUN, and is deleted "
             "before exit; runs/ stays empty"),
            "no RL, SFT, checkpoint update, controller training or final-holdout use",
            "no verifier throughput, latency or capacity claim",
        ],
        "authorizations": {
            "V4_FORMAL_SCREENING_AUTHORIZED": "NO",
            "V4_SECOND_STAGE_AUTHORIZED": "NO",
            "note": ("this transcript is the evidence the owner asked for; the authorizations stay "
                     "NO until the owner says otherwise in writing"),
        },
    }
    return transcript


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="V4-P001 owner-§19 dry run: the whole execution package, no formal generation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--out", default=DEFAULT_OUT, help="where the transcript goes")
    parser.add_argument("--skip-suite", action="store_true",
                        help="skip the pytest gate (rehearsal only; the formal transcript runs it)")
    parser.add_argument("--allow-informal-host", action="store_true",
                        help="rehearsal on a non-fly122 host: record the host without failing on it")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="rehearsal from a dirty tree: record the working-tree state without "
                             "failing on it")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    transcript = build_transcript(run_suite=not args.skip_suite,
                                  allow_informal=args.allow_informal_host,
                                  allow_dirty=args.allow_dirty)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    S.write_json_atomic(out, transcript)
    failed = [entry for entry in transcript["checks"] if not entry["pass"]]
    print(json.dumps({"transcript": str(out), "n_checks": transcript["n_checks"],
                      "n_failed": transcript["n_failed"],
                      "status": transcript["status_of_this_transcript"],
                      "informal_host": transcript["informal_host"],
                      "failed": [entry["check"] for entry in failed]}, indent=2))
    if transcript["informal_host"]:
        print("[rehearsal] this host is not fly122; the transcript is marked informal and means "
              "nothing until the script runs on the formal node")
    return 0 if transcript["n_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
