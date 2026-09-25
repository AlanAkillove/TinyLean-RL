#!/usr/bin/env python3
"""V4-P001 frozen execution contract, shared by the rollout runner and the analyzer.

Same reason as V3's spec module: the runner and the analyzer must consume *exactly* the same
objects. The frozen settings, the raw-row schema, the screening/repair status vocabularies and every
hash check live here once, so the two executables cannot drift apart and only discover it after the
labels exist.

What this module deliberately is NOT: no GPU code, no vLLM, no HTTP, no statistics, no derangement
implementation. The design objects are *imported* from the frozen library modules
(`tinylean_rl.evaluation.v4_seeds`, `v4_schedule`, `v4_derange`, `v4_taxonomy`, `v4_stats`) rather
than restated, so a runner that used a different formula than the one that was frozen would fail on
import instead of on a claim. Nothing here re-derives a number that is already committed: it reads
the committed artifacts and *checks* them.

Every check in `load_frozen()` is fail-closed: it raises `FrozenViolation` rather than warning,
because a silently-wrong check on a preregistered gate is worse than no run at all.

Amendment A (four-arm design) is the authority for the arm set, the derangement, the censoring
guard, the guard order and the balanced arm schedule; the preregistration is the authority for the
pool, the taxonomy, the normalizer, the verifier policy and the thresholds.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "scripts", ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tinylean_rl.evaluation.v4_derange import DERANGEMENT_VERSION
from tinylean_rl.evaluation.v4_prompts import RENDERER_VERSION
from tinylean_rl.evaluation.v4_schedule import (
    ARM_ORDER,
    arm_order,
    schedule_hash,
)
from tinylean_rl.evaluation.v4_seeds import (
    MAX_SCREENING,
    N_PRIMARY,
    SECOND_STAGE_OFFSET,
    V4_BASE_SEED,
    first_stage_seed,
    second_stage_seed,
)
from tinylean_rl.evaluation.v4_stats import (
    ALPHA,
    BOOTSTRAP_REPS,
    BOOTSTRAP_SEED,
    CENSORING_RANGE_MAX,
    DATA_GUARD_MIN,
    DELTA_CA_MIN,
    DELTA_CB_MIN,
)
from tinylean_rl.verifier.policy import VerifyOutcome

EXPERIMENT_ID = "V4-P001"
TITLE = "Paired verifier-guided repairability probe, four-arm (A/B/C/D)"
BRANCH = "v3-jev-rl-controller"
PREREG_COMMIT = "84ab51b5cec14b9e01f36ec3db78e1271194f234"
PREREG_DOC = "docs/v4/V4-P001_preregistration.md"
AMENDMENT_DOC = "docs/v4/V4-P001_amendment_A_four_arm.md"

# --- frozen artifacts this runner reads and verifies ---------------------------------------------
POOL = "experiments/manifests/v4/v4_p001_pool.json"
SEEDS = "experiments/manifests/v4/v4_p001_seeds.json"
SCHEDULE = "experiments/manifests/v4/v4_p001_arm_schedule.json"
PROMPT_AUDIT = "experiments/manifests/v4/v4_p001_prompt_audit.json"
PROMPT_PROVENANCE = "experiments/manifests/v4/v4_p001_prompt_provenance.json"
CONTEXT_SMOKE = "experiments/manifests/v4/v4_p001_context_smoke.json"
CONTEXT_FOUR_ARM = "experiments/manifests/v4/v4_p001_context_four_arm.json"
VERIFIER_PLAN = "experiments/manifests/v4/v4_p001_verifier_plan.json"
POWER = "experiments/manifests/v4/v4_p001_power.json"
REGISTRY = "experiments/manifests/v4/registry.yaml"

TRAIN_PARQUET = "data/processed/p3_promptset/prompt_sets/AI-MO/Kimina-Prover-Promptset/train.parquet"
MODEL_DIR = "models/weights/kimina_distill_0_6b"

# --- frozen generation settings (preregistration §6, §9; Amendment A §7, §12) --------------------
N_ARMS = 4
POSITIONS = 4
SECOND_STAGE_CANDIDATES = N_PRIMARY * N_ARMS                     # 512
WORST_CASE_FORMAL_GENERATIONS = MAX_SCREENING + SECOND_STAGE_CANDIDATES   # 1152
TEMPERATURE = 1.0
TOP_P = 1.0
MAX_RESPONSE_TOKENS = 4096
MAX_MODEL_LEN = 10240
GPU_MEMORY_UTILIZATION = 0.85
MAX_NUM_BATCHED_TOKENS = 10240
MAX_NUM_SEQS = 256
N_SAMPLES = 1                                                   # one first attempt per theorem
DIAGNOSTIC_TOKEN_BUDGET = 512                                   # v4_diagnostics; restated only to
                                                                # assert it, never to apply it
SCREENING_SEED_HASH = "2efb53d8525b7ced44b13dc5aab388bbbf73ed920a65763918ef0e23ab0b8c51"
PAIRED_SEED_HASH = "06ba33b803201e8102b5537011bd242eca59b24768d69f256ac92335a6e6986d"

MODEL = {
    "repo_id": "AI-MO/Kimina-Prover-Distill-0.6B",
    "revision": "332e8a5259d1bdfda19d7c7f339f30804813cd3a",
    "dir": MODEL_DIR,
    "weights_filename": "model.safetensors",
    "weights_sha256": "34e6e630f564d330c79424c404ab0494558a0a659e6201b47d9bd88ccd640fe2",
}
EXPECTED_SMOKE_PEAK_VRAM_MIB = 9208     # §I smoke measurement; a budget reference, not a gate

# --- verifier policy (frozen B0 policy, shipped settings; §15) -----------------------------------
VERIFIER = {
    "server_timeout_s": 120.0,
    "first_timeout_s": 600.0,
    "client_slack_s": 60.0,
    "batch_size": 1,
    "canary_timeout_s": 60.0,
    "canary_retries": 1,
    "max_single_retries": 2,
    "policy": ("tinylean_rl.verifier.policy.VerificationSession: batched pass, isolation pass with "
               "bounded retries, canary gate after every infrastructure event, fail-close on an "
               "unhealthy server"),
}

# --- C′ dedicated verifier instance (Amendment A §16; plan `frozen`) -----------------------------
# Infrastructure, not science: no value here enters the reward, the prompts, the seeds or a gate.
# V4 reuses the *same* dedicated C′ instance V3-R001 used -- same pinned image, same MAX_REPLS = 1,
# same published loopback port -- rather than standing up a second container: a new container would
# have to rebuild the image's Mathlib cache before its first /verify could succeed, and the frozen
# verification semantics are guaranteed to be identical only if the server is identical. What V4
# changes is client-side only: the recovery ceiling is the one the verifier plan froze (192 per run,
# 3 per theorem) instead of R001's 16, because this run has a different candidate count and a
# per-theorem rule that R001 did not have.
VERIFIER_INFRA = {
    "endpoint": "http://127.0.0.1:8010",
    "container": "tinylean-rl-lean-server-r001",
    "image": "projectnumina/kimina-lean-server:2.0.0",
    "max_repls": 1,
    "max_wait_s": 60,
    "max_repl_mem": "8G",
    "verification_concurrency": 1,
    "compose_file": "infra/lean-server/r001/compose.yaml",
    "max_recoveries_per_run": 192,
    "max_recoveries_per_theorem": 3,
    "restart_timeout_s": 120.0,
    "health_poll_timeout_s": 180.0,
    "health_poll_interval_s": 2.0,
    "restart_grace_s": 10,
}

# --- screening statuses (preregistration §6), mutually exclusive ---------------------------------
INITIAL_SUCCESS = "INITIAL_SUCCESS"
PRIMARY_SEMANTIC_FAILURE = "PRIMARY_SEMANTIC_FAILURE"
SYNTAX_FAILURE = "SYNTAX_FAILURE"
FORMAT_FAILURE = "FORMAT_FAILURE"
TIMEOUT_OR_RESOURCE = "TIMEOUT_OR_RESOURCE"
INFRA_CENSORED = "INFRA_CENSORED"
CONTEXT_INELIGIBLE = "CONTEXT_INELIGIBLE"

SCREENING_STATUSES = (
    INITIAL_SUCCESS, PRIMARY_SEMANTIC_FAILURE, SYNTAX_FAILURE, FORMAT_FAILURE,
    TIMEOUT_OR_RESOURCE, INFRA_CENSORED, CONTEXT_INELIGIBLE,
)
#: Statuses that select the primary cohort: a real Lean-level diagnostic in a primary category.
COHORT_SELECTING_STATUSES = (PRIMARY_SEMANTIC_FAILURE,)
#: Statuses that are neither a success nor a model failure: missing data, never a hard negative.
SCREENING_MISSING_STATUSES = (INFRA_CENSORED, CONTEXT_INELIGIBLE)

GROUP_COMPLETE = "COMPLETE"
GROUP_INCOMPLETE = "INCOMPLETE"

FORMAT_ERROR = "format_error"
INFRA_STATUSES = {o.value for o in VerifyOutcome if o.is_infrastructure}
CANDIDATE_FAILURE_STATUSES = {o.value for o in VerifyOutcome if o.is_candidate_failure} | {
    FORMAT_ERROR}
CONCLUSIVE_STATUSES = {VerifyOutcome.VERIFIED.value} | CANDIDATE_FAILURE_STATUSES

SCREENING_SCHEMA_VERSION = "v4-p001-screening-1"
REPAIR_SCHEMA_VERSION = "v4-p001-repair-1"

#: Where the Arm-C/D diagnostic text of a screened candidate came from. Frozen vocabulary; only the
#: first two can produce text.
#:   verify_item      the first error-severity Lean message of the raw /verify item, with its
#:                    ``line L, column C:`` prefix (preregistration §8.1) -- the normal path
#:   verdict_message  the policy's own verdict message, used only when the raw item carries no
#:                    error-severity message (reachable for a ``sorry`` verdict, which Lean reports
#:                    as a warning): the only attributable text the verifier produced
#:   none             nothing attributable -- never a cohort member
DIAGNOSTIC_SOURCES = ("verify_item", "verdict_message", "none")

# --- raw record schemas --------------------------------------------------------------------------
# Screening: one row per screened theorem. The first-attempt completion is stored in full (it is the
# repair context of the second stage), plus the hash of the extracted proof the four arms receive and
# the *bounded, normalized diagnostic* Arms C and D will carry -- both are needed to render the
# second-stage prompts, and both are re-checked against the row when the boundary is validated. The
# diagnostic's provenance (`diagnostic_source`) travels with it so no renderer has to guess.
SCREENING_FIELDS = [
    "schema_version", "run_id", "experiment_id",
    "screening_rank", "formal_rank", "statement_id", "component_id", "name", "source",
    "tier", "classes", "prompt_tokens", "prompt_sha256", "prompt_token_count", "seed",
    "model_sha256", "model_revision",
    "completion_text", "completion_sha256", "generated_tokens", "truncated",
    "format_ok", "has_lean_block", "extracted_proof", "extracted_proof_sha256",
    "extracted_proof_present", "extracted_proof_tokens",
    "diagnostic_text", "diagnostic_sha256", "diagnostic_tokens", "diagnostic_source",
    "truncated_diagnostic", "context_fits", "context_tokens_with_response",
    "verified", "score", "acc", "verify_status", "lean_message",
    "error_category", "screening_status", "primary_eligible",
    "generation_time", "verification_time", "host", "gpu", "created_at",
]

# Repair: one row per arm candidate. ``formal_rank`` is the frozen cohort rank; the four arms of
# that rank share ``seed`` (common random numbers) and differ only in ``arm``, ``prompt_sha256``,
# ``prompt_token_count`` and ``position`` (the frozen balanced execution order).
#
# ``censored`` marks the verifier-plan rule "a theorem that exceeds its per-theorem recovery exposure
# has its remaining arm candidates marked INFRA_CENSORED": such a row was never generated and never
# verified, its ``score`` is None (missing data, never a failure) and ``censored_reason`` says why.
REPAIR_FIELDS = [
    "schema_version", "run_id", "experiment_id",
    "formal_rank", "screening_rank", "statement_id", "component_id", "name", "source",
    "error_category", "arm", "position", "seed",
    "prompt_sha256", "prompt_token_count", "context_tokens_with_response",
    "renderer_version", "normalization_version", "diagnostic_sha256", "diagnostic_source",
    "failed_proof_sha256",
    "model_sha256", "model_revision",
    "completion_text", "completion_sha256", "generated_tokens", "truncated",
    "format_ok", "has_lean_block", "extracted_proof_present",
    "verified", "score", "acc", "verify_status", "lean_message",
    "error_category_second", "success", "censored", "censored_reason",
    "generation_time", "verification_time", "host", "gpu", "created_at",
]

SCREENING_RAW_BASENAME = "v4_p001_screening_raw.jsonl"
SECOND_STAGE_RAW_BASENAME = "v4_p001_second_stage_raw.jsonl"
SCREENING_SUMMARY_BASENAME = "v4_p001_screening_summary.json"
SUMMARY_BASENAME = "v4_p001_run_summary.json"
RECOVERY_LOG_BASENAME = "v4_p001_recovery_log.jsonl"
VERIFIER_EVENTS_BASENAME = "v4_p001_verifier_events.jsonl"

PRIMARY_COHORT_BASENAME = "V4-P001_primary_cohort.json"
DERANGEMENT_BASENAME = "V4-P001_diagnostic_derangement.json"
SECOND_STAGE_PLAN_BASENAME = "V4-P001_second_stage_plan.json"
BOUNDARY_VALIDATION_BASENAME = "V4-P001_boundary_validation.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha(obj) -> str:
    """The canonical-object hash recipe used by every V4 freeze script."""
    return sha256_text(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


class FrozenViolation(RuntimeError):
    """A frozen object is not what was committed. Nothing may proceed past this."""


def candidate_score(status: str) -> int | None:
    """1 verified, 0 conclusive failure, None when the verifier could not decide.

    `None` is the point: an unresolved verifier outcome leaves the label missing. It is never
    silently written as a failure (the bug that made V1's '# System Error:' rows look like hard
    negatives), and the sensitivity reading that completes it with a failure is reported separately
    and never redefines the primary endpoint (Amendment A §5).
    """
    if status == VerifyOutcome.VERIFIED.value:
        return 1
    if status in CANDIDATE_FAILURE_STATUSES:
        return 0
    return None


def screening_status(
    *,
    verify_status: str,
    category: str,
    primary_eligible: bool,
    context_fits: bool,
) -> str:
    """Map one first-attempt verdict to its frozen screening status. Exactly one status per row."""
    if verify_status == VerifyOutcome.VERIFIED.value:
        return INITIAL_SUCCESS
    if verify_status in INFRA_STATUSES:
        return INFRA_CENSORED
    if not context_fits:
        return CONTEXT_INELIGIBLE
    if category in {TIMEOUT_OR_RESOURCE, }:
        return TIMEOUT_OR_RESOURCE
    if category == "format_no_code":
        return FORMAT_FAILURE
    if category == "syntax_parser":
        return SYNTAX_FAILURE
    if primary_eligible:
        return PRIMARY_SEMANTIC_FAILURE
    # `infra` from the taxonomy can only arrive with an infra verify_status, which is handled above;
    # anything else is a Lean rejection the taxonomy could not place in a primary category.
    return TIMEOUT_OR_RESOURCE if category == "timeout_resource" else SYNTAX_FAILURE


def screening_group_status(rows: list[dict]) -> str:
    """One screening row per theorem, so the group is complete iff that row exists and is tagged."""
    if len(rows) != 1 or rows[0].get("screening_status") not in SCREENING_STATUSES:
        return GROUP_INCOMPLETE
    return GROUP_COMPLETE


def repair_group_status(rows: list[dict]) -> str:
    """All four arms of one theorem, in the frozen arm set, one row each."""
    arms = sorted(str(r.get("arm")) for r in rows)
    return GROUP_COMPLETE if arms == sorted(ARM_ORDER) else GROUP_INCOMPLETE


# --- frozen-object loading and verification ------------------------------------------------------

@dataclass
class Frozen:
    """The V4-P001 design as committed: pool, order, seeds, schedule, prompts, verifier plan."""

    pool: dict
    seeds: dict
    schedule: dict
    prompt_audit: dict
    provenance: dict
    context: dict
    verifier_plan: dict
    registry_pins: dict
    members: list                  # pool members, in frozen screening order (rank 1..1371)
    member_by_rank: dict
    member_by_statement: dict
    file_hashes: dict
    checks: list

    @property
    def screening_theorems(self) -> list:
        """The theorems a formal run may screen: frozen ranks 1..MAX_SCREENING, in frozen order."""
        return self.members[:MAX_SCREENING]


def load_json(rel: str):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def registry_file_pins() -> dict:
    """File-level sha256 pins recorded in the V4 registry, read without a YAML dependency."""
    text = (ROOT / REGISTRY).read_text(encoding="utf-8")
    pins = {}
    for key in ("pool_sha256", "seeds_sha256", "schedule_sha256", "prompt_audit_sha256",
                "context_smoke_sha256", "verifier_plan_sha256"):
        m = re.search(rf"^\s*{key}:\s*([0-9a-f]{{64}})\s*$", text, re.MULTILINE)
        if not m:
            raise FrozenViolation(f"{key} is not pinned in {REGISTRY}")
        pins[key] = m.group(1)
    return pins


def _canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _check(name: str, ok: bool, detail: str, checks: list) -> None:
    checks.append({"check": name, "pass": bool(ok), "detail": detail})
    if not ok:
        raise FrozenViolation(f"FROZEN CHECK FAILED -- {name}: {detail}")


def _sha_canonical(obj) -> str:
    return sha256_text(_canonical_json(obj))


# The two recipe literals below are the ones `scripts/v4_p001_pool_build.py` hashed the pool with.
# They are restated rather than imported because the build script is not a library, and they are
# *checked*: if the build script's recipe ever drifted, the recomputation below would no longer
# reproduce the committed `pool_hash` / `order_hash` and `load_frozen` would fail closed. A check
# that reproduced its own recipe would prove nothing.
_POOL_HASH_RECIPE = {
    "unit": "component_id",
    "representative_rule": "min sha256(statement_id) then first passing the 1024-token prompt filter",
}
_ORDER_HASH_RULE = ("tier blocks (tier1, then tier2 classes in frozen order), each sorted by "
                    "sha256(f'{seed}|{statement_id}')")


def pool_hash_recompute(pool: dict, members: list) -> str:
    """Recompute the frozen pool hash from the committed member list."""
    return _sha_canonical({
        **_POOL_HASH_RECIPE,
        "tiers": {"tier1": sorted(m["component_id"] for m in members if m["tier"] == "tier1"),
                  "tier2": sorted(m["component_id"] for m in members if m["tier"] == "tier2")},
        "max_prompt_length": pool["max_prompt_length"],
        "components": sorted(m["component_id"] for m in members),
        "statements": sorted(m["statement_id"] for m in members),
        "component_to_statement": {m["component_id"]: m["statement_id"]
                                   for m in sorted(members, key=lambda m: m["component_id"])},
    })


def order_hash_recompute(pool: dict, members: list) -> str:
    """Recompute the frozen screening-order hash, and the per-member order keys with it."""
    return _sha_canonical({
        "base_seed": pool["base_seed"],
        "rule": _ORDER_HASH_RULE,
        "order": [m["statement_id"] for m in members],
    })


def load_frozen(verify_files: bool = True) -> Frozen:
    """Load the frozen design and prove it is the design that was committed. ABORT on any mismatch."""
    checks: list = []
    pool = load_json(POOL)
    seeds = load_json(SEEDS)
    sched = load_json(SCHEDULE)
    prompt_audit = load_json(PROMPT_AUDIT)
    provenance = load_json(PROMPT_PROVENANCE)
    context = load_json(CONTEXT_FOUR_ARM)
    verifier_plan = load_json(VERIFIER_PLAN)
    load_json(POWER)                                   # present and parseable

    file_hashes = {
        rel: sha256_file(ROOT / rel) for rel in
        (POOL, SEEDS, SCHEDULE, PROMPT_AUDIT, PROMPT_PROVENANCE, CONTEXT_SMOKE,
         CONTEXT_FOUR_ARM, VERIFIER_PLAN)
    }
    pins = registry_file_pins()

    # 1. the pool is the frozen pool, and its own checks passed when it was built
    members = pool["members"]
    _check("pool_size", len(members) == pool["capacity"]["pool_components"]
           == pool["capacity"]["pool_statements"],
           f"{len(members)} members vs capacity {pool['capacity']['pool_components']}", checks)
    _check("pool_hash_recomputes", pool_hash_recompute(pool, members) == pool["pool_hash"],
           f"recomputed pool hash != committed {pool['pool_hash'][:16]}...", checks)
    _check("order_hash_recomputes", order_hash_recompute(pool, members) == pool["order_hash"],
           f"order hash != {pool['order_hash'][:16]}...", checks)
    _check("screening_order_keys_recompute",
           [m["screening_order_key"] for m in members]
           == [sha256_text(f"{pool['base_seed']}|{m['statement_id']}") for m in members],
           "a screening_order_key is not sha256(f'{base_seed}|{statement_id}')", checks)
    _check("screening_ranks_are_the_frozen_sequence",
           [m["screening_rank"] for m in members] == list(range(1, len(members) + 1)),
           "screening_rank is not the contiguous frozen order", checks)
    _check("one_representative_per_component",
           len({m["component_id"] for m in members}) == len(members)
           and len({m["statement_id"] for m in members}) == len(members),
           "duplicate component_id or statement_id in the pool", checks)
    _check("pool_self_checks_passed",
           all(int(v) == 0 for k, v in pool["checks"].items() if k.endswith("_touched"))
           and all(v is True for k, v in pool["checks"].items() if not k.endswith("_touched")),
           str(pool["checks"]), checks)
    # The sealed reserve is the reason the pool exists: assert the exclusion here too, because this
    # is the module a formal run trusts.
    _check("sealed_reserve_untouched",
           pool["checks"]["sealed_components_touched"] == 0
           and pool["checks"]["sealed_statements_touched"] == 0,
           "the pool touched a sealed V3-FINAL-HOLDOUT component or statement", checks)
    _check("v3_formal_sample_untouched",
           pool["checks"]["v3_formal_sample_components_touched"] == 0
           and pool["checks"]["v3_formal_sample_statements_touched"] == 0,
           "the pool overlaps the frozen V3 formal sample", checks)

    # 2. the screening budget and the seed streams are the frozen ones
    _check("screening_budget_frozen",
           pool["screening_budget"]["n_primary_required"] == N_PRIMARY
           and pool["screening_budget"]["max_screens"] == MAX_SCREENING,
           str(pool["screening_budget"]), checks)
    _check("screening_seed_stream_matches_the_formula",
           [row["seed"] for row in seeds["screening_schedule"]]
           == [first_stage_seed(rank) for rank in range(1, MAX_SCREENING + 1)],
           "the frozen screening schedule is not first_stage_seed(1..640)", checks)
    _check("screening_schedule_covers_the_frozen_order",
           [row["statement_id"] for row in seeds["screening_schedule"]]
           == [m["statement_id"] for m in members[:MAX_SCREENING]],
           "the seed schedule's statement order is not the pool's screening order", checks)
    _check("second_stage_seed_stream_matches_the_formula",
           list(seeds["paired_seeds_by_rank"])
           == [second_stage_seed(rank) for rank in range(1, N_PRIMARY + 1)],
           "the frozen paired seeds are not second_stage_seed(1..128)", checks)
    _check("frozen_seed_hashes",
           seeds["screening_seed_hash"] == SCREENING_SEED_HASH
           and seeds["paired_seed_hash"] == PAIRED_SEED_HASH,
           f"screening={seeds['screening_seed_hash'][:16]}... paired={seeds['paired_seed_hash'][:16]}...",
           checks)
    _check("second_stage_seeds_do_not_repeat_screening_seeds",
           not (set(seeds["paired_seeds_by_rank"])
                & {row["seed"] for row in seeds["screening_schedule"]}),
           "a second-stage seed repeats a screening seed", checks)

    # 3. the balanced arm schedule is the frozen one, and it is a permutation of the arm set
    _check("arm_schedule_hash_recomputes", schedule_hash() == sched["schedule_hash"],
           f"recomputed {schedule_hash()[:16]}... != committed {sched['schedule_hash'][:16]}...", checks)
    _check("arm_schedule_hash_is_the_frozen_constant",
           sched["schedule_hash"] == "9e090325ac19cf54cb5a7d68507fb97d632a49802e15c2b5dc21ce4ae617da73",
           sched["schedule_hash"], checks)
    _check("arm_schedule_is_a_permutation_at_every_rank",
           all(sorted(arm_order(rank)) == sorted(ARM_ORDER) for rank in range(1, N_PRIMARY + 1)),
           "an arm order is not a permutation of the four arms", checks)
    _check("arm_schedule_is_position_balanced",
           sched["balance"]["positions_per_arm"]["A_FRESH_RETRY"] == [32, 32, 32, 32]
           and all(counts == [32, 32, 32, 32]
                   for counts in sched["balance"]["positions_per_arm"].values()),
           str(sched["balance"]["positions_per_arm"]), checks)
    _check("arm_schedule_predecessors_balanced",
           sched["balance"]["predecessor_min"] >= 30 and sched["balance"]["predecessor_max"] <= 33,
           f"min={sched['balance']['predecessor_min']} max={sched['balance']['predecessor_max']}",
           checks)

    # 4. the prompt surface: four arms, enforced invariants, frozen hashes
    frozen_example = prompt_audit["frozen_example"]
    _check("four_arms_in_the_prompt_audit",
           sorted(frozen_example["prompt_hashes"]) == sorted(ARM_ORDER),
           str(sorted(frozen_example["prompt_hashes"])), checks)
    _check("arm_a_hash_is_the_frozen_three_arm_hash",
           frozen_example["prompt_hashes"]["A_FRESH_RETRY"]
           == "f2e2f2c2b9caec88a3955c3a47b922ff3a8afa7988fca817736c0a0e41b01811",
           "the Arm-A rendered template moved", checks)
    _check("arm_d_invariants_hold", all(frozen_example["arm_d_invariants"].values()),
           str(frozen_example["arm_d_invariants"]), checks)
    _check("arm_d_equals_arm_c_with_the_donor_diagnostic",
           frozen_example["arm_d_equals_arm_c_with_own_replaced_by_donor"] is True
           and frozen_example["donor_diagnostic_is_a_different_diagnostic"] is True,
           "Arm D is not Arm C with the own diagnostic replaced by a donor diagnostic", checks)
    _check("arm_a_has_no_previous_attempt_reference",
           frozen_example["arm_a_has_no_previous_attempt_reference"] is True, "Arm A mentions a "
           "previous attempt", checks)
    _check("arm_b_c_suffix_invariant", frozen_example["arm_b_c_suffix_invariant"] is True,
           "C is not B plus the diagnostic", checks)
    _check("prompt_lengths_hold_for_the_whole_pool",
           prompt_audit["prompt_lengths"]["renderer_matches_pool_prompt_tokens"] is True
           and prompt_audit["prompt_lengths"][
               "arm_d_rendering_is_c_with_the_donor_diagnostic_on_every_theorem"] is True,
           "the pool-wide prompt checks did not all hold when the audit ran", checks)
    _check("arm_a_canonical_reproduction_is_complete",
           all(v == 128 for v in
               provenance["arm_a_canonical_reproduction"]["v3_r001_archive"]["checks"].values())
           and provenance["arm_a_canonical_reproduction"]["v1_rollout_dumps"][
               "n_distinct_inputs"]
           == provenance["arm_a_canonical_reproduction"]["v1_rollout_dumps"][
               "n_byte_identical_to_arm_a"]
           == provenance["arm_a_canonical_reproduction"]["v1_rollout_dumps"][
               "n_matched_to_a_pinned_theorem"]
           and provenance["arm_a_canonical_reproduction"]["e023_holdout"]["n_records"]
           == provenance["arm_a_canonical_reproduction"]["e023_holdout"][
               "n_prompt_tokens_matched"],
           "Arm A did not reproduce the canonical historical prompt on 100% of the audited rows",
           checks)
    _check("renderer_is_not_claimed_as_the_native_chat_format",
           "NEW inference-only" in provenance["multiturn_renderer_audit"]["decision"]
           and "not described anywhere" in provenance["multiturn_renderer_audit"]["decision"],
           provenance["multiturn_renderer_audit"]["decision"][:200], checks)
    _check("all_provenance_checks_passed", all(provenance["checks"].values()),
           str({k: v for k, v in provenance["checks"].items() if not v}), checks)

    # 5. the rendering / normalization versions this run will use are the frozen ones
    _check("renderer_version_frozen", prompt_audit["renderer_version"] == RENDERER_VERSION,
           f"{prompt_audit['renderer_version']} != {RENDERER_VERSION}", checks)
    _check("normalization_version_frozen", prompt_audit["normalization_version"] == "v4-diag-1",
           str(prompt_audit["normalization_version"]), checks)
    _check("derangement_version_frozen", DERANGEMENT_VERSION == "v4-derange-1",
           DERANGEMENT_VERSION, checks)

    # 6. context: every arm fits, and the smoke that measured the VRAM is this design's
    _check("context_all_arms_fit", all(context["checks"].values()),
           str({k: v for k, v in context["checks"].items() if not v}), checks)
    _check("worst_case_context_is_within_the_window",
           context["worst_case_by_arm"]["C_VERIFIER_REPAIR"]["worst_case_total_tokens"]
           <= context["max_model_len"],
           f"worst case {context['worst_case_by_arm']['C_VERIFIER_REPAIR']} vs "
           f"{context['max_model_len']}", checks)
    _check("smoke_engine_matches_this_run",
           context["smoke_evidence"]["engine"]["max_model_len"] == MAX_MODEL_LEN
           and context["smoke_evidence"]["engine"]["gpu_memory_utilization"] == GPU_MEMORY_UTILIZATION,
           str(context["smoke_evidence"]["engine"]), checks)
    _check("smoke_prompt_is_this_designs_worst_case",
           context["checks"]["worst_case_c_reproduces_smoke_prompt_sha256"] is True,
           "the measured smoke prompt is not the worst-case Arm-C prompt of this design", checks)

    # 7. the verifier ceilings are the frozen ones
    _check("verifier_recovery_ceiling",
           verifier_plan["frozen"]["max_recoveries_per_run"] == VERIFIER_INFRA["max_recoveries_per_run"]
           and verifier_plan["frozen"]["max_recoveries_per_theorem"]
           == VERIFIER_INFRA["max_recoveries_per_theorem"],
           str(verifier_plan["frozen"]), checks)
    _check("recovery_is_not_a_candidate_retry",
           verifier_plan["frozen"]["recovery_is_not_a_candidate_retry"] is True,
           "the plan does not forbid adding an attempt through recovery", checks)

    # 8. the design gates this runner must not be able to move
    _check("gate_constants_are_the_frozen_ones",
           (DELTA_CA_MIN, DELTA_CB_MIN, ALPHA, BOOTSTRAP_REPS, BOOTSTRAP_SEED, DATA_GUARD_MIN,
            CENSORING_RANGE_MAX) == (0.08, 0.05, 0.05, 10_000, 20260925, 103, 0.05),
           "a gate constant moved", checks)
    _check("budget_arithmetic",
           SECOND_STAGE_CANDIDATES == 512 and WORST_CASE_FORMAL_GENERATIONS == 1152,
           f"{SECOND_STAGE_CANDIDATES} / {WORST_CASE_FORMAL_GENERATIONS}", checks)
    _check("model_identity_frozen",
           len(MODEL["revision"]) == 40 and len(MODEL["weights_sha256"]) == 64,
           str(MODEL), checks)

    if verify_files:
        # the theorem text the prompts render from, pinned by the pool
        pinned_inputs = pool["inputs"]
        for rel in (TRAIN_PARQUET,):
            on_disk = sha256_file(ROOT / rel) if (ROOT / rel).exists() else "missing"
            _check(f"pool_pin::{rel.split('/')[-1]}", on_disk == pinned_inputs[rel],
                   f"{rel} is {on_disk[:16]}... but the frozen pool pins "
                   f"{str(pinned_inputs[rel])[:16]}...", checks)
        weights = ROOT / MODEL["dir"] / MODEL["weights_filename"]
        _check("theta0_on_disk_matches_frozen",
               weights.exists() and sha256_file(weights) == MODEL["weights_sha256"],
               f"{weights} hash did not equal the frozen {MODEL['weights_sha256'][:16]}...", checks)
        for key, rel in (("pool_sha256", POOL), ("seeds_sha256", SEEDS),
                         ("schedule_sha256", SCHEDULE), ("prompt_audit_sha256", PROMPT_AUDIT),
                         ("context_smoke_sha256", CONTEXT_SMOKE),
                         ("verifier_plan_sha256", VERIFIER_PLAN)):
            _check(f"registry_pin::{key}", file_hashes[rel] == pins[key],
                   f"{rel} is {file_hashes[rel][:16]}... but {REGISTRY} pins {pins[key][:16]}...",
                   checks)

    by_rank = {m["screening_rank"]: m for m in members}
    by_statement = {m["statement_id"]: m for m in members}
    return Frozen(pool=pool, seeds=seeds, schedule=sched, prompt_audit=prompt_audit,
                  provenance=provenance, context=context, verifier_plan=verifier_plan,
                  registry_pins=pins, members=members, member_by_rank=by_rank,
                  member_by_statement=by_statement, file_hashes=file_hashes, checks=checks)


# --- host / git environment (checked, never assumed) ---------------------------------------------

def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True,
                          check=False).stdout.strip()


def _git_ok(*args: str) -> bool:
    return subprocess.run(["git", "-C", str(ROOT), *args], check=False).returncode == 0


def collect_env() -> dict:
    """Best-effort environment snapshot. Empty string where a probe is unavailable."""
    def run(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip()
        except OSError:
            return ""

    return {
        "hostname": run(["hostname"]),
        "ips": run(["hostname", "-I"]),
        "git_revision": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "status_porcelain": _git("status", "--porcelain"),
        "gpu_names": run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]),
        "gpu_compute_apps": run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"]),
        "prereg_is_ancestor": _git_ok("merge-base", "--is-ancestor", PREREG_COMMIT, "HEAD"),
    }


def check_environment(env: dict, require_formal_host: bool = True) -> list:
    """The fail-closed host/repo checks the runner must pass before it may touch a model."""
    checks: list = []
    if require_formal_host:
        _check("host_is_fly122", "10.3.25.122" in env.get("ips", ""),
               f"ips={env.get('ips')!r} (fly122 is the formal node)", checks)
        _check("gpu_is_rtx3080", "NVIDIA GeForce RTX 3080" in env.get("gpu_names", ""),
               f"gpu={env.get('gpu_names')!r}", checks)
        _check("gpu_is_otherwise_idle", env.get("gpu_compute_apps", "") == "",
               f"another compute job is on the GPU: {env.get('gpu_compute_apps')!r}", checks)
    _check("branch_is_the_v3_branch", env.get("branch") == BRANCH, f"branch={env.get('branch')!r}",
           checks)
    _check("prereg_commit_is_ancestor", env.get("prereg_is_ancestor") is True,
           f"HEAD {env.get('git_revision')} does not contain the preregistration {PREREG_COMMIT}",
           checks)
    _check("working_tree_is_clean", env.get("status_porcelain", "") == "",
           f"uncommitted changes: {env.get('status_porcelain')!r}", checks)
    return checks


def script_version_hash(*rels: str) -> dict:
    """sha256 of the executable bytes, so a result can always be traced to the code that made it."""
    return {rel: sha256_file(ROOT / rel) for rel in rels}


def rows_jsonl(path: Path):
    """Read a raw artifact line by line, skipping an unfinished torn tail rather than crashing."""
    if not path.exists():
        return
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def append_rows_durable(path: Path, rows: list[dict]) -> None:
    """Append one whole group as a single write, then fsync.

    Append-only + one group per call + fsync is what makes the raw artifact crash-safe: a reader sees
    either all rows of a group or fewer, and "fewer" is INCOMPLETE, which the resume path re-runs
    from the same deterministic seed. A row is never rewritten.
    """
    payload = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)
    if not payload:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())


def write_json_atomic(path: Path, payload: dict) -> None:
    """Write-by-temp-then-rename: a run summary is never observable half-written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(path)


def validate_screening_row(row: dict) -> None:
    """Every screening row must be self-identifying and internally consistent before it is written."""
    missing = [k for k in SCREENING_FIELDS if k not in row]
    if missing:
        raise FrozenViolation(f"screening row is missing schema fields {missing}")
    if row["schema_version"] != SCREENING_SCHEMA_VERSION:
        raise FrozenViolation(f"screening row carries schema_version={row['schema_version']!r}")
    if row["experiment_id"] != EXPERIMENT_ID:
        raise FrozenViolation(f"screening row carries experiment_id={row['experiment_id']!r}")
    status = str(row["verify_status"])
    if status not in CONCLUSIVE_STATUSES | INFRA_STATUSES:
        raise FrozenViolation(
            f"screening row verify_status={status!r} is outside the frozen B0 vocabulary; an "
            "unclassified verdict may not be written to a preregistered raw artifact")
    derived = candidate_score(status)
    if row["score"] != derived:
        raise FrozenViolation(
            f"screening row score={row['score']!r} contradicts verify_status={row['verify_status']!r}")
    if bool(row["verified"]) != (derived == 1):
        raise FrozenViolation(f"screening row verified={row['verified']!r} contradicts score")
    if row["screening_status"] not in SCREENING_STATUSES:
        raise FrozenViolation(f"screening_status={row['screening_status']!r} is not frozen")
    if row["screening_status"] == PRIMARY_SEMANTIC_FAILURE and not row["primary_eligible"]:
        raise FrozenViolation("a non-primary failure was tagged as cohort-selecting")
    if row["screening_status"] != PRIMARY_SEMANTIC_FAILURE and row["primary_eligible"]:
        raise FrozenViolation(
            f"primary_eligible={row['primary_eligible']!r} contradicts "
            f"screening_status={row['screening_status']!r}")
    if row["diagnostic_source"] not in DIAGNOSTIC_SOURCES:
        raise FrozenViolation(f"diagnostic_source={row['diagnostic_source']!r} is not frozen")
    if row["screening_status"] == PRIMARY_SEMANTIC_FAILURE and not str(row["diagnostic_text"]).strip():
        raise FrozenViolation(
            "a primary semantic failure carries no attributable Lean diagnostic; Arms C and D would "
            "have nothing to show, so the row may not select the cohort")
    if (row["screening_status"] == CONTEXT_INELIGIBLE) == bool(row["context_fits"]):
        raise FrozenViolation(
            f"context_fits={row['context_fits']!r} contradicts "
            f"screening_status={row['screening_status']!r}")


def validate_repair_row(row: dict) -> None:
    missing = [k for k in REPAIR_FIELDS if k not in row]
    if missing:
        raise FrozenViolation(f"repair row is missing schema fields {missing}")
    if row["schema_version"] != REPAIR_SCHEMA_VERSION:
        raise FrozenViolation(f"repair row carries schema_version={row['schema_version']!r}")
    if row["experiment_id"] != EXPERIMENT_ID:
        raise FrozenViolation(f"repair row carries experiment_id={row['experiment_id']!r}")
    if row["arm"] not in ARM_ORDER:
        raise FrozenViolation(f"repair row carries arm={row['arm']!r}")
    status = str(row["verify_status"])
    if status not in CONCLUSIVE_STATUSES | INFRA_STATUSES:
        raise FrozenViolation(f"repair row verify_status={status!r} is outside the frozen vocabulary")
    derived = candidate_score(status)
    if row["score"] != derived:
        raise FrozenViolation(
            f"repair row score={row['score']!r} contradicts verify_status={row['verify_status']!r}")
    if row["success"] != (derived == 1):
        raise FrozenViolation(f"repair row success={row['success']!r} contradicts score")
    if row["censored"] and (row["verify_status"] not in INFRA_STATUSES or derived is not None
                            or row["completion_text"]):
        raise FrozenViolation(
            "a censored repair row must be an unattempted, infrastructure-censored candidate "
            f"(status={row['verify_status']!r} score={row['score']!r} "
            f"completion={len(str(row['completion_text']))} chars)")


def read_grouped(path: Path, validate) -> dict:
    """screening_rank / formal_rank -> its rows, with the never-merge rule enforced on read."""
    groups: dict = {}
    for row in rows_jsonl(path):
        validate(row)
        key = row.get("formal_rank") if "formal_rank" in row else row.get("screening_rank")
        groups.setdefault(key, []).append(row)
    return groups


__all__ = [
    "ALPHA",
    "AMENDMENT_DOC",
    "ARM_ORDER",
    "BOOTSTRAP_REPS",
    "BOOTSTRAP_SEED",
    "BOUNDARY_VALIDATION_BASENAME",
    "CENSORING_RANGE_MAX",
    "COHORT_SELECTING_STATUSES",
    "CONCLUSIVE_STATUSES",
    "CONTEXT_FOUR_ARM",
    "CONTEXT_INELIGIBLE",
    "DATA_GUARD_MIN",
    "DELTA_CA_MIN",
    "DELTA_CB_MIN",
    "DERANGEMENT_BASENAME",
    "DIAGNOSTIC_SOURCES",
    "DIAGNOSTIC_TOKEN_BUDGET",
    "EXPERIMENT_ID",
    "FORMAT_ERROR",
    "FORMAT_FAILURE",
    "FROZEN_SETTINGS",
    "GROUP_COMPLETE",
    "GROUP_INCOMPLETE",
    "INFRA_CENSORED",
    "INFRA_STATUSES",
    "INITIAL_SUCCESS",
    "MAX_MODEL_LEN",
    "MAX_RESPONSE_TOKENS",
    "MAX_SCREENING",
    "MODEL",
    "MODEL_DIR",
    "N_ARMS",
    "N_PRIMARY",
    "N_SAMPLES",
    "PAIRED_SEED_HASH",
    "POOL",
    "PREREG_COMMIT",
    "PRIMARY_COHORT_BASENAME",
    "PRIMARY_SEMANTIC_FAILURE",
    "PROMPT_AUDIT",
    "PROMPT_PROVENANCE",
    "RECOVERY_LOG_BASENAME",
    "REGISTRY",
    "REPAIR_FIELDS",
    "REPAIR_SCHEMA_VERSION",
    "SCHEDULE",
    "SCREENING_FIELDS",
    "SCREENING_MISSING_STATUSES",
    "SCREENING_RAW_BASENAME",
    "SCREENING_SCHEMA_VERSION",
    "SCREENING_SEED_HASH",
    "SCREENING_STATUSES",
    "SCREENING_SUMMARY_BASENAME",
    "SECOND_STAGE_CANDIDATES",
    "SECOND_STAGE_OFFSET",
    "SECOND_STAGE_PLAN_BASENAME",
    "SECOND_STAGE_RAW_BASENAME",
    "SEEDS",
    "SUMMARY_BASENAME",
    "SYNTAX_FAILURE",
    "TEMPERATURE",
    "TIMEOUT_OR_RESOURCE",
    "TOP_P",
    "TRAIN_PARQUET",
    "V4_BASE_SEED",
    "VERIFIER",
    "VERIFIER_EVENTS_BASENAME",
    "VERIFIER_INFRA",
    "VERIFIER_PLAN",
    "WORST_CASE_FORMAL_GENERATIONS",
    "Frozen",
    "FrozenViolation",
    "append_rows_durable",
    "arm_order",
    "candidate_score",
    "check_environment",
    "collect_env",
    "first_stage_seed",
    "load_frozen",
    "load_json",
    "read_grouped",
    "registry_file_pins",
    "repair_group_status",
    "rows_jsonl",
    "screening_group_status",
    "screening_status",
    "script_version_hash",
    "second_stage_seed",
    "sha",
    "sha256_file",
    "sha256_text",
    "validate_repair_row",
    "validate_screening_row",
    "write_json_atomic",
]

FROZEN_SETTINGS = {
    "experiment_id": EXPERIMENT_ID,
    "arms": list(ARM_ORDER),
    "n_primary": N_PRIMARY,
    "n_arms": N_ARMS,
    "max_screening": MAX_SCREENING,
    "second_stage_candidates": SECOND_STAGE_CANDIDATES,
    "worst_case_formal_generations": WORST_CASE_FORMAL_GENERATIONS,
    "temperature": TEMPERATURE,
    "top_p": TOP_P,
    "max_response_tokens": MAX_RESPONSE_TOKENS,
    "max_model_len": MAX_MODEL_LEN,
    "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
    "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
    "max_num_seqs": MAX_NUM_SEQS,
    "n_samples_first_attempt": N_SAMPLES,
    "seed_formulas": {
        "first_stage": "V4_BASE_SEED + (screening_rank - 1), screening_rank in 1..640",
        "second_stage": ("V4_BASE_SEED + SECOND_STAGE_OFFSET + (formal_rank - 1), formal_rank in "
                         "1..128, shared by A/B/C/D"),
    },
    "renderer_version": RENDERER_VERSION,
    "derangement_version": DERANGEMENT_VERSION,
    "verifier": VERIFIER,
    "verifier_infra": VERIFIER_INFRA,
    "gates": {"delta_ca_min": DELTA_CA_MIN, "delta_cb_min": DELTA_CB_MIN, "alpha": ALPHA,
              "bootstrap_reps": BOOTSTRAP_REPS, "bootstrap_seed": BOOTSTRAP_SEED,
              "data_guard_min": DATA_GUARD_MIN, "censoring_range_max": CENSORING_RANGE_MAX},
    "model": MODEL,
    "prereg_commit": PREREG_COMMIT,
    "screening_schema_version": SCREENING_SCHEMA_VERSION,
    "repair_schema_version": REPAIR_SCHEMA_VERSION,
}
