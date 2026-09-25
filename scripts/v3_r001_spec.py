#!/usr/bin/env python3
"""V3-R001 frozen execution contract, shared by the rollout runner and the analyzer.

Why one module for both: the runner and the analyzer must consume *exactly* the same objects. If
each carried its own copy of the settings, the schema or the block rule, they could drift apart and
the drift would only show up after the labels exist. So the schema, the group-finalization rule and
every frozen hash check live here once, and both executables import it.

What this module deliberately is NOT: it has no GPU code, no vLLM, no HTTP and no statistics of its
own. Statistics come from `scripts/v3_r001_gate.py` (the frozen gate), hashing comes from the same
`sha` recipe that froze the sample, and nothing here re-derives a number that is already committed.

Every check in `load_frozen()` is fail-closed: it raises `FrozenViolation` rather than warning,
because a silently-wrong check on a preregistered gate is worse than no run at all.
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

# the hash recipe and the block rule that froze the sample and the gate, imported not restated.
# `sha` in scripts/v3_r001_gate.py is byte-identical to the one in v3_r001_prescore.py (pinned by
# test_the_two_freeze_scripts_hash_identically), so a hash computed here is the same object.
from v3_r001_gate import TOP_FRACTION, block_size, sha

from tinylean_rl.verifier.policy import VerifyOutcome

EXPERIMENT_ID = "V3-R001"
READING = "consumed_only"                                    # owner decision 1
SAMPLE_KEY = f"{READING}|N=128"                              # the only analyzable sample
BRANCH = "v3-jev-rl-controller"
PREREG_COMMIT = "d9874d3102c519244d6b8e28a6520b108affe6b7"    # the frozen preregistration
PREREG_DOC = "docs/v3/V3-R001_preregistration.md"

POOL = "experiments/manifests/v3/V3-R001_family_clean_pool.json"
PREDICTIONS = "experiments/manifests/v3/V3-R001_predictions.json"
FORMAL_SAMPLE = "experiments/manifests/v3/v3_r001_formal_sample.json"
RESERVE = "experiments/manifests/v3/v3_final_holdout_reserve.json"
GATE = "experiments/manifests/v3/V3-R001_gate.json"
SOURCES = "experiments/manifests/v3/v1_rollout_sources.json"  # frozen theta0 identity lives here
REGISTRY = "experiments/manifests/v3/registry.yaml"
# the theorem text the prompts are rendered from. Its hash is pinned inside the frozen pool
# artifact's `inputs`, so the runner verifies it before it sends a single prompt.
RAW_PARQUET = "data/raw/kimina_promptset/data/train-00000-of-00001.parquet"
TRAIN_PARQUET = "data/processed/p3_promptset/prompt_sets/AI-MO/Kimina-Prover-Promptset/train.parquet"

# --- frozen generation settings (preregistration section 6; a test pins these to the document) ---
N_NOMINAL = 128
N_SAMPLES = 8
# TOP_FRACTION is imported from the frozen gate rather than restated: the block size used to
# interpret the rollout must be the block size that was committed before it.
TEMPERATURE = 1.0
TOP_P = 1.0
MAX_RESPONSE_TOKENS = 4096
MAX_MODEL_LEN = 5120
GPU_MEMORY_UTILIZATION = 0.85
CHUNK_THEOREMS = 16
SEED_BASE = 20260924            # the directive date, frozen in preregistration section 6
SEED_GROUP_SIZE = 8             # canonical n=8 schedule, as defined in scripts/e024_minif2f_eval.py
DRAW_SEED = 20260924            # scripts/v3_r001_prescore.py:DRAW_SEED; a wrong value here makes the
                                # recomputed sample hash fail, so it is checked, not trusted
VERIFIER = {                    # preregistration section 6 timeouts, applied through the B0 policy
    "server_timeout_s": 120.0,      # explicit per-request server budget (E024's --verify-timeout)
    "first_timeout_s": 600.0,       # cold-start budget, used only by the prewarm/canary before group 1
    "client_slack_s": 60.0,         # B0 rule: client timeout must exceed the server timeout
    "batch_size": 4,
    "canary_timeout_s": 60.0,
    "canary_retries": 1,
    "max_single_retries": 2,
    "policy": ("tinylean_rl.verifier.policy.VerificationSession: batched pass, isolation pass with "
               "bounded retries, canary gate after every infrastructure event, fail-close on an "
               "unhealthy server"),
    "worker_note": ("B0's session is deliberately sequential: its canary must not run while another "
                    "verification is in flight (a queued canary on a saturated REPL pool returns 429 "
                    "and would be misread as unhealth). E024's 4-worker path cannot be canary-gated "
                    "mid-flight, so R001 accepts lower verification throughput for the guarantee that "
                    "no infrastructure event is silently absorbed."),
}

# --- frozen host policy (owner decision 16; fly90 is archive/coordination only) ------------------
FORMAL_HOST_IP = "10.3.25.122"
FORMAL_GPU_NAME = "NVIDIA GeForce RTX 3080"
EXPECTED_TOTAL_CANDIDATES = N_NOMINAL * N_SAMPLES      # 1024
SCHEMA_VERSION = "v3-r001-raw-1"

# --- the one settings object both executables hash (owner 2: config must be checked, not assumed) --
FROZEN_SETTINGS = {
    "experiment_id": EXPERIMENT_ID,
    "reading": READING,
    "sample_key": SAMPLE_KEY,
    "n_nominal": N_NOMINAL,
    "n_samples": N_SAMPLES,
    "top_fraction": TOP_FRACTION,
    "temperature": TEMPERATURE,
    "top_p": TOP_P,
    "max_response_tokens": MAX_RESPONSE_TOKENS,
    "max_model_len": MAX_MODEL_LEN,
    "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
    "chunk_theorems": CHUNK_THEOREMS,
    "seed_base": SEED_BASE,
    "seed_group_size": SEED_GROUP_SIZE,
    "seed_formula": "seed_base + (formal_sample_rank - 1) * seed_group_size + sample_index",
    "seed_rank_source": ("v3_r001_formal_sample.json theorems[].formal_sample_rank -- the frozen "
                         "draw order 1..128, independent of q_B2, q_B1, source and any outcome "
                         "(owner amendment B, 2026-09-25)"),
    "draw_seed": DRAW_SEED,
    "verifier": VERIFIER,
    "prereg_commit": PREREG_COMMIT,
    "formal_host_ip": FORMAL_HOST_IP,
    "formal_gpu_name": FORMAL_GPU_NAME,
    "schema_version": SCHEMA_VERSION,
}

# --- raw record schema (owner 2026-09-24 review section 6) --------------------------------------
# V1's rollout dump is 10 keys with no theorem id, which is why D001 had to recover identity by
# regex-matching the prompt text against the parquet. R001 carries identity in every row. The raw
# prompt text is not stored 1024 times: `prompt_sha256` pins the canonical generation string and
# `frozen_prompt_sha256` pins the string the frozen q was computed on, so either is reconstructible
# from (statement_id, the pinned parquet, the pinned chat template). They are not the same string --
# one carries the chat template's special tokens -- and the exact relation between them is what the
# runner's prompt check verifies before any theorem is generated.
CANDIDATE_FIELDS = [
    "schema_version", "run_id",
    "experiment_id", "theorem_rank", "formal_sample_rank", "statement_id", "component_id",
    "name", "source",
    "sample_index", "seed",
    "prompt_sha256", "frozen_prompt_sha256", "formal_statement_sha256",
    "model_sha256", "model_revision",
    "completion_text", "completion_sha256", "generated_tokens", "truncated",
    "format_ok", "verified", "score", "acc",
    "verify_status", "verifier_error_category", "lean_message",
    "generation_time", "verification_time",
    "host", "gpu", "created_at",
]

GROUP_COMPLETE = "COMPLETE"
GROUP_INFRA_CENSORED = "INFRA_CENSORED"
GROUP_INCOMPLETE = "INCOMPLETE"
GROUP_STATUSES = [GROUP_COMPLETE, GROUP_INFRA_CENSORED, GROUP_INCOMPLETE]

# The status vocabulary is IMPORTED from the B0 policy module, not restated: R001 must classify
# infrastructure trouble and proof failure with the same words the verifier reliability audit used,
# or the two datasets could not be compared. `FORMAT_ERROR` is the runner's single addition -- a
# candidate with no extractable proof body is never sent to the verifier, so it is a conclusive
# candidate failure (a real zero), explicitly NOT an infrastructure state.
FORMAT_ERROR = "format_error"

INFRA_STATUSES = {o.value for o in VerifyOutcome if o.is_infrastructure}
CANDIDATE_FAILURE_STATUSES = {o.value for o in VerifyOutcome if o.is_candidate_failure} | {
    FORMAT_ERROR}
CONCLUSIVE_STATUSES = {VerifyOutcome.VERIFIED.value} | CANDIDATE_FAILURE_STATUSES


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class FrozenViolation(RuntimeError):
    """A frozen object is not what was committed. Nothing may proceed past this."""


def candidate_score(status: str) -> int | None:
    """The PRIMARY reward label: 1 verified, 0 conclusive failure, None when undecidable.

    `None` is the point: an unresolved verifier outcome leaves the label missing. It is never
    silently written as 0 (owner decision 5), which is the failure mode that made V1's
    '# System Error:' rows look like hard negatives.
    """
    if status == VerifyOutcome.VERIFIED.value:
        return 1
    if status in CANDIDATE_FAILURE_STATUSES:
        return 0
    return None



def candidate_seed(formal_sample_rank: int, sample_index: int) -> int:
    """Deterministic per-candidate seed: base + (rank-1)*8 + sample_index.

    `formal_sample_rank` is the frozen 1..128 **draw order** rank from the formal sample manifest
    (`formal_sample_rank`), NOT `rank_by_q_B2`: the generation RNG stream must not be a function of
    what the controller predicted or of how it ranked the theorems (owner amendment B, 2026-09-25).
    Because the rank is a frozen constant per theorem, the seed of a candidate depends on no
    scheduling, batching or process order -- which is what makes a resumed group provably identical
    to an uninterrupted one.
    """
    if not 1 <= formal_sample_rank <= N_NOMINAL:
        raise FrozenViolation(f"formal_sample_rank {formal_sample_rank} outside 1..{N_NOMINAL}")
    if not 0 <= sample_index < N_SAMPLES:
        raise FrozenViolation(f"sample_index {sample_index} outside 0..{N_SAMPLES - 1}")
    return SEED_BASE + (formal_sample_rank - 1) * SEED_GROUP_SIZE + sample_index


def is_infra_candidate(record: dict) -> bool:
    """True when this candidate carries no decidable reward because of the verifier, not the proof."""
    return candidate_score(str(record.get("verify_status", ""))) is None


def finalize_group(records: list[dict]) -> dict:
    """One explicit group status per theorem; nothing is silently all-fail.

    COMPLETE         all 8 candidates carry a decidable reward.
    INFRA_CENSORED   >= 1 candidate is an unresolved infrastructure failure -> label is missing.
    INCOMPLETE       the runner did not finish this group (missing/repeated candidates).
    """
    n = len(records)
    ids = [r.get("sample_index") for r in records]
    if n != N_SAMPLES or sorted(ids) != list(range(N_SAMPLES)):
        return {
            "group_status": GROUP_INCOMPLETE, "y": None, "n_candidates": n,
            "n_pos_score": None, "n_infra_candidates": None,
            "reason": f"expected {N_SAMPLES} distinct sample_index 0..{N_SAMPLES - 1}, saw {ids}",
        }
    infra = [r["sample_index"] for r in records if is_infra_candidate(r)]
    if infra:
        return {
            "group_status": GROUP_INFRA_CENSORED, "y": None, "n_candidates": n,
            "n_pos_score": None, "n_infra_candidates": len(infra),
            "reason": f"unresolved verifier outcome on sample_index {infra}; label stays missing, "
                      "the group is excluded and NOT counted all-fail (owner decision 5)",
        }
    n_pos = sum(int(candidate_score(str(r["verify_status"]))) for r in records)
    return {
        "group_status": GROUP_COMPLETE, "y": 1 if 0 < n_pos < N_SAMPLES else 0,
        "n_candidates": n, "n_pos_score": n_pos, "n_infra_candidates": 0,
        "reason": f"y = 1 iff 0 < sum(score) < 8; observed sum = {n_pos}",
    }


# --- frozen-object loading and verification -----------------------------------------------------

@dataclass
class Frozen:
    """The R001 design as committed: sample, ranking, block, gate, model identity, hashes."""

    formal: dict
    gate: dict
    predictions: dict
    pool: dict
    reserve: dict
    theta0: dict
    theorems: list                    # the 128 rows, in frozen q-rank order
    draw_order: list                  # statement_ids in frozen draw order
    block_ids: list                   # the frozen top-20% block, q-rank order
    q_B2: list
    q_B1: list
    file_hashes: dict
    checks: list


def load_json(rel: str):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def registry_file_pins() -> dict:
    """The file-level sha256 pins recorded in the V3 registry, read without a YAML dependency."""
    text = (ROOT / REGISTRY).read_text(encoding="utf-8")
    pins = {}
    for key in ("formal_sample_sha256", "final_reserve_sha256", "gate_sha256"):
        m = re.search(rf"^\s*{key}:\s*([0-9a-f]{{64}})\s*$", text, re.MULTILINE)
        if not m:
            raise FrozenViolation(f"{key} is not pinned in {REGISTRY}")
        pins[key] = m.group(1)
    return pins


def _check(name: str, ok: bool, detail: str, checks: list) -> None:
    checks.append({"check": name, "pass": bool(ok), "detail": detail})
    if not ok:
        raise FrozenViolation(f"FROZEN CHECK FAILED -- {name}: {detail}")


def load_frozen(verify_files: bool = True) -> Frozen:
    """Load the frozen design and prove it is the design that was committed. ABORT on any mismatch."""
    checks: list = []
    formal = load_json(FORMAL_SAMPLE)
    gate = load_json(GATE)
    predictions = load_json(PREDICTIONS)
    pool = load_json(POOL)
    reserve = load_json(RESERVE)
    theta0 = load_json(SOURCES)["theta0"]

    file_hashes = {
        rel: sha256_file(ROOT / rel) for rel in
        (FORMAL_SAMPLE, GATE, RESERVE, POOL, PREDICTIONS, SOURCES)
    }

    # 1. the sample is the one analyzable sample, at the frozen size
    _check("sample_key_is_the_frozen_one", formal["predictions_sample_key"] == SAMPLE_KEY,
           f"manifest says {formal['predictions_sample_key']}", checks)
    theorems = formal["theorems"]
    _check("formal_sample_size", len(theorems) == N_NOMINAL, f"{len(theorems)} rows", checks)
    _check("gate_agrees_on_size", gate["design"]["N_nominal"] == N_NOMINAL
           and gate["design"]["top_fraction"] == TOP_FRACTION, str(gate["design"]), checks)

    # 2. one theorem per family, unique ids, no historical label, every q present
    comps = [t["component_id"] for t in theorems]
    stmts = [t["statement_id"] for t in theorems]
    _check("component_uniqueness", len(set(comps)) == N_NOMINAL, f"{len(set(comps))} distinct", checks)
    _check("statement_uniqueness", len(set(stmts)) == N_NOMINAL, f"{len(set(stmts))} distinct", checks)
    _check("one_theorem_per_component", formal["coverage"]["one_theorem_per_component"] is True,
           str(formal["coverage"]), checks)
    _check("no_historical_label", not any(t["has_historical_label"] for t in theorems),
           "a candidate in the formal sample already carries a label", checks)
    q_B2 = [t["q_B2_controller"] for t in theorems]
    q_B1 = [t["q_B1_handcrafted"] for t in theorems]
    _check("all_q_finite", all(0.0 <= v <= 1.0 for v in q_B2 + q_B1),
           "a q value is missing or outside [0,1]", checks)

    # 3. the frozen hashes: sample identity, draw order, q vectors, block membership
    pred_sample = predictions["prospective_samples"][SAMPLE_KEY]
    draw_order = list(pred_sample["statement_ids"])          # the frozen draw order, not a sort
    _check("draw_order_is_the_frozen_sample", sorted(draw_order) == sorted(stmts)
           and len(draw_order) == N_NOMINAL, f"{len(set(draw_order))} distinct of {len(draw_order)}",
           checks)
    pair = {t["statement_id"]: t["component_id"] for t in theorems}
    _check("draw_pairing_matches_the_manifest",
           all(pair.get(s) == c for s, c in zip(draw_order, pred_sample["component_ids"])),
           "a statement/component pairing differs between predictions and formal sample", checks)
    _check("draw_seed_reproduces_the_sample_sha",
           sha({"reading": READING, "seed": DRAW_SEED, "components": pred_sample["component_ids"]})
           == pred_sample["sample_sha256"] == formal["sample_sha256"],
           f"recomputed from seed {DRAW_SEED}; committed {formal['sample_sha256'][:16]}...", checks)
    _check("sample_hash_matches_the_gate", gate["design"]["formal_sample_sha256"]
           == formal["sample_sha256"], "gate and manifest disagree", checks)
    # the q vectors are hashed in DRAW order by the freeze script, so the comparison has to be too:
    # this is what proves the ranking the block was cut from is the ranking that was committed.
    q2_by_stmt = {t["statement_id"]: t["q_B2_controller"] for t in theorems}
    q1_by_stmt = {t["statement_id"]: t["q_B1_handcrafted"] for t in theorems}
    idp = formal["identity_with_the_power_sample"]
    _check("q_B2_vector_hash", sha([q2_by_stmt[s] for s in draw_order])
           == pred_sample["q_B2_vector_sha256"] == idp["q_B2_vector_sha256"]
           == idp["committed_q_B2_vector_sha256"],
           idp["q_B2_vector_sha256"], checks)
    _check("q_B1_vector_hash", sha([q1_by_stmt[s] for s in draw_order])
           == pred_sample["q_B1_vector_sha256"] == idp["q_B1_vector_sha256"]
           == idp["committed_q_B1_vector_sha256"],
           idp["q_B1_vector_sha256"], checks)

    block_ids = formal["top20pct_block"]["statement_ids"]
    _check("block_size_is_frozen", formal["top20pct_block"]["size"] == len(block_ids)
           == block_size(N_NOMINAL), f"{len(block_ids)} vs m={gate['design']['top_block_size']}",
           checks)
    _check("block_matches_the_gate", len(block_ids) == gate["design"]["top_block_size"],
           "the gate's frozen m differs from the manifest block", checks)
    _check("block_hash_recomputes", sha(block_ids) == formal["top20pct_block"]["sha256"],
           formal["top20pct_block"]["sha256"], checks)
    # the block must be exactly the top of the frozen q ranking, so no outcome can move membership
    ranked = [t["statement_id"] for t in sorted(
        theorems, key=lambda t: (-t["q_B2_controller"], t["statement_id"]))]
    _check("block_is_the_frozen_q_top", ranked[:len(block_ids)] == list(block_ids),
           "the committed block is not the top-m of the frozen q_B2 ranking", checks)
    _check("ranks_are_consecutive", sorted(t["rank_by_q_B2"] for t in theorems)
           == list(range(1, N_NOMINAL + 1)), "rank_by_q_B2 is not a 1..128 permutation", checks)
    # Amendment B: the generation-seed rank must be the outcome-free draw order, so that the RNG
    # stream cannot depend on the controller's ranking (or on anything a q touched). The check is on
    # the artifact, and re-derived from `draw_order`, not trusted to be consistent with itself.
    seed_ranks = sorted(t.get("formal_sample_rank", -1) for t in theorems)
    order_by_rank = sorted(theorems, key=lambda t: t.get("formal_sample_rank", -1))
    _check("formal_sample_rank_is_the_frozen_draw_order",
           seed_ranks == list(range(1, N_NOMINAL + 1))
           and [t["statement_id"] for t in order_by_rank] == list(draw_order),
           f"formal_sample_rank must be 1..128 in the frozen draw order (amendment B); saw "
           f"{seed_ranks[:3]}..{seed_ranks[-1]}", checks)

    # 4. the reserve stays sealed and disjoint from this sample
    reserve_comps = {c["component_id"] for c in reserve["components"]}
    _check("reserve_sealed", reserve["status"] == "SEALED", str(reserve["status"]), checks)
    _check("sample_touches_no_sealed_component", not (set(comps) & reserve_comps),
           f"{len(set(comps) & reserve_comps)} formal components are in the sealed reserve", checks)
    _check("partition_is_exact", reserve["membership_hashes"]["formal_plus_reserve_equals_pool"]
           is True, str(reserve["membership_hashes"]), checks)

    # 5. model identity is the frozen theta0, and the file on disk is that model
    _check("theta0_identity_present",
           len(theta0.get("weights_sha256", "")) == 64 and len(theta0.get("revision", "")) == 40,
           str(theta0), checks)
    if verify_files:
        weights = ROOT / theta0["dir"] / "model.safetensors"
        _check("theta0_on_disk_matches_frozen", weights.exists()
               and sha256_file(weights) == theta0["weights_sha256"],
               f"{weights} hash did not equal the frozen {theta0['weights_sha256'][:16]}...", checks)
        pins = registry_file_pins()
        for key, rel in (("formal_sample_sha256", FORMAL_SAMPLE),
                         ("final_reserve_sha256", RESERVE), ("gate_sha256", GATE)):
            _check(f"registry_pin::{key}", file_hashes[rel] == pins[key],
                   f"{rel} is {file_hashes[rel][:16]}... but {REGISTRY} pins {pins[key][:16]}...",
                   checks)
        # the theorem text is a frozen input too: a silently different parquet would mean the
        # statement whose q was frozen is not the statement being proved.
        pinned_inputs = pool["inputs"]
        for rel in (RAW_PARQUET, TRAIN_PARQUET):
            on_disk = sha256_file(ROOT / rel) if (ROOT / rel).exists() else "missing"
            _check(f"pool_pin::{rel.split('/')[-1]}", on_disk == pinned_inputs[rel],
                   f"{rel} is {on_disk[:16]}... but the frozen pool pins "
                   f"{str(pinned_inputs[rel])[:16]}...", checks)
    return Frozen(formal=formal, gate=gate, predictions=predictions, pool=pool, reserve=reserve,
                  theta0=theta0, theorems=theorems, draw_order=draw_order, block_ids=block_ids,
                  q_B2=q_B2, q_B1=q_B1, file_hashes=file_hashes, checks=checks)


def frozen_block_for(frozen: Frozen, stratum: str | None = None) -> list:
    """The frozen block membership of a source stratum, by the frozen q ranking only.

    `m` is computed from the FROZEN stratum size, never from the analyzed size: shrinking the block
    after seeing which theorems got censored would be re-selecting the top 20% post hoc.
    """
    rows = [t for t in frozen.theorems if stratum is None or t["source"] == stratum]
    ranked = sorted(rows, key=lambda t: (-t["q_B2_controller"], t["statement_id"]))
    m = block_size(len(ranked))
    return [t["statement_id"] for t in ranked[:m]]


# --- host / git environment (checked, never assumed) --------------------------------------------

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
        _check("host_is_fly122", FORMAL_HOST_IP in env.get("ips", ""),
               f"ips={env.get('ips')!r} (owner decision 16: fly122 is the formal node)", checks)
        _check("gpu_is_rtx3080", env.get("gpu_names", "").count(FORMAL_GPU_NAME) >= 1
               and FORMAL_GPU_NAME in env.get("gpu_names", ""),
               f"gpu={env.get('gpu_names')!r}", checks)
        _check("gpu_is_otherwise_idle", env.get("gpu_compute_apps", "") == "",
               f"another compute job is on the GPU: {env.get('gpu_compute_apps')!r} "
               "(E022 root cause: one GPU-heavy job at a time)", checks)
    _check("branch_is_v3", env.get("branch") == BRANCH, f"branch={env.get('branch')!r}", checks)
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
    """Read a raw artifact line by line, skipping an unfinished torn tail rather than crashing.

    A torn last line can only ever be the tail of an append that was interrupted mid-write, and a
    torn group is never a valid group (see `finalize_group`), so it is dropped here and the whole
    theorem is re-run deterministically by the resume path.
    """
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

    Append-only + one group per call + fsync is what makes the raw artifact crash-safe: a reader
    either sees all 8 rows of a group or fewer than 8, and "fewer than 8" is INCOMPLETE to
    `finalize_group`, which the resume path re-runs from the same deterministic seeds. A row is
    never rewritten, so a concurrent reader cannot be shown a half-updated group.
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


def validate_row(row: dict) -> None:
    """Every raw row must be self-identifying and internally consistent before it can be written."""
    missing = [k for k in CANDIDATE_FIELDS if k not in row]
    if missing:
        raise FrozenViolation(f"raw row is missing schema fields {missing}")
    status = str(row["verify_status"])
    if status not in CONCLUSIVE_STATUSES | INFRA_STATUSES:
        raise FrozenViolation(
            f"raw row verify_status={status!r} is outside the frozen B0 vocabulary "
            f"{sorted(CONCLUSIVE_STATUSES | INFRA_STATUSES)}; an unclassified verdict may not be "
            "written to a preregistered raw artifact")
    derived = candidate_score(status)
    if row["score"] != derived:
        raise FrozenViolation(
            f"raw row score={row['score']!r} contradicts verify_status={row['verify_status']!r} "
            f"(the score is always candidate_score(verify_status); a hand-written score is the "
            "all-fail-by-accident bug this schema exists to prevent)")
    if bool(row["verified"]) != (derived == 1):
        raise FrozenViolation(f"raw row verified={row['verified']!r} contradicts score")
    if row["experiment_id"] != EXPERIMENT_ID:
        raise FrozenViolation(f"raw row carries experiment_id={row['experiment_id']!r}")
    if row["schema_version"] != SCHEMA_VERSION:
        raise FrozenViolation(f"raw row carries schema_version={row['schema_version']!r}")


def read_groups(path: Path) -> dict:
    """statement_id -> its candidate rows, with the never-merge rule enforced on read.

    One (statement_id, sample_index) pair may appear exactly once in a raw artifact. A duplicate can
    only mean candidates from two runs were merged into one group, which owner 8 forbids; the
    resume path drops whole groups instead, so reaching this branch is real corruption.
    """
    groups: dict = {}
    for row in rows_jsonl(path):
        validate_row(row)
        sid = row["statement_id"]
        groups.setdefault(sid, []).append(row)
    for sid, rows in groups.items():
        seen = [r["sample_index"] for r in rows]
        if len(set(seen)) != len(seen):
            raise FrozenViolation(
                f"group {sid} has repeated sample_index {sorted(seen)}: candidates from different "
                "runs have been merged, which the protocol forbids (owner 8)")
        ranks = {r["theorem_rank"] for r in rows}
        if len(ranks) != 1:
            raise FrozenViolation(f"group {sid} carries more than one theorem_rank {ranks}")
    return groups


