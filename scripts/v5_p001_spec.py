#!/usr/bin/env python3
"""Frozen specification for V5-P001 -- Offline Process-Reward Recoverability Audit.

Machine-readable twin of ``docs/v5/V5-P001_preregistration.md``. Every constant
here is frozen in Phase A, *before* any historical process label is inspected
(owner §22, §27). V5-P001 performs no training and no model generation: this
module only parameterises a read-only audit of historical V1 rollouts.

Vocabulary, gates and the classification taxonomy follow the owner directive
for V5-P001 verbatim; the paper-compatible process credit is d1 = -0.05 for a
locally verified tactic strictly before the first error and d2 = -0.10 for the
first erroneous tactic and everything after it (first-error propagation).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXPERIMENT_ID = "V5-P001"

# --- historical surface (owner §4-§6) -----------------------------------------------------------
# The primary surface is the existing V1 RLVR rollout dumps only: no new
# generation, no theta0-only rollouts, no V3/V4 prospective samples.
SEED_DIRS = {
    "seed1": ROOT / "runs" / "p3b_pilot" / "rollout_data",
    "seed2": ROOT / "runs" / "m1_seed2" / "rollout_data",
    "seed3": ROOT / "runs" / "m1_seed3" / "rollout_data",
}
DATASET = ROOT / "data" / "raw" / "kimina_promptset" / "data" / "train-00000-of-00001.parquet"
REGISTRY = ROOT / "experiments" / "manifests" / "v2" / "family_component_registry.json"
#: Frozen V3 manifest with per-file sha256 of every V1 rollout file (verification target).
V1_SOURCES_MANIFEST = ROOT / "experiments" / "manifests" / "v3" / "v1_rollout_sources.json"

GROUP_N = 8  # rollout.n of the historical V1 runs.

#: Sentinel ``pred`` values: no Lean code was ever submitted for the candidate
#: (historical ``formal_rewards`` skips them), so the process oracle must not
#: fabricate tactic labels for them (owner §10).
SENTINEL_NO_PROOF = "No proof found in the output."
SENTINEL_NO_STATEMENT = "Theorem statement couldn't be parsed from statement."
PRED_SENTINELS = (SENTINEL_NO_PROOF, SENTINEL_NO_STATEMENT)

#: Group-level contamination policy (established V3 policy, reused verbatim):
#: a candidate is infra-contaminated when its ``tool_feedback`` starts with the
#: system-error marker; any contaminated candidate excludes the whole group.
SYSTEM_ERROR_MARKER = "# System Error:"

GROUP_ALL_FAIL = "ALL_FAIL"
GROUP_MIXED = "MIXED"  # historical label name: "informative"
GROUP_ALL_SUCCESS = "ALL_SUCCESS"
GROUP_INFRA_EXCLUDED = "INFRA_EXCLUDED"

#: Frozen historical binary counts to reproduce (docs/v3/data_audit.md):
#: 603 all_fail / 110 informative / 7 all_success over 720 groups; 686 groups
#: survive the contamination exclusion.
EXPECTED_GROUPS_TOTAL = 720
EXPECTED_GROUPS_PRIMARY = 686
EXPECTED_ALL_FAIL = 603
EXPECTED_MIXED = 110
EXPECTED_ALL_SUCCESS = 7

# --- process oracle infrastructure (owner §7, §21, §26) ----------------------------------------
# A dedicated Kimina instance on the formal GPU host, same pinned image digest
# as the canonical verifier, so the whole-proof semantics are identical while
# the pooled/production verifier stays untouched. Infra outcomes (timeout /
# server error / transport loss) are *censored*, never counted as failures.
ORACLE_INFRA = {
    "endpoint": "http://127.0.0.1:8020",
    "container": "tinylean-rl-lean-oracle-v5",
    "host": "fly122",
    "image": "projectnumina/kimina-lean-server:2.0.0",
    "image_digest": (
        "sha256:588a2cbbd10da509ed13f53ac136f8463fabff02dfe4eca535e7c47ae6e3ffd9"
    ),
    "max_repls": 1,
    "server_timeout_s": 120,  # server-side per-request budget (kills first)
    "client_slack_s": 60,  # client timeout = server_timeout + slack
    "batch_size": 1,  # one candidate per request: infotree stays unambiguous
    "max_single_retries": 2,  # bounded isolation retries for infra outcomes
    "canary_timeout_s": 300,  # cold canary (first request re-imports Mathlib)
    "warm_canary_timeout_s": 120,
    "canary_retries": 1,
    "max_recoveries_per_run": 192,  # bounded container restarts per Phase-B run
    "restart_timeout_s": 120.0,
    "health_poll_timeout_s": 180.0,
    "health_poll_interval_s": 2.0,
    "restart_grace_s": 10,
}

# --- tactic -> token mapping (owner §9) ---------------------------------------------------------
TOKENIZER_DIR = ROOT / "models" / "weights" / "kimina_distill_0_6b"
TOKENIZER_EXPECTED_VOCAB = 151643
#: The generated continuation is tokenized as a standalone sequence; no BOS or
#: other special token is inserted when re-tokenizing stored response text.
TOKENIZER_ADD_SPECIAL_TOKENS = False

# --- process credit (owner §2, §10, §23) --------------------------------------------------------
# Canonical, paper-compatible first-error propagation.
D1_CANONICAL = -0.05
D2_CANONICAL = -0.10
#: Sensitivity settings, evaluated after the primary classification only.
SENSITIVITY_SETTINGS = {
    "canonical_d1_d2": (D1_CANONICAL, D2_CANONICAL),
    "stronger_gap": (-0.05, -0.50),
    "equal_penalty": (-0.10, -0.10),
}

PROCESS_STATUSES = (
    "SUCCESS",
    "PREFIX_BEARING_FAILURE",
    "FIRST_TACTIC_FAILURE",
    "PARSE_OR_SYNTAX_FAILURE",
    "FORMAT_NO_CODE",
    "PROCESS_ORACLE_INFRA",
)

#: Tactic parser-kind prefix and the pure-sequence wrappers that are not
#: themselves tactics for scoring purposes.
TACTIC_NAME_PREFIX = "Lean.Parser.Tactic."
TACTIC_WRAPPER_NAMES = frozenset(
    {
        "Lean.Parser.Tactic.tacticSeq",
        "Lean.Parser.Tactic.tacticSeq1Indented",
        "Lean.Parser.Tactic.tacticSeqBracketed",
    }
)

# --- frozen gates (owner §22) -------------------------------------------------------------------
GATE_E1_MIN = 0.95  # exact tactic->token mapping among process-parsed nodes
GATE_E2_MIN = 0.90  # deterministic-oracle success among code-extractable non-infra
GATE_G1_RATE_MIN = 0.25  # pooled RecoveryRate
GATE_G1_CI_LOWER_MIN = 0.15  # family/component bootstrap CI lower bound
GATE_G2_PER_SEED_MIN = 0.20  # point estimate in every seed
GATE_G3_RATE_MIN = 0.15  # structured recovery inside a quartile
GATE_G3_QUARTILES_MIN = 3  # of 4 quartiles
GATE_QUARTILE_MIN_N = 5  # technical guard: a quartile below this n cannot pass

CLASSIFICATIONS = (
    "PROCESS_SIGNAL_GO",
    "NO_PROCESS_RECOVERY",
    "PROCESS_SIGNAL_SEED_UNSTABLE",
    "PROCESS_SIGNAL_LENGTH_CONFOUNDED",
    "INCONCLUSIVE_PROCESS_ORACLE",
)

# --- bootstrap (owner §13) ----------------------------------------------------------------------
BOOTSTRAP = {
    "n_resamples": 10000,
    "seed": 20260926,
    "cluster": "component_id",  # family/component-aware; statement fallback when unmapped
    "alpha": 0.05,
    "method": "percentile",
}

# --- output artifacts (owner §33) ---------------------------------------------------------------
HISTORICAL_SURFACE = ROOT / "experiments" / "manifests" / "v5" / "v5_historical_surface.json"
ORACLE_VALIDATION = ROOT / "experiments" / "manifests" / "v5" / "v5_process_oracle_validation.json"
PREREG_MANIFEST = ROOT / "experiments" / "manifests" / "v5" / "V5-P001.yaml"
PROCESS_RUN_DIR = ROOT / "runs" / "v5_p001_process"
PROCESS_LABELS = PROCESS_RUN_DIR / "v5_p001_process_labels.jsonl"
PROCESS_RAW_DIR = PROCESS_RUN_DIR / "raw"
PROCESS_RAW_MANIFEST = PROCESS_RUN_DIR / "v5_p001_process_raw_manifest.json"
PROCESS_FREEZE = PROCESS_RUN_DIR / "v5_p001_process_freeze.json"
PROCESS_RESULTS = ROOT / "experiments" / "manifests" / "v5" / "V5-P001_results.json"
PROCESS_VALIDATION = PROCESS_RUN_DIR / "v5_p001_process_validation.json"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(obj: object) -> str:
    """Stable serialization for hashing oracle outputs."""

    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
