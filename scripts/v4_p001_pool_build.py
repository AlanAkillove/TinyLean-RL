#!/usr/bin/env python3
"""V4-P001 §B/§C — build the DEVELOPMENT POOL, freeze the screening order and the screening budget.

Owner §B: Tier 1 is the already-consumed development/training families; Tier 2 is used ONLY IF Tier 1
is insufficient; a family component is the sampling unit; one deterministic theorem representative per
family; the screening order must be outcome-free; eligible components, statement ids, sources,
screening order and pool hash are frozen before any V4 formal generation.

Component-level class rules, taken from frozen registries only (never re-derived by hand):

  Tier 1  ``contains.v1_used >= 1`` — at least one member statement was actually rolled out during V1
          training/development. Family closure: the whole component is treated as a training family.
  Tier 2  completed V2 development classes whose material was explicitly released, plus the Track C
          class the owner released in V3 (`V2-track-C-reservation-release`):
          ``a001_selection`` (A001 checkpoint selection, COMPLETE, A-track terminated),
          ``b002_pilot`` / ``b002_calibration`` (V2-B002 pilot + calibration, COMPLETE),
          ``b003_eval`` (V2-B003 evaluation, COMPLETE - the 24-component buffer stays excluded),
          ``e023_holdout`` (E023 final-holdout evaluation, COMPLETE and reported),
          ``c_joint_holdout`` (never executed; owner released the reservation on 2026-09-24).

Excluded at component level, asserted rather than assumed:

  * the V3 sealed final reserve (93 components) — ``sealed_components_touched`` must be 0;
  * the V3-R001 formal sample (128 nominal components) — the previous prospective sample; excluding it
    keeps "no previous result is reopened" true at the statement level, not just in prose;
  * ``a_reserve`` (reserved for the conditional second A-track decision, never taken — an explicit
    unused reservation);
  * ``b1_audit_reserved`` and the ``b1_reserved_statement_ids`` carve-out (reservation-flagged);
  * the 24 never-consumed V2-B003 buffer components.

Representative rule (deterministic, outcome-free, no human choice): members are ordered by
``sha256(statement_id)`` — the V2 registry's own representative ordering — and the first member that
passes the trainer's frozen 1,024-token prompt filter is the representative. No outcome, source,
difficulty, error category or repairability history is read.

Screening order: Tier 1 block first, then the Tier-2 classes in the fixed order above; inside a block
by ``sha256(V4_BASE_SEED || statement_id)``. Uniform hash permutations — verifiably independent of
source, family, difficulty and every outcome label. Tier 2 is therefore only reached when Tier 1 is
exhausted, which is exactly §B's "only if Tier 1 is insufficient".

Screening budget (§C): ``ceil(N_PRIMARY / PRIMARY_RATE_FLOOR)`` = 128 / 0.20 = 640 screens. The floor
is below the 95 % lower bound of both theta0 estimates measured in ``v4_p001_audit.json``
(0.3125, n=96 step-1 proxy; 0.2891, n=512 E023), so the budget is not a point prediction of the
observed rate. If 128 primary failures are not collected within 640 screens the run STOPS and reports.

Read-only over every input; no model, no GPU, no verifier, no rollout.

Output: experiments/manifests/v4/v4_p001_pool.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

REGISTRY = "experiments/manifests/v2/family_component_registry.json"
TRAIN_PARQUET = "data/processed/p3_promptset/prompt_sets/AI-MO/Kimina-Prover-Promptset/train.parquet"
RAW_PARQUET = "data/raw/kimina_promptset/data/train-00000-of-00001.parquet"
RESERVE = "experiments/manifests/v3/v3_final_holdout_reserve.json"
FORMAL_SAMPLE_V3 = "experiments/manifests/v3/v3_r001_formal_sample.json"
FAMILY_CLEAN_POOL_V3 = "experiments/manifests/v3/V3-R001_family_clean_pool.json"
V1_SOURCES = "experiments/manifests/v3/v1_rollout_sources.json"
B002_PILOT_SET = "experiments/manifests/v2/v2_b002_pilot_set.json"
B002_CALIBRATION_SET = "experiments/manifests/v2/v2_b002_calibration_set.json"
B003_EVAL_SET = "experiments/manifests/v2/v2_b003_eval_set.json"
B1_RESERVED_IDS = "experiments/manifests/v2/b1_reserved_statement_ids.json"
TOKENIZER = "models/weights/kimina_distill_0_6b"
MAX_PROMPT_LENGTH = 1024
V4_BASE_SEED = 20260925
N_PRIMARY = 128
PRIMARY_RATE_FLOOR = 0.20
MAX_SCREENING = math.ceil(N_PRIMARY / PRIMARY_RATE_FLOOR)

TIER1_FLAG = "v1_used"
# Tier-2 class order is part of the frozen screening order, not a ranking: it follows the V2 track
# sequence (Track A selection, Track B pilot/calibration/evaluation, E023, then the released Track C).
TIER2_FLAGS = ("a001_selection", "e023_holdout", "c_joint_holdout")
TIER2_SETS = ("b002_pilot", "b002_calibration", "b003_eval")
TIER2_ORDER = ("a001_selection", "b002_pilot", "b002_calibration", "b003_eval", "e023_holdout", "c_joint_holdout")
EXCLUDED_FLAGS = ("a_reserve", "b1_audit_reserved")

OUT = "experiments/manifests/v4/v4_p001_pool.json"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def order_key(seed: int, statement_id: str) -> str:
    return sha256_text(f"{seed}|{statement_id}")


def git_revision() -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()


def statement_ids(payload: dict, key: str) -> set[str]:
    return {str(entry.get("statement_id") or entry.get("representative_statement_id")) for entry in payload[key]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=OUT)
    args = parser.parse_args()

    registry = json.loads((ROOT / REGISTRY).read_text())
    reserve = json.loads((ROOT / RESERVE).read_text())
    sample_v3 = json.loads((ROOT / FORMAL_SAMPLE_V3).read_text())
    v1_sources = json.loads((ROOT / V1_SOURCES).read_text())
    b003 = json.loads((ROOT / B003_EVAL_SET).read_text())
    b1_reserved = json.loads((ROOT / B1_RESERVED_IDS).read_text())

    components = registry["components"]
    by_id = {c["component_id"]: c for c in components}
    stmt_index = {stmt: c["component_id"] for c in components for stmt in c["member_statement_ids"]}

    sealed_components = {c["component_id"] for c in reserve["components"]}
    sealed_statements = {c["statement_id"] for c in reserve["components"]}
    v3_formal_components = {m["component_id"] for m in sample_v3["theorems"]}
    v3_formal_statements = {m["statement_id"] for m in sample_v3["theorems"]}

    class_statements: dict[str, set[str]] = {
        "b002_pilot": statement_ids(json.loads((ROOT / B002_PILOT_SET).read_text()), "theorems"),
        "b002_calibration": statement_ids(json.loads((ROOT / B002_CALIBRATION_SET).read_text()), "theorems"),
        "b003_eval": {str(c["representative_statement_id"]) for c in b003["evaluation_components"]},
    }
    class_statements["b001_audit_reserved"] = {str(s) for s in b1_reserved["statement_ids"]}
    class_components: dict[str, set[str]] = {
        name: {stmt2comp for stmt in stmts if (stmt2comp := stmt_index.get(stmt))}
        for name, stmts in class_statements.items()
    }

    # --- exclude-first, then tier assignment (component-level contact) -------------------------
    excluded_components: dict[str, set[str]] = {"sealed_v3_reserve": sealed_components, "v3_formal_sample": v3_formal_components}
    excluded_components["b003_buffer"] = {str(c) for c in b003["buffer_component_ids"]}
    for flag in EXCLUDED_FLAGS:
        excluded_components[flag] = {c["component_id"] for c in components if (c.get("contains") or {}).get(flag, 0) > 0}
    excluded_components["b001_reserved_ids"] = class_components["b001_audit_reserved"]
    exclusion_union = set().union(*excluded_components.values())

    tier_of: dict[str, str] = {}
    class_of: dict[str, list[str]] = {}
    for c in components:
        cid = c["component_id"]
        if cid in exclusion_union:
            continue
        contains = c.get("contains") or {}
        classes = []
        if contains.get(TIER1_FLAG, 0) > 0:
            classes.append(TIER1_FLAG)
        for flag in TIER2_FLAGS:
            if contains.get(flag, 0) > 0:
                classes.append(flag)
        for name in TIER2_SETS:
            if cid in class_components[name]:
                classes.append(name)
        if not classes:
            continue
        class_of[cid] = classes
        tier_of[cid] = "tier1" if TIER1_FLAG in classes else "tier2"

    tier1 = [by_id[cid] for cid in class_of if tier_of[cid] == "tier1"]
    tier2 = [by_id[cid] for cid in class_of if tier_of[cid] == "tier2"]
    if not tier1:
        raise SystemExit("Tier 1 is empty; the pool rule is broken")

    # --- theorem text + prompt filter ---------------------------------------------------------
    df = pd.read_parquet(ROOT / TRAIN_PARQUET)
    prompt_len: dict[str, int] = {}
    seen: dict[str, int] = {}
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(ROOT / TOKENIZER))

    def chat_len(messages) -> int:
        return len(tok.apply_chat_template(messages, add_generation_prompt=True))

    for sid, p in zip(df["statement_id"].astype(str), df["prompt"].tolist(), strict=True):
        msgs = p.tolist() if hasattr(p, "tolist") else list(p)
        msgs = [dict(m) if not isinstance(m, dict) else m for m in msgs]
        key = json.dumps(msgs, default=str, sort_keys=True)
        if key not in seen:
            seen[key] = chat_len(msgs)
        if seen[key] <= MAX_PROMPT_LENGTH:
            prompt_len[sid] = seen[key]
    prompt_ok = set(prompt_len)

    raw = pd.read_parquet(ROOT / RAW_PARQUET, columns=["statement_id", "source", "name", "formal_statement"])
    source_of = dict(zip(raw["statement_id"].astype(str), raw["source"].astype(str), strict=True))
    name_of = dict(zip(raw["statement_id"].astype(str), raw["name"].astype(str), strict=True))

    # --- one deterministic representative per family ------------------------------------------
    pool: list[dict] = []
    dropped_no_member_in_filter: dict[str, list[str]] = {"tier1": [], "tier2": []}
    dropped_no_source: list[str] = []
    for tier, group in (("tier1", tier1), ("tier2", tier2)):
        for c in group:
            members = sorted(c["member_statement_ids"], key=lambda sid: (sha256_text(sid), sid))
            chosen = next((sid for sid in members if sid in prompt_ok), None)
            if chosen is None:
                dropped_no_member_in_filter[tier].append(c["component_id"])
                continue
            if chosen not in source_of:
                dropped_no_source.append(c["component_id"])
                continue
            pool.append(
                {
                    "component_id": c["component_id"],
                    "statement_id": chosen,
                    "name": name_of.get(chosen, ""),
                    "source": source_of[chosen],
                    "prompt_tokens": prompt_len[chosen],
                    "component_size": c["size"],
                    "tier": tier,
                    "classes": sorted(class_of[c["component_id"]]),
                    "representative_is_registry_representative": chosen == c.get("representative_statement_id"),
                }
            )

    tier_rank = {"tier1": 0, "tier2": 1}
    class_rank = {name: index for index, name in enumerate(TIER2_ORDER)}

    def block_key(row: dict) -> tuple:
        classes = [c for c in row["classes"] if c != TIER1_FLAG]
        if not classes:
            return (tier_rank[row["tier"]], 0)
        return (tier_rank[row["tier"]], 1 + class_rank[classes[0]])

    pool.sort(key=lambda r: (block_key(r), order_key(V4_BASE_SEED, r["statement_id"])))
    for rank, row in enumerate(pool, start=1):
        row["screening_rank"] = rank
        row["screening_order_key"] = order_key(V4_BASE_SEED, row["statement_id"])

    pool_statements = {r["statement_id"] for r in pool}
    pool_components = {r["component_id"] for r in pool}

    # --- assertions ---------------------------------------------------------------------------
    checks = {
        "sealed_components_touched": len(pool_components & sealed_components),
        "sealed_statements_touched": len(pool_statements & sealed_statements),
        "v3_formal_sample_components_touched": len(pool_components & v3_formal_components),
        "v3_formal_sample_statements_touched": len(pool_statements & v3_formal_statements),
        "a_reserve_components_touched": len(pool_components & excluded_components["a_reserve"]),
        "b1_audit_components_touched": len(pool_components & excluded_components["b1_audit_reserved"]),
        "b1_reserved_ids_components_touched": len(pool_components & excluded_components["b001_reserved_ids"]),
        "b003_buffer_components_touched": len(pool_components & excluded_components["b003_buffer"]),
        "all_representatives_prompt_filter_pass": all(r["statement_id"] in prompt_ok for r in pool),
        "screening_ranks_contiguous": [r["screening_rank"] for r in pool] == list(range(1, len(pool) + 1)),
        "screening_keys_unique": len({r["screening_order_key"] for r in pool}) == len(pool),
        "one_representative_per_component": len(pool_components) == len(pool),
        "tier1_block_precedes_tier2_block": [r["tier"] for r in pool] == sorted(r["tier"] for r in pool),
    }
    failing_zero = (
        "sealed_components_touched",
        "sealed_statements_touched",
        "v3_formal_sample_components_touched",
        "v3_formal_sample_statements_touched",
        "a_reserve_components_touched",
        "b1_audit_components_touched",
        "b1_reserved_ids_components_touched",
        "b003_buffer_components_touched",
    )
    if any(checks[k] != 0 for k in failing_zero):
        raise SystemExit(f"V4 pool touches a frozen or reserved set: {checks}")
    if not all(
        checks[k]
        for k in (
            "all_representatives_prompt_filter_pass",
            "screening_ranks_contiguous",
            "screening_keys_unique",
            "one_representative_per_component",
            "tier1_block_precedes_tier2_block",
        )
    ):
        raise SystemExit(f"V4 pool invariants failed: {checks}")

    pool_hash = sha256_text(
        canonical_json(
            {
                "unit": "component_id",
                "representative_rule": "min sha256(statement_id) then first passing the 1024-token prompt filter",
                "tiers": {"tier1": sorted(r["component_id"] for r in pool if r["tier"] == "tier1"),
                          "tier2": sorted(r["component_id"] for r in pool if r["tier"] == "tier2")},
                "max_prompt_length": MAX_PROMPT_LENGTH,
                "components": sorted(pool_components),
                "statements": sorted(pool_statements),
                "component_to_statement": {r["component_id"]: r["statement_id"] for r in sorted(pool, key=lambda r: r["component_id"])},
            }
        )
    )
    order_hash = sha256_text(
        canonical_json(
            {
                "base_seed": V4_BASE_SEED,
                "rule": "tier blocks (tier1, then tier2 classes in frozen order), each sorted by sha256(f'{seed}|{statement_id}')",
                "order": [r["statement_id"] for r in pool],
            }
        )
    )

    sources = Counter(r["source"] for r in pool)
    tier_counts = Counter(r["tier"] for r in pool)
    class_counts = Counter(name for r in pool for name in r["classes"])
    tier1_capacity = tier_counts["tier1"]
    expected = {
        str(rate): {
            "expected_primary_failures_in_tier1": round(rate * tier1_capacity, 1),
            "screens_needed_for_128_at_this_rate": math.ceil(N_PRIMARY / rate),
            "within_frozen_budget": math.ceil(N_PRIMARY / rate) <= MAX_SCREENING,
            "within_pool_capacity": math.ceil(N_PRIMARY / rate) <= len(pool),
        }
        for rate in (0.20, 0.25, 0.29, 0.3125)
    }
    out = {
        "artifact_type": "v4_p001_development_pool",
        "status": "FROZEN-PENDING-PREREGISTRATION-COMMIT",
        "purpose": "V4-P001 paired verifier-guided repairability probe - screening pool, order and budget",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": subprocess.run(["hostname"], capture_output=True, text=True, check=False).stdout.strip(),
        "git_revision": git_revision(),
        "tier_rule": {
            "tier1_definition": f"component contains.{TIER1_FLAG} >= 1 (already-consumed training families)",
            "tier2_definition": "completed V2 development classes with released material + the Track-C class the owner released in V3",
            "tier2_classes_in_order": list(TIER2_ORDER),
            "tier2_used": tier_counts["tier2"] > 0,
            "tier2_admission_rule": "Tier 2 is only reached after the whole Tier 1 block is screened; the block order is frozen in the screening order below",
        },
        "excluded_classes": {name: len(ids) for name, ids in sorted(excluded_components.items())},
        "excluded_union_components": len(exclusion_union),
        "unit": "component_id",
        "representative_rule": "members ordered by sha256(statement_id); first member passing the frozen 1024-token prompt filter",
        "max_prompt_length": MAX_PROMPT_LENGTH,
        "base_seed": V4_BASE_SEED,
        "screening_order_rule": "tier blocks (Tier 1 first, then Tier-2 classes in the frozen order), each sorted by sha256(f'{base_seed}|{statement_id}') - outcome-free hash permutation",
        "screening_budget": {
            "n_primary_required": N_PRIMARY,
            "primary_rate_floor": PRIMARY_RATE_FLOOR,
            "max_screens": MAX_SCREENING,
            "rule": "ceil(128 / floor); if 128 primary failures are not collected within max_screens the run STOPS and reports before any second-stage generation",
            "expected_screens": expected,
        },
        "capacity": {
            "tier1_components_before_prompt_filter": len(tier1),
            "tier2_components_before_prompt_filter": len(tier2),
            "components_dropped_no_member_in_prompt_filter": {k: len(v) for k, v in dropped_no_member_in_filter.items()},
            "components_dropped_no_source": len(dropped_no_source),
            "pool_components": len(pool),
            "pool_components_by_tier": dict(sorted(tier_counts.items())),
            "pool_components_by_class": dict(sorted(class_counts.items())),
            "pool_statements": len(pool_statements),
            "sources": dict(sorted(sources.items())),
        },
        "capacity_detail": {
            "dropped_no_member_in_filter_components": dropped_no_member_in_filter,
            "dropped_no_source_components": dropped_no_source,
            "excluded_component_ids": {name: sorted(ids) for name, ids in sorted(excluded_components.items())},
        },
        "prompt_tokens": {
            "min": min(r["prompt_tokens"] for r in pool),
            "median": sorted(r["prompt_tokens"] for r in pool)[len(pool) // 2],
            "max": max(r["prompt_tokens"] for r in pool),
        },
        "checks": checks,
        "pool_hash": pool_hash,
        "order_hash": order_hash,
        "members": pool,
        "inputs": {
            rel: sha256_file(ROOT / rel)
            for rel in (REGISTRY, TRAIN_PARQUET, RAW_PARQUET, RESERVE, FORMAL_SAMPLE_V3, FAMILY_CLEAN_POOL_V3,
                        V1_SOURCES, B002_PILOT_SET, B002_CALIBRATION_SET, B003_EVAL_SET, B1_RESERVED_IDS)
        },
        "tokenizer": TOKENIZER,
        "prohibitions_respected": [
            "no rollout, generation, verifier call, GPU work, RL optimizer step or smoke",
            "no V1/V2/V3 frozen artifact modified - all inputs are hashed as received",
            "no selection by outcome, difficulty, source, family size or historical repairability: the order is a hash permutation and the representative rule reads only sha256(statement_id) and prompt length",
            "the 93-component sealed reserve is untouched and asserted disjoint",
        ],
        "v1_sources_note": (v1_sources.get("note", "")[:400] if isinstance(v1_sources, dict) else ""),
    }
    destination = ROOT / args.out
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(out, indent=2) + "\n")
    print(f"pool components: {len(pool)}  tier1={tier_counts['tier1']} tier2={tier_counts['tier2']}  sources: {dict(sources)}")
    print(f"pool_hash: {pool_hash}")
    print(f"order_hash: {order_hash}")
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
