#!/usr/bin/env python3
"""V3-R001 §5 — build the PROSPECTIVE FAMILY-CLEAN POOL and freeze its hash.

Owner §5 requires a pool from which every family component "touched" by earlier work has been
removed, closed at component level, and explicitly forbids reclaiming families for sample size
(不要为了样本量偷回这些 families). Owner §4 / the deployment-pool audit fixes the unit: one unique
statement_id == one canonical prompt (duplicate rows are byte-identical), and a candidate must
survive the trainer's own 1,024-token prompt filter, because a theorem the sampler could never
draw is not a legitimate test theorem for a sampler-derived claim.

The hard part is the word "touched". The V2 theorem-role registry assigned a ROLE to essentially
the whole eligible pool, but roles that were never consumed are not leakage, so this script
separates the two and reports the cost of each reading instead of silently choosing one:

  CONSUMED  - a rollout / evaluation / selection step actually ran on the item:
              V1 seed1/2/3 rollouts (the controller's own label source), the registry's wider
              v1_used_union (8 dumps), the sealed E023 holdout, V2-A001's 512 selection items,
              V2-B001's 16 audit items, V2-B002's 512 pilot components + 64 calibration items,
              V2-B003's 192 evaluation components.
  RESERVED  - parked for a downstream decision that never happened:
              V2-B003's 24 buffer components, the 162 A-reserve statements, the 680
              C-joint-holdout statements (Track C was superseded before launch: docs/v2/
              track_b_closeout.md §6, "V2-B004 = NOT RUN", "no allocator training corpus was
              generated"), and the B-train/B-validation/B-test role mass that B004 would have
              consumed but never did.

Everything is read from the frozen artifacts below; nothing is re-derived, and no rollout, model
or GPU work happens here.

Output: experiments/manifests/v3/V3-R001_family_clean_pool.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from v3_d001_lib import FORMAL_BLOCK_RE, _load_statement_map, _normalize, build_records

TRAIN_PARQUET = "data/processed/p3_promptset/prompt_sets/AI-MO/Kimina-Prover-Promptset/train.parquet"
RAW_PARQUET = "data/raw/kimina_promptset/data/train-00000-of-00001.parquet"
REGISTRY = "experiments/manifests/v2/family_component_registry.json"
ROLE_REGISTRY = "experiments/manifests/v2/theorem_role_registry.json"
POOL_AUDIT = "experiments/manifests/v3/V3-R001_deployment_pool_audit.json"
TOKENIZER = "models/weights/kimina_distill_0_6b"
MAX_PROMPT_LENGTH = 1024

# Every rollout dump readable on this host, including the ones V3-D001 deliberately excluded. Used
# only to answer a provenance question in owner §5: which statements have ACTUAL generation output
# on disk ("contact") as opposed to a role assignment for a track that was stopped ("reservation").
ROLLOUT_DIRS = ["runs/p3b_pilot/rollout_data", "runs/m1_seed2/rollout_data",
                "runs/m1_seed3/rollout_data", "runs/m2_qwen_smoke/rollout_data",
                ".cache/m1_seed3_r2r3_dumps", "experiments/p3_0_batch",
                "experiments/p3_b_n8_batch"]

ARTIFACTS = {
    "e023_holdout": "experiments/manifests/m1_final_holdout.json",
    "a001_selection": "experiments/manifests/v2/v2_a001_selection_set.json",
    "b001_audit": "experiments/manifests/v2/b1_reserved_statement_ids.json",
    "b002_pilot": "experiments/manifests/v2/v2_b002_pilot_set.json",
    "b002_calibration": "experiments/manifests/v2/v2_b002_calibration_set.json",
    "b003_eval": "experiments/manifests/v2/v2_b003_eval_set.json",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(rel: str):
    return json.loads((ROOT / rel).read_text())


def stmt_ids(items, key: str) -> list[str]:
    out = []
    for it in items:
        v = it[key] if isinstance(it, dict) else it
        out.append(str(v))
    return out


def usage_classes() -> dict:
    """Every exclusion class, with the artifact it came from and how many items it names."""
    cls: dict[str, dict] = {}
    role = load(ROLE_REGISTRY)

    # --- CONSUMED -----------------------------------------------------------------------------
    built = build_records(ROOT, RAW_PARQUET, REGISTRY)
    recs = built["valid"] + built["infra_censored"]
    v1_seed123 = sorted({r["statement_id"] for r in recs})
    cls["v1_seed123_rollouts"] = {
        "status": "consumed", "statements": v1_seed123, "components": [],
        "why": "the controller's own label source (V3-D001's 612 statements / 720 groups)",
        "source": "runs/*/rollout_data via scripts/v3_d001_lib.build_records"}

    cls["v1_used_union_wider"] = {
        "status": "consumed",
        "components": sorted(str(c["component_id"]) for c in load(REGISTRY)["components"]
                             if "v1_used" in (c.get("exclusion_reasons") or [])),
        "statements": [],
        "why": "superset of the above: the 8 historical V1 usage dumps "
               "(E013/E016/P3C/IGR-mechanism/m1_seed2/m1_seed3/m2_qwen_smoke/p3b_pilot). Taken at "
               "COMPONENT level from the frozen family registry's exclusion_reasons='v1_used', which "
               "is the closure the owner §5 requires (a touched statement removes its whole family).",
        "source": f"{REGISTRY}#components[].exclusion_reasons == 'v1_used'",
        "role_registry_named_statement_count": role["exclusions"]["v1_used_union"],
        "role_registry_v1_used_sources": role["exclusions"]["v1_used_sources"]}

    e023 = load(ARTIFACTS["e023_holdout"])
    cls["e023_final_holdout"] = {
        "status": "consumed", "statements": stmt_ids(e023["theorems"], "statement_id"),
        "components": [], "sealed": e023.get("sealed"),
        "why": "128 sealed project-unseen final-evaluation theorems",
        "source": ARTIFACTS["e023_holdout"]}

    a001 = load(ARTIFACTS["a001_selection"])
    cls["a001_selection"] = {
        "status": "consumed", "statements": stmt_ids(a001["theorems"], "statement_id"),
        "components": [], "sealed": a001.get("sealed"),
        "why": "V2-A001 checkpoint-selection set; A001 was terminated with selection_outcome NONE, "
               "but rollouts were generated on it, so it is consumed rather than merely reserved",
        "source": ARTIFACTS["a001_selection"]}

    b001 = load(ARTIFACTS["b001_audit"])
    cls["b001_audit_reserved"] = {
        "status": "consumed", "statements": b001["statement_ids"], "components": [],
        "out_of_pool_ids": b001.get("out_of_pool_ids", []),
        "why": "V2-B001 B1 budget-semantics audit development subset (16; 2 outside the V2 pool)",
        "source": ARTIFACTS["b001_audit"]}

    b002 = load(ARTIFACTS["b002_pilot"])
    cls["b002_pilot_components"] = {
        "status": "consumed", "statements": stmt_ids(b002["theorems"], "statement_id"),
        "components": stmt_ids(b002["theorems"], "component_id"),
        "why": "V2-B002 hindsight-oracle pilot: 512 family-clean components, rollout-generated",
        "source": ARTIFACTS["b002_pilot"]}

    cal = load(ARTIFACTS["b002_calibration"])
    cls["b002_calibration"] = {
        "status": "consumed", "statements": stmt_ids(cal["theorems"], "statement_id"),
        "components": [], "why": "64-item pipeline-calibration replay (a subset of E023)",
        "source": ARTIFACTS["b002_calibration"]}

    b003 = load(ARTIFACTS["b003_eval"])
    cls["b003_eval_components"] = {
        "status": "consumed",
        "statements": stmt_ids(b003["evaluation_components"], "representative_statement_id"),
        "components": stmt_ids(b003["evaluation_components"], "component_id"),
        "why": "V2-B003 independent evaluation set: 192 components",
        "source": ARTIFACTS["b003_eval"]}

    # --- RESERVED but never consumed -----------------------------------------------------------
    cls["b003_buffer_components"] = {
        "status": "reserved-never-used", "statements": [],
        "components": list(b003["buffer_component_ids"]),
        "why": "24 reserve components kept as buffer by the B003 freeze rule; never evaluated",
        "source": ARTIFACTS["b003_eval"]}

    cls["a_reserve"] = {
        "status": "reserved-never-used", "statements": list(role["roles"]["A-reserve"]),
        "components": [],
        "why": "the registry itself: 'reserved for the conditional second A-track decision "
               "(V2-A002-class) only' - that decision was never taken (A001 terminated)",
        "source": f"{ROLE_REGISTRY}#roles.A-reserve"}

    cls["c_joint_holdout"] = {
        "status": "reserved-never-used", "statements": list(role["roles"]["C-joint-holdout"]),
        "components": [],
        "why": "Track C's 2x2 design was superseded before launch and V2-B004 = NOT RUN, so these "
               "680 statements were never rolled out; they are a reservation, not contact",
        "source": f"{ROLE_REGISTRY}#roles.C-joint-holdout"}

    consumed_representatives = (set(cls["b002_pilot_components"]["statements"])
                                | set(cls["b003_eval_components"]["statements"])
                                | set(cls["b001_audit_reserved"]["statements"])
                                | set(cls["a001_selection"]["statements"])
                                | set(cls["b002_calibration"]["statements"]))
    b_roles = (set(map(str, role["roles"]["B-train"])) | set(map(str, role["roles"]["B-validation"]))
               | set(map(str, role["roles"]["B-test"])))
    cls["b_role_mass_unconsumed"] = {
        "status": "reserved-never-used", "statements": sorted(b_roles - consumed_representatives),
        "components": [],
        "why": "B-train/B-validation/B-test role assignments (5,361 statements) that V2-B004 would "
               "have turned into an allocator training corpus; B004 is formally stopped and 'no "
               "allocator training corpus was generated', so no rollout ever touched them",
        "source": f"{ROLE_REGISTRY}#roles.B-*",
        "cross_check_b_roles_superset_of_consumed": {
            "pilot_reps_inside_b_roles": len(set(cls["b002_pilot_components"]["statements"]) & b_roles),
            "b003_reps_inside_b_roles": len(set(cls["b003_eval_components"]["statements"]) & b_roles),
            "pilot_reps": len(cls["b002_pilot_components"]["statements"]),
            "b003_reps": len(cls["b003_eval_components"]["statements"])}}
    return cls


READINGS = {
    "owner_literal": {
        "exclude": ["v1_seed123_rollouts", "e023_final_holdout", "a001_selection",
                    "b001_audit_reserved", "b002_pilot_components", "b002_calibration",
                    "b003_eval_components", "c_joint_holdout"],
        "gloss": "owner §5 read literally: the named usages, plus the C/final-reserved components "
                 "the owner also named. 'V1 seed1/2/3 rollout_data' is taken exactly = the three "
                 "V1 RL seeds (612 statements). Reservations other than C are not in §5's list."},
    "owner_literal_wider_v1": {
        "exclude": ["v1_used_union_wider", "e023_final_holdout", "a001_selection",
                    "b001_audit_reserved", "b002_pilot_components", "b002_calibration",
                    "b003_eval_components", "c_joint_holdout"],
        "gloss": "same, but 'V1 rollout data' = the registry's 8-dump v1_used_union (763 "
                 "statements -> 527 family components by closure) rather than only the three V1 RL "
                 "seeds (612 statements)."},
    "owner_literal_all_reservations_honored": {
        "exclude": ["v1_used_union_wider", "e023_final_holdout", "a001_selection",
                    "b001_audit_reserved", "b002_pilot_components", "b002_calibration",
                    "b003_eval_components", "c_joint_holdout", "a_reserve",
                    "b003_buffer_components", "b_role_mass_unconsumed"],
        "gloss": "every §5 usage AND every remaining reservation: the A-reserve, B003's untouched "
                 "buffer, and the B-train/B-validation/B-test role mass that the formally stopped "
                 "V2-B004 never turned into an allocator corpus. Maximal subtraction."},
    "consumed_only": {
        "exclude": ["v1_used_union_wider", "e023_final_holdout", "a001_selection",
                    "b001_audit_reserved", "b002_pilot_components", "b002_calibration",
                    "b003_eval_components"],
        "gloss": "leakage-minimal: remove only components whose members were actually rolled out, "
                 "evaluated or selected; reservations for stopped tracks survive. C-joint-holdout "
                 "is NOT excluded here, which is exactly why this reading needs an owner decision."},
    "registry_labels_maximal": {
        "exclude": ["__registry_any_exclusion_reason__", "__b002_b003_any_component__"],
        "gloss": "the strictest computable reading: drop any component the frozen registry flags "
                 "with ANY exclusion reason, plus every component B002/B003 named. Reported to show "
                 "that what the V2 design left over is exhausted, i.e. that any surviving pool "
                 "necessarily contains reserved-never-used material."},
}


def build_component_maps(reg: dict) -> tuple[dict, dict]:
    stmt_to_comp: dict[str, str] = {}
    comps: dict[str, dict] = {}
    for c in reg["components"]:
        cid = str(c["component_id"])
        comps[cid] = {"size": int(c["size"]),
                      "members": [str(s) for s in c["member_statement_ids"]],
                      "representative": str(c["representative_statement_id"]),
                      "sources": c.get("sources") or {},
                      "exclusion_reasons": list(c.get("exclusion_reasons") or []),
                      "eligible_for_b2": bool(c.get("eligible_for_b2"))}
        for s in comps[cid]["members"]:
            stmt_to_comp[s] = cid
    return stmt_to_comp, comps


def local_contact_scan(sid_map: dict) -> dict:
    """Which statement_ids have ACTUAL generation output somewhere on this host's disk.

    This is the evidence for the consumed/reserved distinction: a role assignment in a registry is
    a bookkeeping act, but a rollout jsonl is contact with an outcome. Parsing reuses V3-D001's own
    statement-resolution path (normalized '# Formal Statement:' block -> statement_id).
    """
    per_dir: dict[str, dict] = {}
    contacted: set[str] = set()
    for rel in ROLLOUT_DIRS:
        d = ROOT / rel
        ids: set[str] = set()
        unparsed = 0
        files = sorted(d.glob("*.jsonl")) if d.is_dir() else []
        for fp in files:
            seen_inputs: set[str] = set()
            with fp.open(encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    try:
                        inp = json.loads(line)["input"]
                    except (ValueError, KeyError):
                        continue
                    if inp in seen_inputs:
                        continue                      # one group = n=8 lines sharing one prompt
                    seen_inputs.add(inp)
                    mm = FORMAL_BLOCK_RE.search(inp)
                    sid = sid_map.get(_normalize(mm.group(1)), (None, None))[0] if mm else None
                    if sid:
                        ids.add(sid)
                    else:
                        unparsed += 1
        contacted |= ids
        per_dir[rel] = {"present": d.is_dir(), "files": len(files),
                        "unique_statement_ids": len(ids), "unmatched_prompts": unparsed}
    return {"dirs_scanned": per_dir, "statements_contacted_on_disk": len(contacted),
            "statement_ids": sorted(contacted)}


def resolve_components(cls: dict, stmt_to_comp: dict, comps: dict, known: set) -> dict:
    """Statement lists -> their components (component closure); unknown ids reported, never dropped."""
    out = {}
    for name, c in cls.items():
        ids = set(c.get("components") or [])
        unknown_comp = sorted(i for i in ids if i not in comps)
        extra = []
        for s in c.get("statements") or []:
            if s in stmt_to_comp:
                ids.add(stmt_to_comp[s])
            elif s not in known:
                extra.append(s)
        c = dict(c)
        c["component_ids"] = sorted(ids)
        c["n_statements_named"] = len(set(c.get("statements") or []))
        c["n_components_resolved"] = len(ids)
        c["unknown_component_ids"] = unknown_comp
        c["statement_ids_not_in_registry"] = extra
        out[name] = c
    return out


def pool_hash_of(result: dict, reading: str) -> str:
    return hashlib.sha256(json.dumps(
        {"components": result["candidate_component_ids"],
         "candidates": result["candidate_statement_ids"],
         "map": sorted(result["component_to_candidate"].items()),
         "unit": "statement_id", "max_prompt_length": MAX_PROMPT_LENGTH,
         "reading": reading}, sort_keys=True).encode()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/manifests/v3/V3-R001_family_clean_pool.json")
    ap.add_argument("--primary", default="owner_literal_wider_v1", choices=sorted(READINGS))
    args = ap.parse_args()

    reg = load(REGISTRY)
    stmt_to_comp, comps = build_component_maps(reg)
    known = set(stmt_to_comp)
    cls = resolve_components(usage_classes(), stmt_to_comp, comps, known)

    # --- does any "reserved, never used" class actually have rollout output on disk? -------------
    scan = local_contact_scan(_load_statement_map(ROOT, RAW_PARQUET))
    contacted = set(scan.pop("statement_ids"))
    per_class_contact = {}
    for name, c in cls.items():
        named = set(c.get("statements") or [])
        if not named:                       # the wider V1 class is recorded at component level
            named = {s for cid in c["component_ids"] for s in comps[cid]["members"]}
        per_class_contact[name] = {
            "declared_status": c["status"], "statements_in closure": len(named),
            "with_local_rollout_output": len(named & contacted)}
    reserved_with_contact = sorted(n for n, v in per_class_contact.items()
                                   if v["declared_status"] == "reserved-never-used"
                                   and v["with_local_rollout_output"] > 0)
    contact = {"dirs_scanned": scan["dirs_scanned"],
               "unique_statements_with_local_rollout_output": scan["statements_contacted_on_disk"],
               "per_class": per_class_contact,
               "sanity_check_v1_seed123_fully_contacted":
                   per_class_contact["v1_seed123_rollouts"]["with_local_rollout_output"]
                   == per_class_contact["v1_seed123_rollouts"]["statements_in closure"],
               "reserved_classes_with_any_local_contact": reserved_with_contact,
               "limits": [
                   ("E013/E016/P3C(E018)/IGR-mechanism dumps are named by the frozen V2 registries "
                    "but are no longer on this host, so their contact is accepted from the registry "
                    "rather than re-derived; the scan covers every dump that IS present."),
                   ("A class can show 0 local contact because its rollouts happened on another node, "
                    "not because they never happened: the authoritative record for the 8-dump V1 "
                    "usage is the frozen family registry's exclusion_reasons.")]}

    # --- prompt-token eligibility (same rule as the deployment-pool audit) ----------------------
    df = pd.read_parquet(ROOT / TRAIN_PARQUET)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(ROOT / TOKENIZER))

    def chat_len(p) -> int:
        msgs = [dict(m) if not isinstance(m, dict) else m
                for m in (p.tolist() if hasattr(p, "tolist") else list(p))]
        return len(tok.apply_chat_template(msgs, add_generation_prompt=True))

    seen: dict[str, int] = {}
    prompt_len: dict[str, int] = {}
    kept_rows = 0
    for sid, p in zip(df["statement_id"].astype(str), df["prompt"].tolist(), strict=True):
        key = json.dumps(p.tolist() if hasattr(p, "tolist") else list(p), default=str, sort_keys=True)
        if key not in seen:
            seen[key] = chat_len(p)
        if seen[key] <= MAX_PROMPT_LENGTH:
            kept_rows += 1
            prompt_len[sid] = seen[key]
    prompt_ok = set(prompt_len)
    audit = load(POOL_AUDIT)
    consistency = {
        "audit_sampled_population_rows": audit["answer"]["sampled_population_rows"],
        "audit_sampled_population_unique_statements": audit["answer"]["sampled_population_unique_statements"],
        "this_run_rows_kept": kept_rows,
        "this_run_statements_kept": len(prompt_ok),
        "rows_match": kept_rows == audit["answer"]["sampled_population_rows"],
        "statements_match": len(prompt_ok) == audit["answer"]["sampled_population_unique_statements"],
    }
    if not (consistency["rows_match"] and consistency["statements_match"]):
        raise SystemExit(f"FATAL: token-filter disagrees with the committed pool audit: {consistency}")

    # formal-statement token lengths (owner §5 "formal-token distribution")
    raw = pd.read_parquet(ROOT / RAW_PARQUET)
    formal = dict(zip(raw["statement_id"].astype(str), raw["formal_statement"].astype(str)))
    source_of = dict(zip(raw["statement_id"].astype(str), raw["source"].astype(str)))
    if raw.groupby("statement_id")["source"].nunique().max() > 1:
        raise SystemExit("FATAL: statement_id maps to more than one source label; the source "
                         "stratification of owner §11 would be ambiguous")

    def tok_dist(ids) -> dict:
        texts = [formal[i] for i in ids if i in formal]
        n = sorted(len(tok(t)["input_ids"]) for t in texts)
        if not n:
            return {"n": 0}

        def q(f: float) -> int:
            return n[min(len(n) - 1, int(f * len(n)))]

        return {"n": len(n), "min": n[0], "p25": q(.25), "median": statistics.median(n),
                "p75": q(.75), "p90": q(.90), "p99": q(.99), "max": n[-1],
                "mean": round(statistics.fmean(n), 3)}

    def evaluate(exclude: list[str]) -> dict:
        removed_by: dict[str, list[str]] = {}
        blocked: set[str] = set()
        for name in exclude:
            if name == "__registry_any_exclusion_reason__":
                ids = [cid for cid, c in comps.items() if c["exclusion_reasons"]]
            elif name == "__b002_b003_any_component__":
                ids = sorted(set(cls["b002_pilot_components"]["component_ids"])
                             | set(cls["b003_eval_components"]["component_ids"])
                             | set(cls["b003_buffer_components"]["component_ids"]))
            else:
                ids = cls[name]["component_ids"]
            removed_by[name] = sorted(ids)
            blocked |= set(ids)
        kept_comps = [cid for cid in comps if cid not in blocked]
        kept_stmts_all = [s for cid in kept_comps for s in comps[cid]["members"]]
        kept_stmts_elig = [s for s in kept_stmts_all if s in prompt_ok]
        comps_elig = [cid for cid in kept_comps
                      if any(s in prompt_ok for s in comps[cid]["members"])]
        # one-candidate-per-component reduction used by the §7 sampling unit: the lexicographically
        # smallest eligible member id of the component (deterministic, no outcome information)
        reps = {cid: min(s for s in comps[cid]["members"] if s in prompt_ok)
                for cid in comps_elig}
        cands = sorted(reps.values())
        # how many theorems could be drawn if the owner relaxed "one per component"?
        eligible_per_comp = [sum(1 for s in comps[cid]["members"] if s in prompt_ok)
                             for cid in comps_elig]
        return {
            "excluded": exclude,
            "components_removed_by_class": {k: len(v) for k, v in removed_by.items()},
            "components_blocked_total": len(blocked),
            "components_remaining": len(kept_comps),
            "statements_remaining_component_closed": len(kept_stmts_all),
            "components_remaining_after_prompt_filter": len(comps_elig),
            "statements_remaining_after_prompt_filter": len(kept_stmts_elig),
            "one_candidate_per_component": len(reps),
            "max_N_if_k_theorems_per_component": {str(k): sum(min(k, n) for n in eligible_per_comp)
                                                  for k in (1, 2, 3)},
            "one_per_component_eligible": len(cands),
            "components_removed_by_reading": len(blocked),
            "singleton_components": sum(1 for cid in comps_elig if len(comps[cid]["members"]) == 1),
            "component_size_hist": dict(sorted(Counter(
                len(comps[cid]["members"]) for cid in comps_elig).items())),
            "sources_of_one_per_component_eligible": dict(sorted(
                Counter(source_of.get(s, "unknown") for s in cands).items())),
            "formal_token_length_one_per_component_eligible": tok_dist(cands),
            "prompt_token_range_one_per_component_eligible": {
                "min": min((prompt_len[s] for s in cands), default=None),
                "max": max((prompt_len[s] for s in cands), default=None)},
            "candidate_statement_ids": sorted(reps.values()),
            "candidate_component_ids": sorted(comps_elig),
            "component_to_candidate": {cid: reps[cid] for cid in sorted(comps_elig)},
        }

    readings = {name: {"gloss": spec["gloss"], "result": evaluate(spec["exclude"])}
                for name, spec in READINGS.items()}

    prim = readings[args.primary]["result"]
    # what does EACH carve-out in the primary reading individually cost? (leave-one-out)
    marginal_cost = {name: {
        "status": cls[name]["status"],
        "candidates_returned_if_this_class_were_not_excluded":
            evaluate([e for e in READINGS[args.primary]["exclude"] if e != name])
            ["one_candidate_per_component"]}
        for name in READINGS[args.primary]["exclude"]}
    payload = {
        "artifact_type": "v3_r001_family_clean_pool",
        "status": "POOL FROZEN (read-only) - no rollout, no RL step, no model inference, no sample "
                  "drawn yet",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorized_by": "owner directive 2026-09-24 §5 and §17 step 4",
        "host": {"hostname": os.uname().nodename,
                 "git_revision": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                                capture_output=True, text=True,
                                                check=False).stdout.strip()},
        "question": "Which family components are still usable for a PROSPECTIVE test, once every "
                    "component touched by earlier work is removed - and how large is what is left?",
        "unit": "one unique statement_id == one canonical prompt (proven by the deployment-pool "
                "audit: duplicate rows are byte-identical)",
        "eligibility_rule": [
            ("component-level closure: a component survives only if NO member statement/component is "
             "in any excluded class"),
            (f"prompt filter: len(apply_chat_template(prompt, add_generation_prompt=True)) <= "
             f"{MAX_PROMPT_LENGTH} (the trainer's own rule, so a candidate is actually drawable)"),
            ("one candidate per component: lexicographically smallest eligible statement_id "
             "(deterministic, outcome-blind)"),
        ],
        "inputs": {rel: sha256_file(ROOT / rel) for rel in
                   [REGISTRY, ROLE_REGISTRY, POOL_AUDIT, ARTIFACTS["e023_holdout"],
                    ARTIFACTS["a001_selection"], ARTIFACTS["b001_audit"], ARTIFACTS["b002_pilot"],
                    ARTIFACTS["b002_calibration"], ARTIFACTS["b003_eval"], TRAIN_PARQUET,
                    RAW_PARQUET]},
        "tokenizer": TOKENIZER,
        "usage_classes": {k: {kk: vv for kk, vv in v.items() if kk != "statements"}
                          | {"n_statements_named": v["n_statements_named"],
                             "n_components_resolved": v["n_components_resolved"],
                             "statement_ids_sample": (v.get("statements") or [])[:3],
                             "unmapped": {"unknown_component_ids": v["unknown_component_ids"],
                                          "statement_ids_not_in_registry":
                                              v["statement_ids_not_in_registry"]}}
                          for k, v in cls.items()},
        "local_rollout_contact_evidence": contact,
        "readings": {k: {"gloss": v["gloss"],
                         "result": {kk: vv for kk, vv in v["result"].items()
                                    if kk != "candidate_statement_ids"
                                    and kk != "component_to_candidate"}}
                     for k, v in readings.items()},
        "primary_reading": args.primary,
        "marginal_cost_of_each_carveout_in_primary_reading": marginal_cost,
        "pool": {k: v for k, v in prim.items() if k != "component_to_candidate"},
        "pool_component_to_candidate": prim["component_to_candidate"],
        "pools_by_reading": {
            k: {"pool_hash": pool_hash_of(v["result"], k),
                "one_candidate_per_component": v["result"]["one_candidate_per_component"],
                "max_N_if_k_theorems_per_component":
                    v["result"]["max_N_if_k_theorems_per_component"],
                "sources_of_one_per_component_eligible":
                    v["result"]["sources_of_one_per_component_eligible"],
                "formal_token_length_one_per_component_eligible":
                    v["result"]["formal_token_length_one_per_component_eligible"],
                "candidate_statement_ids": v["result"]["candidate_statement_ids"],
                "candidate_component_ids": v["result"]["candidate_component_ids"],
                "component_to_candidate": v["result"]["component_to_candidate"]}
            for k, v in readings.items()},
        "pool_hash": pool_hash_of(prim, args.primary),
        "extraction_union": sorted(set().union(*(set(v["result"]["candidate_statement_ids"])
                                                 for v in readings.values()))),
        "token_filter_consistency_with_pool_audit": consistency,
        "capacity_verdict": {
            "owner_preferred_N": 128,
            "capacity_by_reading": {k: v["result"]["one_candidate_per_component"]
                                    for k, v in readings.items()},
            "primary_reading": args.primary,
            "primary_reading_capacity": prim["one_candidate_per_component"],
            "primary_reading_can_support_N128_without_reusing_a_family":
                prim["one_candidate_per_component"] >= 128,
            "readings_that_can_support_N128": sorted(
                k for k, v in readings.items()
                if v["result"]["one_candidate_per_component"] >= 128),
            "structural_point": (
                "Capacity is decided ENTIRELY by which carve-outs count. The difference between the "
                "readings is not argument about leakage mechanics - the same statement sets are "
                "involved in every reading - it is whether a V2 role assignment for a track that was "
                "formally stopped (Track C's 2x2 superseded before launch, V2-B004 = NOT RUN) makes a "
                "family unusable. Only the owner can settle that, and reclaiming those families "
                "without an owner decision would be exactly the '偷回 families for sample size' the "
                "directive forbids."),
            "note": "See owner_decision_required §R001-pool-reading in the memo: under the strict "
                    "reading the pool cannot reach N=128 one-per-component.",
        },
        "prohibitions_respected": [
            "no family reclaimed for sample size (every reading is a pure subtraction)",
            "no rollout, no generation, no verifier call, no GPU use",
            "no frozen V2 / D001 artifact modified",
            "no sample drawn yet (owner §7 sequencing: pool -> power -> sample)",
        ],
    }
    path = ROOT / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"capacity_by_reading": payload["capacity_verdict"]["capacity_by_reading"],
                      "primary": args.primary,
                      "can_support_N128": payload["capacity_verdict"]["readings_that_can_support_N128"],
                      "contact": {k: v for k, v in contact.items() if k != "dirs_scanned"},
                      "dirs": contact["dirs_scanned"],
                      "pool_hash": payload["pool_hash"],
                      "consistency": consistency}, indent=2))
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
