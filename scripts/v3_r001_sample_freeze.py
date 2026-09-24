#!/usr/bin/env python3
"""V3-R001 owner decisions 1-4 (2026-09-24) — freeze the FORMAL sample and seal the FINAL RESERVE.

The owner settled the two open choices: the contamination reading is `consumed_only`, and the 221
clean family components are allocated 128 to R001 and 93 to a sealed V3 final-capacity reserve. This
script performs that allocation and writes the two manifests.

The load-bearing discipline: the 128 are NOT picked again. They are exactly the sample that §14's
power calculation already used -- recomputed with the same frozen draw function and compared to the
committed `sample_sha256`, and the script refuses to write anything unless the hashes match. The 93
reserve components are the pool minus that draw, so reserve membership is the complement of a
uniform permutation and cannot have been influenced by a q value or by an outcome.

Writes:
  experiments/manifests/v3/v3_r001_formal_sample.json
  experiments/manifests/v3/v3_final_holdout_reserve.json

No rollout, no generation, no verifier, no GPU, no label. Nothing here consumes a result.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

# the draw and the hash come from the script that froze them, not from a re-implementation:
# identity with §14's power sample has to be checked against the real mechanism.
from v3_r001_prescore import DRAW_SEED, draw_components, quantiles, sha

POOL = "experiments/manifests/v3/V3-R001_family_clean_pool.json"
PREDICTIONS = "experiments/manifests/v3/V3-R001_predictions.json"
FORMAL_OUT = "experiments/manifests/v3/v3_r001_formal_sample.json"
RESERVE_OUT = "experiments/manifests/v3/v3_final_holdout_reserve.json"

READING = "consumed_only"                         # owner decision 1
N_NOMINAL = 128                                   # owner decision 4 (N=192 rejected)
TOP_FRACTION = 0.20
FORMAL_KEY = f"{READING}|N={N_NOMINAL}"

OWNER_TRACK_C_RELEASE = (
    "V2 Track C reserved families were released by owner for V3 because Track C was never executed "
    "and no outcomes from those families were observed. This is a governance change in reservation "
    "status, not a reuse of previously evaluated data.")

RESERVE_FORBIDDEN_USES = [
    "R001 rollout (not one generation of a reserve theorem)",
    "controller fitting of any kind, now or later",
    "sampler tuning, including the lambda diagnostics",
    "power tuning, or any revision of the gate",
    "R001 analysis, including any post-hoc stratification or robustness check",
    "future hyperparameter or checkpoint selection",
    "changing reserve membership on the basis of q values",
]

Q_FIELDS = ["q_B2_controller", "q_B1_handcrafted", "q_B2_fold_ensemble", "q_B1_fold_ensemble"]


def load(rel: str):
    return json.loads((ROOT / rel).read_text())


PROVENANCE = ("bytes are a pure function of the frozen pool, the frozen predictions and the draw "
              "rule; no wall-clock, host or git field, so the sha256 pinned in registry.yaml is "
              "re-checkable on any host by re-running. Freeze time, host and commit live in "
              "registry.yaml and in the preregistration commit, not here.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--formal-out", default=FORMAL_OUT)
    ap.add_argument("--reserve-out", default=RESERVE_OUT)
    args = ap.parse_args()

    pool, pred = load(POOL), load(PREDICTIONS)
    if READING not in pool["pools_by_reading"]:
        raise SystemExit(f"FATAL: reading {READING!r} is not in the frozen pool artifact")
    p = pool["pools_by_reading"][READING]
    pool_components = sorted(p["candidate_component_ids"])
    comp_to_stmt = p["component_to_candidate"]
    if len(pool_components) != p["one_candidate_per_component"]:
        raise SystemExit(f"FATAL: {READING} lists {len(pool_components)} component ids but "
                         f"claims {p['one_candidate_per_component']} candidates")

    frozen = pred["prospective_samples"].get(FORMAL_KEY)
    if frozen is None:
        raise SystemExit(f"FATAL: no frozen candidate sample {FORMAL_KEY} in the predictions "
                         "artifact -- the sample the owner froze cannot be reconstructed")

    # --- the identity check that makes this a freeze and not a re-draw ---------------------------
    recomputed = draw_components(pool_components, N_NOMINAL, seed=DRAW_SEED)
    if recomputed != list(frozen["component_ids"]):
        raise SystemExit(f"FATAL: recomputed draw differs from the frozen {FORMAL_KEY} sample; the "
                         "first divergence is at "
                         f"{next(i for i,(a,b) in enumerate(zip(recomputed, frozen['component_ids'])) if a != b)}")
    recomputed_sha = sha({"reading": READING, "seed": DRAW_SEED, "components": recomputed})
    if recomputed_sha != frozen["sample_sha256"]:
        raise SystemExit(f"FATAL: sample hash {recomputed_sha} != the committed "
                         f"{frozen['sample_sha256']}; refusing to call this the power-calculation "
                         "sample")

    stmts = [comp_to_stmt[c] for c in recomputed]
    cand = pred["candidate_scores"]
    missing = [s for s in stmts if s not in cand]
    if missing:
        raise SystemExit(f"FATAL: {len(missing)} sampled theorems have no frozen q (owner §6): "
                         f"{missing[:3]}")

    # the top-20% block, frozen now: rank by q_B2, ties by (-q, statement_id) so the block is a
    # function of committed objects and not of a sort's incidental order
    m = round(TOP_FRACTION * N_NOMINAL)
    order = sorted(range(N_NOMINAL), key=lambda i: (-cand[stmts[i]]["q_B2_controller"], stmts[i]))

    theorems = []
    for pos, i in enumerate(order):
        sid, cid = stmts[i], recomputed[i]
        c = cand[sid]
        theorems.append({
            "rank_by_q_B2": pos + 1,
            "in_top20pct_block": pos < m,
            "component_id": cid, "statement_id": sid, "source": c["source"],
            "prompt_token_count": c["prompt_token_count"],
            "formal_char_count": c["formal_char_count"],
            "step_norm_used": c["step_norm_used"],
            **{k: c[k] for k in Q_FIELDS},
            "has_historical_label": c["has_historical_label"],
        })

    reserve_components = [c for c in pool_components if c not in set(recomputed)]
    reserve_stmts = [comp_to_stmt[c] for c in reserve_components]
    if set(reserve_components) & set(recomputed) or set(reserve_stmts) & set(stmts):
        raise SystemExit("FATAL: formal sample and reserve are not disjoint")
    if sorted(reserve_components + recomputed) != pool_components:
        raise SystemExit("FATAL: formal + reserve does not reproduce the 221-component pool")
    if len(set(reserve_stmts)) != len(reserve_stmts):
        raise SystemExit("FATAL: reserve statement ids are not unique")
    if set(reserve_stmts) & set(stmts):
        raise SystemExit("FATAL: a statement appears in both the formal sample and the reserve")

    mix = {}
    for s in stmts:
        mix[cand[s]["source"]] = mix.get(cand[s]["source"], 0) + 1
    rmix = {}
    for s in reserve_stmts:
        rmix[cand[s]["source"]] = rmix.get(cand[s]["source"], 0) + 1
    q2 = [cand[s]["q_B2_controller"] for s in stmts]
    q1 = [cand[s]["q_B1_handcrafted"] for s in stmts]
    if sha(q2) != frozen["q_B2_vector_sha256"]:
        raise SystemExit("FATAL: the frozen q_B2 vector of this sample does not hash to the "
                         "committed value -- the ranking is not the one that was frozen")
    if sha(q1) != frozen["q_B1_vector_sha256"]:
        raise SystemExit("FATAL: the frozen q_B1 vector of this sample does not hash to the "
                         "committed value")

    formal = {
        "artifact_type": "v3_r001_formal_sample",
        "status": ("FROZEN_AND_LAUNCH_PENDING — the one sample R001 may analyze, fixed before any "
                   "outcome; no rollout, no generation, no verifier call, no label exists"),
        "provenance": PROVENANCE,
        "authorized_by": "owner decisions 2026-09-24 §2, §3, §4 and directive §7, §17 step 6",
        "question": ("Which 128 family-clean theorems does R001 draw, and is that set provably the "
                     "same set the §14 power calculation was run on?"),
        "reading": READING, "N_nominal": N_NOMINAL, "pool_capacity": len(pool_components),
        "predictions_sample_key": FORMAL_KEY,
        "sample_sha256": recomputed_sha,
        "identity_with_the_power_sample": {
            "claim": ("this is byte-for-byte the sample used for the frozen power/gate calculation, "
                      "not a fresh draw that happens to have the same size"),
            "draw_seed": DRAW_SEED,
            "recomputed_component_order_matches": True,
            "recomputed_sample_sha256": recomputed_sha,
            "committed_sample_sha256": frozen["sample_sha256"],
            "q_B2_vector_sha256": sha(q2),
            "committed_q_B2_vector_sha256": frozen["q_B2_vector_sha256"],
            "q_B1_vector_sha256": sha(q1),
            "committed_q_B1_vector_sha256": frozen["q_B1_vector_sha256"],
            "provenance_of_the_frozen_hash": f"{PREDICTIONS}#prospective_samples[{FORMAL_KEY}]",
        },
        "draw_rule": {
            "unit": "family component -> its deterministic theorem representative (one theorem per "
                    "family, which is also what makes the §14 test exact)",
            "mechanism": f"order by SHA256('<seed>|<component_id>'), take the first {N_NOMINAL}",
            "seed": DRAW_SEED,
            "why_not_a_library_rng": ("a hash order is reproducible on any numpy/python version, so "
                                      "'fixed before the result' stays checkable"),
        },
        "governance": {
            "contamination_definition": READING,
            "owner_release_of_track_c": OWNER_TRACK_C_RELEASE,
            "v2_history_untouched": ("no V2 frozen manifest or result was modified; the release is "
                                     "recorded as V3 provenance only"),
            "allocation": {"r001_components": N_NOMINAL,
                           "final_reserve_components": len(reserve_components),
                           "reserve_manifest": RESERVE_OUT},
            "single_sample_rule": ("whatever the result, only this sample may be analyzed. The other "
                                   "nine frozen candidate samples are DESIGN-ONLY / NEVER ANALYZE "
                                   "(owner decision 3); switching samples after seeing an outcome is "
                                   "prohibited."),
        },
        "top20pct_block": {
            "fraction": TOP_FRACTION, "size": m,
            "defined_by": "q_B2_controller descending, ties by statement_id ascending",
            "note": ("the block is a function of frozen objects only; no outcome can move a theorem "
                     "in or out of it"),
            "statement_ids": [t["statement_id"] for t in theorems if t["in_top20pct_block"]],
            "sha256": sha([t["statement_id"] for t in theorems if t["in_top20pct_block"]]),
        },
        "source_distribution": dict(sorted(mix.items())),
        "source_distribution_sha256": sha(dict(sorted(mix.items()))),
        "prompt_token_count": quantiles([t["prompt_token_count"] for t in theorems]),
        "context_check": {
            "max_prompt_token_count": max(t["prompt_token_count"] for t in theorems),
            "max_response_tokens": 4096, "max_model_len": 5120,
            "fits": max(t["prompt_token_count"] for t in theorems) + 4096 <= 5120,
        },
        "q_summary": {"q_B2_controller": quantiles(q2), "q_B1_handcrafted": quantiles(q1)},
        "coverage": {"n_theorems": len(theorems), "n_components": len(set(recomputed)),
                     "n_unique_statement_ids": len(set(stmts)),
                     "one_theorem_per_component": len(set(recomputed)) == len(theorems),
                     "every_theorem_has_a_q": all(0.0 <= t[k] <= 1.0 and np.isfinite(t[k])
                                                  for t in theorems for k in Q_FIELDS),
                     "any_historical_label": any(t["has_historical_label"] for t in theorems),
                     "n_missing_q": 0},
        "membership_hashes": {
            "component_ids_sha256": sha(recomputed),
            "statement_ids_sha256": sha(stmts),
            "pool_hash_of_this_reading": p["pool_hash"],
            "union_of_formal_and_reserve_equals_pool": True,
        },
        "theorems": theorems,
        "inputs_read_only": [POOL, PREDICTIONS],
        "authorizes_next": ("§17 step 6's remaining part (write the preregistration around this "
                            "sample) and then the owner's launch decision. It authorizes no GPU "
                            "work and starts no rollout."),
        "prohibitions_respected": [
            "no rollout, no generation, no verifier, no label, no n=8 sampling",
            "no RL training, no optimizer step, no gradient",
            "the 10 GB training smoke remains NOT AUTHORIZED and was not run",
            "no re-selection among the ten frozen candidate samples (owner decision 3)",
            "no reserve component appears in this sample",
            "no V2 / V3-D001 frozen artifact modified",
        ],
    }

    reserve = {
        "artifact_type": "v3_final_holdout_reserve",
        "status": "SEALED",
        "provenance": PROVENANCE,
        "purpose": "future final family-clean capability evaluation",
        "authorized_by": "owner decisions 2026-09-24 §2 and §13",
        "reading": READING,
        "n_components": len(reserve_components),
        "membership_rule": {
            "how": "the 221-component consumed_only pool MINUS the frozen N=128 formal sample",
            "consequence": ("membership is the complement of a uniform hash-ordered permutation, so "
                            "it cannot have been chosen by q value, by difficulty, or by any "
                            "outcome (owner decision 2)"),
            "formal_sample_sha256": recomputed_sha,
        },
        "seal": {
            "forbidden_uses": RESERVE_FORBIDDEN_USES,
            "mechanical_guards": [
                "no R001 script reads this manifest except to assert disjointness",
                "the analysis script refuses any sample member whose component_id is in this list",
                "membership is hash-committed, so a later edit is detectable",
            ],
            "disclosed_limitation": ("q values for these components exist in the already-committed "
                                     "V3-R001_predictions.json, because owner §6 required scoring "
                                     "every candidate theorem before any outcome. The seal therefore "
                                     "binds USE, not existence: no R001 decision, gate, ranking or "
                                     "reserve membership may be derived from them."),
            "if_r001_stops": ("the reserve stays sealed and available for a final external/"
                              "confirmatory evaluation of the paper, per owner decision 13"),
        },
        "source_distribution": dict(sorted(rmix.items())),
        "membership_hashes": {
            "component_ids_sha256": sha(reserve_components),
            "statement_ids_sha256": sha(reserve_stmts),
            "pool_hash_of_this_reading": p["pool_hash"],
            "disjoint_from_formal_sample_components": True,
            "disjoint_from_formal_sample_statements": True,
            "formal_plus_reserve_equals_pool": True,
        },
        "what_is_recorded": ("component and statement ids plus hashes and the source label only. No "
                             "q values are copied here, and none are needed: the seal is about not "
                             "using them."),
        "components": [{"component_id": c, "statement_id": comp_to_stmt[c],
                        "source": cand[comp_to_stmt[c]]["source"]} for c in reserve_components],
        "inputs_read_only": [POOL, PREDICTIONS, FORMAL_OUT],
        "prohibitions_respected": [
            "no generation, no verifier outcome, no reward observation, no training",
            "no checkpoint or hyperparameter selection touched these families",
            "no membership change based on q values",
        ],
    }

    for path, payload in ((Path(args.formal_out), formal), (Path(args.reserve_out), reserve)):
        path = ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))
        print("wrote", path)
    print(json.dumps({"formal_sample_sha256": recomputed_sha,
                      "matches_committed": recomputed_sha == frozen["sample_sha256"],
                      "N": len(theorems), "top20_block": m, "reserve": len(reserve_components),
                      "formal_source_mix": dict(sorted(mix.items())),
                      "reserve_source_mix": dict(sorted(rmix.items())),
                      "disjoint_and_covering_pool": True}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
