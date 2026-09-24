# V3-R001 §5 — the prospective family-clean pool, and what is actually left

Artifact: `experiments/manifests/v3/V3-R001_family_clean_pool.json`
Producer: `scripts/v3_r001_pool_build.py` (read-only; no rollout, no RL step, no model inference)
Authority: owner directive 2026-09-24 §5 and §17 step 4. Depends on §17 step 3
(`docs/v3/V3-R001_deployment_pool_audit.md`) for the sampling unit.

Headline: **owner §5's exclusion list, honored literally, leaves 56 family-clean components — not
the 128 theorems of §7.** The pool cannot reach the planned sample size without an owner decision
about whether V2 reservations for tracks that were formally stopped still bind. This document
measures that instead of quietly choosing a reading (`不要为了样本量偷回这些 families`).

---

## 1. What a candidate has to satisfy

Three rules, all applied to frozen inputs, none of them re-derived:

1. **Component-level closure.** A family component survives only if *no* member statement and no
   component id appears in any excluded usage class. Removing one statement of a component removes
   the whole component, so the statement counts of §5's list understate the cost:
   V1's 612 labelled statements are 443 components, and the 8-dump `v1_used` union is 527.
2. **The trainer's own prompt filter** — `len(apply_chat_template(prompt, add_generation_prompt=True))
   ≤ 1024`. A theorem the deployment sampler could never draw is not a legitimate test theorem for a
   sampler-derived claim. This run reproduces step 3's numbers exactly (24,246 rows / 7,613
   statements), which is the cross-check that the filter here and the filter in the audit are the
   same computation; the script hard-fails if they ever disagree.
3. **One candidate per component**, the lexicographically smallest eligible member id. Deterministic
   and outcome-blind, so the §7 "family component first → deterministic theorem representative"
   unit is one physical draw per family.

Every usage class was resolved against the frozen registries with **zero** unmapped ids
(`usage_classes[*].unmapped.statement_ids_not_in_registry = 0`, `unknown_component_ids = 0`), so no
carve-out silently evaporated because an id was not recognized.

## 2. "Touched" is not one concept — consumed vs reserved

§5's word is *touched*. A role assignment in a registry is a bookkeeping act; a rollout dump is
contact with an outcome. The script separates the two and, for the first time, checks contact
directly: it parses **every** rollout jsonl present on this host (`local_rollout_contact_evidence`)
and resolves each prompt to a `statement_id` through V3-D001's own normalization path.

| directory | files | unique statements |
|---|---|---|
| `runs/p3b_pilot/rollout_data` | 60 | 227 |
| `runs/m1_seed2/rollout_data` | 60 | 229 |
| `runs/m1_seed3/rollout_data` | 60 | 227 |
| `runs/m2_qwen_smoke/rollout_data` | 5 | 20 |
| `.cache/m1_seed3_r2r3_dumps` | 7 | 28 |
| `experiments/p3_0_batch`, `experiments/p3_b_n8_batch` | 3 + 3 | 0 (GRPO batch fixtures, no prompt) |

Union: **612** statements with real generation output on disk — precisely V3-D001's labelled set
(`sanity_check_v1_seed123_fully_contacted = true`), and zero additional statements from the smoke
run or the aborted r2/r3 backups. All four `reserved-never-used` classes show
`with_local_rollout_output = 0`.

The honest limit, recorded in the artifact: absence on *this* host is not proof of absence. The
E013 / E016 / P3C(E018) / IGR-mechanism dumps are named by the frozen V2 registries but no longer
exist here, so their contact is accepted from the registry rather than re-derived. That is why the
`v1_used_union_wider` class is taken from `family_component_registry.json`'s per-component
`exclusion_reasons == "v1_used"` (527 components) instead of from disk. The authoritative record for
the reserved classes is different in kind: `docs/v2/track_b_closeout.md` §6 states V2-B004 = NOT RUN
and that no allocator training corpus was generated, and Track C's 2x2 was superseded before launch.

## 3. Five readings, because the answer is the owner's

The script does not pick. It evaluates the carve-out sets §5 names under five defensible readings:

| reading | components blocked | one-per-component capacity | N if 2 per family | N if 3 | sources (auto / human / synth) |
|---|---|---|---|---|---|
| `owner_literal` (§5 exactly, "V1 seed1/2/3" = the 3 RL seeds) | 1,619 | **86** | 106 | 126 | 44 / 24 / 18 |
| `owner_literal_wider_v1` (as above, V1 = 8-dump union) | 1,649 | **56** | 68 | 80 | 28 / 17 / 11 |
| `consumed_only` (remove only what was actually used; C survives) | 1,483 | **221** | 309 | 397 | 101 / 42 / 78 |
| `owner_literal_all_reservations_honored` (+ A-reserve, B003 buffer, B-role mass) | 1,706 | **0** | 0 | 0 | — |
| `registry_labels_maximal` (any registry exclusion flag, or any B002/B003 name) | 1,706 | **0** | 0 | 0 | — |

Read left to right, this is a nested chain: the wider-V1 pool is a subset of the literal pool
(56 ⊆ 86), and `literal ∩ consumed_only = 56` exactly. Statements remaining after the prompt filter:
247 (literal), 150 (wider), 970 (consumed_only). Formal-statement token medians are similar across
readings (167.5 / 173.0 / 151), and prompt lengths stay inside the filter by construction
(149–1017 tokens for the wider pool) — so the capacity difference is families, not length.

What each carve-out individually costs, measured by leaving it out of the primary reading:

| carve-out | status | capacity if it were NOT excluded |
|---|---|---|
| `b002_pilot_components` (512) | consumed | 567 |
| `v1_used_union_wider` (527 comps) | consumed | 251 |
| `b003_eval_components` (192) | consumed | 245 |
| `c_joint_holdout` (680 stmts / 504 comps) | **reserved-never-used** | 221 |
| `a001_selection` (512) | consumed | 167 |
| `e023_final_holdout` (128) | consumed | 63 |
| `b001_audit_reserved` (16) | consumed | 59 |
| `b002_calibration` (64) | consumed | 56 |

So V2-B002's pilot is the dominant sink — it alone holds 512 of the 1,706 components, which is why
`b002_pilot_components` was legitimately the largest carve-out in §5's list and is excluded under
*every* reading. The one carve-out that decides whether N=128 exists at all is Track C's
`c_joint_holdout`: 56 → 221.

## 4. The consequence for §7, stated plainly

* Under the literal §5 reading the experiment can draw at most **56** theorems from distinct
  families (or 86 under the narrow "V1 = seed1/2/3" reading). N=128 is unreachable there, and
  reaching it by taking 2–3 theorems per family would contradict §7's "prefer one component one
  theorem" *and* concentrate the sample in ≤ 12 multi-member families (component size histogram of
  the wider pool: `{1: 44, 5: 1, 7: 2, 8: 1, 9: 1, 10: 7}`).
* The **only** reading that supports §7's preferred N=128 with one theorem per family — and even
  N=192 — is `consumed_only`, which differs from the literal reading in exactly one respect: it
  treats Track C's 680 reserved statements as a reservation rather than as contact, because Track C
  was superseded before launch and V2-B004 = NOT RUN.
* Honoring *every* remaining V2 reservation (A-reserve 162, B003's 24 buffer components, and the
  5,361-statement B-train/validation/test role mass) leaves **nothing**: 0 components. That is not an
  argument that the project is out of data; it is the mark of how thoroughly V2 partitioned the
  promptset for an allocator experiment that was stopped. It does mean any usable prospective pool
  necessarily contains material that was *assigned* a role but never *used*.
* The strictest computable reading (`registry_labels_maximal`) is also 0, i.e. the V2 design's own
  eligibility labels already describe an exhausted pool: 728 eligible components / 1,073 statements,
  of which B002's pilot and B003's evaluation then consumed 704.

This is an owner decision, not an engineering one. `consumed_only` is the recommended option on the
evidence in §2 (no on-disk contact, plus a closeout that says the consuming experiment never ran),
but it releases a class §5 explicitly named ("C/final reserved components"), so it requires the
owner's signature. Nothing was reclaimed for sample size: every reading is a pure subtraction, and
the `owner_literal_wider_v1` reading remains the artifact's primary, so the conservative pool is the
one that is committed and hashed today.

## 5. Frozen hashes

`pool_hash` = sha256 over the canonical JSON of `{candidate components, candidate statements,
component→candidate map, unit="statement_id", max_prompt_length=1024, reading}`. All five readings
are frozen now, before any sample is drawn and before any rollout exists, so the owner's choice
cannot be made after seeing an outcome. Re-running the script reproduces the artifact byte-for-byte
apart from `created_at_utc` and `host`.

| reading | pool_hash |
|---|---|
| `owner_literal` | `bea0b85da4d5f05d658ae9c4482c910c583dcda9154ddde520b8498d0ab31b63` |
| `owner_literal_wider_v1` (primary) | `d2d3b58e0ce26680667d1128644ade281cacdf5f9a2a934eba91f9dca5b97edb` |
| `owner_literal_all_reservations_honored` | `b3467a16aecd5ad1aea55c92934a4aea8565c3d4c11be2314fd181f499209925` |
| `consumed_only` | `72b5bd4922e70d4a79e28470b7f34e86026cd341315b0e50232018553364fd5a` |
| `registry_labels_maximal` | `61f08e5c97b4b7c2a326bacd7c43970ca07055cae2829ff2ad7ac08499a878ba` |

The union of candidates over all readings is **251** statement ids
(`extraction_union`), and it is a superset of every pool the owner might approve. §17 step 7's θ0
forward pass and `q_i` freeze can therefore cover all options in one reading-agnostic run on fly122,
with the draw itself made afterwards from the frozen, outcome-blind rule.

## 6. Prohibitions respected

* No family reclaimed for sample size: five readings are all pure subtractions of frozen sets, and
  the conservative literal reading is the committed primary.
* No rollout, no generation, no verifier call, no GPU, no RL optimizer step, no 10 GB smoke.
* No frozen V2 / D001 artifact modified — the registries and the six carve-out manifests are inputs,
  hashed as received (`inputs` block records each sha256).
* No sample drawn (owner §7 sequencing: pool → power → sample), and no `epsilon`-style support hack
  anywhere in this file (§13 abandoned ε=0 permanently; this pool is a list of eligible theorems,
  not a probability distribution).
