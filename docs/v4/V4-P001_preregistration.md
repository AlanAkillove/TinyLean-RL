# V4-P001 — Paired Verifier-Guided Repairability Probe

**Preregistration.** Written and committed **before any V4 formal generation exists**: no V4
theorem rollout, no V4 verifier call, no V4 arm outcome, no V4 label. Every object named below is
`FROZEN`; the only generation that has run under this directive is the §I infrastructure smoke on
already-consumed material, which produces no analyzed outcome.

| | |
|---|---|
| Authority | owner directive 2026-09-25, "TinyLean-RL enters V4", §A–§S |
| Direction | `Verifier-Guided Repair for Sub-Billion Lean Provers` (V4) |
| Question | owner §B/§E, verbatim in §1 below |
| Pool + order | `experiments/manifests/v4/v4_p001_pool.json` (producer `scripts/v4_p001_pool_build.py`) |
| Seeds | `experiments/manifests/v4/v4_p001_seeds.json` (producer `scripts/v4_p001_freeze_seeds.py`) |
| Historical audit | `experiments/manifests/v4/v4_p001_audit.json` (producer `scripts/v4_p001_audit.py`) |
| Prompt audit | `experiments/manifests/v4/v4_p001_prompt_audit.json` (producer `scripts/v4_p001_prompt_audit.py`) |
| Power | `experiments/manifests/v4/v4_p001_power.json` (producer `scripts/v4_p001_power.py`) |
| Verifier plan | `experiments/manifests/v4/v4_p001_verifier_plan.json` (producer `scripts/v4_p001_verifier_plan.py`) |
| Context smoke | `experiments/manifests/v4/v4_p001_context_smoke.json` (+ `v4_p001_smoke_prompt.json`) |
| Registry | `experiments/manifests/v4/registry.yaml` |
| Taxonomy | `src/tinylean_rl/evaluation/v4_taxonomy.py` (`v4-taxonomy-1`) |
| Normalizer | `src/tinylean_rl/evaluation/v4_diagnostics.py` (`v4-diag-1`) |
| Renderer | `src/tinylean_rl/evaluation/v4_prompts.py` (`v4-prompt-1`) |
| Statistics | `src/tinylean_rl/evaluation/v4_stats.py` |
| Host | fly122, RTX 3080 10 GB, n=1 (owner §I) |
| Launch state | **NOT LAUNCHED.** Screening requires the owner's explicit go after this commit reaches `origin` (`FORMAL_V4_P001_AUTHORIZED: NO`). |

---

## 1. What is being tested, and what is not

The research question is the owner's, unchanged:

> Given a Lean-level failed proof produced by frozen Kimina-Distill-0.6B theta0, does the exact Lean
> verifier diagnostic improve second-attempt proof success beyond (A) a fresh retry and (B)
> self-revision without the diagnostic?

V4-P001 is a **mechanism probe**, not training and not a capability claim. It is explicitly **not**:
RL, SFT, distillation, checkpoint update, controller training, optimizer work of any kind (owner
§B, §R); a final-holdout experiment (the 93-component V3-FINAL-HOLDOUT stays `SEALED / NEVER USED`);
a reopening of V1, V2 or V3 (no V3-D001 or V3-R001 result is touched, re-analyzed or reinterpreted);
a verifier-throughput or latency study (owner §P forbids the claim); and not V4-R001, which exists
only as a downstream branch in §12 below and is not part of this experiment.

What it *is*: three paired second-attempt arms on the same frozen cohort of 128 theorems that theta0
failed at Lean level on the first attempt, with one shared sampling seed per theorem across arms
(owner §J), differing **only** in what the second user turn contains (owner §E/§F).

## 2. Frozen inputs, and the sealed sets

The V4 namespace is new (`experiments/manifests/v4/`). No V1/V2/V3 artifact is edited. The V2 family
component registry is read as received and hashed in the pool artifact's `inputs` block.

The sealed reserve is asserted, not assumed: `v4_p001_pool.json.checks` records
`sealed_components_touched = 0` and `sealed_statements_touched = 0` against
`experiments/manifests/v3/v3_final_holdout_reserve.json`, plus
`v3_formal_sample_components_touched = 0`, `v3_formal_sample_statements_touched = 0`,
`a_reserve_components_touched = 0`, `b1_audit_components_touched = 0`,
`b1_reserved_ids_components_touched = 0`, `b003_buffer_components_touched = 0` — eight disjointness
checks, all `0`. The union of the excluded classes is 330 components. The pool artifact carries
thirteen checks in total and every one of them passes.

This is the owner's §A requirement — V3-FINAL-HOLDOUT 93 components remain `SEALED / NEVER USED`
and are excluded from every V4 pool by construction — and the frozen pool's hashes are the proof.

## 3. The development pool (owner §B)

**Unit.** The family component of the frozen V2 registry: 1,706 components, component-level
closure, `representative_statement_id = min sha256(statement_id)` (the registry's own ordering).

**Tier 1** = already-consumed development/training families: `contains.v1_used >= 1` (453 components
before the prompt filter). These are the families V1 actually rolled out.

**Tier 2** = completed V2 development classes with released material, plus the Track-C class the
owner released in V3, in a frozen order:
`a001_selection` → `b002_pilot` → `b002_calibration` → `b003_eval` → `e023_holdout` →
`c_joint_holdout`. Tier 2 is admitted **only after the whole Tier 1 block is screened**; the block
order is frozen in the screening order itself and asserted by
`tier1_block_precedes_tier2_block: true`. Tier 2 was admissible here for two reasons recorded in the
V3 pool analysis: the material is *assigned a role but never used* (V2-B004 = NOT RUN, Track C
superseded before launch), and the owner's §B Tier-2 clause authorises explicitly non-reserved
development families when Tier 1 is insufficient — which it is: Tier 1 caps at 453 screens while the
frozen screening budget is 640.

**Excluded at component level** (each asserted as a zero intersection, never inferred): the 93
sealed reserve components; the 128-component V3 formal sample; `a_reserve` (145);
`b1_audit_reserved` + its 16 reserved ids; the 24-component `b003_buffer`.

**Representative rule** (deterministic, outcome-free, no human choice): members ordered by
`sha256(statement_id)`; the first member passing the trainer's frozen 1,024-token prompt filter is
the theorem. No outcome, source, difficulty, family size or historical repairability is read — this
is exactly the owner's §B prohibition. The field
`representative_is_registry_representative` records whether the chosen theorem is also the
registry's own representative.

**Pool result.** 1,371 components (Tier 1: 453; Tier 2: 918 — a001_selection 329, b002_pilot 511,
b002_calibration 50, b003_eval 189, e023_holdout 98, c_joint_holdout 281); 5 Tier-2 components had
no member inside the prompt filter and 0 had no source record. Sources: autoformalizer 686,
synthetic 462, human 223. Prompt lengths: min 112, median 271, p95 481, max 914 tokens.

    pool_hash  7533c94b4d1969f195ba7af81bc00023ef829a0495d67e5180cbf9fdfd3ffc2c
    order_hash ce329b36b5e7ee4712fa9bdd1a8ac2f42d919129297475cc9387828b197ecc15

`pool_hash` is the sorted component set, statement set and component→statement map; `order_hash` is
the base seed plus the screening order. Both were computed before any V4 generation.

**Screening order** (owner §B): sort by `sha256(f"{V4_BASE_SEED}|{statement_id}")` with
`V4_BASE_SEED = 20260925`, then cut into the frozen tier/class blocks. A uniform hash permutation is
verifiably independent of source, family, difficulty and every outcome label — it is the same
construction V3 used for its draw rule. `screening_ranks_contiguous` and `screening_keys_unique`
are both `true`.

## 4. Historical prevalence and the screening budget (owner §C)

Before fixing the budget, the failure prevalence of theta0 was measured on **already-consumed**
corpora only (no generation), with the frozen extractor re-run over recorded completions:

| corpus | n | primary | syntax | format | timeout | infra | verified |
|---|---|---|---|---|---|---|---|
| V1 training population (all steps) | 5,760 | 0.2498 | 0.0080 | 0.6339 | 0.0146 | 0.0139 | 0.0799 |
| V1 step-1 dump (theta0 proxy, Tier-1 families) | 96 | 0.3125 | 0.0208 | 0.6042 | 0.0417 | 0.0104 | 0.0104 |
| E023 holdout, theta0 | 512 | 0.2891 | 0.0137 | 0.4395 | 0.0137 | 0.0098 | 0.2344 |
| V3-R001 attempt-2, theta0 | 1,024 | 0.2910 | 0.0127 | 0.5537 | 0.0195 | 0.0273 | 0.0957 |

Three independent theta0 estimates agree at 0.289–0.313 (central 0.3008). The step-1 corpus is the
theta0 rollout that started training and is the closest available proxy on Tier-1 families. The
R001 classification **re-extracts** the proof body from the recorded completion with the project's
frozen extractor and asserts the result against the recorded `extracted_proof_present` flag: 0
mismatches over 1,024 rows. All 567 format failures there had no fenced Lean block, i.e. the format
class is genuine and not an artefact of re-extraction.

**Frozen budget (owner §C).** Planning floor `PRIMARY_RATE_FLOOR = 0.20` — deliberately below every
measured estimate. `MAX_SCREENING = ceil(128 / 0.20) = 640`. Anything in
`{0.20, 0.25, 0.29, 0.3125}` needs 410–640 screens, all inside both the budget and the pool
capacity; expected primary failures inside Tier 1 alone are 90.6–141.6. If 128 primary-eligible
failures are **not** collected within 640 screens, the run **STOPS and reports** before any
second-stage generation. N is never lowered after seeing repair outcomes; no theorem is ever
replaced from another family after second-stage outcomes exist (owner §C, §K).

## 5. The frozen error taxonomy (owner §C)

`src/tinylean_rl/evaluation/v4_taxonomy.py`, version `v4-taxonomy-1`, fixed before any V4 outcome
exists. Eleven categories, priority-ordered; each candidate receives exactly one:

| category | primary? |
|---|---|
| `elaboration_type_mismatch` | **primary** |
| `unsolved_goals` | **primary** |
| `tactic_failure` | **primary** |
| `unknown_identifier` | **primary** |
| `typeclass_synthesis` | **primary** |
| `other_semantic_lean_failure` | **primary** |
| `syntax_parser` | no — counted descriptively |
| `format_no_code` | no — the model never produced Lean code |
| `timeout_resource` | no — resource exhaustion, not a wrong proof |
| `infra` | no — verifier/infrastructure, never a candidate failure |
| `verified` | success |

The precedence order is deliberate and documented in the module: resource hints are tested first
("maximum number of heartbeats" can appear inside a semantic-looking message), name resolution
before syntax ("unknown identifier 'k'" vs "unexpected token ':'; expected ')'"), and the prose
heuristic runs only when proof text was actually stored. A Lean rejection with **no** diagnostic is
not screenable and is never promoted into the primary cohort.

**Primary eligible failure** means exactly: not verified, a real Lean-level diagnostic is available,
and the category is primary. Syntax, format, timeout/resource and infra are excluded from the
cohort by construction (owner §C).

**Ambiguity is measured, not assumed.** `matched_primary_rules` recomputes how many primary rules
the message text alone would have matched; the audit reports the fraction. Observed: 0.42 % (V1
population), 0.78 % (E023), 0.29 % (R001) — the corpus is overwhelmingly unambiguous, and the
figure is recorded rather than claimed to be zero.

Pass-rate check of the taxonomy against the observed corpora: the six primary rules absorb the great
majority of real Lean rejections (tactic failure dominating at 54–81 % of primary failures), and the
`other_semantic` bucket is small (V1 68/5,760; R001 15/1,024) with its top messages reported
verbatim in the audit for coverage review. No category was added after seeing a failure.

## 6. First attempt (owner §D)

Frozen at: model `AI-MO/Kimina-Prover-Distill-0.6B` revision
`332e8a5259d1bdfda19d7c7f339f30804813cd3a`, weights sha256
`34e6e630f564d330c79424c404ab0494558a0a659e6201b47d9bd88ccd640fe2`; temperature 1.0, top_p 1.0,
`n = 1`, max response 4096 tokens; the frozen chat template; the project's format-gated Lean
verification semantics.

**Theorem-level seeds** are deterministic and independent of source, family, error category and
every historical outcome:
`first_stage_seed(rank) = 20260925 + (rank − 1)` for `rank ∈ 1..640` (owner §D), frozen in
`v4_p001_seeds.json` with `screening_seed_hash 2efb53d8525b7ced44b13dc5aab388bbbf73ed920a65763918ef0e23ab0b8c51`.

**Screening statuses** (owner §D), all mutually exclusive: `INITIAL_SUCCESS`,
`PRIMARY_SEMANTIC_FAILURE`, `SYNTAX_FAILURE`, `FORMAT_FAILURE`, `TIMEOUT_OR_RESOURCE`,
`INFRA_CENSORED`, `CONTEXT_INELIGIBLE`. The screening flow is **descriptive**, not a hypothesis
test: it selects the cohort and is never quoted as evidence about repair.

`CONTEXT_INELIGIBLE` (owner §I) is a screening status, not a failure: a theorem whose Arm C context
cannot be carried losslessly inside the frozen window is excluded from the cohort and reported. The
rule is applied uniformly, before any second-stage generation.

## 7. Three paired arms (owner §E, §F, §G)

| arm | second-turn context |
|---|---|
| `A_FRESH_RETRY` | original theorem prompt only — byte-identical to the first attempt |
| `B_SELF_REVISION` | theorem + the same failed proof + a generic correction request |
| `C_VERIFIER_REPAIR` | identical to B **plus the exact normalized Lean diagnostic** |

Message structure (owner §F, preferred form): `user: theorem` → `assistant: failed proof` →
`user: correction request [ + diagnostic for C ]` → `assistant: generated repair`. The renderer uses
only `system` / `user` / `assistant` roles; the frozen template does support a `tool` role, and the
renderer **does not use it** — recorded as `tool_role_supported: true`,
`tool_role_used_by_renderer: false`.

The frozen correction request is one sentence and identical for B and C:

> The previous proof attempt was rejected. Revise it into a correct and complete Lean proof of the
> same theorem.

so that **Arm B and Arm C differ only by the diagnostic content**. This is asserted mechanically:
`arm_c_equals_arm_b_plus_diagnostic_after_removal: true` (removing the diagnostic block from the
rendered C prompt reproduces the rendered B prompt exactly, up to the template's trailing generation
prefix) and `arm_b_c_suffix_invariant: true`. Arm A is asserted to contain no reference to a previous
attempt (`arm_a_has_no_previous_attempt_reference: true`).

Template hashes are frozen in `v4_p001_prompt_audit.json` for the example theorem
(`algebra_10592`, statement `bd53fe26-1fb5-46c7-b827-c4a9bc470408`, screening rank 1):

    A_FRESH_RETRY    f2e2f2c2b9caec88a3955c3a47b922ff3a8afa7988fca817736c0a0e41b01811
    B_SELF_REVISION  e64361448265d559e8a0ab34cd48efdae4310b5b02ae547b89b0d226aae6ab3a
    C_VERIFIER_REPAIR ee268b3ef181d910e54092318a34a5601cf1037dd633e6ac149739583a569dde

The renderer is `v4-prompt-1`; the audit additionally records that the frozen template renders a
mid-conversation assistant turn as plain content (no extra scaffolding), which is what makes the
"failed proof" turn a faithful copy of what the model produced rather than a summary of it.

**Repair context is the extracted Lean proof, not wrapper or log text** (owner §G). The record keeps
the original response hash and the extracted-proof hash per candidate. No manual simplification,
completion or repair of the failed proof is performed at any point — it is copied byte-for-byte from
what the frozen extractor returned. When extraction yields nothing, the theorem is a `FORMAT_FAILURE`
and never enters the cohort.

## 8. Diagnostic normalization (owner §H)

`src/tinylean_rl/evaluation/v4_diagnostics.py`, version `v4-diag-1`, frozen before any formal
outcome. It does exactly two things:

1. **Deterministic first-block extraction** — the first error-severity Lean message of the verifier
   response, with its `line L, column C:` prefix, is taken as the diagnostic. No later block is
   consulted and no block is chosen by content.
2. **Infrastructure-noise removal** — absolute paths, file/line frames, request / trace / span /
   correlation ids, timestamps, `<time>` / `<id>` placeholders, HTTP URLs and server-wrapper lines
   (`# system error`, `lean server error`, `repl crashed`, tracebacks). Lines are tested *after*
   placeholder substitution, so `<time> lean server error: ...` is removed, but a line that carries
   Lean information is never allowed to match.

**Preserved, byte-for-byte:** the Lean message, the goal state, the local context, expected vs
actual types, identifier and tactic names, and line/column positions. The audit states this as
`information_preserved` and the tests assert it (goal state and `⊢` survive; a context line with
`⊢` is never stripped; normalization is idempotent and deterministic).

**No human-written explanation, no generated hint, no proposed fix** is ever added (owner §H). The
normalizer cannot generate text.

**Token budget ≤ 512** with a single frozen tail marker
`[diagnostic truncated at the frozen 512-token budget]`, and truncation is **reported**, never
silent. Measured over 3,289 observed diagnostics: raw median 220 → normalized median 144; p95 383;
**truncation rate 2.68 %** (88 of 3,289). This rate is reported before launch as §H requires.

## 9. Context feasibility (owner §I) — declared, then proved

Worst case, by construction: theorem prompt 914 (max over the pool) + failed proof 4,096 (the frozen
response budget) + diagnostic 512 + response 4,096 = **9,618 tokens ≤ 10,240** (the frozen engine
`max_model_len`). Headroom: 622 tokens at the cap, 590 measured in the smoke.

The smoke was infrastructure-only, on already-consumed material, on the formal host: a worst-case
Arm C prompt built by longest-token selection (theorem 914 tokens + a 4,096-token consumed failed
proof + a 512-token truncated diagnostic = 5,554 tokens,
`prompt_sha256 5094b2c2a9d82f9451c30fb73aa13099820f8005800f55c3e8b00bc8a68eb30f`), one generation,
no verifier, no analysis. Result: **PASS** — the prompt fits with the full response budget, the run
generated 4,096 tokens in 28.51 s (`finish_reason: length`), peak VRAM 9,208 MiB of 10,240 MiB, and
the on-disk weights matched the frozen hash exactly. Engine: vLLM 0.9.1, torch 2.7.0+cu126,
`max_model_len 10240`, `max_num_batched_tokens 10240`, `gpu_memory_utilization 0.85`,
`max_num_seqs 256`, load 118.77 s. (`peak_torch_allocated_mib: 0.0` is expected — vLLM 0.9.1 runs
generation in a worker process, so `nvidia-smi` is the reliable figure and is the one quoted.)

The response budget is identical in A, B and C and is not changed between arms (owner §I). The smoke
is non-formal and is never reported as a result.

## 10. Common random numbers (owner §J)

One shared second-stage seed per formal rank across all three arms:

    second_stage_seed(i) = 20260925 + 1_000_000 + (i − 1),   i ∈ 1..128

derived only from the frozen formal rank and the fixed V4 base seed — no outcome, source, family,
error category or diagnostic enters the formula. Frozen **before** any A/B/C generation exists:
`paired_seed_hash 06ba33b803201e8102b5537011bd242eca59b24768d69f256ac92335a6e6986d`, with all
eight invariance checks `true`:

* `seeds_recompute_identically` — the artifact is reproducible from the frozen formulas;
* `arms_share_every_seed` — A, B and C use the identical seed for each theorem;
* `diagnostic_content_does_not_move_a_seed` — changing the diagnostic changes no seed;
* `source_or_error_label_does_not_move_a_seed` — changing source or error label changes no seed;
* `first_stage_seeds_unique`, `second_stage_seeds_unique`, `first_and_second_stage_disjoint`,
  `all_seeds_below_2_pow_31`.

## 11. Outcome definition and the paired analysis (owner §K, §L)

**Success** = canonical score `1` under the same format-gated Lean verification semantics in all
three arms; there is no arm-specific verifier, prompt budget, sampling parameter or success rule. A
candidate that never reaches the verifier (no extractable proof) is a failure with score 0, exactly
as in V3-R001's frozen semantics.

**Primary comparisons**, over the theorems with complete valid A/B/C verification only:

* `Delta_repair = mean(C − A)` with one-sided exact paired McNemar (`p(C > A)`) and a paired
  theorem-level bootstrap 95 % CI;
* `Delta_diagnostic = mean(C − B)`, same two statistics;
* `B − A` is **descriptive only** and carries no gate.

The verifier has no third outcome: it returns verified or not. `INFRA_CENSORED` is **missing data,
never a failure** (owner §K) — an arm result censored by infrastructure is dropped from the paired
computation, the theorem is counted as incomplete, and the incompleteness is reported per arm and
per reason. The data guard is evaluated on complete triplets.

Implementation: `src/tinylean_rl/evaluation/v4_stats.py` — `mcnemar_exact_greater` sums the exact
binomial tail over discordant pairs (no scipy, no normal approximation);
`paired_bootstrap_ci` is percentile-based, `BOOTSTRAP_REPS = 10000`, `BOOTSTRAP_SEED = 20260925`
(frozen), resampling theorems. Deterministic and re-runnable from the frozen raw artifact alone.

## 12. The gate and the outcome taxonomy (owner §M, §N)

**Data guard (evaluated first).** If complete valid A/B/C triplets `< 103` (nominal 128 minus 20 %
infrastructure/censoring allowance), the outcome is `D_INCONCLUSIVE_BY_DATA` and the GO taxonomy is
**not applied**. The guard cannot be waived, and the cohort cannot be topped up with replacement
theorems.

**A_VERIFIER_SPECIFIC_GO** requires **all six** conditions:

| endpoint | threshold | statistic |
|---|---|---|
| `C − A` | `≥ +0.08` | mean over complete triplets |
| `C > A` | one-sided exact McNemar `p ≤ 0.05` | exact binomial tail on discordant pairs |
| `C − A` | bootstrap 95 % CI lower bound `> 0` | paired theorem-level percentile bootstrap |
| `C − B` | `≥ +0.05` | mean over complete triplets |
| `C > B` | one-sided exact McNemar `p ≤ 0.05` | exact binomial tail on discordant pairs |
| `C − B` | bootstrap 95 % CI lower bound `> 0` | paired theorem-level percentile bootstrap |

**B_SELF_REVISION_ONLY** — the repair gain over the fresh retry is established but the
verifier-specific gain is not (i.e. `C` beats `A` while the `C` vs `B` conditions fail): the
advantage may come from giving the model a second attempt with its own failure, not from the
diagnostic. **C_NO_REPAIR_GAIN** — the `C` vs `A` conditions fail: the diagnostic does not
demonstrate a repair gain. **D_INCONCLUSIVE_BY_DATA** — reachable **only** through the §M guard.

`decide_outcome` implements the guard-before-gates order and is unit-tested on the threshold edges
(including that `Δ = 0.08` exactly passes the delta condition and `p = 0.05` exactly passes the
exact test).

## 13. Power, stated honestly (owner §S step 9)

Exact enumeration plus a 2,000-repetition simulation (1,000 bootstrap replicates each) under stated
hypothetical arm success rates — `v4_p001_power.json`. The two thresholds are not equally costly:
Δ ≥ +0.08 over 128 pairs is a discordance-scale shift of about 10.2 pairs, and the bootstrap CI
lower-bound condition binds harder than McNemar (Δ̂ must exceed roughly 2 SE, not 1.645 SE).

| p(A) / p(C) | Δ | exact McNemar power | **full-gate power** |
|---|---|---|---|
| 0.10 / 0.20 | 0.10 | 0.688 | 0.602 |
| 0.10 / 0.25 | 0.15 | 0.938 | 0.878 |
| 0.15 / 0.25 | 0.10 | 0.593 | 0.501 |
| 0.15 / 0.30 | 0.15 | 0.890 | 0.823 |
| 0.20 / 0.28 | 0.08 | 0.376 | 0.307 |
| 0.20 / 0.30 | 0.10 | 0.529 | 0.438 |
| 0.20 / 0.35 | 0.15 | 0.849 | 0.733 |

Read plainly: **at the frozen threshold Δ = +0.08 the full gate has only ~0.31–0.41 power; at
Δ = +0.10 it is ~0.44–0.60; it exceeds 0.8 only near Δ = +0.15.** A null result at these thresholds
is therefore **not** evidence that the diagnostic does nothing — it is a statement that N = 128 with
this strict a gate can only detect a large effect. This is disclosed before launch and **is not an
argument for changing the frozen thresholds**; the thresholds are the owner's. The minimal
favorable-discordance table (`min_favorable_discordant_for_p005`: 0→5, 1→7, 2→9, 3→10, 4→12) makes
the McNemar condition auditable by hand. `frozen_code_cross_check` reproduces a hand-checkable
configuration (`p_A = 0.15`, `p_C = 0.25`, n = 128 → Δ = 0.09375, p = 0.0442, CI lower −0.0078, 27
favourable vs 15 adverse discordant pairs) — included precisely to show the code can **fail** the
gate on the CI condition despite passing both others.

## 14. Secondary mechanism analysis (owner §O)

Preregistered, **descriptive only, and it may not modify the GO gate**: by error category and by
source; same-error vs different-error transitions; per-theorem success transitions across arms;
edit distance and changed-token fraction between the failed proof and the repair; copied-prefix
fraction; generated token counts; wall-clock time; and the arm-to-arm transition matrix. Every
figure is reported with its denominator, and the analysis is re-runnable from the frozen raw
artifact. No AUPRC endpoint exists in this study (owner §L).

## 15. Verifier infrastructure (owner §P)

Separate V4 verifier infrastructure that reuses the C′ lessons from V3-R001 without changing them:
a dedicated container, `MAX_REPLS = 1`, sequential verification, server-side timeout (120 s + 60 s
client slack), mandatory `restart → /health → cold nonformal canary` between a non-conclusive
candidate and the next one, and fail-closed behaviour on an identity/health/canary failure. This is
the frozen C′ lifecycle, unchanged.

**The safety recovery ceiling is pre-registered here, before formal generation, from the observed
V3 density** (`experiments/manifests/v4/v4_p001_verifier_plan.json`, producer
`scripts/v4_p001_verifier_plan.py`, read-only over the V3 attempt-2 archive, which lives on fly90
outside git; the derivation is frozen in the committed artifact so it does not need re-running):

| quantity | value |
|---|---|
| V3-R001 attempt-2 executions | 3 (two aborted, one complete; deduplicated by `run_id`) |
| candidates verified | 1,024 |
| recoveries attempted / succeeded | 47 / 45 |
| pooled density | 0.0459 per candidate |
| worst execution density | 0.1328 per candidate (`20260925T062406Z`) |
| Poisson 99.9 % upper bound at V4 scale (λ = 136) | **173** |
| **frozen ceiling** | **192 per run** (1.11× that bound) |
| frozen per-theorem cap | **3** (one per arm candidate) |
| recovery wall-clock | median 7.55 s, max 17.34 s |

Two facts drive the number, and both are measured rather than assumed. First, **the old ceiling of
16 was itself the cause of both aborts** — the abort reasons are literally
`recovery budget exhausted: 16 of 16 restarts already used in this launch` — so a low per-session
ceiling is exactly the failure mode §P tells us to avoid. Second, recoveries are **not** evenly
spread: one theorem (rank 67) took 7 of the 17 recoveries in the worst execution, because V3
verified 8 samples of the same theorem and each re-poisoned the singleton REPL. V4 verifies one
candidate per theorem per arm, so the per-theorem exposure is structurally ≤ 3; a theorem that
exceeds the per-theorem cap has its remaining arm candidates marked `INFRA_CENSORED` (missing, never
a failure) and the run continues, so a single pathological theorem cannot consume the run budget.

Recovery remains **infrastructure, not a candidate retry** (owner §6 in C′): it never adds an
attempt to a candidate's frozen budget, and an infrastructure error remains missing/censored.
No throughput claim is made from this study. Infrastructure errors remain **missing/censored**,
never failures.

## 16. Compute plan and cost (owner §Q)

| item | value |
|---|---|
| screening candidates (worst case) | 640 first attempts (Tier 1 block 453, then Tier 2 in the frozen order) |
| expected screening at the planning floor | 640; at the central estimate (0.3008) ≈ 426 |
| cohort (nominal) | 128 theorems |
| second-stage candidates | **384** (3 arms × 128) |
| worst-case formal generations | 640 + 384 = **1,024** |
| expected verifier calls | ≤ 1,024 (only candidates with an extractable proof reach the verifier) |
| measured per-generation cost | single capped request (smoke): 28.51 s for 4,096 tokens; V3 batched (8 sequences/call): 6.75 s per candidate |
| estimated GPU time | **≈1.9–8.1 GPU-h** for 1,024 generations — the two endpoints are measured (`1,024 × 6.75 s` and `1,024 × 28.51 s`), plus ≤ 2 min of engine load per session; a small fraction of a day on one RTX 3080 |
| peak memory (measured) | 9,208 MiB of 10,240 MiB |
| training cost | **zero** — no optimizer, no gradient, no checkpoint |

V4-P001 remains a **low-cost mechanism probe** by construction: one host, one model, n = 1, no
training, and a screening budget that is capped before any outcome exists.

## 17. Prohibitions honored (owner §A, §B, §R, §S)

* No V1/V2/V3 artifact was modified; all inputs are hashed as received in the pool artifact.
* No final-holdout / sealed-reserve contact: all eight disjointness checks are `0`.
* No family selected or ordered by historical repairability, error outcome, difficulty or source:
  the representative rule reads `sha256(statement_id)` and prompt length only, and the order is a
  uniform hash permutation.
* No generation during preregistration. The pool build, historical audit, prompt audit, seed freeze,
  verifier plan, power analysis and context smoke are all read-only over consumed material; the
  single §I smoke is infrastructure-only and non-formal.
* No RL, SFT, checkpoint update, controller training, or final-holdout use.
* No lowering of N and no cohort top-up once second-stage outcomes exist (owner §C, §K).
* The taxonomy, the normalizer, the renderer, the seeds, the pool and the order were all frozen
  before any formal V4 generation exists — which is the state of the repository at this commit.

## 18. Downstream branches — recorded, not implemented (owner §R)

| outcome | what the owner may do next |
|---|---|
| `A_VERIFIER_SPECIFIC_GO` | owner may design **V4-R001** (verifier-integrated training). The conceptual target — verifier-feedback tokens excluded from the policy loss — is **not** part of P001 and is not implemented here. |
| `B_SELF_REVISION_ONLY` | a separate self-revision direction, only after owner review; not authorised by this document. |
| `C_NO_REPAIR_GAIN` | close the verifier-feedback line; no follow-up experiment is implied. |
| `D_INCONCLUSIVE_BY_DATA` | owner review; no rescue by lowering N, changing the cohort, or re-screening. |

## 19. Known risks, stated plainly

1. **Power at the frozen threshold is modest** (§13). A null is weak evidence of absence.
2. **Format failures dominate theta0 output** (44–63 %). The primary-eligible rate depends on the
   model producing extractable Lean code; the 0.20 floor is deliberately conservative and the
   measured estimates sit 45–56 % above it, but the floor is what the 640-screen budget is built on.
3. **Tier 2 is reachable** if Tier 1 yields fewer than 128 primary failures (its measured ceiling is
   ≈142 at 0.3125, ≈91 at 0.20). Tier 2 material is *assigned a role but never used*; the argument
   for its admissibility is recorded in §3 and is the owner's §B Tier-2 clause, not an assumption.
4. **Very long diagnostics** (max 18,790 tokens raw; 2.68 % over budget) are truncated at the frozen
   512-token marker. Truncation is reported and never silent; a truncated diagnostic is still the
   *exact* Lean diagnostic, just clipped at the tail.
5. **Bootstrap-CI binding**: the CI condition, not McNemar, is the binding constraint at the
   threshold — deliberately, since the owner froze it.

## 20. Launch checklist (all must hold before the first formal generation)

1. This document, the registry, and every artifact it names are committed and pushed, and fly122 is
   synced to the same revision.
2. The owner has given an explicit `FORMAL_V4_P001_AUTHORIZED: YES` after this commit reaches
   `origin` — this document itself ends `NO`.
3. The V4 verifier infrastructure is deployed and the §15 recovery ceiling (192 per run, 3 per
   theorem) is implemented in the runner and recorded in the run summary before the first formal
   verification.
4. The frozen pool, order and seed artifacts are re-hashed on fly122 and match this document.
5. The model weights on the formal host match the frozen hash (already verified once in the §I
   smoke).
6. No other V4 experiment is in flight on the formal host.
