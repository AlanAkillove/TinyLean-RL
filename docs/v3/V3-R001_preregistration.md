# V3-R001 — Prospective Family-Clean Validation of the Informative-Group Controller

**Preregistration.** Written and committed **before any R001 outcome exists**: no rollout, no
generation, no verifier call, no label. Status of every object below is `FROZEN`.

| | |
|---|---|
| Authority | owner directive 2026-09-24 §3–§15 and §17 step 6; owner decisions 2026-09-24 §1–§16 (the review response to the pre-reg report) |
| Question source | owner §3, verbatim: *"Can a controller trained entirely on historical RLVR rollouts prospectively identify reward-informative groups on previously unseen theorem families?"* |
| Machine-readable gate | `experiments/manifests/v3/V3-R001_gate.json` (producer `scripts/v3_r001_gate.py`) |
| Formal sample | `experiments/manifests/v3/v3_r001_formal_sample.json` (producer `scripts/v3_r001_sample_freeze.py`) |
| Sealed reserve | `experiments/manifests/v3/v3_final_holdout_reserve.json` |
| Controller predictions | `experiments/manifests/v3/V3-R001_predictions.json`, sha256 `070e933d8e45cbec416099a563dd2298e96f9ab1ae25b189b46c43de98de3602` |
| Pool | `experiments/manifests/v3/V3-R001_family_clean_pool.json` (`docs/v3/V3-R001_family_clean_pool.md`) |
| Power | `experiments/manifests/v3/V3-R001_power.json` (`docs/v3/V3-R001_power.md`) |
| Host | fly122, RTX 3080 (owner §16; fly90 is archive/coordination only) |
| Launch state | **NOT LAUNCHED.** Rollout requires the owner's explicit go after this commit reaches `origin`. |

---

## 1. What is being tested, and what is not

Two questions are **co-primary**, and the reason is empirical rather than stylistic.

**Q1 (pooled).** Does the frozen D001 controller, scored on every candidate before any outcome,
enrich reward-informative GRPO groups in the top 20% of previously unseen theorem families?

**Q2 (within-stratum).** Or is the pooled effect mainly *source routing* — does the controller
retain predictive enrichment **within** the synthetic stratum, where the effect would have to come
from theorem content rather than from which corpus a statement came from?

Before a single outcome exists, the pre-outcome diagnostic already shows the confound is live: the
top-20% block of the 251-candidate union is **50/50 synthetic under both arms**, and mean q_B2 is
0.44743 on synthetic against 0.03230 autoformalizer and 0.03122 human. A pooled pass alone therefore
cannot license a "beyond source" claim. Owner decision 6 promotes Q2 from robustness appendix to
co-primary; §9 resolves the two jointly.

R001 is **not**: an RL training run (no optimizer, no gradient, no checkpoint selection — owner §3,
§15); a re-opening of D001's gate (owner §10); a test of theorem-proving ability, compute savings,
or "the controller improves RL" (those claims remain NOT permitted); a search for a good λ (owner
§12); or a step toward V3-R002, which the owner decides about only if the outcome is A.

## 2. Population and contamination definition

Owner decision 1 freezes the contamination definition as **`consumed_only`**: a family component is
eligible iff no member statement was *empirically used* — generated, verified, rewarded, trained on,
or used for selection — by V1 seed1/2/3, E023, A001, B001/B002/B003.

The governance change is recorded in V3 provenance, and the owner's statement is carried verbatim:

> V2 Track C reserved families were released by owner for V3 because Track C was never executed and
> no outcomes from those families were observed. This is a governance change in reservation status,
> not a reuse of previously evaluated data.

No V2 frozen manifest, result, or history was modified (§5 of the directive; the release lives in
`experiments/manifests/v3/registry.yaml` as a V3 `amendments` entry). The pool artifact's other
readings are retained unchanged as evidence of what the choice was worth: under the strict literal
reading the same construction yields 56 components, and honoring every reservation yields 0.

The sampling unit follows the deployment-pool audit (`docs/v3/V3-R001_deployment_pool_audit.md`):
**one unique, stable `statement_id` = one canonical prompt**, deduplicated from the Promptset's
24,418 parquet rows / 7,620 unique statements, and restricted by the trainer's own filter
`len(apply_chat_template(prompt, add_generation_prompt=True)) ≤ 1024` tokens so that every candidate
is actually drawable (24,246 rows / 7,613 statements survive). Family components are the cluster;
each contributes exactly one deterministic representative (lexicographically smallest eligible
statement_id), so **one theorem per family** and the analyzed units are independent.

## 3. Allocation of the 221 clean components

Owner decision 2 partitions the `consumed_only` pool, and the partition is an exact, verified cover
(`scripts/v3_r001_sample_freeze.py` exits if it is not):

| set | n components | source mix | hash of component ids |
|---|---|---|---|
| **V3-R001 formal sample** | 128 | 58 auto / 25 human / 45 synthetic | `840d1dbb7e277c26d40521c8287b5585ca1798535914cd14e31a01f6bce1e8da` |
| **V3-FINAL-HOLDOUT reserve** | 93 | 43 auto / 17 human / 33 synthetic | `d7430ac7fc38dce67d47e356b6eba441814a2698d51e2ce26754a8994f457c6d` |
| union = pool | 221 | 101 / 42 / 78 | pool hash `72b5bd49…64fd5a` |

Component-disjoint, statement-disjoint, union equal to the pool: all three are asserted in code, and
`scripts/v3_r001_gate.py` independently refuses to emit a gate over a sample that touches a sealed
component.

**Reserve governance (owner decision 13).** `SEALED`, purpose *future final family-clean capability
evaluation*. Forbidden: R001 rollout, controller fitting, sampler tuning, power tuning, R001
analysis including post-hoc stratification, future hyperparameter choice, and any change of reserve
membership based on q values. Membership is the pool **minus** a uniform hash-ordered draw, so it
cannot have been picked by q or by difficulty. If V3 stops at R001, the reserve stays sealed and
available for a final confirmatory evaluation.

A disclosed limitation, stated rather than left to be discovered: q values for the reserve
components exist in the already-committed predictions artifact, because owner §6 required scoring
every candidate theorem before any outcome. **The seal binds use, not existence.**

## 4. The formal sample — and the proof it is the power-calculation sample

Owner decision 3 removes the choice: no selection among the ten frozen candidate samples is
permitted; the formal sample **is** the exact `consumed_only | N=128` draw that §14's power
calculation used. This is checked, not asserted — `v3_r001_formal_sample.json#identity_with_the_power_sample`
recomputes the draw through the same frozen `draw_components`/`sha` used at freeze time and compares
three hashes to the committed predictions artifact:

| object | recomputed = committed |
|---|---|
| sample | `0ac7134fdbe03a7d698961e7fb49b288be29d5c62892fc2c56c858a4cbf70451` |
| q_B2 vector (in sample order) | `d83f0bbae694f27c0769c6d15ce40f8d36850531bb1c347ab034201cef8e15bb` |
| q_B1 vector (in sample order) | `3d1fd616ac69a0b52f397f10d0ce060023d20e339f19934a39c8130203349c59` |

Draw rule: order components by `SHA256("20260924|<component_id>")`, take the first 128 — a uniform
permutation with no library-RNG or `PYTHONHASHSEED` dependence (pinned by
`tests/test_v3_r001_extract_prescore.py`). Seed = the directive date, fixed in code.

Frozen properties of the sample: 128 theorems over 128 distinct family components, none with a
historical label, all with a finite q in [0,1]; prompt tokens min 123 / median 273 / max 1010, and
1010 + 4096 = 5106 ≤ `max_model_len` 5120 so every candidate fits; q_B2 min 0.002 / median 0.04364 /
p90 0.60175 / max 0.84014 / mean 0.16712.

**The top-20% block is frozen now:** size `m = round(0.20 × 128) = 26`, ordered by q_B2 descending
with ties broken by statement_id ascending, sha256
`d5791fd52b80e899b563f2048b3d22d2ec560dc4cbc554e928774a5441fcd629`. It is a function of committed
objects only, so no outcome can move a theorem in or out.

The other nine candidate samples are recorded in the gate artifact as
**`DESIGN-ONLY / NEVER ANALYZE`**, with their hashes. Whatever the result, switching sample is not
available.

**File-level pins.** `v3_r001_formal_sample.json` = `733a699e9eeb4f1d8b130c1a87c3455a06748b66aa3909f90995f432877c6beb`,
`v3_final_holdout_reserve.json` = `b60be381b5f64a0394691344b71cd704ed337716f0eea81ea265e05af7761568`,
`V3-R001_gate.json` = `fcb700b508460f5fdaac318f7ce8e9bd56266f7aaaf07c7452d00f08d9c7ac93`. Both moved hashes moved exactly once, before any outcome, in the
2026-09-25 pre-outcome amendments (§18): the sample gained the explicit `formal_sample_rank` and the
gate gained `frozen_block_semantics` — neither changed a frozen number (sample hash `0ac7134f…`, the
q-vector hashes and the block hash `d5791fd5…` are byte-identical to the preregistration commit).
None of the
three carries a wall-clock, host or git field — their bytes are a pure function of the committed pool,
predictions and power artifacts — so re-running either script anywhere reproduces these hashes exactly.
An earlier revision embedded `created_at_utc`/`git_revision`/`host` and therefore could not be
re-hash-verified by a reader; that was corrected before any outcome existed, and it changed only how
provenance is written: no sample, block, gate boundary or design hash above or below moved. Freeze time
and host are recorded in `experiments/manifests/v3/registry.yaml` and in the commit itself.

## 5. The controller: frozen, scored, and proven to be D001's

`q_i = P(informative | theorem_i, step_norm = 0)` from V3-D001's frozen protocol — θ0 block-18
last-token representation (1024-d) plus the step_norm slot, standardized, L2 logistic (Newton/IRLS),
C selected by inner family-grouped CV on AUPRC over D001's grid. **No refit. No new layer, no MLP,
no mean pooling** (owner §6, §15).

Deployment heads (from the committed predictions artifact): C = 0.003 in both arms; inner AUPRC
0.61047 (B2) / 0.50963 (B1); `w_sha256` B2 `b96c68506755fbf298e95885d68063810ca008c238c1885ac54573feda4767d6`,
B1 `872d1b8524d5d538c3b0aa2d4d637ceb1b3e721295aa218cbfd1af71ba398d8d`; standardization sha
`f2d5bff1b312ea6db524df000c4f54fab01cc1370f64d400a16d924257630f09`.

The guard that makes "frozen" more than a word: the script re-derives D001's **per-fold** C and inner
AUPRC for both arms and requires all 5/5 folds to match the frozen artifact before scoring anything —
`frozen_protocol_replication` is 5/5 with zero mismatched folds, so q cannot come from a silently
different model than the one that earned the D001 GO. In-sample AUPRC on the 686 fitting groups is
0.78446 (B2) / 0.48891 (B1); D001's OOF references 0.5728 / 0.42673 are reported beside them and the
in-sample figures are explicitly **not** the prospective estimate.

`step_norm = 0.0` is the pre-update / initial-policy condition (owner §6). The fit data's step_norm
support is 0.0167–1.00, so 0.0 sits just below it; both heads are linear in the features and 0.0 is
the *same* value for every candidate, so it shifts all logits by one constant and cannot reorder
them — which is the only thing §9's statistic depends on.

Coverage: 251/251 candidates scored, `n_missing_q = 0`, `has_historical_label` false for every one.
Representation provenance: the pre-registered layers [9, 18, 27] from θ0 (weights sha
`34e6e630f564…`), extracted on the same RTX 3080 as the reference pass with 8/8 bit-exact recipe
checks and 612/612 historical prompts reconstructed byte-identically; raw reps stay gitignored, only
hashes and provenance are committed. Cross-engine check: bf16 3090-vs-3080 drift moves q by at most
0.0294 but leaves the top-50 ranking and every sample hash identical.

## 6. Rollout protocol (owner decision 5)

Exactly one pass, on fly122's RTX 3080, over the 128 frozen theorems.

| parameter | value |
|---|---|
| model | Kimina-Distill-0.6B **θ0** (`models/weights/kimina_distill_0_6b`, weights sha `34e6e630f564…`) — no trained checkpoint |
| engine | vLLM 0.9.1 (`LLM.generate`, canonical single path) |
| samples per theorem | **n = 8** |
| temperature / top_p | 1.0 / 1.0 |
| max response tokens | 4096 |
| max_model_len | 5120 |
| gpu_memory_utilization | 0.85 |
| chunk_theorems | 16 |
| seed schedule | `seed_base + (formal_sample_rank − 1) * 8 + sample_index`, `seed_base = 20260924` (canonical n=8 schedule). `formal_sample_rank` is the frozen draw-order rank 1..128 written in the sample manifest — independent of `q_B2`, `q_B1`, `source` and any outcome (amendment B, §18) |
| prompt | the canonical chat-template rendering, identical to the string used for representation extraction |
| verifier | canonical Lean 4 `#check`/`by example` semantics as in D001/E023; server timeout 120 s, client timeout 180 s, first verify 600 s, batch 4, canary timeout 60 s, through the **B0 reliability policy** (`VerificationSession`), sequential rather than E024's 4-worker path (amendment C, §18) — health check and canary before group 1, infra taxonomy, fail-close, unresolved infra → censored/missing |

Nothing about the sample may change after the result: no re-draw, no re-ranking, no re-fit, no new
N, no λ selected on the outcome (owner §7, §12).

## 7. The label, and censoring

Primary label per theorem *i*:

```
y_i = 1  iff  0 < Σ_{j=1..8} score_ij < 8
```

`score` is the primary label; it is never swapped for `acc`. A group containing an unresolved
**infrastructure** error (verifier crash, timeout that is not a genuine proof failure) stays
**missing / excluded** under D001's policy — it is not converted to all-fail (owner decision 5).
V1's measured infra-censoring rate was 0.04722 (34/720), so the analyzed set is expected near 122 of
128. Genuine proof failure, including a truncated-without-proof response, is a real outcome and
counts as fail; only infrastructure failure is censoring.

## 8. Primary analyses

On the analyzed sample (all quantities on the same theorems, both arms):

1. `AUPRC(q_B2, y)` and `AUPRC(q_B1, y)`; `AUPRC_prevalence` = π̂, the no-information reference
   (a constant score), reported beside them.
2. Overall prospective **IGR** = (Σ y_i)/N_analyzed; the informative-group rate of the sample itself.
3. **Top-10 / top-20 / top-30% B2 IGR**, and `enrichment = selected IGR / overall IGR` at each
   fraction. Only **top-20%** is confirmatory; 10% and 30% are descriptive, so no multiplicity is
   smuggled in by looking at three blocks.
4. All intervals by **family-component bootstrap**: resample components with replacement, 10,000
   reps, seed 20260924, 2.5/97.5 percentile. With one theorem per component the component and
   theorem resamples coincide, which is stated rather than left to be discovered.
5. The exact conditional test of §9 — this, not the bootstrap, is the confirmatory inference.

## 9. The gate

The gate is the frozen executable in `scripts/v3_r001_gate.py` (`pooled_gate()`), whose constants come
from owner decisions 4, 8 and 9. Inputs: `N = |A|` (the analyzed theorems), `K` (positives in them),
`m = |A ∩ B|` and `x` (positives among those `m`), where `B` is the **preregistered frozen** top-20%
block. Membership is never recomputed after an outcome (amendment A, §18; machine-readable form
`V3-R001_gate.json#frozen_block_semantics`).

### P1 — exact enrichment test (confirmatory)

Take the **preregistered frozen block** `B` — never re-taken, never topped up from labels, never
repaired for a censored theorem — and let `A` be the analyzed theorems, `B_analyzed = A ∩ B`,
`N = |A|`, `m = |B_analyzed|`, `K` the positives in `A` and `x` the positives in `B_analyzed`. Under
H0 (the ranking carries no information), x given K is
**Hypergeometric(N, K, m)**, so the one-sided exact p-value is `P(X ≥ x | N, K, m)`.

```
P1 PASS  <=>  exact p <= 0.05
```

`scripts/v3_r001_power.critical_values` and `scripts/v3_r001_gate.exact_p` agree by construction: p
crosses 0.05 exactly at the boundary x. The test is *conditional on the observed k*, so the rule is
fully determined by frozen code and no human picks the boundary after the fact.

**Attainable level, reported rather than assumed.** A discrete exact test cannot realize 0.05; the
frozen rule's true size at the design anchor (N=128, m=26, π=0.1391) is **0.0274** unconditionally
and **0.0419** conditionally at the anchor k=18. The nominal α is 0.05 and the design's attainable α
is 0.0274 — the gate is *conservative*, and the analysis must report the attainable value, not the
nominal one.

The frozen boundary at N=128, m=26 (full table in `V3-R001_gate.json`; ranges of k that share a
boundary are collapsed):

| observed k | reject if x ≥ | enrichment at the boundary | conditional level |
|---|---|---|---|
| 3 | 3 | 4.923 | 0.00762 |
| 4 | 3 | 3.692 | 0.02626 |
| 5 | 4 | 3.938 | 0.00601 |
| 6 | 4 | 3.282 | 0.01548 |
| 7 | 4 | 2.813 | 0.031 |
| 8–11 | 5 | 3.077 → 2.238 | 0.00878 → 0.04555 |
| 12–14 | 6 | 2.462 → 2.11 | 0.01568 → 0.03739 |
| **15–18** | **7** | 2.297 → **1.915** | 0.01345 → 0.04191 |
| 19–22 | 8 | 2.073 → 1.79 | 0.01635 → 0.04374 |
| 23–26 | 9 | 1.926 → 1.704 | 0.01797 → 0.04368 |
| 27–30 | 10 | 1.823 → 1.641 | 0.01855 → 0.04229 |
| 31–34 | 11 | 1.747 → 1.593 | 0.01833 → 0.03998 |
| 35–39 | 12 | 1.688 → 1.515 | 0.01751 → 0.04625 |
| 40 | 13 | 1.6 | 0.02096 |

At the design anchor k = 18 the rule reads **reject iff x ≥ 7**, i.e. **enrichment ≥ 1.915**. The
boundary is monotone in k over the whole feasible range; the only k without a rejection region are
the degenerate ends (k = 1, and k > 115 where the block is saturated), neither of which is plausible.

### P2 — effect size (confirmatory)

```
P2 PASS  <=>  top-20% enrichment >= 1.5  AND  its component-bootstrap 95% CI lower bound > 1.0
```

**Pooled gate = P1 AND P2** (owner decision 8). Requiring both is deliberate: P1 alone can be
tripped by a large block count on a small base rate, P2 alone is an interval without a controlled
type-I error.

### S1 / S2 — the within-synthetic co-primary (owner decision 9)

Restrict to the frozen synthetic stratum of the formal sample: **n = 45**, block
**m = round(0.20 × 45) = 9**, ranked **within** the stratum by the frozen q_B2 so the source label
itself never enters the ranking. Design anchor: π = 0.3444 (V1's synthetic informative rate),
expected positives 15.5, k = 15.

```
S1 PASS  <=>  exact p <= 0.05   (at n=45, m=9, k=15 this is x >= 6, enrichment >= 2.00)
S2 PASS  <=>  within-synthetic top-20% enrichment >= 1.5 with CI lower bound > 1.0
```

Attainable level for S1: **0.0229** unconditional, 0.0263 conditional at the anchor. MDE at 80%
power: **2.1617**. If the synthetic stratum turns out to have too few positives for the preregistered
test to be identifiable, that resolves as **INCONCLUSIVE-BY-DATA for this co-primary — never a FAIL,
and never a silent fallback to the pooled result.**

### Power, stated as the design's honest envelope

| quantity | pooled (N=128) | within-synthetic (n=45) |
|---|---|---|
| block size m | 26 | 9 |
| prevalence anchor π | 0.1391 | 0.3444 |
| expected positives | 17.8 | 15.5 |
| boundary at the anchor | x ≥ 7 (E ≥ 1.915) | x ≥ 6 (E ≥ 2.00) |
| attainable α (unconditional) | 0.0274 | 0.0229 |
| MDE at 80% power | 2.3324 | 2.1617 |

π is a **design anchor, not a measurement**: V1's per-source rates applied to this sample's own
source mix. Its known bias is that V1's rates were measured on theorems it chose to re-draw, so they
may not transfer — which is exactly what R001 tests. D001's measured top-20% B2 enrichment was 3.467
(family-held-out OOF), comfortably above both MDEs.

### The four outcomes (owner decision 7)

| outcome | condition | meaning and consequence |
|---|---|---|
| **A — GO-SEMANTIC** | pooled PASS **and** within-synthetic PASS | prospective enrichment survives stratification; the controller is not merely identifying its source. The only outcome under which the owner may consider **V3-R002 decision-guided RL**. |
| **B — SOURCE-DRIVEN-ONLY** | pooled PASS, within-synthetic FAIL | enrichment exists but cannot be separated from source-level routing. A valid scientific result; **no expensive RL intervention**. |
| **C — NO-GO** | pooled FAIL | the controller did not prospectively enrich on unseen families; the Jev-inspired RL line stops. |
| **D — INCONCLUSIVE-BY-DATA** | N_analyzed < 0.8 × 128, **or** analyzed positives < 5, **or** the synthetic co-primary is unidentifiable | the sample could not carry the test. **Not a NO-GO.** No re-drawing, no sample switching, no redefining the label rule to rescue it. |

## 10. What is explicitly NOT a gate

**`AUPRC_B2 > AUPRC_B1` significance is NOT A GATE** (owner decision 10). D001's full-procedure
ΔAUPRC was +0.111 with 95% CI [−0.024, +0.239] on 433 components (replicated on fly122 at 500 reps,
commit `2f0480d`); a component-level interval scales as 1/√n, so at whole-pool sizes R001's half-width
would be ≈0.18–0.36. Requiring significance would manufacture a NO-GO regardless of whether the
controller works. It is decided by power, not forced.

B1 stays what owner §10 asks for: the identical sample, zero extra GPU, reported as
`AUPRC_B1`, `ΔAUPRC = AUPRC_B2 − AUPRC_B1` with its bootstrap CI, and top-20 IGR for both arms.

**Interpretation rule (owner decision 11).** If B1 ≈ B2 prospectively, that is **not** read as a
controller failure. The reading turns on the pooled-vs-within-synthetic contrast and on whether the
semantic representation supplies *usable* ranking structure beyond a coarse source/difficulty proxy.
Both pooled and within-synthetic B1/B2 pairs are reported side by side.

## 11. Source-concentration audit (preregistered, owner §11 + decision 6)

Reported for the formal sample: source × prevalence; within-synthetic AUPRC and top-20 enrichment
for both arms (co-primary, §9); within-autoformalizer and within-human AUPRC and enrichment
**descriptively** — at n=58 (π≈0.0304, ≈2 expected positives) and n=25 (π≈0.0215, ≈1 expected) these
strata cannot carry a test: the maximum coherent enrichment gives power 0.5445 and 0.0927
respectively, so an empty result there is **not** a null and must not be written as one.
Source-stratified B2 ranking is secondary robustness. The question the owner asked — *"is B2 just
learning synthetic > other source?"* — is answered by S1/S2 plus these strata, and the pre-outcome
evidence in §1 is reported alongside so the reader can see the confound was known before the labels.

## 12. Mixture-policy diagnostics (owner decision 12, §13)

No two-arm sampling happens in R001. For λ ∈ {0.5, 0.8}, `P_λ(i) = (1−λ)/N + λ·q_i/Σq` is evaluated
as an offline estimand diagnostic only, reporting **expected IGR, ESS, entropy, max probability, min
probability**. Positivity is asserted in code — `P_λ(i) > 0` for every eligible theorem — because
that is the property the old `epsilon = 0` sampler destroyed; **ε = 0 is permanently abandoned**
(owner §13). Frozen values on this sample's q:

| λ | min P | max P | ESS | entropy (bits) |
|---|---|---|---|---|
| 0.5 | 0.003953 | 0.023543 | 85.88 / 128 | 6.7096 |
| 0.8 | 0.001637 | 0.032982 | 56.75 / 128 | 6.2787 |

Uniform ESS is 128 by definition, so the table *is* the guidance-cost statement. λ is never selected
on R001 results; if R002 ever happens, λ is preregistered separately.

## 13. Compute plan and host policy

E023's measured θ0 pass on fly122 is the anchor: 128 theorems × n=4 in 1,114.5 s generation +
1,349.8 s verification = 2.177 s + 2.636 s per candidate, 1,530.7 useful tok/s, mean response 3,332
tokens, 43.75% truncated at 4096, **peak 9,202 MiB sampled on the same 10 GB card**.

Per n=8 group ≈ 8 × 4.813 s ≈ **38.5 s** ⇒ **N=128 ≈ 1.4 h**, ≈3.41 M generated tokens. Honest band
**1–3.5 GPU-hours**, widened because (i) n=4→n=8 scaling is an assumption — no fly122 n=8 rollout has
ever been measured, (ii) verification has historically ranged 1.0–7.4 s per candidate and can
dominate, and (iii) vLLM preallocates by `gpu_memory_utilization`, so the binding risk on a 10 GB
card is KV-cache pressure causing preemption/recompute, which lengthens *time*, not feasibility.
Weights ≈1.5 GB plus ≈4.6 GB of KV for 8 concurrent 5.1k-token sequences fit inside 10 GB with no
offloading experiment and **no 10 GB training smoke** (owner decision 15: NOT AUTHORIZED, not run).

Host: fly122 RTX 3080 only (owner §16, decision 16). Not moved back to fly90's 3090 because the card
is tighter — and any vLLM 0.9.1 outcome would not be bit-reproducible across engines anyway
(`docs/v2/b1_budget_semantics.md` records 0/4 internal determinism and 0/4 prefix-cache agreement;
vLLM 0.9.1 on fly122 is the registered engine). At temperature 1.0 the outcome is stochastic **by
design**; what is frozen is the sample, the ranking, and the label rule, not the tokens.

If anything in the run cannot stay within this recipe on fly122: **STOP → report the owner → the
owner decides** rent-24GB / terminate. No silent fallback.

Any pre-launch timing probe must use statements **outside** the 251-statement extraction union, so a
dry run cannot manufacture R001 labels.

## 14. Analysis surface

| artifact | state |
|---|---|
| `scripts/v3_r001_sample_freeze.py` → `v3_r001_formal_sample.json`, `v3_final_holdout_reserve.json` | committed, runs CPU-only |
| `scripts/v3_r001_gate.py` → `V3-R001_gate.json` | committed, runs CPU-only |
| `scripts/v3_r001_rollout.py` | committed: reads the formal sample manifest, refuses any component in the sealed reserve, re-verifies every prompt against the frozen rendering before any generation, seeds each candidate from its frozen draw-order rank, verifies through the B0 policy, appends whole groups to `runs/v3_r001/rollout/` (gitignored). `--dry-run` performs every check and generates nothing; a formal run additionally requires `--i-have-owner-launch-authorization` |
| `scripts/v3_r001_analyze.py` | committed: reads the frozen gate artifact and the raw rollout, computes §8's quantities, calls `pooled_gate()`, emits `V3-R001_results.json`. It must refuse to analyze anything but `consumed_only|N=128` |
| `scripts/v3_r001_spec.py` | committed: the shared frozen contract both executables import (hash-pinned loading, the raw schema, the group-finalization rule, the seed schedule) |
| `tests/test_v3_r001_fixtures.py` | committed: fixtures A–G — A/B/C/D outcomes, infra censoring, block-hash tamper, manifest tamper — run against the real frozen design on synthetic labels |
| `tests/test_v3_r001_hypergeom.py` | committed: the exactness sweep over the preregistered N/K/m/x range, against exact rational arithmetic |
| `tests/test_v3_r001_gate_freeze.py` | committed: identity with the power sample, partition/disjointness, seal, boundary-vs-p equivalence, taxonomy resolution, positivity invariant, and that the freeze scripts hash identically |

## 15. Prohibitions honored (owner §15)

No RL optimizer training; no 10 GB training smoke; no offload experiment; no MLP; no new layers; no
mean pooling; no TinyJev/Kev; no new controller task; no Track B; no candidate ranking task; no repair
policy. V2 frozen artifacts untouched; the Track C release is V3 provenance only. Raw rollout data is
never committed — hashes and provenance only. Nothing in this document was written after seeing an
R001 outcome, because none exists.

## 16. Known risks, stated plainly

1. **Prevalence may not transfer.** The whole design anchor π = 0.1391 rests on V1's per-source rates.
   If the clean pool's true prevalence is materially lower, positives fall and the test can resolve D
   rather than A/B/C. That is the honest cost of a prospective test, and D is a permitted outcome.
2. **The pool is small, and that is structural.** 221 usable families is what remains after every
   earlier experiment's claim is honored. A 56-family strict reading would have produced a descriptive
   study with ~1 expected positive in the block, not a test.
3. **Source confounding is real and pre-registered, not discovered later.** Both arms' top blocks are
   50/50 synthetic before any outcome. B could be the true result.
4. **n=8 timing is extrapolated** from an n=4 measurement on the same card.
5. **The reserve's q values exist.** The seal is on use; the commitment is mechanical (guards in the
   gate and analysis scripts), not merely declarative.
6. **bf16 non-reproducibility** means exact token sequences are not repeatable across engines; the
   frozen objects are the sample, the ranking and the label rule.

## 17. Launch checklist (all must hold before the first generation)

- [x] contamination reading `consumed_only` recorded with the owner's Track C release statement
- [x] 221 → 128 + 93 partition verified disjoint and covering, hashed
- [x] formal sample proven identical to the §14 power sample by three hashes
- [x] gate frozen: boundaries, attainable α, P1/P2, S1/S2, taxonomy A/B/C/D
- [x] every candidate scored with a q, no refit, D001 protocol replication 5/5
- [x] nine alternative samples marked DESIGN-ONLY / NEVER ANALYZE
- [x] reserve marked SEALED
- [x] three pre-outcome amendments — A frozen block semantics, B draw-order seed rank, C B0 verifier
      execution — implemented, tested and hash-pinned before any outcome (§18)
- [x] **preregistration commit present on `origin/v3-jev-rl-controller`** — checked after the push of
      the commit that contains this file, verified with `git ls-remote` (remote == local HEAD)
- [ ] **owner's explicit launch authorization** ← currently withheld

## 18. Pre-outcome amendments (owner approved 2026-09-25)

Marker for all three: **PRE-OUTCOME CLARIFICATION — no prospective labels existed.** No R001 candidate
had been generated, no formal theorem had been sent to the verifier and no label existed when these
were made. They change wording, the seed-rank rule and the verifier execution path — never the sample,
the block, the boundary, the label rule or any threshold.

**Amendment A — frozen block semantics (approved).** The pooled and the within-synthetic top-20%
blocks are the **preregistered frozen memberships**. The earlier phrasing `m = round(0.20 × N_analyzed)`
is deprecated, because it could be misread as re-taking the top 20% once labels exist. Formal
definition of both cells:

```
A = the analyzable theorems          B = the preregistered frozen top-20% block
B_analyzed = A ∩ B

N = |A|     m = |B_analyzed|     K = positives in A     x = positives in B_analyzed
X ~ Hypergeometric(N, K, m)      p = P(X >= x)
```

Within the synthetic stratum the same rule reads `A_syn`, `B_syn`, `m_syn = |A_syn ∩ B_syn|`,
`X ~ Hypergeometric(|A_syn|, K_syn, m_syn)`. Invariants: membership is never recomputed; the block is
never topped up from observed labels; no replacement theorem is drawn in for an infra-censored one;
`m_variant_if_the_block_were_recomputed` is a descriptive disclosure that never enters a gate.

**Amendment B — the generation seed formula (changed).** The earlier implementation seeded from
`rank_by_q_B2`, which made the generation RNG stream a function of the controller's ranking. Not
approved. The frozen rule is now

```
seed = seed_base + (formal_sample_rank − 1) * 8 + sample_index,   seed_base = 20260924
```

where `formal_sample_rank` is written into `v3_r001_formal_sample.json` as 1..128 in the artifact's
already-frozen, outcome-free **draw order** (`SHA256('<draw_seed>|<component_id>')` ascending — the
order the membership hash was already taken in). Nothing was re-shuffled and nothing was sorted by
statement_id, q or source: the existing order was written down. Enforced in code and tests: the rank
is independent of `q_B2`, `q_B1`, `source` and any outcome; changing any q value moves no seed; all
128 × 8 = 1024 seeds are distinct. Membership unchanged, ordering explicitly pinned — the file hash
moved (`7b4371e1…` → `733a699e…`) while `sample_sha256` (`0ac7134f…`), both q-vector hashes, the
block hash (`d5791fd5…`) and all 128 component and statement memberships are byte-identical.

**Amendment C — verifier execution (approved).** The formal R001 verifier runs the **B0 reliability
policy** (`tinylean_rl.verifier.policy.VerificationSession`): sequential and conservative, with
server-side timeout, client timeout, health check, canary, the infra taxonomy, fail-close, and
unresolved infra → censored/missing. E024's 4-worker concurrency is **not** used, because a canary
cannot be gated while verifications are in flight. This is recorded as an **infrastructure execution
choice, not a treatment difference** — generation scientific semantics, reward semantics and both
gates are unchanged. Limitation accepted with it: R001's verifier wall-clock may not be compared to
E023's concurrent verifier wall-clock as an exact efficiency claim; R001's question is prospective
informativeness, not verifier throughput.

**Artifact hashes after the amendments (before → after).** `v3_r001_formal_sample.json`
`7b4371e1f8211c41c3e158f8a9201f28948d29befc9a8462ae5f843c851d3d88` →
`733a699e9eeb4f1d8b130c1a87c3455a06748b66aa3909f90995f432877c6beb`; `V3-R001_gate.json`
`b308038093e7260f7f15d774a7f428d1f913e2555e7f885b9b88d9102452a3fc` →
`fcb700b508460f5fdaac318f7ce8e9bd56266f7aaaf07c7452d00f08d9c7ac93`; `v3_final_holdout_reserve.json`
unchanged (`b60be381…`). The frozen settings block changed with the seed formula, so its sha256 moved
too; the runner stamps the settings hash it actually ran under, and the analyzer re-checks it.

## 19. Infrastructure amendment C′ (2026-09-25, post-attempt-1)

**Status of record: `V3-R001 attempt-1: INFRASTRUCTURE_ABORT / NO_SCIENTIFIC_OUTCOME / NOT_ANALYZED`**
(owner decision of 2026-09-25). Attempt-1 stopped three times on a verifier-lifecycle fault of the
shared Lean server; its 384 candidates (ranks 1–48) are quarantined provenance, never combined with a
later execution and never analyzed. Nothing in sections 1–18 of this preregistration is changed.

The amendment and its evidence are frozen before attempt-2 in

- narrative: `docs/v3/V3-R001_infrastructure_amendment_Cprime.md` (audit, root cause, C′ design,
  the owner §7 review, the §8/§9 validation);
- machine-readable: `experiments/manifests/v3/V3-R001_Cprime.yaml`;
- validation artifact: `experiments/manifests/v3/V3-R001_Cprime_validation.json`.

The only frozen-settings entry that moves is `VERIFIER.batch_size` 4 → 1 (verification concurrency
1 against a dedicated `MAX_REPLS=1` instance); the settings hash moves `978566da…` → `bf069ecc…`.
Sample, block, seeds, model, generation, label rule, censoring, gate and analysis are unchanged, and
no timeout was lengthened. **The launch checklist item "owner's explicit launch authorization"
remains withheld**: C′ is implemented and validated, attempt-2 still runs only on a separate owner
instruction.
