# [ARCHIVED / SUPERSEDED] Decision-Guided RLVR Mechanism Trial

> **Status: `DRAFT / NOT LAUNCHED` — archived 2026-09-24 by owner review §2. This design was NOT
> approved and is NO LONGER the next experiment. Kept on purpose as the historical record of a design
> that was drafted, audited and then superseded; nothing here was ever executed (no training, no
> rollout, no verifier call, no GPU job) and nothing here may be run without a fresh owner decision.**
>
> **The `V3-R001` identifier has been vacated and reassigned.** It now names
> **V3-R001 — Prospective Family-Clean Validation of the Informative-Group Controller**
> (`docs/v3/V3-R001_preregistration.md`), a frozen-θ0 *rollout* experiment with **no optimizer update**.
> Where this file says "R001" it means the old draft; where the new preregistration says "R001" it means
> the prospective validation. The RL intervention sketched below would, if the owner ever authorizes it,
> be numbered from **V3-R002** onward — not R001.
>
> **Owner reasons for not approving this draft (2026-09-24 §2), recorded verbatim in substance:**
> 1. D001's B1-superiority arm now carries material full-procedure uncertainty (A3: level-2 ΔAUPRC CI
>    [−0.025, 0.237] includes 0), so the novelty premise this trial leaned on is qualified.
> 2. The frozen sampler had a **support bug**: with `epsilon* = 0`, theorems with `q = 0` get probability
>    exactly 0 (the offline freeze's own `starved_statement_fraction = 0.02041`, i.e. 12 of 588 labelled
>    theorems), and `q` is *undefined* for the 72.6 % of pool rows that were never labelled. Permanently
>    abandoned in favour of a uniform-mixture policy with guaranteed positive support (§13).
> 3. **Deployment pool semantics were never frozen** — "uniform over 24,418 rows" was inherited, not
>    proven from the trainer source, so the control arm of this trial was not a defined object.
> 4. **Source concentration**: informative groups are ~90 % synthetic, so a guided sampler shifts the
>    training distribution in a way the offline probe never evaluated.
> 5. **10 GB RL training on fly122 is unresolved** (θ0 fits; the optimizer state does not) and the
>    10 GB compatibility smoke was explicitly NOT to be run.
> 6. **No need to take the training cost first** — a cheaper, cleaner prospective validation of the
>    controller exists, and it must succeed before any intervention is considered.
>
> Scientific content that SURVIVES archiving and is reused by the new R001: the informative-group label
> definition, the frozen D001 controller/`q` recipe, the level-1 vs level-2 bootstrap discipline, the
> power analysis machinery, and the two corrections recorded in §5.1/§6 (the exact duplicate-draw bound
> and the 3.8 h GPU-hour basis, which replaced an unsupported 5.6 h).
>
> Anything below this banner is **as-drafted on 2026-09-24, before the owner review**, including the
> `CANONICAL_GO` and "beyond handcrafted difficulty features" wording that §1 of the review has since
> superseded (`CANONICAL_GO_WITH_QUALIFICATION`).

---

# [original draft text begins here]

# V3-R001 — Decision-Guided RLVR Mechanism Trial

> **Status at drafting time: `PREREGISTRATION-DRAFT` — authored for owner review. NOT LAUNCHED. NO RL RUN.**
> Nothing in this document has been executed: no training, no rollout, no verifier call, no GPU job.
> Authorization to run requires an explicit owner GO on this file plus the §10 pre-flight checklist.
> *(Superseded — see the archive banner above.)*

```yaml
experiment: V3-R001
name: Decision-Guided RLVR Mechanism Trial
kind: rl-intervention (mechanism trial)
status: PREREGISTRATION-DRAFT / NOT LAUNCHED
branch: v3-jev-rl-controller
default_host: fly90 (RTX 3090 24 GB)          # registry V3-R### default
depends_on:
  - V3-D001 = CANONICAL_GO (docs/v3/V3-D001_canonical_audit.md)
  - offline freeze  experiments/manifests/v3/archive/V3-R001-training-draft_offline_freeze.json
question: >
  Does D001-guided theorem sampling increase the realized density of informative
  GRPO groups under a fixed RL compute budget?
claim_level: MECHANISM ONLY (not capability, not Pass@k, not wall-clock savings at scale)
```

---

## 1. Question, estimand, and what a result means

**Question (owner Part B).** Under a fixed number of optimizer steps and a fixed candidate budget per
step, does drawing training theorems from a distribution informed by the frozen V3-D001 controller
produce a **higher realized fraction of reward-informative GRPO groups** than drawing them the way V1
did?

**Primary estimand.** For arm `a`, `IGR_a = E[ 1{ 0 < Σ_{j=1..8} score_j < 8 } ]` over the groups that
arm `a` actually rolls out, with the *identical* label definition as V3-D001 (§1 of the D001
preregistration: `score`, never `acc`). The trial quantity is the difference `IGR_treat − IGR_control`.

**A mechanism GO would license** the sentence "controller-informed sampling raises informative-group
density at fixed compute". It would still **not** license "improves proving", "improves Pass@k", or
"saves compute" — those need V3-R002 (capability endpoint + a budget-matched comparison at scale).

**A null is informative and is not to be rescued.** Per §10 the horizon, gate, `alpha/epsilon`, layer,
label and folds are all frozen before the first rollout; a failure to clear the gate ends V3 for the
sampling-mechanism question.

---

## 2. Frozen dependency on V3-D001 (nothing re-fit here)

V3-D001 (offline, 686 valid V1 rollout groups, 433 family components, nested family-grouped CV) came out
**GO** on G1∧G2∧G3 with the calibration criteria met, and the canonical audit classifies it
`CANONICAL_GO` (reproduction bit-identical; `LEAKAGE_AUDIT: PASS` 24/24; provenance 5/5; A3 committed).
The A3 consequence for this draft is specific: the full-procedure bootstrap **kept** B2's absolute top-20 enrichment
(1000/1000 replicates ≥ 1.75, worst 2.4546) but **lost** the two-sided ΔAUPRC-against-B1 claim
(level-2 CI [−0.02504, 0.23679], one-sided p = 0.048). R001's estimand is treatment-vs-**uniform**, which
depends on the surviving arm; nothing in this trial may be framed as "beats a handcrafted difficulty
sampler", and no B1 comparison arm is added on that account (a new baseline after seeing D001 outcomes
would be exactly the post-hoc move the owner forbade).

The owner-frozen claim wording is carried into R001 **with the A3 qualification attached** (it is not
silently relaxed, and it is not silently dropped — see §A5 of the audit doc):

> *Frozen Kimina representations contain family-generalizable signal for predicting reward-informative
> RLVR groups beyond handcrafted difficulty features.*

R001 exists **only** to test whether that offline signal has a sampling-mechanism consequence. The D001
artifacts (`V3-D001.yaml`, `v3_d001_folds.json`, `V3-D001_results.json`) are inputs, never re-derived.

---

## 3. Arms and the single-difference rule (owner B4)

Two arms, run **serially on the same machine, same binary, same environment, same pinned commit**, and
with **matched seeds**: `seed A` and `seed B` for each arm (4 runs total).

| Held identical in both arms | Value (the V1 / P3-B recipe that produced the D001 labels) |
|---|---|
| policy init / backbone | `AI-MO/Kimina-Prover-Distill-0.6B`, weights sha256 `34e6e630…640fe2` (= theta0, the same model the controller reads) |
| training mode | full-parameter update (V1 choice; no LoRA) |
| algorithm | GRPO with DrGRPO mean-only centring (`norm_adv_by_std_in_grpo=false`), `loss_agg_mode=seq-mean-token-sum-norm`, clip 0.2 / 0.3 / dual-clip 3.0, `entropy_coeff=0`, no KL loss, no KL reward, reference policy disabled |
| candidates per theorem | `rollout.n = 8` (unchanged — this is the label's unit) |
| prompts per optimizer step | `train_batch_size = 4` (→ 32 sequences/step), `ppo_mini_batch_size = 4`, `ppo_micro_batch_size_per_gpu = 2` |
| optimizer | AdamW, lr `2e-6`, V1 schedule |
| lengths | `max_prompt_length = 1024`, `max_response_length = 4096`, `max_model_len = 5120`, `max_num_batched_tokens = 5120` |
| decoding | `temperature = 1.0`, `top_p = 1.0` |
| prompt pool | `AI-MO/Kimina-Prover-Promptset` train split: **24,418 rows / 7,620 unique statements**, duplicate rows retained (row multiplicity is part of the V1 prior) |
| verifier | identical Lean verification path, identical timeout, identical reward semantics |
| steps | same `H` per arm (§6) |
| env / code | `pinned_verl_commit = e16b605e8186614c685875c9b57eb19e841b521a`, same `scripts/run_p3_pilot.sh`-derived runner, same container/env, `dataloader_num_workers = 0` |
| **the only difference** | **the theorem sampling distribution `P(i|t)`** |

**Both arms run the same sampler class** (`src/v3/decision_sampler.py`, new), so any implementation
artifact is shared: control passes weights `w ≡ 1`, treatment passes `w_i(t) = ε + q_i(t)^α` (§5).

Documented fidelity caveat: V1 used torch's `RandomSampler` (a shuffled pass over the 24,418 rows, i.e.
*without* replacement within an epoch). A weight-driven multinomial draws *with* replacement, so the
control arm is **not byte-identical to V1's index sequence**. The expected number of repeated index *pairs*
over a whole control arm is `n(n−1)/2K` with `K = 24 418`, i.e. **0.20 at `H = 25`** (100 draws) and
**0.29 at `H = 30`** (120 draws) — far below one duplicate per arm. The pre-flight test in §10 item 10.2 must show
the control arm's row distribution is statistically indistinguishable from V1's. Keeping one sampler
implementation for both arms is preferred over "exact V1 indices for control / different algorithm for
treatment", which would confound sampler *code* with the intervention.

---

## 4. Controller (owner B1, B2)

* **Recipe:** the frozen D001 B2 model — theta0 **block-18 last-token** representation (1024-d) ⊕
  `step_norm` → L2-regularized logistic (Newton/IRLS), exactly `v3_d001_lib.fit_logistic`.
* **Fit set:** **all 686 V1 seed1/2/3 valid groups** (history only). No rollout, reward, verifier or
  trial data enters it. `C` selected by the identical inner family-grouped AUPRC criterion on that fit
  set (selected `C = 0.003`, mean inner AUPRC 0.610).
* **Artifact:** weights + standardizer persisted in `experiments/manifests/v3/archive/V3-R001-training-draft_offline_freeze.json`
  (`controller.weights_mu_sd`, `controller.weights_sha256`) and hashed into the run manifest at launch.
* **Deployment score:** `q_i(t) = P(informative | theorem i, step_norm(t))`. The theorem representation is
  **precomputed once** for the whole pool before launch; **only the scalar `step_norm` changes between
  steps** — one matrix-vector product + sigmoid per step over 7,620 statements, microseconds, no GPU.
* **No online retraining, ever**, inside R001: no refresh, no EMA, no bandit, no re-fit on the trial's own
  rollouts, no acceptance of treatment-arm data as training data.
* **Honesty about `step_norm`:** the fitted `q(t)` surface is nearly flat in `step_norm` (it is 1 of 1025
  features), so the frozen sampler is effectively time-stationary within a ≤30-step trial. The step input is
  kept for recipe identity with D001, not because it drives the arm.

---

## 5. Sampler (owner B3) — stochastic, never deterministic top-k

```
w_i(t) = epsilon + q_i(t)^alpha          (i over the 24,418 POOL ROWS; a row inherits its statement's q)
P(i|t) = w_i(t) / Σ_j w_j(t)             b_t ~ Multinomial(4, P(·|t))    WITHOUT the top-k shortcut
```

* Deterministic top-k is **prohibited** (owner B3): it collapses coverage and destroys the group-diversity
  assumption behind GRPO advantages.
* Row multiplicity is applied **at row level**, so the treatment arm keeps V1's "statement appears as often
  as it has rows" prior and only *modulates* it.
* Seeded per run (`seed A`/`seed B`), recorded, and reproducible from the manifest.

`epsilon` and `alpha` are **frozen offline** from the D001 out-of-fold prediction distribution and the
prompt-pool structure — never swept on RL outcomes (owner B3). The offline search is in
`experiments/manifests/v3/archive/V3-R001-training-draft_offline_freeze.json` (`grid_search`, 5 × 7 = 35 pairs). Rules, all
declared before any RL data existed:

* **objective:** maximize the multiplicity-weighted OOF-predicted treated IGR (an off-policy estimate on
  the 686 labelled groups);
* **ties:** toward the larger `epsilon` (flatter = safer);
* **anti-collapse constraints, five of them, evaluated at row level and expressed RELATIVE TO THE CONTROL
  ARM'S OWN ROW-LEVEL DISTRIBUTION:** normalized sampling entropy ≥ 0.85 · ESS ≥ 3 % of the samplable
  support (see the denominator note in §5.1) · max theorem sampling probability ≤ 2 × control's ·
  expected distinct family components **per batch** ≥ 0.9 × control's · expected distinct family components
  **over the whole trial** ≥ 0.8 × control's.

Why control-relative: pool row multiplicity spans 1…54 (median 6 among labelled statements), so the
**uniform control arm itself** already concentrates ~4.8× relative to statement-uniform. An absolute cap
measures the pool's copy structure, not controller-induced concentration, and it rejects the treatment for
doing what the control already does. (An earlier draft of this freeze used an absolute 5× cap; it left only
7/35 grid points feasible and was corrected before any RL outcome existed — recorded in
`grid_search.constraint_rationale` and in §14 risk 4.)

### 5.1 Frozen numeric values

From `experiments/manifests/v3/archive/V3-R001-training-draft_offline_freeze.json` (offline freeze
`2026-09-24T01:08:59Z`, 653.6 s; 31 of 35 grid pairs feasible). No RL data existed at any point in
this computation. *The artifact records no execution host*, so the node this ran on is a workflow fact
(the dev node, fly90) rather than something the artifact certifies — noted because §10.1's reps-artifact
content hash is what actually pins the input.

| Quantity | Value |
|---|---|
| `alpha*`, `epsilon*` | **1.0, 0.0** (`P(i\|t) ∝ q_i(t)`, **no floor**). Consequence, stated plainly: with `epsilon=0` a statement whose `q` is exactly 0 gets probability **exactly 0**. The freeze artifact's own `starved_statement_fraction = 0.02041` says **12 of the 588 labelled theorems are in that state**, so "no theorem has zero support" is **false** for this pick; what holds is that every *family component* retains positive support (`min_group_probability` = 5.113e-5 over the 433 components). A floor (`epsilon > 0`) is the mechanism that would restore zero-support theorems, and the tie-break rule toward larger `epsilon` did not select one — see §14 risks 8 and 9. |
| row-level diagnostics (**support = the 588 labelled statements, weighted by their pool row multiplicity**; `K_units = 588` in the artifact's own `row_level` block) | ESS **216.379** · normalized entropy **0.90944** · max theorem sampling probability **1.9561 ×** the control's · expected distinct family components per batch **3.9408** (0.9952 × control's 3.9599) · expected distinct components over the trial **68.667** vs control 74.444 → trial coverage ratio **0.9224** |
| ⚠ denominator caveat | the artifact field is named `ESS_fraction_of_pool` = 0.36799, but its denominator is `K_units = 588` (labelled statements), **not** the 24 418-row pool. Against the pool rows the same ESS is **0.89 %**, which would *fail* the ≥ 3 % constraint. The constraint as frozen and evaluated is "≥ 3 % of the samplable support actually used by the treatment distribution"; if the owner intends it pool-relative, that is a different constraint and §10.3 must re-run it that way before launch. |
| controller `q` reference numbers | max prob over *statement-uniform* would read 9.2993, of which 4.754 is the pool's own multiplicity (hence the control-relative rule, §5) |
| OOF-predicted treated IGR | record-level **0.42913** · multiplicity-weighted **0.22704** |
| control (uniform) realized IGR on the same labelled set | record-level **0.1516** · multiplicity-weighted **0.03257** |
| **OOF-predicted uplift (primary estimand, multiplicity-weighted)** | **+19.447 pp**, component-bootstrap 95 % CI **[+13.463, +26.243] pp** |
| OOF-predicted uplift (record-level contrast, reported not chosen) | +27.753 pp, CI [+22.575, +32.762] pp |
| deployment controller | frozen B2 recipe (theta0 block-18 last-token 1024-d ⊕ `step_norm`) → L2 logistic; C = 0.003 by inner family-grouped CV (mean inner AUPRC 0.61047); fit on **all 686** V1 valid groups; **no online retraining**; weights sha256 `dc0c24ed523a1dd4ca63ef19848419a0d3c65a934d59d902a27fa858a513589a` |
| `q(t)` time-dependence | essentially flat: mean abs shift 0.00106 across `step_norm` 0.083→0.50 (1 of 1025 features) — the frozen sampler is effectively time-stationary within a ≤ 30-step trial |
| sanity | the deployment recipe re-derives D001's OOF AUPRC **0.5728** (bit-equal to the committed value); B1 reference 0.42673; in-sample 0.78446 reported as descriptive only |
| sensitivity: pick under the record-level objective (reported, never used) | `alpha=2.0, epsilon=0.025`, record-level treated IGR 0.44781 but multiplicity-weighted uplift only **+15.081 pp** → **the two objectives disagree**, so the objective had to be — and is — frozen in advance (§14 risk 3) |

---

## 6. Horizon and compute budget (owner B5)

The horizon comes from the **historical per-step IGR variance**, not from taste. Pooled over the three V1
runs' 180 per-step IGR values (E019/E020/E022, 4 groups/step = the same cluster shape R001 will produce):
mean 0.1528, **sd 0.1857**. Treating the **training step as the inference cluster**,
`SE(Δ IGR) = sqrt(2)·sd/sqrt(H)`.

Steps per arm for 80 % power (one-sided α = 0.05), from `power_analysis`:

| true uplift in realized IGR | +5 pp | +10 pp | +15 pp | +20 pp | +26 pp |
|---|---|---|---|---|---|
| steps/arm for 80 % power | 171 | 43 | 19 | 11 | 7 |
| power at `H = 25` | 0.244 | 0.602 | 0.887 | 0.985 | 0.9995 |

Equivalently, the minimum detectable effect at 80 % power is **13.06 pp at `H = 25`** (14.60 pp at 20,
11.92 pp at 30; `SE(Δ IGR) = 0.0525` at `H = 25`).

The offline-predicted uplift is **+19.447 pp** with component-bootstrap CI **[+13.463, +26.243] pp**, so
`H = 25` sits *just* above the power floor at the CI's lower edge (13.06 pp vs 13.46 pp). That is tight,
and it is tight in the wrong direction if the true effect is smaller than the offline estimate — which §13
argues is likely, since the offline number is an upper bound.

**Proposed horizon `H = 25` optimizer steps per arm** (owner range 20–30), i.e. 100 theorem draws and
800 candidate rollouts per arm; 4 runs (2 seeds × 2 arms) run serially on one card = 4 × 25 × 136 s =
13 600 s ≈ **3.8 h wall ≈ 3.8 GPU-hours** at V1's measured 136 s/step (`p3b_pilot.yaml` mean, median 138,
range 78–245; `p3_0_complete.yaml` total 136.4). If instead the two arms of a seed are co-resident on the
24 GB card the wall time shrinks but GPU-hours do not, and per-step time must be re-measured rather than
assumed. `H` is frozen here and is **not** to be extended after seeing data (§10).

---

## 7. Endpoints

**Primary (the gate quantity).** Realized IGR per arm = fraction of the arm's valid groups with
`0 < Σ score < 8`, pooled over both seeds, plus the difference.

**Secondary (all descriptive or gate-supporting; none may replace the primary):**

| Endpoint | Why |
|---|---|
| informative groups per 1 000 candidates, per 1 000 response tokens, **per GPU-hour** | the sample-efficiency mechanism, in the units the budget is paid in |
| mean reward per step; all-fail fraction; all-success fraction | the two degenerate directions (`Σ=0`, `Σ=8`) the IGR excludes; a rise in all-success is a *different* good news story and must not be silently counted as IGR |
| realized sampling entropy vs designed entropy; unique theorems; unique family components | did the frozen sampler actually do what the freeze said |
| IGR by step bin (1–5 / 6–12 / 13–19 / 20–25) and per-arm IGR trend | **curriculum-collapse / use-up check**: informative density that decays toward zero within the trial is a mechanism failure even if the pooled mean clears the gate |
| source composition per arm (synthetic / autoformalizer / human) | D001's sharpest limitation: the signal is ~90 % synthetic, so the treatment arm may act through source shift. Reported as a **mediator**, not hidden |
| response length, truncation rate, format-failure rate, verifier error rate, grad-norm finiteness | pathology gates (§9 R4) |
| per-step `q_i(t)` quantiles of drawn theorems | whether the arm drifted off the controller's fitted support |

Raw per-candidate dumps (prompt, response, score) are written for both arms and hashed into a provenance
manifest; **the raw rollout data is never committed to git** (hashes only), per the standing §17 rule.

---

## 8. Gate (owner B6) — frozen, and it does NOT look at Pass@k

All four must hold on the pooled 2-seed data. Failure of any one → **STOP**, report the owner, no
V3-R002 discussion.

| # | Rule | Test |
|---|---|---|
| **R1 mechanism** | treated realized IGR **clearly exceeds** uniform | one-sided **step-cluster permutation test** (permute arm labels across steps within seed, ≥10 000 label permutations, seed frozen in the manifest) at α = 0.05, **and** both seeds point the same direction (a second seed that reverses the sign is a FAIL, not a wash) |
| **R2 efficiency** | informative groups per GPU-hour improved | treated > control on the ratio, with the same direction in both seeds (compute is measured, not assumed: wall time + candidate count + token count per arm) |
| **R3 no coverage collapse** | the treatment did not narrow the curriculum | treated unique family components ≥ 0.8 × control's over the trial, and realized per-step sampling normalized entropy ≥ 0.85 of the designed value in ≥ 90 % of steps |
| **R4 no verifier/format pathology** | the arms are comparable as learning problems | verifier error rate not worse than control; format-failure and truncation rates within control ± declared tolerance; all grad-norms finite; no step with a zero-advantage-all-steps stall |

Explicitly **out of the gate**: minif2F / fixed-set Pass@k, final reward level, or any capability metric.
R1–R4 are about *what the sampler did to the training signal*, not about whether the model got better — a
25-step trial cannot answer the second question and will not be described as if it could.

Also pre-declared: **R1's practical power floor.** At `H = 25`, 80 % power requires an effect of
**13.06 pp** (§6 table). A true uplift below that will read "no clear signal" — that is a power statement,
not a
falsification of D001, and the report must say so in those words.

---

## 9. Analysis plan and no-peeking discipline

1. No interim analysis for any decision; the gate is evaluated once, on the full frozen horizon.
2. No horizon extension, no `alpha`/`epsilon` re-selection, no layer/label/fold change, no re-weighting
   rule change after trial data exists. Any deviation is logged in the run manifest and the result is
   reported as exploratory, not confirmatory.
3. The permutation test, the tie rules, the IGR label (`score`), the cluster unit (step), and the seed
   pairing are fixed by this file.
4. V1's own runs (same recipe, uniform sampler, steps 1–25 of E019/E020/E022) are available as an
   **external reference** for the control arm; they are a sanity check on control reproduction, never a
   third "arm" in the comparison (different wall-clock period, and seed1 is a stitched E017+E019 run).
5. Reporting: the memo must state the primary estimate, R1–R4 individually, the IGR-by-step-bin trend, the
   source composition, and every failed secondary — no selective reporting.

---

## 10. Pre-flight artifacts required BEFORE any launch (owner §11 checklist)

| # | Artifact | Acceptance test |
|---|---|---|
| 10.1 | **theta0 representations for the entire prompt pool** (7,620 unique statements, block 18, last non-padding token, same extractor `v3_d001_extract.py`, weights hash `34e6e630…`) — R001 cannot score what it has never encoded. ≈6 min GPU (612 prompts took 28.4 s, peak 1.53 GB) | content hash recorded; 100 % of pool rows have a `q_i(t)`; the 612 already-extracted statements are **bit-identical** to the frozen D001 artifact |
| 10.2 | Sampler module + unit tests | control (`w ≡ 1`) reproduces the row-multiplicity prior and is statistically indistinguishable from V1's `RandomSampler` draws; weights normalized over rows; deterministic given the recorded seed; `sampler.update()` called once per optimizer step; `dataloader_num_workers = 0` (required by verl's `create_rl_sampler` hook) |
| 10.3 | **Full-pool** sampling diagnostics with the frozen `alpha*, epsilon*` | the §5 constraints hold on the 24,418-row pool (not only on the labelled 588-statement support); **every §5 number is reported twice, once over the pool rows and once over the 588-statement support, with the denominator named**, and the count of rows with *exactly zero* sampling probability is reported explicitly (§14 risk 8); numbers written to the launch manifest before step 1 |
| 10.4 | Frozen config diff between arms | `git diff` shows exactly the sampler weight source differing; the diff is committed |
| 10.5 | Memory compatibility smoke (§11) | done on fly122, non-scientific, result recorded |
| 10.6 | Commit this file + `V3-R001.yaml` + the freeze manifest, **before** the first rollout | registry entry `PREREGISTERED / NOT LAUNCHED`; git rev pinned in the run manifest |
| 10.7 | Rollout provenance plan | per-step dumps + hash manifest path declared; raw data gitignored |

---

## 11. Non-scientific 10 GB memory compatibility smoke (owner B7)

Purpose: **feasibility measurement only** — can the *logical* recipe be hosted on a 10 GB card. It
produces no scientific number, enters no analysis, and its result cannot change the formal recipe.

* Preserve exactly: `rollout.n = 8`, effective batch 4 prompts/step (32 sequences), reward semantics and
  DrGRPO mean-only normalization, optimizer semantics (AdamW, lr 2e-6, no KL), prompt/response lengths,
  verifier, `temperature/top_p`, 1 optimizer step only.
* **Allowed memory levers:** gradient checkpointing (already on in the V1 recipe), CPU offload of
  optimizer state, activation-memory optimization (recomputation), micro-batch splitting **that leaves the
  effective batch and the update math unchanged** (gradient accumulation).
* **STOP and report the owner** if fitting the recipe would require changing `n`, the effective batch, the
  sampling semantics, the reward normalization, or any other training semantics. Never silently mutate them
  (standing rule from `docs/v3/data_audit.md`).

**Host decision (needs the owner's confirmation).** `docs/dual_server_collaboration.md` §13 forbids
CPU-offload / LoRA / optimizer / batch changes **on fly122** precisely because they are research
confounds, allowing only a single-optimizer-step memory smoke. Since the owner's B7 list *permits* CPU
optimizer offload for the smoke, the two documents are consistent only under this reading, which this
draft adopts:

* **fly122 (10 GB): non-scientific memory smoke only** (1 optimizer step, feasibility recorded).
* **fly90 (24 GB): the formal R001 arms**, with the V1-faithful recipe and no offload — matching the
  registry default (`V3-R###: default-host fly90`) and keeping the comparison free of memory-workaround
  confounds.

If the owner intends fly122 to host the formal trial instead, that is a **recipe change relative to V1**
(offload on by necessity) and must be decided and written into this file *before* launch, not discovered
mid-run.

---

## 12. Explicitly out of scope for R001

Any capability claim (minif2F, fixed-set Pass@k) · compute-saving claims at scale · a second controller
iteration · `epsilon`/`alpha` sweeps against RL outcomes · MLPs, new layers, mean-pooling changes, new
backbones, TinyJev/Kev models, Track B allocators, candidate rankers, repair controllers · multi-arm
curriculum ladders · restoring V2/A001 artifacts (V2 is frozen) · any change to V2 frozen
manifests/results. **Only this one pre-registered comparison.**

---

## 13. Predicted effect — the honest statement

The offline estimate of the treated arm's informative density depends on the weighting, and the trial
draws under the pool's row distribution while labels exist for only part of it:

* **record level** (each of the 686 labelled groups equally): treated IGR **0.42913** vs **0.1516** on the
  same support — the D001 realized prevalence (the V1 pooled RL mean, over all 180 per-step values, is
  0.1528, which agrees);
* **multiplicity-weighted** (pool rows, i.e. closer to what an arm draws): treated IGR **0.22704** vs
  **0.03257** on the same support.

Both come from the same OOF predictions and the same frozen `alpha*, epsilon*`; they differ because pool
row multiplicity is **negatively** related to measured informativeness (labelled statements cover 27.4 % of
pool rows, median multiplicity 6 vs pool median 1, and their multiplicity-weighted informative rate is
3.3 % against a 15.2 % group-level rate). This is a genuine **support/estimand gap**: 72.6 % of the pool
rows have never been rolled out, so no offline calculation can pin the trial's absolute IGR. The primary
estimand is the **multiplicity-weighted** contrast (**+19.447 pp, CI [+13.463, +26.243] pp**), because that
is the quantity a row-drawing arm actually realizes; the record-level contrast (+27.753 pp) is reported as
a description of the labelled set, not as a prediction. The trial is therefore powered against a range
(true effects from 5 pp, where 25 steps give only 24 % power, to 26 pp, where they give >99.9 % — §6)
rather than a point prediction, and the pre-registered
reading is: **R1 clears ⇒ mechanism evidence; R1 does not clear ⇒ the trial cannot distinguish a small
real effect from no effect, and V3 sampling work stops.**

---

## 14. Open risks (ranked)

1. **Support/estimand gap (§13)** — predicted uplift is only computable on 27 % of pool rows; the
   plausible effect spans the region where a 25-step trial is well-powered and where it is not.
2. **Source concentration.** D001's informative class is ~90 % synthetic; the treated arm will likely shift
   the source mix. That makes "decision-guided sampling" partly a source-reweighting intervention, so the
   mechanism claim is about *the controller's distribution*, not about semantic understanding per se.
3. **Non-stationarity / use-up.** `q` was fit on a moving policy's rollouts; repeatedly sampling the same
   high-`q` theorems can exhaust their informativeness as the policy improves. The IGR-by-step-bin trend is
   the pre-registered diagnostic, and a decaying trend is a mechanism failure even if the pooled mean
   clears R1.
4. **Constraint yardstick.** `max_prob_over_control` is a choice; the absolute-vs-relative correction made
   in §5 changed the feasible set materially (**7/35 → 31/35** grid points) and the frozen pick from
   `alpha=2.0, epsilon=0.1` (row-level uplift **+5.03 pp**) to `alpha*=1.0, epsilon*=0.0` (**+19.45 pp**).
   It was fixed offline with no RL data, and the alternative rule is recorded, but the owner should know
   the frozen sampler is sensitive to it.
5. **Two seeds only.** With `n = 1` run per arm per seed, the step-cluster test assumes exchangeability of
   steps across arms; run-level (seed) variation is only probed by the "both seeds same direction" rule.
6. **10 GB vs 24 GB host tension (§11)** — unresolved between owner B7 and `dual_server_collaboration.md`.
7. **Controller provenance.** seed1's V1 data is a stitched E017+E019 run (`data_audit.md` risk R4): fine
   for pooled training data, not re-runnable — the trial inherits this as a provenance footnote only.
8. **`epsilon* = 0` creates literal zero support.** A theorem with `q = 0` is then unreachable, and the
   frozen artifact already shows 12 of 588 labelled theorems (2.041 %) in that state. On the deployment
   pool the exposure is much larger: `q` is *undefined* for the 72.6 % of pool rows whose statement has
   never been labelled, and any implementation that treats "undefined" as `q = 0` turns the treatment arm
   into a sampler over ~6 679 of 24 418 rows while the control samples all of them. That is a coverage
   collapse of precisely the kind gate R3 forbids, and the frozen diagnostics cannot detect it because
   they were computed on the labelled support. Mitigations, in order of preference: (a) §10.1 gives every
   pool statement a real `q`, then (b) §10.3 re-runs the diagnostics over all 24 418 rows and counts
   zero-probability rows explicitly, and (c) if the zero-support fraction is material, the owner decides
   between a positive floor and restricting the pool — **before** launch, as a documented amendment, not
   by tuning `epsilon` after seeing RL outcomes.
9. **The ESS constraint's denominator is ambiguous in the frozen artifact.** `ESS_fraction_of_pool`
   = 0.36799 divides by the 588 labelled statements, not by the 24 418-row pool (where the same ESS is
   0.89 %, i.e. below the 3 % bar). Whichever reading the owner intends, §10.3 must report both numbers
   under their explicit denominators rather than inherit the field name.

---

## 15. Owner decision points (needed before this draft can become `PREREGISTERED`)

* **D-a** confirm `alpha*, epsilon*` as frozen by §5 (or direct the alternative rule).
* **D-b** confirm `H = 25` and 2 seeds × 2 arms (≈ 3.8 GPU-h on fly90, serial), or 1 seed (≈ 1.9 GPU-h, weaker R1).
* **D-c** confirm fly90 hosts the formal arms and fly122 only the memory smoke (§11).
* **D-d** confirm R1 as significance-only, or add an absolute effect floor (e.g. "and observed uplift ≥
  **13.06** pp" — the 80 %-power MDE at `H = 25`, §6/§8), which converts an underpowered null into a cleaner
  negative statement.
* **D-e** confirm the prompt pool stays the full 24,418-row pool (in §13's spirit: restricting it to the
  labelled support would make the treated arm sample theorems the controller was trained on, i.e. in-sample
  — the full pool is what makes R001 an out-of-sample test of the controller).
* **D-f** decide how literal zero support under `epsilon* = 0` is handled, **after** reading the §10.3
  full-pool diagnostics and **before** launch: a positive floor, a restricted pool, or accepting the
  starvation and writing it into the gate (§14 risk 8). This must not be settled by tuning `epsilon` after
  RL outcomes exist.
* **D-g** state the intended denominator of the `ESS ≥ 3 %` constraint — 588 labelled statements (as the
  freeze actually evaluated it, 36.8 %) or 24,418 pool rows (as its field name claims, 0.89 %, which would
  **fail**) (§14 risk 9).

---

## 16. Status

`PREREGISTRATION-DRAFT`. **No RL has been launched and none will be** without the owner's explicit GO plus
items 10.1–10.7 complete. If the owner declines, V3's contribution stays exactly what D001 established,
and the sampling-mechanism question remains open rather than answered negatively.
