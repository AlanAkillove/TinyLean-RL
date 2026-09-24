# V3-D001 — Canonical methodological audit (A1–A5)

Experiment: **V3-D001** · Audit scope: *methodology only — no new fitting recipe, no new folds, no new label, no RL*
Owner directive: 2026-09-24 Part A (A1 complete cross-seed report · A2 leakage audit · A3 full-procedure
bootstrap · A4 representation sanity · A5 classification)
Machine record of the audit: `experiments/manifests/v3/V3-D001_canonical_audit_fly90.json` and
`..._fly122.json` (formal node) · level-2 bootstrap: `V3-D001_fullproc_bootstrap.json` +
`V3-D001_fullproc_3rep_fly122.json` (`runs/v3_d001/` holds the same files, but `runs/` is gitignored)
Auditor tooling (new, read-only w.r.t. the frozen pipeline): `scripts/v3_d001_audit.py`,
`scripts/v3_d001_fullproc_boot.py`
Frozen inputs consumed unchanged: `experiments/manifests/v3/V3-D001.yaml`,
`docs/v3/V3-D001_preregistration.md`, `experiments/manifests/v3/v3_d001_folds.json`,
`experiments/manifests/v3/V3-D001_results.json`, `runs/v3_d001/theta0_reps.npz`

> **Headline, stated up front.** No leakage was found — the frozen numbers reproduce bit-identically and all
> 24 preprocessing checks pass, so `D001_FINAL_STATUS` remains **CANONICAL_GO** and the frozen G1–G3 gate is
> unchanged. The one substantive change comes from **A3**: once fold re-partition, inner-CV C reselection and
> refit variability are propagated (1000 full-procedure replicates), the ΔAUPRC(B2 − B1) interval
> **widens from [0.051, 0.241] to [−0.025, 0.237] and includes 0** (952/1000 replicates positive, one-sided
> p = 0.048). The top-20 enrichment arm instead holds in **1000/1000** replicates. So the evidence that
> survives procedure uncertainty is *"the representation finds informative groups at enrichment far above
> prevalence"*; the evidence that does not, at two-sided 95 %, is *"and better than the handcrafted
> difficulty features"*. See §A3 and the claim qualification in §A5.

---

## 0. Reproduction first (precondition for every A-item)

Before auditing anything, **every number in the committed `V3-D001_results.json` was recomputed from the
raw artifacts** (rollout parquet + family registry + frozen fold manifest + frozen theta0 reps) with the
frozen code path (`scripts/v3_d001_run.py` imported, never edited):

| Surface | Result |
|---|---|
| dataset counts (686 groups / 433 components / 104 positives / prevalence 0.1516) | reproduced bit-identical |
| B0 / B1 / B2(block18) / B2(block9) / B2(block27) metric dicts (AUPRC, AUROC, Brier, ECE, top-10/20/30 IGR + enrichment) | reproduced bit-identical |
| all six family-component bootstrap CIs | reproduced bit-identical (same seed 20260923, 10k reps) |
| cross-seed shared fields (n_test_groups, test_positives, prevalence, AUPRCs, chosen C) | reproduced bit-identical |
| gate inputs G1/G2/G3 and the derived GO + calibration claim | reproduced bit-identical |

`R_reproduction_vs_committed.all_pass = true`. The audit therefore operates on the same numbers that were
reported to the owner, and nothing in this document is a re-analysis of a *different* run.

**Cross-node reproduction (A5 provenance).** The audit was run twice independently — on fly90 (RTX 3090,
64 cores, canonical/dev node) and on fly122 (RTX 3080, 16 cores, the owner-designated formal node) — from
two different git revisions (`4e1ddce` and `bab2104`). The two JSON records were then compared field by
field: **every numeric and verdict field is identical**; the only differing field in the whole document is
the audit's own recorded `A4...audit_host_git_revision`. Both nodes report
`LEAKAGE_AUDIT: PASS (24/24)`, `A4 all_pass (5/5)` and `D001_CANONICAL: CANONICAL_GO`. Wall time differs
(724 s vs 424 s for nested CV + cross-seed) but no timing enters any verdict.

---

## A1 — Complete cross-seed report (G3 in full, not just "PASS")

Protocol per fold: train on the two non-held seeds, **delete every group whose family component appears
anywhere in the held-out seed** (that is the family-isolation cost, reported explicitly below), fit B1 and
B2 with inner family-grouped CV for C, evaluate on all held-out-seed groups. Nothing was re-tuned on the
test seed.

| Held-out seed | train seeds | groups kept for training | dropped by family exclusion | train comps / pos / prev | test groups / comps / pos / prev | B1 AUPRC | B2 AUPRC | **ΔAUPRC** | B1 top20 IGR | B2 top20 IGR | B1 top20 enr | B2 top20 enr | B2>B1 | top20>prev | frozen G3 fold rule |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| seed1 | 2+3 | 315 (of 455) | **−140** | 238 / 49 / 0.1556 | 231 / 195 / 37 / 0.1602 | 0.6288 | 0.6707 | **+0.0419** | 0.5652 | 0.6087 | 3.53× | 3.80× | yes | yes | **PASS** |
| seed2 | 1+3 | 285 (of 452) | **−167** | 230 / 39 / 0.1368 | 234 / 203 / 34 / 0.1453 | 0.3753 | 0.5032 | **+0.1279** | 0.4043 | 0.4681 | 2.78× | 3.22× | yes | yes | **PASS** |
| seed3 | 1+2 | 322 (of 465) | **−143** | 251 / 53 / 0.1646 | 221 / 182 / 33 / 0.1493 | 0.4869 | 0.4713 | **−0.0157** | 0.3636 | 0.4545 | 2.44× | 3.04× | **no** | yes | **FAIL** |

top-20 `k` per fold: 46 / 47 / 44 groups. Family disjointness train↔test: **0 shared components in all
three folds**. Per-fold chosen regularization: C=0.03/0.003, 0.003/0.003, 0.003/0.003 (B1/B2).

### Is G3 2/3 or 3/3 same-direction? — the honest answer

Splitting the frozen conjunctive rule into its parts:

| Criterion | folds satisfied |
|---|---|
| AUPRC(B2) > AUPRC(B1) | **2/3** (seed3 is −0.0157) |
| top-20 IGR > that fold's prevalence | **3/3** |
| top-20 IGR(B2) > top-20 IGR(B1) | **3/3** |
| both frozen conditions together (the §14 rule) | **2/3** → meets the ≥2/3 bar |

ΔAUPRC across folds: signs **+ / + / −**, mean **+0.0514**, median **+0.0419**, min **−0.0157**.

So **G3 is genuinely 2/3, not 3/3.** The *ranking-accuracy* direction is seed-dependent (one of three
holdouts is slightly negative, magnitude −0.016 AUPRC on 33 positives); the *top-20 enrichment* direction
— the quantity R001 actually depends on — is 3/3. Reported as 2/3; no reinterpretation of the frozen rule,
and the seed-3 fold is disclosed in the memo's limitations.

---

## A2 — Itemized preprocessing-leakage audit

24 itemized checks. The strongest is the *active* probe: for each of the 5 outer folds and both models, the
held-out rows' **features were replaced by large random noise** and, separately, their **labels were
flipped**; the fold's chosen C, standardizer `mu`/`sd`, fitted weight vector and held-out predictions must be
**bit-identical** to the unperturbed fit. Any leakage of held-out information into a fitted quantity would
move at least one of them.

| # | Check | Verdict | Evidence |
|---|---|---|---|
| 1–10 | `all_preprocessing_outer_train_local` — B1 × 5 outer folds, B2(block18) × 5 outer folds | PASS | e.g. fold0 (548 train / 138 test, 346 vs 87 comps): C unchanged 0.003 under test-feature scramble **and** under test-label flip; `max_abs_weight_drift = 0.0`; `max_abs_prediction_drift = 0.0` |
| 11 | `B0_prevalence_baseline_is_outer_train_local` | PASS | per-fold train prevalence 0.15146 / 0.15118 / 0.15118 / 0.15301 / 0.15118 — none equals the pooled 0.1516 |
| 12 | `inner_CV_keyed_strictly_within_outer_train` | PASS | inner-fold keys = exactly the outer-train component set per fold (346/346, 346/346, 347/347, 347/347, 346/346); `inner_keys_overlapping_outer_test = 0` for all folds |
| 13 | `family_component_hard_disjointness` | PASS | every component appears in exactly one outer fold; the 5 fold component counts sum to 433 = total |
| 14 | `design_matrices_independent_of_all_label_fields` | PASS | with `y_score`, `y_acc`, `n_pos_score`, `n_pos_acc`, `n_infra_candidates` all replaced by junk before featurization, B1 and all three B2 design matrices are bit-identical |
| 15 | `source_encoding_is_frozen_level_onehot_no_target_signal` | PASS | 14 frozen B1 features in the frozen order; source levels {synthetic, autoformalizer, human} as 0/1 one-hot only; no frequency/target/mean encoding; 0 records with unknown source |
| 16 | `training_step_normalization_is_deterministic_and_label_free` | PASS | `step_norm = rollout_step / 60` with the frozen horizon constant — no per-seed or per-fold rescaling; it is the last B2 column; known at deployment from the trainer step counter |
| 17 | `B2_input_surface_is_frozen_representation_plus_step_only` | PASS | columns 0..1023 = theta0 block last-token vector, column 1024 = step_norm; nothing else (no reward, no verifier output, no rollout statistic) |
| 18 | `no_class_weight_or_resampling_in_fit_or_selection` | PASS | `class_weight` / `sample_weight` absent from `fit_logistic`, `select_C`, `nested_oof`, `fit_predict`; deterministic full-batch IRLS (max_iter 50, tol 1e-7), bias unpenalized |
| 19 | `C_grid_criterion_tiebreak_match_frozen_manifest` | PASS | grid {0.003…10}, criterion = mean inner AUPRC, ties keep the smaller (more regularized) C — all three as written in the frozen `V3-D001.yaml` |
| 20 | `topk_enrichment_denominator_is_evaluation_label_mean` | PASS | enrichment recomputed from unrounded arrays matches the committed value; denominator is the 686-group evaluation label mean only and is never fed to any fit |
| 21 | `primary_label_is_y_score_never_acc` | PASS | `y = y_score` (104 positives) and `y ≠ y_acc` (93 positives, 13 groups differ) — the `acc` variant exists in the record builder but enters no fit, no selection, no gate |
| 22 | `folds_frozen_before_any_outcome_and_reproducible` | PASS | recomputed outer fold map is bit-identical to the committed manifest (`folds_hash e0e0d30c…`); fold manifest has **exactly one** commit, `2026-09-23T23:17:49+08:00`, the same commit as the preregistration, and its blob is unchanged at HEAD; results were committed later (`23:35:00`) |
| 23 | `cross_seed_train_test_family_isolation` | PASS | 0 shared components train↔test in all three folds (195/203/182 test comps vs 238/230/251 train comps) |
| 24 | `committed_bootstrap_is_fixed_OOF_prediction_level` | PASS (classification) | the committed CIs resample family components → row indices of an **already computed** OOF prediction vector; no refit, no C reselection, no fold rebuild per replicate — hence A3 |

```
LEAKAGE_AUDIT: PASS            (24/24)
```

Consequences checked and confirmed absent: no target/frequency encoding, no test-fold standardization
statistics, no pooled-prevalence baseline inside a fold, no cross-fold component sharing, no
label-dependent feature, no post-outcome fold/C/layer/label change, and the reported evaluation-side
denominators never enter a fit.

---

## A3 — Full-procedure component bootstrap (post-hoc robustness; cannot touch the frozen gate)

### Classification of the committed interval

The frozen, committed ΔAUPRC CI **[+0.0510, +0.2415]** is a
**fixed-OOF-prediction (level-1 / "prediction-level") family-component bootstrap**:
components → row indices of the already-computed out-of-fold prediction vector, 10k reps, seed 20260923.
It propagates **sampling variability of the evaluation set only**. It does *not* propagate the variability of
the fold partition, of inner-CV regularization selection, or of the logistic fit itself — which is exactly
what a shallow 1025-feature / 104-positive model is sensitive to.

### Added robustness layer (new, post-hoc, non-gating)

`scripts/v3_d001_fullproc_boot.py` — **level-2 full-procedure family-component bootstrap**:

1. resample the 433 family components **with replacement** (same cluster unit as the frozen bootstrap);
2. **rebuild** outer-5 / inner-4 family-grouped folds from the resampled components
   (deterministic `stratified_group_kfold`, per-rep seeds derived from the replicate index);
3. **reselect C** on the inner folds of each outer fold (frozen grid / criterion / tiebreak);
4. **refit** B1 and B2(block18) and **regenerate** out-of-fold predictions;
5. recompute ΔAUPRC = AUPRC(B2) − AUPRC(B1) on the resampled evaluation surface.

**Status of the in-flight runs (recorded while in flight, re-measured 2026-09-24T11:45+08:00).** Each
artifact is produced by a single end-of-run write, so partial replicate sets are not recoverable. Two
runs are in flight:

| Run | Launched | Reps / workers | Measured state at 11:45 |
|---|---|---|---|
| fly90 `runs/v3_d001/v3_d001_fullproc_boot.json` | 08:15 | 1000 / 40 | 3 h 44 m elapsed, 40 workers each `nlwp = 1`, aggregate CPU 515 300 core-s → **95.9 % of 40 cores**, ≈535 reps done, ≈3.4 h remaining |
| fly122 `runs/v3_d001/v3_d001_fullproc_500rep_fly122.json` | 02:35 UTC | 500 / 15 | 1 h 09 m elapsed, 15 workers `nlwp = 1`, 52 100 core-s → ≈84–93 % utilization; ETA band 1.5–6 h (wider, because 15 threads share 8 physical cores and the marginal per-rep cost there is not directly measurable without stealing a core) |

The per-rep cost on fly90 is a **direct measurement, not an extrapolation**: a single-replicate probe
(`--reps 1 --workers 1`) launched alongside the 40-worker run consumed **≥ 1 046 core-seconds without
finishing**, against 765 s/rep mean for this machine's lighter-load 3-rep reference and 244 s/rep on
fly122 uncontended. That is the basis of the ≈0.040 reps/s figure above. The probe was then killed: it
occupies one of 40 cores and its remaining information value is exhausted.

*Rejected accelerations, with reasons.* Reducing fly90's worker count cannot remove a BLAS thread
explosion that does not exist (all workers are single-threaded) and would discard ≈535 completed
replicates, since nothing is written until the run ends; float32 / GPU / eigen-decomposition re-use would
change rounding and break the bit-identical recovery of the committed B1/B2 AUPRC that makes this variant
comparable to the frozen gate; no third host exists and both nodes are saturated; and dropping below
500 replicates violates the frozen 500–1000 range. The overlap between the two runs is not waste:
replicate seeds are `20260924 … 20260923 + N`, so fly122's 500-rep set is a **deterministic prefix subset**
of the fly90 1000-rep set, and whichever lands first is a complete answer at the frozen floor while the
other turns the shared 500 seeds into a cross-node replication of the level-2 procedure.

### A3 result — 1000 replicates, committed

Artifact `experiments/manifests/v3/V3-D001_fullproc_bootstrap.json` (sha256 `5e6697ca…`, 525 KB, from
`runs/v3_d001/v3_d001_fullproc_boot.json`): **1000 requested / 1000 valid / 0 dropped**, fly90, 40
single-threaded workers, 13 825 s wall (08:15 → 12:08 +08:00), mean 548.7 s wall per replicate
(4.34 reps/min). The in-flight estimate above was ≈1.9× pessimistic — a freshly spawned replicate process on
a saturated 2-NUMA box gets worse placement than the steady-state workers, so the probe's marginal cost was
not representative. Recorded rather than rewritten.

**The full-procedure interval is wider and it crosses zero.**

| Quantity | frozen level-1 (fixed OOF, 10 k reps) | **level-2 full procedure (1000 reps)** |
|---|---|---|
| ΔAUPRC point | 0.14607 | reference recomputed inside this run = **0.14607** (`matches_committed: true`); bootstrap mean **0.10869**, median 0.10800 |
| ΔAUPRC 95 % interval | [0.05096, 0.24145] — **excludes 0** | **[−0.02504, 0.23679] — includes 0** |
| interval width | 0.19049 | 0.26183 (**+37 %**) |
| sign consistency | — | **952 / 1000** replicates with ΔAUPRC > 0 (48 ≤ 0); one-sided p = **0.048** |
| B2 top-20 enrichment | 3.46659, CI [3.02435, 3.92865] | CI **[2.79858, 3.87505]**; **1000 / 1000** replicates ≥ 1.75 *and* > 1.0; smallest single replicate **2.4546** |
| Δ top-20 enrichment (B2 − B1) | — | CI [−0.09932, 1.08192]; 939 / 1000 > 0 |
| ΔBrier (B2 − B1) | — | CI [−0.02963, +0.00415]; 943 / 1000 < 0 (one-sided p 0.057) |

**Reading, stated as the negative result it partly is.**

1. **G1's margin was procedure-uncertainty-limited, and the level-1 interval understated it.** Propagating
   fold re-partition + inner-CV C reselection + Newton refit widens the interval by 37 % and moves the lower
   bound from +0.051 to −0.025. `ΔAUPRC > 0` survives as a *directional* statement at one-sided 5 %
   (p = 0.048, and only just) and does **not** survive a two-sided 95 % interval.
2. **The frozen gate is not re-opened** (owner directive; see the counterfactual below). The GO stands as
   preregistered on the statistic that was preregistered.
3. **What is robust is enrichment, not ranking accuracy.** B2's absolute top-20 informative-group enrichment
   clears the frozen 1.75 bar in *every one* of 1000 full-procedure replicates, worst case 2.4546. The
   *B2-minus-B1 difference* in enrichment is not robust (CI includes 0). So "the frozen representation
   identifies informative groups at a rate far above prevalence" survives level 2; "it does so *better than
   the handcrafted difficulty features*" is the part level 2 cannot support at 95 %.
4. **The calibration claim is directional, not two-sided-significant.** ΔBrier is negative in 94.3 % of
   replicates (0.0877 vs 0.11024 on the original sample) but its 95 % interval reaches +0.0042.
5. **The frozen point estimate sits on the favourable side of its own resampling distribution.** The
   bootstrap mean is 0.10869 against a reference of 0.14607 — a bias of **−0.03738**, with **71.2 %** of
   replicates below the reported value. Not an error (a with-replacement bootstrap of a
   1025-feature/104-positive model is expected to shift down as folds become unbalanced), but it means the
   headline ΔAUPRC is an optimistic draw, not a median.

**The interval really does move the design, not just the labels.** All 1000 replicates self-attest
`folds_rebuilt: true` and `C_reselected: true`. Resampled evaluation surfaces span 610–756 rows (orig. 686),
253–293 unique family components (orig. 433 — a component is drawn ~38 % of the time), 75–139 positives
(orig. 104) and prevalence 0.11241–0.20182 (orig. 0.1516; median of replicates 0.15041).

**Code provenance of this artifact.** `host.git_revision` reads **1f15c8e**, but that field is sampled when
the artifact is *written*, not when the run started, so it certifies nothing about the executed blob — the
run loaded a pre-lint working tree (launched 08:15, before `bab2104` at 08:50). The certificate is instead
that the run **re-derives the frozen numbers from scratch inside itself**: B1 0.42673, B2 0.5728,
ΔAUPRC 0.14607, `matches_committed: true`. Independently, the three replicate seeds shared with the
earlier fly122 pre-lint run are **bit-identical across nodes on every compared field**
(ΔAUPRC 0.13811 / 0.11443 / 0.16276, plus B1/B2 AUPRC, both top-20 enrichments, ΔBrier, prevalence, `n_rows`,
`n_unique_components`, `positives`). fly122's 500-replicate run of the same seed prefix is still in flight
and will extend that replication from 3 seeds to 500.

**Cross-node determinism and lint-invariance of the procedure (measured).** The same script was run with
`--reps 3 --workers 3` on fly122 (from committed `bab2104`, pre-lint) and on fly90 (working tree with this
session's cosmetic lint edits). All 3 replicates agree **bit-for-bit on every scientific field**
(`delta_auprc` = **0.13811 / 0.11443 / 0.16276**, all three positive; plus B1/B2 AUPRC, top-20 enrichment
pair, Brier, prevalence, component counts); the only differing field is each replicate's own wall-clock
`seconds`. This
simultaneously establishes (i) cross-node determinism of the full-procedure bootstrap and (ii) that the lint
edits are numerically inert. The fly122 run also re-derives the reference sample exactly
(`matches_committed: true`, ΔAUPRC 0.14607 == committed 0.14607).

Rule respected: this is a **robustness statement about interval width**, reported alongside the frozen
numbers. It is *not* a gate and it did not re-open G1–G3. Two honesty notes about that status. First, the
artifact field `comparison_to_frozen_claim.gate_outcome_changed` is a **hard-coded structural constant**
(`False` at `scripts/v3_d001_fullproc_boot.py:236`), not a measurement — by the owner's directive this run
*cannot* change the gate, so quoting that field as a result would be circular. The informative statement is
the explicit **counterfactual, now computed: had the level-2 interval been the preregistered gating
statistic, G1 would have FAILED** (lower bound −0.02504 ≤ 0, vs the frozen +0.05096 > 0), while **G2 would
have passed unchanged and more strongly** (B2 top-20 enrichment lower bound 2.79858 > 1.0 and point
1000/1000 ≥ 1.75). G3 is untouched by the bootstrap — it is a per-seed-fold direction check, not an
interval. So the conjunctive GO would have become "enrichment-only GO": the ranking-accuracy arm is the one
that does not survive procedure uncertainty. Second, both in-flight runs loaded their code from
working trees that predate the lint edits (`bab2104` was committed 08:50 while fly90 launched 08:15; fly122
checked out `78cdc45` at 02:50 UTC, after its 02:35 UTC launch), so git alone cannot certify the exact blob
each run executed. That gap is closed empirically rather than asserted: each artifact re-derives the
non-resampled reference sample and reports `matches_committed` against the committed B1/B2 AUPRC, and the
500 shared replicate seeds can be compared byte-for-byte across nodes.

---

## A4 — Representation and provenance sanity

| Check | Verdict | Evidence |
|---|---|---|
| theta0 backbone is the frozen checkpoint | PASS | expected `34e6e630f564…640fe2`, frozen identically in the preregistration, `V3-D001.yaml` and `scripts/v3_d001_extract.py`; local `model.safetensors` (1,503,300,328 B) hashed at audit time and equal |
| reps artifact is the fly122 formal extraction | PASS | recomputed content hash `d0c6b7b211d7…9ecf4c` == committed; `formality = FORMAL`; device `NVIDIA GeForce RTX 3080`; formal git revision `41c5730` (the preregistration commit), i.e. extraction ran on frozen code |
| one representation per statement, computed once | PASS | 612 rows / 612 unique `statement_id`s for 612 statements appearing in the 720 groups (588 of them inside the 686 valid groups); keyed by statement only — not by seed, step or label; single `no_grad` forward per statement at the pre-registered layer indices |
| extraction path never reads labels | PASS | prompt text byte-identical across all copies of a statement (0 divergent); the 3 statements observed with **both** labels in the valid set have **bit-identical** representations; extractor inputs are `tokenizer(prompt) → bf16 forward → hidden_states[layer+1][-1]` |
| block 18 is pre-registered, not outcome-selected | PASS | primary = `int(2/3 × num_hidden_layers)`, robustness = `NL//3` and `NL−1`, asserted at extraction time (the script aborts on any mismatch); the gate code reads only `PRIMARY_LAYER`; blocks 9/27 (AUPRC 0.502 / 0.564 vs 0.573) are reported as robustness only and enter **no** gate |

Provenance completeness: preregistration + YAML + fold manifest committed **before** extraction (§18,
commit `41c5730`); §17 rollout-source hash manifest committed (`bafdc92`) with raw `rollout_data` never in
git; the audit's own numbers derive from the hashed reps artifact, not from a re-extraction.

---

## A5 — Classification

```
D001_CANONICAL_AUDIT:      reproduction PASS · A1 complete · LEAKAGE_AUDIT PASS (24/24) · A4 PASS (5/5)
                            A3 COMPLETE (1000/1000 replicates, committed)
D001_FINAL_STATUS:         CANONICAL_GO            (offline probe; gate GO G1^G2^G3 + calibration criteria met)
                            CANONICAL_GO with the A3 qualification below — the frozen gate is unchanged, but
                            the level-2 interval does not support one of the three arms at two-sided 95 %.
permitted_claim (owner-frozen wording, unchanged):
  "Frozen Kimina representations contain family-generalizable signal for predicting reward-informative RLVR
   groups beyond handcrafted difficulty features."
claim_qualification_from_A3 (added by this audit; narrowing, not relaxing):
  - robust under the full-procedure bootstrap: high top-20 informative-group enrichment of B2, relative to
    prevalence (1000/1000 replicates >= 1.75, worst replicate 2.4546).
  - NOT robust at two-sided 95 %: the "beyond handcrafted difficulty features" clause. Delta-AUPRC(B2-B1)
    has a level-2 interval of [-0.02504, 0.23679] (one-sided p = 0.048); the Delta-enrichment and Delta-Brier
    intervals also include 0. Direction is consistent in 95.2 % / 93.9 % / 94.3 % of replicates respectively.
  - proposed narrower wording, for the owner to accept or reject, not adopted here:
    "Frozen Kimina representations contain family-generalizable signal that identifies reward-informative
     RLVR groups at enrichment far above prevalence."
NOT permitted (unchanged): "controller improves RL" · "controller improves theorem proving" ·
                            "controller saves compute"   -> these require V3-R001 / V3-R002 evidence
launched_or_modified:      nothing. No RL, no rollouts, no fold/C/layer/label change, no V2 touch.
```

**What this audit changed: nothing about the frozen outcome.** It added (i) the complete per-fold cross-seed
table with the exclusion cost and the split of the G3 rule into its two conditions, (ii) an active
leakage-probe suite, (iii) a level-2 full-procedure bootstrap, which turned out to *weaken* one arm of the
frozen claim, and (iv) a provenance/sanity pass over the representation artifact. G1–G3, the GO, the
calibration claim, the fold manifest, the label definition and the primary layer are byte-identical to what
was reported on 2026-09-23.

### Residual weaknesses a reader should weigh (disclosed, not fixed by this audit)

- **Source concentration.** Informative groups are ~90% `synthetic` (prevalence 0.344 vs 0.030 / 0.022), so a
  B2-driven sampler is effectively a synthetic-oversampler. The probe shows the signal generalizes across
  *families*, not across *sources*.
- **G3 direction is not uniform.** 2/3 on AUPRC; −0.0157 on the seed-3 holdout (33 positives).
- **Small positive count, and the headline number is an optimistic draw.** 104 positives. The committed
  level-1 interval is a *prediction-level* interval and carries neither fold-partition nor C-selection
  variability. The level-2 A3 run (1000 replicates, committed) shows what that omitted: the ΔAUPRC interval
  **widens from [0.051, 0.241] to [−0.025, 0.237] and crosses zero**, and the bootstrap mean (0.10869) sits
  0.037 *below* the reported point estimate, with 71.2 % of replicates below it. The frozen G1 verdict is
  unchanged by directive; the *strength* of the ranking-accuracy evidence is weaker than the committed
  interval implies, and a reader should treat ΔAUPRC = 0.146 as a favourable draw rather than a central value.
- **Uncalibrated for deployment drift.** Everything is off-policy on V1 rollouts under a fixed theta0; the
  deployment controller's `step_norm` input is the only online quantity and the fitted `q(t)` surface is
  nearly flat in it.
- **No causal claim.** "Predicts informative groups" ≠ "increases informative groups when sampled more",
  because informativeness interacts with the policy's own learning trajectory. That is precisely the R001
  question, still untested.
