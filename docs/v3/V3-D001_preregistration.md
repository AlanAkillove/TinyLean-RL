# V3-D001 — Preregistration (FROZEN before any test outcome)

Experiment id: **V3-D001** (offline informative-group decision probe)
Namespace: `V3-D###` (offline probes); `V3-R###` reserved for post-GO RL intervention
Branch: `v3-jev-rl-controller`
Status: **PREREGISTERED / NOT YET RUN**
Prepared: 2026-09-23 · Owner directive V3-D001 §0–§21 (verbatim decisions encoded below)

> **This document is the frozen protocol.** Every threshold, feature set, layer, fold,
> seed, label, and gate below was fixed from the owner directive *before* any B2
> out-of-fold result was inspected. Nothing here may be changed after outcomes are
> seen. §20 step 14: **STOP after D001; no RL regardless of GO/NO-GO.**

---

## 0. Scope and hard constraints

- **No new RL training. No new rollout generation.** D001 reuses *only* the frozen
  V1 seed1/seed2/seed3 rollout data (`runs/*/rollout_data`) → frozen theta0
  representations → shallow decision heads → family-clean evaluation.
- **B3 tiny MLP is NOT AUTHORIZED** even if B2 GOs (§16). A GO does not by itself
  unlock anything: results return to the owner before any V3-R001 decision (§20 step 14).
- **Formal run host**: owner-designated **fly122 / RTX 3080 10GB** worker node. Any
  fly90 / RTX 3090 artifact is `NON-FORMAL` (feasibility smoke only); its timing/VRAM
  are not the formal compute result. Representation numerics may not be mixed across
  backbones/hashes.
- V2 is frozen and untouched: no modification of V2 frozen manifests/results, no
  restoration of A001 or the Track B allocator.

## 1. Task, labels, and censoring

- **Unit of observation** = one reconstructed n=8 GRPO group (a `theorem × step × seed`).
- **Primary label** `y_score = 1 iff 0 < Σ_j score_j < 8` (format-gated GRPO reward).
  This is the *only* primary label; `acc` may not replace it after seeing results.
- **Secondary sensitivity label** `y_acc = 1 iff 0 < Σ_j acc_j < 8` (verifier-only),
  reported only to ask whether the controller tracks the RL-reward frontier vs the
  solving frontier.
- **Strict complete-group policy** (§2): drop every group containing ≥1
  `# System Error:` candidate.
  `720 total − 34 infra-censored = 686 valid` primary groups.
  A *secondary* partial-identifiability count (censored groups whose observed cells
  already contain both 0 and 1) may be reported but never enters primary train/eval.
- Repeated theorems are **not merged**: each group is one Bernoulli observation;
  all copies of a statement share the same family fold; the bootstrap unit is the
  `family_component_id`. Representations are extracted once per unique prompt and cached.

## 2. Frozen identifiers (provenance)

| Item | Value |
|---|---|
| Rollout source manifest (hashes only, no raw) | `experiments/manifests/v3/v1_rollout_sources.json` (commit bafdc92) |
| theta0 weights sha256 | `34e6e630f564d330c79424c404ab0494558a0a659e6201b47d9bd88ccd640fe2` (rev `332e8a52`) |
| theta0 config sha256 | `f823a08cdadbe0e04ae184b6ea55a75bbabef8499151a170e90e01972d742ba5` |
| Promptset parquet sha256 | `68f33cee8d4d69479069117880be096fbfebb01af8c46cee0e54201fb28e1750` (rev `3009c548`) |
| Family-component registry sha256 | `f5074c962ca33975154291f675c50bc55aa481785ad9401e5f0b6bf1ac5a0024` |
| **Nested fold manifest** | `experiments/manifests/v3/v3_d001_folds.json` |
| **folds_hash** | `e0e0d30cfd81aed21d9bda91ae21be5700f644ddf77a198e7729d3fd93ede0d1` |
| valid groups / components | 686 / 433 |

## 3. Evaluation design — nested family-grouped CV (§3)

The original single 70/15/15 split is **cancelled** (a 15% family-clean test holds only
~21 positives). Primary evaluation is nested cross-validation:

- group unit = `family_component_id` (never statement id);
- **outer 5 folds / inner 4 folds**, seed **20260923**;
- implemented as a **deterministic numpy StratifiedGroupKFold equivalent** (sklearn is
  absent on fly122 and is not a project dependency) — quadratic-load balanced partition,
  rarest-class-largest-first, hard group disjointness; pinned by
  `tests/test_v3_d001_lib.py` (group disjointness, determinism, stratification, no
  statement split) and `tests/test_v3_d001_folds.py` (folds_hash reproducibility,
  rebuild-matches-frozen).
- Hard constraints, asserted at build time:
  `outer_train_components ∩ outer_test_components = ∅` and
  `inner_train_components ∩ inner_dev_components = ∅`.
- Folds are **read from the committed manifest** (component-keyed), not rebuilt by the
  fitting script, so fly90 and fly122 use the identical partition. **Folds are never
  changed after seeing outcomes.**

**Primary evaluation surface = the 686 out-of-fold (OOF) predictions**, each produced by
a model that never saw its own family. All headline metrics (§11) are computed on this
OOF vector.

## 4. Baseline ladder

### B0 — prevalence (§6)
Per outer fold, `p̂ = mean(y)` on that fold's outer-training rows; every outer-test row
gets the same `p̂`. This is the calibration / Brier null.

### B1 — handcrafted difficulty (§7)
A serious (not straw-man) baseline answering *"is informativeness just a simple
difficulty proxy?"* Frozen feature vector (statement-only; **no rollout/verifier/reward
signal**), extracted deterministically by `v3_d001_lib.extract_b1`:

```
formal_char_count, formal_line_count, formal_word_count,
n_paren, n_binders, n_commas, n_quant_syms, n_arrows,
has_existential, prompt_token_count, step_norm,
source=synthetic, source=autoformalizer, source=human
```

(`prompt_token_count` = exact theta0-tokenized prompt length, produced once in the
extraction pass so B1 and B2 share a single deterministic tokenizer source.)
Model: **L2 logistic regression**, **no** `class_weight` (calibration preserved);
regularization `C` chosen **only** on inner family-grouped CV (§6 criterion below).

### B2 — Jev-inspired frozen representation probe (PRIMARY, §8)
Frozen backbone `Kimina-Prover-Distill-0.6B theta0` (sha §2). **No backbone parameter is
updated.** Representation = **last non-padding-token hidden state of Transformer block 18**
(~2/3 depth), `h(x) ∈ R^1024`, concatenated with the normalized training step (deploy-time
controller knows the current step):

```
x_B2 = [ h_block18(x) , step_norm ]        # dim 1025
```

Model: **L2 logistic regression**; `C` chosen on inner family-grouped CV.
**Forbidden inputs** (leakage): seed id, reward history, verifier result, rollout output,
current group reward, any future-success information.

### B2 robustness (§9)
Two pre-registered secondary representations, **block 9** and **block 27**, identical
last-token pooling + step concat. Used *only* to ask whether signal depends on depth.
**Primary is always block 18**: no cherry-picking the best layer on outer-test, no
concatenating 9+18+27 and re-declaring the main method, no scanning all 28 layers.
Mean pooling is not in the formal main table.

## 5. Model fitting details (frozen)

- Standardization (z-score) is fit on the outer-training split only and applied to the
  held-out fold; recomputed inside every inner split (no leakage).
- L2 logistic solved by Newton/IRLS (`fit_logistic`, ridge on non-bias weights,
  `max_iter=50, tol=1e-7`, intercept unpenalized) — deterministic, dependency-free.
- **`C` selection criterion (frozen):** maximize the mean **inner AUPRC** across the
  4 inner folds of the outer-training partition; grid
  `C ∈ {0.003,0.01,0.03,0.1,0.3,1.0,3.0,10.0}`; ties break to the smaller (more
  regularized) `C`. Selected per outer fold; recorded in the results artifact.
- BLAS threads are capped (this build's multithreaded LAPACK solve is pathologically
  slow); this is an implementation performance choice, numerics unchanged.

## 6. Calibration (§10)

The raw logistic probability is the primary calibrated output. No isotonic / Platt /
temperature calibration is fit post hoc. Report **Brier**, **equal-mass ECE (10 bins)**,
and a **reliability curve**. A separate scalar-calibration follow-up may only be
preregistered later if B2 ranking is clearly effective but calibration is poor.

## 7. Metrics (§8/§11)

- **Primary ranking metric: AUPRC** (informative prevalence ≈ 15%).
- Secondary: **AUROC**, **Brier**, **ECE**.
- System metric: **top-k informative-group enrichment** at `k ∈ {10%, 20%, 30%}`,
  reported as both selected IGR and enrichment ratio `= selected IGR / overall prevalence`.
- Prevalence reported throughout.

## 8. Uncertainty (§12)

All formal CIs use **family-component bootstrap** (not group-row bootstrap),
**10 000 replicates**, fixed seed **20260923**, computed on the OOF predictions.
Paired component-bootstrap CIs are reported for
`ΔAUPRC(B2 − B1)`, `top-20 enrichment(B2)`, and `ΔBrier(B2 − B1)`.

## 9. Cross-seed generalization (§13, secondary robustness)

Three family-isolated transfers: `train {1,2}→test {3}`, `train {1,3}→test {2}`,
`train {2,3}→test {1}`. For each held-out seed the test set is that seed's valid groups;
every training group whose `component_id` overlaps a test component is **removed**
(no family crosses the seed boundary). Report point estimates + bootstrap CIs only;
per-fold statistical significance is *not* required (each fold has ~15–20 positives).

## 10. Label-stationarity diagnostics (§5, descriptive only — may NOT retune the model)

- **A. IGR vs training progress**: step bins 1–10, 11–20, …, 51–60, per seed and pooled.
- **B. repeated-theorem label consistency**: within-seed and cross-seed repeats →
  always-non-informative / always-informative / flips.
- **C. source × label**: prevalence for synthetic / autoformalizer / human (+ other).

These are entered into the memo to judge whether theorem-only signal could be confounded
by training step or source; they never change the primary representation or model.

## 11. GO / NO-GO gate (§14, FROZEN — the expensive-RL-intervention bar)

Judged **B2 block-18 vs B1** on the OOF surface. All three must hold:

- **G1** — `ΔAUPRC = AUPRC(B2) − AUPRC(B1)`: family-component bootstrap **95% CI lower
  bound > 0**.
- **G2** — top-20% B2 enrichment: point estimate **≥ 1.75×** pooled prevalence **and**
  bootstrap **95% CI lower bound > 1.0×**.
- **G3** — family-isolated cross-seed: in **at least 2 of 3** held-out-seed folds,
  `AUPRC(B2) > AUPRC(B1)` **and** `top-20 IGR > fold prevalence`.

**Any one failure ⇒ NO-GO** for expensive RL intervention. The gate may **not** be
rescued by switching layer, adding an MLP, or changing the label.

## 12. Calibration claim, judged separately (§15)

*"Useful RL sampler"* and *"calibrated controller"* are distinct. **Additionally**, if
`Brier(B2) ≤ Brier(B1)` **and** `ECE(B2) ≤ 0.10`, the result may be called a
**calibrated semantic controller**. If the ranking gate GOs but calibration fails, only a
**rank-based theorem sampler** claim is permitted (no probability-calibration claim).

## 13. Compute plan (§13/§19)

- **Representation extraction** (GPU, fly122): ~612 unique prompts, batch=1 forward,
  `output_hidden_states=True`; expected peak well under 10 GB (bf16 0.6B). Fresh 1–3
  prompt smoke on the 3080 re-establishes formal timing. Records host id + GPU + hashes.
- **Fitting / evaluation** (CPU, fly122): shallow logistic heads + CV + bootstrap.
- This is a *cheap* stage. If any future formal RL intervention cannot preserve the
  original logical recipe on 10 GB, **stop and report** — do not silently change n=8,
  effective batch, reward normalization, or any training semantics beyond theorem
  distribution.

## 14. Execution order (§20)

```
1 verify host → 2 commit V3-0 audit → 3 commit rollout-source manifest
→ 4 write protocol/manifest (THIS) → 5 freeze nested folds → 6 tests+lint
→ 7 commit preregistration → 8 extract theta0 reps (fly122) → 9 fit B0/B1/B2 nested CV
→ 10 component bootstrap → 11 cross-seed → 12 write memo
→ 13 classify GO/NO-GO by G1-G3 → 14 STOP (no RL either way; report owner)
```

## 15. Analysis surface (scripts)

- `scripts/v3_d001_lib.py` — records, deterministic grouped CV, L2 logistic, metrics, bootstrap, B1 features
- `scripts/v3_d001_folds.py` → `experiments/manifests/v3/v3_d001_folds.json` (frozen folds, §20 step 5)
- `scripts/v3_d001_extract.py` — theta0 block {9,18,27} last-token reps + token lengths (§20 step 8)
- `scripts/v3_d001_run.py` — nested CV fit/eval/bootstrap/cross-seed/gate → `runs/v3_d001/v3_d001_results.json` (§20 steps 9–13)
- `tests/test_v3_d001_lib.py`, `tests/test_v3_d001_folds.py`

## 16. Known risks carried from V3-0

- **R1 (power):** mitigated by using all 110 informative positives on the OOF surface
  rather than a ~21-positive single split; component-bootstrap CIs quantify residual width.
- **R2 (difficulty confound):** all-fail dominance (83.75%) makes informativeness ≈
  capability-frontier difficulty. This is exactly why the gate is **B2 vs the handcrafted
  B1 baseline (enrichment over difficulty), not raw accuracy**: B2 must beat B1, or the
  "semantic controller beyond difficulty" hypothesis fails.
- **R7 (split coarseness):** fixed by the committed `folds_hash`; never re-drawn.
