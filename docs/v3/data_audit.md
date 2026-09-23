# V3-0 — Data audit: is reward-informativeness predictable from theorem semantics?

**Status:** Stage V3-0 audit **ACCEPTED by owner (2026-09-23)**; the five pending decisions are now locked (§15). No formal V3-D001 split is committed **yet**; no probe is trained, no hidden states are extracted beyond the (non-formal, fly90/3090) 3-prompt memory smoke, no RL is launched.
**Branch:** `v3-jev-rl-controller`. V2 stays frozen at `7e99a01` (`v2-research-freeze-20260923`); nothing here mutates any V2 manifest/result.
**Formal V3-D001 node:** **fly122 / RTX 3080 10 GB** (see §0.1 host record).
**Reproduce:** `scripts/v3_data_audit.py` → `docs/v3/data_audit_stats.json`; feasibility smoke `scripts/v3_theta0_smoke.py`. All read-only over existing seed1/2/3 dumps.

V3 research question (unchanged from the plan):
> Can a lightweight Jev-inspired calibrated decision model predict which Lean theorems yield **reward-informative** GRPO groups, and can that signal later improve RLVR sample efficiency?

Stage V3-0 asks only the *prior* question: **does the signal exist at all** in the existing offline data, at near-zero cost. If it does not, V3 stops before any expensive intervention.

---

## 0. Repo state & host identity (formal-node record)

- Working branch `v3-jev-rl-controller`, tree clean at checkout, same commits as the frozen `worker/v2-rl` head (`7e99a01`).
- V2 closeout (`docs/v2/v2_closeout.md`): trainability reproducible; **IGR ≈ 0.158 / 0.150 / 0.150 (seed1/2/3)** is a stable phenomenon; held-out gain small & seed-sensitive; Track B closed (B004 predictor stopped by preregistered gate); **family-component split is mandatory** for any supervised eval (exact-statement split underestimates family leakage).
- New V3 files created this round (all under `docs/v3/` and `scripts/v3_*`, none are V2 artifacts): `data_audit.md`, `data_audit_stats.json`, `v3_data_audit.py`, `v3_theta0_smoke.py`.

### 0.1 Host identity — the hidden-state smoke ran on fly90, NOT the formal node

Per the dual-server canonical role table (`docs/dual_server_collaboration.md` §0/§2), the `V3-D001` **formal** run must execute on **fly122** (evaluation worker). Host verification:

| node | role | hostname | address | GPU | VRAM |
| --- | --- | --- | --- | --- | --- |
| fly90 (this shell) | producer / canonical analysis | `fly` | 10.3.25.90 | RTX 3090 | 24 GB |
| **fly122 (formal V3-D001)** | evaluation worker | `ubuntu` | 10.3.25.122 | **RTX 3080** | **10 GB** |

- The §5 feasibility smoke was executed on **fly90 / RTX 3090 24 GB** (GPU-485e894c…). It is therefore recorded as a **NON-FORMAL feasibility smoke**: its timing/VRAM numbers establish *architecture feasibility only* and are **not** used as the formal V3 compute result.
- fly122 confirmed reachable via `ssh fly@10.3.25.122`: repo at `/home/fly/ZJQ/TinyLean-RL`, branch `v3-jev-rl-controller` @ `7e99a01`, working tree clean; theta0 weights present; all three seed rollout dirs present (60 files each); `torch 2.7.0+cu126` CUDA active. Spot sha256 of rollout files is **byte-identical fly90↔fly122** (provenance match).
- **`sklearn` is absent on fly122**, so V3-D001 uses a **deterministic StratifiedGroupKFold equivalent** implemented in-repo (owner §3) rather than mutating the worker environment; representation numerics are **never mixed across backbones/hashes** — a formal 1–3 prompt smoke is re-run on fly122/3080 before extraction.

---

## 1. Historical dataset (rollout dumps)

Three full GRPO runs, identical recipe (DrGRPO full-FT, `rollout.n=8`, `train_batch_size=4` prompts/step, temp 1.0/top_p 1.0, lr 2e-6, no KL, 60 steps, `max_prompt_length=1024`, `max_response_length=4096`).

| seed | run dir | `data.seed` | steps | files | candidate records | groups (n=8) |
| --- | --- | --- | --- | --- | --- | --- |
| seed1 | `runs/p3b_pilot/rollout_data` | default (1); **two-run stitched** E017 (1–30) + E019 (31–60) | 1–60 | 60 | 1,920 | 240 |
| seed2 | `runs/m1_seed2/rollout_data` | 20260918 | 1–60 | 60 | 1,920 | 240 |
| seed3 | `runs/m1_seed3/rollout_data` | 20260919 | 1–60 | 60 | 1,920 | 240 |
| **pooled** | | | | **180** | **5,760** | **720** |

Excluded from the study: `runs/m2_qwen_smoke` (Qwen3-Base, 5 steps, not a Kimina training seed) and `.cache/m1_seed3_r2r3_dumps/*` (aborted seed3 attempts r2/r3, same seed/statements, different RNG stream — **must never be merged** with `runs/m1_seed3`).

**Q1 — theorems/groups per seed:** each seed = 240 groups over 227 / 229 / 227 **distinct theorems** (seed1/2/3) respectively (some theorems repeat across steps — §4).

**Q2 — are n candidate rewards fully saved?** Yes. Every one of the 5,760 records carries exactly the 10-key VERL dump schema `{input, output, gts, score, step, pred, acc, tool_feedback, response, format_error}`. Group reconstruction is by **contiguity + byte-identical `input`** (there is no stored group id). Verification: **all 720 groups have exactly size 8; 0 irregular blocks; 0 unparsable lines** across all 180 files. `reward.py` returns `score = proof_rw * format_rw`; advantages are computed in-memory only and not dumped.

**Q7/Q8 — rewards never missing.** `score` is strictly binary — observed values only `{0.0, 1.0}` (460 ones / 5,300 zeros pooled). No NaN, no absent reward, no fractional credit. So the group reward-sum is a clean integer in 0..8.

---

## 2. Label reconstruction (informative / all-fail / all-success)

Group label from the summed binary score over n=8, exactly per the plan:

```
all_fail    : sum = 0
informative : 0 < sum < 8        (nonzero relative-advantage signal exists)
all_success : sum = 8
```

**Q3 — precisely recoverable. Q6 — base rates:**

| seed | all-fail | informative | all-success | **informative rate** |
| --- | --- | --- | --- | --- |
| seed1 | 201 (83.75%) | 38 | 1 | **0.158** |
| seed2 | 201 (83.75%) | 36 | 3 | **0.150** |
| seed3 | 201 (83.75%) | 36 | 3 | **0.150** |
| **pooled (720)** | **603 (83.75%)** | **110** | **7** | **0.153** |

This reproduces the published V1 IGR (0.158/0.150/0.150) — independent confirmation the reconstruction is faithful.

**The single most important framing fact for V3:** the reward-degenerate mass is **almost entirely all-fail (83.75%), not all-success (0.97%)**. So the binary task "informative vs not" is effectively **"is this theorem near the 0.6B model's solvability frontier (sometimes provable in 8 samples) vs beyond it (never provable)."** It is a *capability-edge / difficulty* signal — coherent and genuinely useful for a sampler, but it means:

- The **B1 handcrafted-difficulty baseline is a serious competitor, not a strawman.** Beating prevalence is trivial; the honest bar is beating B1.
- The **exploratory 3-class is not viable as a classifier target**: `all_success` has only 7 pooled positives (3-class §11).

**Label source (score vs acc) — a real design fork.** `acc` is the verifier-only reward (ignores the format gate); `score` is the actual GRPO reward. They disagree on **13 / 720 groups** (group-level) and **36 groups** differ in raw sum. Candidate-level: 460 `score=1` vs 523 `acc=1` (63 proofs verified but format-gated to 0). Decision: **primary label = `score`** (that is what produces in-group advantage in the RL loop and what a sampler would actually act on); report `acc`-based as a robustness re-run. Not yet frozen — owner confirm.

---

## 3. Anomalies and exclusion policy

There is **no boolean error flag**; infra failures are string-matched under the `tool_feedback` prefix `"# System Error:"`.

| anomaly | seed1 | seed2 | seed3 | pooled |
| --- | --- | --- | --- | --- |
| system-error candidates (timeout / retry-exhausted) | 26 | 14 | 63 | **103** (1.8%) |
| groups containing ≥1 system-error candidate | 9 | 6 | 19 | **34** |
| groups that are *entirely* system-error | 0 | 0 | 1 | 1 |
| irregular / non-size-8 groups | 0 | 0 | 0 | 0 |

**Exclusion rule (proposed, fail-closed, follows Track B precedent `v2_b002_infra_adjudication.py`, `docs/v2/b0_verifier_reliability.md`): a system error is *censoring*, not a true 0-reward.** Therefore **any group with ≥1 system-error candidate is dropped from the supervised label set** (not silently converted to all-fail). Sensitivity already computed: dropping those 34 groups moves pooled informative rate **0.153 → 0.152** and removes ~2 positives per seed — negligible but principled. Final scope after censoring: **686 groups**.

Not anomalies (kept as genuine negatives): format-gate failures. `format_error ≠ "No error."` on 3,123 candidates, dominated by **"Could not find a valid lean4 code block" (3,123)** — i.e. the model emitted no parseable proof at temp 1.0. These are legitimate 0-rewards and are the *substance* of the all-fail class, not noise. (`Tactics/Lean4 do not match` 260, `too many lines` 183, etc.)

**Truncation is unrecoverable in `rollout_data`.** `output`/`response` are `skip_special_tokens=True` decodes, so `max_response_length=4096` cutoff info is destroyed (0 `"Generation over length limit."` observed). Consequence: **no length-conditioned label** can be built from these dumps; if a length feature is ever wanted it must come from a fresh (cheap, prompt-only) theta0 pass, not from the rollout.

---

## 4. Theorem identity, recurrence, and family mapping

**Identity chain (reused verbatim from repo analyzers):** `FORMAL_BLOCK_RE` on `input` → `normalize` → join to Promptset parquet (`data/raw/kimina_promptset/…/train-00000-of-00001.parquet`, `statement_id`) → invert `experiments/manifests/v2/family_component_registry.json` (`component_id`). The dumps carry **no theorem id** — only this text join, which is exact here.

**Q9 — coverage is 100%:** `groups_unmapped_to_statement_id = 0`, `groups_unmapped_to_component_id = 0`. Pooled: **612 distinct `statement_id` → 443 distinct `component_id`** (L3 source-family ∪ L2 skeleton ∪ L4 identical-NL connected components — the V2-frozen §5.3 definition). The hard family-isolation requirement is fully satisfiable on this data.

**Q4 — cross-step recurrence within a seed (real paired observations, non-independent):**

| seed | distinct theorems | theorems in >1 step | step-count histogram |
| --- | --- | --- | --- |
| seed1 | 227 | 13 | {1:214, 2:13} |
| seed2 | 229 | 10 | {1:219, 2:9, 3:1} |
| seed3 | 227 | 13 | {1:214, 2:13} |

**Q5 — cross-seed recurrence.** Exact-statement overlap is modest; **family overlap is large**:

| pair | shared `statement_id` | shared `component` |
| --- | --- | --- |
| seed1∩seed2 | 27 | 66 |
| seed1∩seed3 | 17 | 47 |
| seed2∩seed3 | 32 | 69 |
| all three | 5 | 25 |

**132 of the 443 components appear in more than one seed.** This is decisive for the §10 cross-seed test: it can *not* be a naive "train seeds 1+2, test seed 3" split — the seed-3 families leak into training. The cross-seed generalization test must impose family isolation *simultaneously* with the seed holdout (§10 quantifies the resulting collapse).

Component group-size distribution (over the 443): **287 singletons, 90 with 2 groups, 36 with 3, 15 with 4, 8 with 5, 5 with 6, 1 with 7, 1 with 8.** So 65% of families contribute exactly one labeled group → the *effective independent sample is small* (see power, §10).

---

## 5. theta0 representation — architecture & feasibility (B2)

- **theta0 = `models/weights/kimina_distill_0_6b`** (AI-MO/Kimina-Prover-Distill-0.6B, revision `332e8a52…`), architecture `Qwen3ForCausalLM`: **28 layers, hidden_size 1024, intermediate 3072, 16 heads (head_dim 128), 8 KV heads, vocab 151,936, bf16, tie_word_embeddings.**
- **Controller-time input = the exact chat-rendered prompt `input`** (system + user + `# Problem` NL + `# Formal Statement` fenced lean4 + `assistant`). It is available *before* generation, contains **no rollout reward, no verifier outcome, no rollout-history feature** → leakage-free by construction, and is byte-identical to what theta0 saw during RL.
- We embed **one vector per unique theorem prompt** (a theorem's prompt is identical across its re-samples and across seeds) → **612 forwards total**, not 5,760.

**Feasibility smoke — NON-FORMAL (fly90 / RTX 3090 24 GB; architecture-feasibility only, not a formal V3 compute number; `scripts/v3_theta0_smoke.py`):**

| quantity (fly90 3090, non-formal) | value |
| --- | --- |
| model load time / peak VRAM (bf16) | 10.6 s / **1.50 GB** |
| forward peak VRAM (214-token prompt, `output_hidden_states`) | **1.50 GB** (weights dominate; activations negligible) |
| 3-prompt forward wall time | 1.52 s (~0.5 s/prompt incl. overhead) |
| pooled vector (last non-pad token) per layer | dim **1024**, fp32 |

Even at `max_prompt_length=1024` and batch 8, activations stay far under 10 GB. **612 extractions ≈ a few minutes, ~1.5 GB.** B2 compute is *not* a bottleneck; the **formal** extraction runs on fly122/RTX 3080 10 GB and will re-measure timing/VRAM there (the fly90 3090 numbers above are non-formal). (Full extraction deliberately NOT run this round, per owner constraint.)

**Representation choice (frozen by owner for V3-D001, §8/§9):** **primary = block 18** (≈2/3 of the 28 Qwen3 blocks), **last non-padding-token** pooling → `h(x) ∈ R^1024`, concatenated with the **normalized training step** (∈[0,1]; a deploy-time controller knows the current step). **Secondary robustness reps = block 9 and block 27** (same last-token pooling), used only to test depth-dependence — **never concatenated for the headline, never a 28-layer sweep.** Pooling is fixed to last-token; mean-pooling is reserved as clearly-exploratory only. Backbone frozen, no parameter update.

---

## 6. Proposed supervised task (V3-1)

- **Unit of analysis:** a *group* = one n=8 rollout of one theorem at one step (686 after censoring). Repeated groups of the same theorem are *replicate* observations of that theorem's informativeness, not independent points.
- **Input:** theta0 frozen representation of the theorem prompt (§5). No rollout/verifier/history features.
- **Primary target (binary):** `informative = (0 < sum(score) < 8)` vs `not`. Positive rate ≈ 0.15 → imbalanced; ranking + calibration metrics, not accuracy.
- **Exploratory 3-class (all-fail / mixed / all-success):** report *descriptively only*. With 7 pooled all-success positives it is **not a viable learning target** and cannot enter the gate.

**Split rule (HARD): split on `component_id`** (not statement, not record). All groups of every member theorem of a component move together to exactly one side. train∩dev = train∩test = dev∩test = ∅ at component level. A deterministic sha256-of-`component_id` hash split (seed fixed, recorded) is proposed; the actual assignment is **not** committed until the owner freezes the GO/NO-GO.

Illustrative 70/15/15-by-component partition (numbers only to size the problem, **not** the frozen split):

| split | components | groups | all-fail | informative | all-success | inf. rate |
| --- | --- | --- | --- | --- | --- | --- |
| train | 310 | 504 | 425 | 73 | 6 | 0.145 |
| dev | 66 | 111 | 95 | 16 | 0 | 0.144 |
| test | 67 | 105 | 83 | **21** | 1 | 0.200 |

> **Note (owner decision, supersedes this illustrative split):** the single 70/15/15 split is **not** frozen. V3-D001 primary evaluation uses **nested family-grouped cross-validation** (outer 5-fold / inner 4-fold, `StratifiedGroupKFold`-equivalent on `component_id`, seed 20260923) producing **686 out-of-fold predictions** as the evaluation surface. The table above is retained only to size the positives (≈20/15% fold). See `docs/v3/V3-D001_preregistration.md`.

---

## 7. Baseline ladder (no complex model first)

- **B0 prevalence:** constant `p = base rate` (report as calibration floor + the enrichment reference).
- **B1 handcrafted logistic regression (the real bar):** token length of prompt/statement, `#` of hypotheses & binders, count of `theorem/lemma` decls, presence of `∃`/`∀`/`↔`, arithmetic vs algebra vs number-theory token in the name, source ∈ {synthetic, autoformalizer, human}, Mathlib tactic-complexity proxies (from the *statement*, never the rollout). Answers: *is informativeness just a cheap difficulty proxy?* B2 must beat B1 on family-clean test, else the frozen representation buys nothing.
- **B2 Jev-inspired frozen-representation probe (main method):** standardized **block-18 last-token `h(x) ∈ R^1024`** + normalized step → **L2-regularized logistic regression** (primary; `C` chosen in inner grouped CV, **no `class_weight=balanced`** to preserve calibration). Backbone frozen. Secondary reps block 9 / block 27 for depth-robustness only.
- **B3 tiny MLP:** only if B2 already clears B0/B1 family-clean, else skipped.

---

## 8. Metrics (calibration is the point, not just label)

- **AUPRC** — primary ranking metric under 15% imbalance.
- **AUROC** — secondary.
- **Brier score**, **reliability curve**, **ECE** (equal-mass bins) — calibration.
- **Informative-group enrichment** (the decisive system metric): sort the family-clean test by `P(informative)`; report the actual informative rate in the **top-10 / 20 / 30%** vs random prevalence (≈0.153). A controller is only valuable if, e.g., overall ≈15% → top-ranked subset ≈30%+. Everything downstream (RL sampler) hinges on this, not on accuracy.

All CIs via **family-cluster bootstrap** (resample `component_id`, 10k, seed fixed) — never record-level — consistent with `docs/v2/family_leakage_audit.md` §5.1 and `v2_a001_analyze.py`.

---

## 9. Cross-seed generalization (the core experiment)

Question: does the predictor learn a *theorem-level* informativeness that transfers to an unseen training seed, or the reward noise of a particular seed? The honest version must hold out a seed **and** isolate families. Under split-train-two-seeds / test-one-seed with component isolation applied on **both** sides:

| held-out seed | family-clean test comps | test groups | **test positives** | train comps / groups / pos |
| --- | --- | --- | --- | --- |
| seed1 | 114 | 125 | **20** | 241 / 331 / 52 |
| seed2 | 95 | 99 | **15** | 238 / 297 / 40 |
| seed3 | 102 | 117 | **15** | 250 / 322 / 50 |

So a family-isolated cross-seed test has only **15–20 positive informative groups**. This is the binding constraint of the whole stage (§10).

---

## 10. Statistical power (drives an honest, non-soft GO/NO-GO)

- Family-clean pooled test ≈ **21 positives** (dev 16). 3-fold family-isolated seed-holdout ≈ **15–20 positives** each.
- At prevalence 0.15, n≈140 groups → the base rate's own Wilson 95% CI is roughly **(0.10, 0.22)** — the *prevalence itself* is only known to ±5 points. AUPRC and top-k enrichment computed on ~20 positives inherit similarly wide CIs; a top-20% slice holds ~21–25 groups ⇒ single-digit positives ⇒ Poisson-level noise.
- Consequences: (a) we **cannot** honestly certify a *small* enrichment; only a *large* one (top-ranked informative rate ≥ ~2× prevalence) is separable from noise here; (b) any CI we report must be the family-cluster bootstrap width, and the GO rule must be phrased on **bootstrap lower bounds**, not point estimates.

---

## 11. Proposed GO/NO-GO (to freeze *before* viewing test — awaiting owner)

> **Note (owner decision 2026-09-23):** the operative gate is the owner-frozen **G1–G3** in the V3-D001 preregistration (§14 of the owner directive): ΔAUPRC(B2−B1) bootstrap CI lower > 0; B2 top-20% enrichment ≥1.75× prevalence with CI lower > 1.0×; ≥2/3 family-isolated cross-seed folds where AUPRC(B2)>AUPRC(B1) and top-20 IGR > fold prevalence. Calibration is judged **separately**. The draft below is kept for provenance only.

Draft, deliberately not tuned to pass. Compute on the family-clean test with family-cluster bootstrap; **all must hold**:

1. **Beats prevalence:** test AUPRC bootstrap CI lower bound > B0 prevalence baseline (AUPRC of a random ranker at that prevalence).
2. **Beats difficulty proxy:** model test AUPRC **and** top-20% enrichment each exceed B1's, with the paired-by-component bootstrap CI of the difference excluding 0 — otherwise the frozen LM adds nothing over handcrafted difficulty.
3. **Real enrichment:** top-20% family-clean informative rate ≥ **1.75×** prevalence (≈0.27) with bootstrap CI excluding 1.0×.
4. **Calibration not broken:** ECE ≤ 0.10 **and** monotone reliability ordering on the top half (a controller emits probabilities, not just a label).
5. **Cross-seed transfer:** at least **2 of 3** family-isolated seed-holdouts independently satisfy (1) with a positive top-20% lift.

**GO** → design *one* pre-registered RL sampling-policy intervention (§13). **NO-GO** (any miss, or effect only on train/dev) → V3 stops before expensive RL, per the plan's own fail-early principle.

---

## 12. Compute estimate (Stage V3-D001)

All cheap; formal run on **fly122 / RTX 3080 10 GB**; no RL.

| step | cost |
| --- | --- |
| theta0 hidden extraction (612 unique prompts; block 18 primary + 9/27 robustness, last-token) | ~minutes, ≪10 GB (formal re-measured on fly122) |
| B0/B1 CPU logistic + 10k family-component bootstrap | CPU, seconds–minutes |
| B2 L2-logistic on standardized block-18 h(x)∈R^1024 (+step) × 686, nested grouped CV + bootstrap | CPU, minutes |
| calibration / enrichment / seed-holdout analysis | CPU, minutes |

No new rollouts, no new RL training, no 20h+ job. Consistent with owner's "第一阶段应该非常便宜."

---

## 13. Next phase only if offline GO (out of scope this round)

Single pre-registered RL comparison — Uniform vs **decision-guided theorem sampling**, everything else frozen: same theta0, same GRPO/DrGRPO, `n=8`, same optimizer / effective batch / reward / max tokens / steps / budget; the *only* intervention is the sampling policy over theorem prompts. Metrics: informative-group ratio, nonzero-advantage groups, useful-groups/token, useful-groups/GPU-hour, reward dynamics, gradient/param drift, family-clean held-out proving. No unbounded sweep. If 10GB cannot preserve the original logical recipe → **stop and report**, never silently mutate `n=8` / batch / reward-normalization / training semantics beyond the theorem distribution.

---

## 14. Open risks (for owner)

- **R1 (severity: high) — power.** 21 family-clean test positives (15–20 per seed-holdout). Only large effects are detectable; mitigation = preregistered enrichment-focused rule + bootstrap-CI language, and a documented option to pool dev into test only under a *frozen* rule if the owner prefers power over a held dev.
- **R2 (severity: high) — informativeness ≈ difficulty.** All-fail dominance (83.75%) makes the task close to a learnability-frontier estimator; B1 is a strong competitor and must be beaten, or the contribution is "a cheap difficulty proxy," not a semantic controller.
- **R3 (severity: medium) — label fork score vs acc** (13 group disagreements). Freeze `score` (primary) before test; `acc` as declared robustness.
- **R4 (severity: medium) — seed1 provenance.** stitched E017+E019, `data.seed` never explicit, not re-runnable. Keep in pooled/train; exclude from any reproducibility-critical confirmatory claim.
- **R5 (severity: low) — host-only data.** `runs/` and `experiments/results/` are gitignored (data lives only on this host). Recommend a one-off sha256 hash manifest of the three dump dirs before any further work (a V3 artifact, not a V2 change).
- **R6 (severity: low) — no length signal** in dumps (truncation destroyed). If curriculum wants length conditioning it needs a fresh prompt-only pass, not the rollouts.
- **R7 — family split is coarse.** 287/443 singleton components; random 15% test can swing the observed informative rate (0.145 train vs 0.200 test). Fix the split by recorded hash *before* viewing outcomes and never re-draw.

---

## 15. Recommendation & owner-decision log

**Advance to Stage V3-D001, scoped as a *power-bounded* offline probe with the owner-frozen enrichment-focused gate.** The enabling facts are real: labels reconstruct exactly and reproduce published IGR; family mapping is 100% complete and every group is size-8; theta0 representation extraction is verified feasible (architecture-feasible on fly90; formal on fly122) and leakage-free by construction. The two genuine threats are **statistical power (R1)** and the **difficulty-proxy confound (R2)** — which is why the ladder starts at B1 and the gate is stated as family-clean *enrichment over the handcrafted baseline*, not accuracy. If B2 cannot beat B1 within these CIs, the correct outcome is a clean **NO-GO** before any RL.

**Owner decisions (2026-09-23) — all five locked; V3-0 audit ACCEPTED:**
1. ✅ **Primary label = `score`** (`y_score=1 iff 0<Σ_j score_j<8`); `acc` only as preregistered secondary sensitivity; never swap primary on the basis of `acc`.
2. ✅ **Strict complete-group infra policy:** `720 − 34 (any "# System Error:") = 686` primary valid groups; a group with ≥1 censored candidate is excluded entirely; a secondary *partial-identifiability* audit (mixed-observed censored groups are mathematically informative) may be reported but never enters primary fit/eval.
3. ✅ **Single 70/15/15 split cancelled** → **nested family-grouped CV** (outer 5 / inner 4, `component_id` groups, seed **20260923**), 686 OOF predictions as the primary surface; folds frozen before results.
4. ✅ **Representation:** primary **block 18** last-token `h(x)∈R^1024` + normalized step; L2-logistic, no `class_weight`, `C` via inner grouped CV; **block 9 / 27** are declared robustness only; B3 MLP **NOT authorized**.
5. ✅ **Cross-seed** = family-isolated 3 holdout folds (§13); GO/NO-GO = owner-frozen **G1–G3** (§11 note); calibration claim judged separately (§15 of owner directive).

Formal V3-D001 executes on **fly122 / RTX 3080**; the fly90 3090 smoke is retained but marked **non-formal**.
