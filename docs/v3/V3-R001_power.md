# V3-R001 §14 — what this project can actually power, computed before the gate was written

Artifact: `experiments/manifests/v3/V3-R001_power.json`
Producer: `scripts/v3_r001_power.py` (read-only; no rollout, no model, no GPU)
Tests: `tests/test_v3_r001_power.py` — the exact power curve is cross-checked against a 60k-replicate
Monte Carlo and the critical values against `scipy.stats.fisher_exact` on every possible total.
Authority: owner directive 2026-09-24 §14 and §17 step 5.

Owner §14 asked for the power calculation *first*, and for the result to decide whether the GO gate
may require B2 > B1 significance. Both questions now have answers, and one of them is unwelcome.

---

## 1. The gate can be exact, and that is a consequence of §7's sampling unit

Because R001 draws **one theorem per family component**, the analyzed units are independent, and the
§9 statistic — how many informative groups land in the top-f block of the frozen `q` ranking — is a
two-cell comparison. Conditional on `k` informative theorems among `N`, the count inside a
block of size `m` is Hypergeometric(*N*, *k*, *m*) under H0, so the α=0.05 rejection boundary is
exact and needs no bootstrap. Power is then a finite convolution of two Binomials, computed
closed-form rather than simulated. (The family-component bootstrap remains the reporting device for
the interval; it is not needed for the test.) If the owner instead approves more than one theorem per
family, this exactness is lost and the interval must be component-clustered.

An effect size is only coherent while the block can hold it without emptying the rest of the pool:
`f·E·π ≤ π`, i.e. **E ≤ 1/f**. Enrichment 3 in the top half is not a large effect, it is an
impossibility, and the script refuses to report a power for it.

## 2. The prevalence anchors, and why they are the real constraint

Everything here is measured on frozen artifacts, not assumed:

| quantity | value | source |
|---|---|---|
| group prevalence in V1 (informative iff 0 < Σscore < 8) | 0.1516 | `V3-D001_results.json#dataset` |
| **theorem-level** informative rate (unweighted over the 588 labelled statements) | 0.1709 | recomputed from `build_records` |
| infra-censoring rate (a group that must stay missing, §8) | 0.0472 (34/720) | V3-D001 audit |
| per-source prevalence: synthetic / autoformalizer / human | 0.3444 / 0.0304 / 0.0215 | `stationarity.source_prevalence` |
| B2 top10 / top20 / top30 enrichment (family-held-out OOF) | 3.824 / **3.467** / 2.818 | `B2_block18_PRIMARY` |
| B1 top10 / top20 / top30 enrichment | 3.155 / 2.937 / 2.690 | `B1_handcrafted` |
| AUPRC B2 / B1 / Δ | 0.5728 / 0.4267 / +0.1461 | same |

The source split is the crux. Informative groups in V1 were almost entirely a **synthetic-source**
phenomenon. R001's clean pool has a different mix, so the pool's prevalence is not V1's:

| reading | pool capacity | source mix (auto / human / synth) | predicted prevalence |
|---|---|---|---|
| `owner_literal_wider_v1` | 56 | 28 / 17 / 11 | **0.089** |
| `owner_literal` | 86 | 44 / 24 / 18 | 0.094 |
| `consumed_only` | 221 | 101 / 42 / 78 | 0.140 |

(The prediction applies V1's per-source rates to the pool's mix. Its bias is stated rather than
hidden: V1's rates were measured on theorems it chose to re-draw up to 54 times, so they may not
transfer. This is also why owner §11's source-concentration audit is not a robustness appendix —
it is the same question as the power question.)

## 3. Minimum detectable enrichment at 80% power (exact; top-20% block)

`UNREACH` (defined in the artifact, used in the appendix blocks) = even the maximum *coherent*
enrichment (every positive inside the block) fails to reach 80% power at that prevalence. No top-20%
cell in this grid is UNREACH; four of the top-30/top-50 cells at π ≤ 0.09 are, and their exact
failure is quoted in `min_detectable_enrichment_at_80pct_power`.

| pool N | π=0.06 | π=0.09 | π=0.14 | π=0.17 |
|---|---|---|---|---|
| 56 | 4.66 | **3.83** | 3.12 | 2.85 |
| 86 | 4.02 | 3.31 | 2.69 | 2.48 |
| 128 | 3.37 | 2.79 | 2.33 | 2.15 |
| 192 | 2.89 | 2.45 | 2.08 | 1.94 |
| 221 | 2.72 | 2.32 | 1.98 | 1.86 |

Read against D001's own measured B2 top-20% enrichment of **3.47**:

| reading | expected positives in the selected block | power at E=3.47 | N needed for 80% power | verdict |
|---|---|---|---|---|
| `owner_literal_wider_v1` (56, π 0.089) | 0.98 | **0.69** | 76 | **not usable as a test** — the pool is smaller than the sample the effect needs |
| `owner_literal` (86, π 0.094) | 1.59 | 0.86 | 71 | usable, but with no margin: it can only detect an effect as large as the one already observed |
| `consumed_only` (221, π 0.140) | 6.14 | ~1.00 | 39 | comfortably usable; the sub-sample N=128 still detects enrichment ≥ 2.33 |

So the strict §5 reading does not merely shrink the experiment, it changes its epistemic status: at
56 families the top-20% block is expected to contain **one** informative theorem, and no property of
the controller can turn that into a test. The preregistration must say which of the two it is,
rather than dressing a 56-theorem sample as a confirmatory trial.

## 4. The B2-vs-B1 requirement is unpowered, and §14 makes that decisive

D001 measured the *same* comparison at 433 family components: full-procedure mean ΔAUPRC +0.109,
95% CI [−0.025, +0.237] — already inconclusive. A component-level bootstrap interval scales as
1/√n_components, so at R001's whole-pool sizes the half-width would be:

| N | approximate ΔAUPRC half-width | would D001's point delta clear zero? |
|---|---|---|
| 56 | 0.364 | no |
| 86 | 0.294 | no |
| 128 | 0.241 | no |
| 192 | 0.197 | no |
| 221 | 0.183 | no |

This is an approximation (it assumes comparable between-component variance), and it is used only to
answer a *no-go-direction* question, where a conservative approximation is enough: no feasible R001
pool can reach significance on B2 > B1. §14 explicitly leaves that requirement to be "decided by
power, not forced", so **the gate does not require it**. B1 stays exactly what §10 asks for — a
zero-extra-GPU comparator on the identical sample, reported with its interval and with the honest
reading that the interval covers zero.

## 5. Two warnings the numbers come with

**Discreteness.** The exact test's realized size is *not* 0.05 at these sample sizes; it is the
largest reachable level below it. Across the whole grid it runs 0.005–0.034, and in the top-20%
column used above 0.018–0.033, reported per cell as
`realized_size_of_the_exact_test_under_H0`. Power is therefore a step function of (N, m) and can be
non-monotone in N. A gate that says only "p < 0.05" without naming N and the block fraction is
under-specified, and the first version of this script produced exactly that non-monotonicity as a
bug report — it classified the null hypothesis (E = 1) as "infeasible" on a floating-point tie and
then reported ~100% power for effects that do not exist. `test_the_null_hypothesis_is_always_a_coherent_effect_size`
now pins that down.

**The censoring tax.** §8 keeps infra-censored groups missing rather than counting them as all-fail,
which costs about 4.7% of the sample: N=128 analyzes as ~122, N=56 as ~53. The grid reports
`expected_analyzable_N_after_censoring` per cell.

## 6. What this recommends, and what is not mine to decide

The power analysis points at **N=128 drawn from the `consumed_only` pool (221 clean families)**: 80%
power against enrichment ≥ 2.33 at the predicted prevalence, a real margin below D001's observed
3.47, and a design whose realized level (0.0274 at the π=0.14 anchor) is defensible. Getting there
requires the owner to release Track C's never-rolled-out reservation, which §5 names as an exclusion —
see `docs/v3/V3-R001_family_clean_pool.md` §3-4. That is the owner's signature, not an engineering
choice, and the alternative (the strict 56-family pool) is a legitimate but *descriptive* study.
The 10 GB resource question that paused the old training draft is not a blocker for this design: R001
generates, it does not train. D001's measured fly122 θ0 pass over 612 statements took 28.4 s at
1.53 GB peak VRAM on the RTX 3080. A rollout-only generation of 128 × 8 = 1,024 sequences at V1's
measured 136 s per 32-sequence step (`experiments/manifests/p3b_pilot.yaml`, `step_time_s.mean`; that
step includes the training update, so it is an upper bound) is ≤ 1.3 h, and n=8 concurrent
5.1k-token sequences of the 0.6B model need roughly 4.6 GB of KV cache plus 1.5 GB of weights —
inside the 10 GB card, with no offloading experiment and no 10 GB training smoke.
