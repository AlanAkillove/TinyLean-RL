# V3 closeout — the controller line is closed by the preregistered prospective outcome

Owner decision of 2026-09-25: `V3-R001 = SOURCE-DRIVEN-ONLY` is accepted as the final scientific
outcome; `V3-R002` is not authorized and not run. This document and the three companions it points to
(`docs/research_evidence_matrix.md`, `docs/paper_contribution_candidates.md`,
`docs/paper_figure_inventory.md`) are consolidation only. No model generation, verifier experiment or
RL training was performed for them, and none is authorized.

## 1. Status of record

```
V3-D001   COMPLETE  /  CANONICAL_GO_WITH_QUALIFICATION
V3-R001   COMPLETE  /  SOURCE-DRIVEN-ONLY
V3-R002   NOT RUN   —  stopped by the preregistered R001 outcome
```

Recorded in `experiments/manifests/v3/registry.yaml` (amendments `V3-R001-attempt2-result` and
`V3-closeout`):

> R002 was not launched because the pooled prospective enrichment did not survive the preregistered
> within-synthetic co-primary test.

**Naming rule.** R001 is *not* a failure. Its canonical classification is `SOURCE-DRIVEN-ONLY`, which
is outcome **B** of the gate frozen before the run (`experiments/manifests/v3/V3-R001_gate.json`):
the pooled prospective enrichment is real while it does not reproduce within the synthetic stratum,
so the effect cannot be separated from source-level distributional structure.

The Jev-inspired controller intervention line is **CLOSED**. The result is not to be rescued with new
layers, an MLP, source-balanced retraining, a larger controller, TinyJev/Kev, an alternative sampling
objective, a new prospective sample, the final reserve, or any RL intervention. The prospective
falsification *is* the result.

## 2. Canonical V3 claim

> Frozen Kimina representations showed strong retrospective and pooled prospective enrichment for
> reward-informative RLVR groups. However, the preregistered within-synthetic prospective test did
> not pass, so the pooled effect could not be separated from source-level distributional structure.

Mandatory implication, carried with the claim wherever it is stated:

> The evidence does not support interpreting the controller as a source-independent semantic
> predictor of RL learning value.

Not claimed, and not to be implied: that the semantic controller works; that guided RL would improve
training; compute savings; capability improvement.

## 3. Positive descriptive results, preserved and distinguished

These numbers are real and are kept — as *descriptive* results, separate from the conclusion. The
distinction matters: read alone, several of them invite exactly the reading the prospective test
ruled out.

### 3.1 D001 (retrospective, offline probe; `experiments/manifests/v3/V3-D001_results.json`)

```
B2 AUPRC (block-18 last-token primary):  0.57280        (B1 0.42673; prevalence/B0 0.1516)
fixed-OOF delta vs B1:                   +0.14607  [0.05096, 0.24145]   (level-1, excludes 0)
full-procedure delta:                    +0.10869  (mean over 1000 reps; one-sided p = 0.048;
                                                    952/1000 replicate signs positive)
full-procedure CI:                       [-0.02504, 0.23679]            (level-2, includes 0)
top-20 enrichment:                       3.46659  [3.02435, 3.92865]    (level-1)
                                         full-procedure: 1000/1000 replicates >= 1.75
```

Level-2 artifact: `experiments/manifests/v3/V3-D001_fullproc_bootstrap.json` (1000 reps, fly90);
cross-node replication: `V3-D001_fullproc_500rep_fly122.json` (mean +0.11118,
CI [-0.02427, 0.23934], sign consistency 0.958). Qualification of record: superiority over
handcrafted features is **not** robust at full-procedure two-sided 95 % — the interval includes zero.

### 3.2 R001 (prospective, family-clean; `experiments/manifests/v3/V3-R001_results.json`)

```
N nominal:      128
N analyzed:     112   (16 groups INFRA_CENSORED; data guard 112 >= 103 passed before any gate)
positives:      24
prevalence:     0.2143

pooled:         enrichment = 2.9867    CI lower = 2.2123    P1 PASS    P2 PASS
within-synthetic: enrichment = 1.5185  CI lower = 0.9762    S1 FAIL    S2 FAIL

AUPRC:          B2 = 0.6831            B1 = 0.5512
```

### 3.3 The methodological result

> High pooled predictive performance alone would have produced a misleading conclusion without the
> source-aware prospective test.

The pooled gate passed with an enrichment of 2.99 and an interval lower bound of 2.21 — a result that
would have been reported as a success under a pooled-only protocol. The preregistered co-primary
(within-synthetic) test failed (1.52, interval lower bound 0.98), and it is that failure which is
canonical. This is a transferable design lesson, not a footnote: for any controller whose training
population is source- or provenance-heterogeneous, a pooled prospective test is not sufficient.

## 4. Infrastructure results stay secondary

The verifier incidents around attempt-1 and the C′ amendment are engineering, not science, and are
kept out of the main narrative:

- a verifier-lifecycle bug was exposed by pathological Lean elaboration (a computation outlived the
  client's request on a reusable REPL; the frozen policy could detect the state but not clear it);
- a bounded, journaled, fail-closed recovery procedure was implemented and validated off-formal
  (`docs/v3/V3-R001_infrastructure_amendment_Cprime.md`;
  `experiments/manifests/v3/V3-R001_Cprime_validation.json`);
- formal attempt-2 then required 45 recoveries across three execution sessions (16 + 16 + 13), all
  successful — no scientific setting was changed to make the run finish;
- the scientific result was computed only after the complete preregistered sample (128/128 groups)
  was frozen, structurally checked, and archived.

Detailed chronology lives in the C′ amendment doc (appendix-grade), the session journal and the
registry entries; the paper should carry at most a short reproducibility note.

## 5. Final reserve

```
V3-FINAL-HOLDOUT:
  93 components
  SEALED
  NEVER USED
```

Artifact `experiments/manifests/v3/v3_final_holdout_reserve.json`
(sha256 `b60be381b5f64a0394691344b71cd704ed337716f0eea81ea265e05af7761568`), status `SEALED`.
No outcome from these components has been observed; none was opened during closeout. The seal's
forbidden uses (rollout, controller fitting, sampler tuning, power tuning, any R001 analysis) stand
unchanged and are not waived by this closeout.

## 6. Unifying thesis: assessment

Proposed thesis:

> In a sub-billion-parameter Lean prover, optimization and control signals can appear strong
> in-distribution yet repeatedly weaken under seed replication, family isolation, and prospective
> source-aware evaluation.

Clause-by-clause mapping to concrete evidence (full citations in `docs/research_evidence_matrix.md`):

| Clause | Concrete evidence | Verdict |
|---|---|---|
| "optimization signal [appears strong in-distribution]" | P3-A/B, E017, E020, E022 — IGR ≈ 0.15 per seed on the training population, reproducible | Supported, with the caveat that "in-distribution" here means *on the training population of the 0.6B distill under this recipe*, not "large" |
| "seed replication [weakens it]" | E018/E019/E023 — held-out capability deltas cross zero (mean ≈ −0.91 pp); optimization-level signal reproduces but does not convert into stable capability | Supported for capability; **not** supported for the optimization signal itself (IGR is stable across seeds — the thesis must not claim the optimization signal weakens under seed replication) |
| "family isolation [weakens it]" | family-component audit (7620 statements / 1706 components; large multi-variant structure, cross-role overlap) plus D001's family-grouped CV and A3's full-procedure bootstrap: the B2−B1 advantage loses its interval under family-level resampling | Supported as *fragility under family-level uncertainty*, with the caveat that the leakage finding is about evaluation units, not a mechanism |
| "prospective source-aware evaluation [weakens it]" | R001: pooled PASS, within-synthetic FAIL; canonical `SOURCE-DRIVEN-ONLY` | Supported — this is the strongest clause, and the only one that is prospective and preregistered |

**Overreach to avoid.** (i) The thesis implies a general law; the evidence covers one model
(Kimina-Distill-0.6B), one recipe and one horizon, so it must be stated as an observed pattern in
this setting. (ii) "Repeatedly" invites counting the same underlying phenomenon twice (D001's
level-2 interval and R001's co-primary failure are related, not independent). (iii) The optimization
signal itself does *not* weaken under seed replication, so the phrase must not be read as
"everything decays". With those three qualifications, the thesis is an accurate, evidence-backed
summary of the programme; without them it overreaches.

## 7. Experiment accounting

See `docs/research_evidence_matrix.md` for the question-level view. Classification used here:

- **scientific experiment** — produced a scientific measurement used in an evidence claim;
- **engineering validation** — exercised infrastructure without producing scientific outcomes;
- **infrastructure recovery** — restoring capacity during a formal run (never a scientific event).

### 7.1 Completed scientific experiments

GPU wall-clock is reported where it was recorded at launch/close. Values are wall time on one
RTX 3080 10 GB (fly122) unless stated; no value here is an efficiency claim.

| Experiment | Work performed | Wall | GPU-h | Result |
|---|---|---|---|---|
| E017 | P3-B pilot, 30 GRPO steps, n=8 | 4,218 s | 1.17 | dynamics as predicted |
| E019 | seed1 extension step30→step60 | 4,534 s | 1.26 | POSITIVE-INCONCLUSIVE |
| E020 | seed2 replication, 60 steps | 8,290 s | 2.30 | IGR 0.150 |
| E022 | seed3 replication, 60 steps | 9,433 s | 2.62 | IGR 0.150 |
| E023 | sealed 128-theorem holdout, 4 checkpoints × 512 candidates | 8,960.95 s (per checkpoint 2,465.50 / 2,524.43 / 1,842.62 / 2,128.41) | 2.49 | capability deltas cross zero |
| V2-B001 | budget-semantics gate (32/32 determinism, 160/160 prefix match) | not recovered here | — | B1 PASS |
| V2-B002 | 512 components × 1 sample, family-unseen pilot | 6h45m | 6.75 | 11/512 adjudicated solved; hindsight upper bounds |
| V2-B003 | 192 components × 8 samples = 1,536 trajectories | ~20h45m | 20.75 | cross-fitted allocator gate reading |
| V3-D001 | probe extraction (612 prompts × blocks {9,18,27}) + CPU fits | 28.38 s GPU + 290.44 s CPU | 0.008 | CANONICAL_GO_WITH_QUALIFICATION |
| V3-D001 | full-procedure bootstrap, 1,000 reps (fly90, CPU, 40 workers) | 13,825.1 s | CPU | mean +0.10869, CI [−0.02504, 0.23679] |
| V3-D001 | full-procedure replication, 500 reps (fly122, CPU, 15 workers) | 13,777.1 s | CPU | mean +0.11118, CI [−0.02427, 0.23934] |
| V3-R001 | attempt-2 formal run, 128 groups × 8 samples = 1,024 candidates, 45 recoveries | 14,698 s over 3 sessions (6,645 + 4,530 + 3,523) | 4.08 | `SOURCE-DRIVEN-ONLY` |

Subtotal, recorded scientific GPU time: **≈ 41.4 GPU-h** plus CPU-only analysis (D001 bootstraps
≈ 7.7 CPU-h across the two hosts) and the small smoke/calibration runs (E010–E016, E018, E020-M,
E021) that are individually recorded in `docs/experiment_log.md` and were not totted here.

### 7.2 Terminated, aborted or infrastructure-only

| Item | Wall | Classification | Disposition |
|---|---|---|---|
| V3-R001 attempt-1 | 2026-09-25T01:03:56 → 03:22:13 (2.30 GPU-h) | infrastructure abort, **no scientific outcome** | 384 candidates quarantined, never read, combined or analyzed; superseded by amendment C′ |
| E022 restarts r2/r3 (seed3, pre-completion attempts) | killed by systemd-oomd at model load / step 3 | external-cause termination, no retry left | bounded retries exhausted; seed3 completed later under the two-unit topology |
| E024 (MiniF2F theta0) | 107.7 min (1.80 GPU-h) | **PAUSED, no result claimed** | partial artifact (512 records) kept on fly122, not committed |
| C′ nonformal validation (T1–T5) | 8/8 recoveries; recovery cost 7.37–17.23 s each | **engineering validation** | `V3-R001_Cprime_validation.json`; not a scientific experiment |
| `v3_r001_stress_recovery.py` pathological-candidate cycles | 131.8–131.9 s per pathological candidate | **engineering validation** | exercises production C′ code paths; never a candidate retry, never a scientific event |

### 7.3 Not run, by gate

| Item | Status | Stopped by |
|---|---|---|
| V2-B004 | NOT RUN | B003 preregistered gate reading |
| V3-R002 | NOT RUN | preregistered R001 outcome (`SOURCE-DRIVEN-ONLY`) |
| 93-component final reserve | SEALED, NEVER USED | never opened, including at closeout |

### 7.4 Models and checkpoints

| Model / checkpoint | Pin | Role |
|---|---|---|
| Kimina-Distill-0.6B `theta0` | rev `332e8a52`, weights sha256 `34e6e630…` | D001 probe extraction, R001 rollouts (1,024 candidates), E023 base evaluation, E024 partial |
| Qwen3-0.6B-Base | `Qwen/Qwen3-0.6B-Base`, pinned revision | E020-M / E021 cold-start diagnostics (reward-dead boundary) |
| GRPO step-60, seed1 / seed2 / seed3 | `runs/p3c_models/step_60`, `runs/m1_seed2_models/step_60`, `runs/m1_seed3_models/step_60` (≈ 7.3 GB with optimizer state) | E023 held-out evaluation; D001 population is their rollout dumps |

No model was generated, retrained or re-evaluated during closeout.

### 7.5 Theorem-family consumption

| Scope | Statements | Components |
|---|---|---|
| V2 component registry (full Promptset) | 7,620 | 1,706 merged (1,710 L3 name-families — different object, must not be conflated) |
| V1 training population (E017–E023) | 612 distinct statement ids | 443 |
| V3-D001 representation corpus | 612 unique prompts | 443 |
| V3-R001 `consumed_only` pool | — | 221 (= 128 formal + 93 reserve) |
| Sealed and remaining | — | **93 components, never used** |

V1 groups: 720 nominal (240 per seed), 686 after ≤system-error censoring, 110 informative at the
pooled base rate 0.1528.

### 7.6 Counting rule

Only §7.1 counts as scientific experiment. §7.2 items are terminated, paused or engineering
validation; §7.3 items produced nothing. **Stress tests are not scientific experiments** and must
not be counted in any experiment total, sample size or power statement.
