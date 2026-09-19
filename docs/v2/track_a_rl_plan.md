# TinyLean-RL V2 — Track A: RL prover plan & recipe decision memo

Status: **A0 complete; A1 complete (this memo); A3 preregistered as `V2-A001`
(evaluation-only, frozen A3-primary set) and running since 2026-09-19; theorem-family
leakage audit complete 2026-09-20 ([`family_leakage_audit.md`](family_leakage_audit.md)).**
No training run has been launched; Candidate 2 (the only conditional training run) stays
dormant.
Track structure: [`research_plan.md`](research_plan.md) (Track A/B/C amendment). V1 evidence
index: [`legacy_evidence.md`](legacy_evidence.md). Discipline: [`experiment_protocol.md`](experiment_protocol.md),
[`../dual_server_collaboration.md`](../dual_server_collaboration.md) §6.1,
[`../seed_control_audit.md`](../seed_control_audit.md) (the only configurable seed in the
pinned build is `+data.seed`).

## 1. Track A objective

Under the affordable single-RTX-3090 compute budget, assemble a final RL prover checkpoint
whose choice is *better justified* than "the V1 60-step recipe endpoint, case closed".

- Deliverable: the frozen `theta_RL*` for Track C — not a new RL algorithm, not more raw
  training compute.
- Track A stops (A4) as soon as we can reasonably answer: *which RL checkpoint / recipe is
  the most suitable representative model under this compute regime?*
- A4 freeze package: checkpoint path; weights hash / revision; training manifest; training
  seed; training compute; selection rationale; evaluation artifact hashes. After A4 the
  checkpoint is not swapped before Track C.
- Checkpoint-selection evidence must come from a dedicated selection set — never from the
  Track C final benchmark (research-plan guardrail).

What "better justified" concretely means here:

1. a single, pre-registered selection protocol on a clean set (never trained on; not the
   E023 holdout; not the Track C benchmark);
2. complete provenance for the chosen checkpoint;
3. an honest statement of what is and is not established — including "no detectable gain vs
   theta0" if that is what the data say.

## 2. A1 — V1 RL evidence synthesis (the five questions)

Sources: frozen V1 records only — E017/E019/E020/E022 dynamics artifacts
(`e019_dynamics`, `e020_seed2_dynamics`, `e022_seed3_dynamics`; cross-checked by the
trainer-metrics parsers), manifests `p3b_pilot.yaml`, `p3c_fixed_eval.yaml`,
`m1_step60.yaml`, `m1_seed_replication.yaml`, `e023_holdout.yaml`, and
`docs/experiment_log.md`. Numbers below are read off those artifacts; nothing is re-run.

### 2.1 Cross-seed trends (seed1 E019 / seed2 E020 / seed3 E022 — 60 steps each)

| seed (exp) | peak window | IGR by 10-step segment (1–60) | overall IGR | overall score | Z / O | resp_len / clip / entropy | grad>0 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| seed1 (E019) | 21–30 (0.225) | .100/.200/.225/.175/.100/.150 | 0.158 | 0.089 | 0.838 / 0.004 | 3558 / 0.540 / 26.7 | 27/60 |
| seed2 (E020) | 41–50 (0.250) | .150/.050/.175/.150/.250/.125 | 0.150 | 0.074 | 0.838 / 0.013 | 3563 / 0.544 / 25.3 | 29/60 |
| seed3 (E022) | 1–10 (0.250) | .250/.125/.125/.100/.150/.150 | 0.150 | 0.076 | 0.838 / 0.013 | 3565\* / 0.545\* / 23.0\* | 27/60 |

\* seed3 trainer-metrics cover steps 1–32 only (dynamics cover 60/60); resp/clip/entropy
averages are over the available steps.

Common trends:

- **Level**: overall IGR ≈ 0.15 in all three seeds (0.158 / 0.150 / 0.150); Z ≈ 0.84; O ≤
  1.3 % — informative groups stay sparse but persistently present.
- **Stability of the generation regime**: response length mean ≈ 3.55–3.57 k of the 4096
  cap; clip ratio ≈ 0.54; entropy 23–27 with no collapse; no format collapse; verifier
  errors near zero during training.
- **Gradient behavior**: gradient is nonzero only in mixed-group steps — 27–29 of 60 steps
  per seed; where present, grad_norm ≈ 0.08–0.19. Zero-gradient steps are all-zero groups,
  not failures.
- **Update magnitude**: the weights move for real but little — rel_L2 vs theta0: seed1
  9.1e-5 (step10) → 1.35e-4 (20) → 1.67e-4 (30) → 2.35e-4 (60); seed2 2.41e-4; seed3
  2.17e-4; all 311/311 keys changed.

### 2.2 What "rise-then-fall" actually is — and what it is not

- seed1: rise (0.100 → 0.200 → 0.225) → fall (0.175 → 0.100) → partial recovery (0.150).
- seed2: early dip (0.050) → late peak (0.250 at 41–50) → final dip (0.125).
- seed3: starts at its peak (0.250) and deflates; never rises again.

What generalizes across seeds **(three robust statements)**:

1. an informative frontier is **transient** — every seed ends below its own peak;
2. the **peak location is seed-dependent** (21–30 / 41–50 / 1–10) — "21–30 peak then
   regression" is *not* a cross-seed fixed pattern;
3. the **overall level is stable** (~0.15).

What does **not** generalize: a fixed rise-then-fall window. Consequence: fixed-step early
stopping has no cross-seed basis; a robust horizon policy must keep checkpoints and select
on evidence rather than assume a privileged step.

### 2.3 Plausible training horizon

- **60 steps**: covers every observed peak; no seed's tail suggests a second wave. E019's
  frozen Case-A rule would have allowed *discussing* 60→100 only after a strong positive —
  the observed outcome was Case B, so >60 is not indicated.
- **30 steps**: E018 gives the best dev-set point estimate at θ30 (+0.98 pp), but E019
  measured θ60 ≈ θ30 (−0.20 pp) — neither 30 nor 60 is evidenced as better on existing
  data. Both remain legitimate candidates for the A3 selection.
- **<30 steps**: not indicated (seed2's peak is at 41–50; no capability data exists early).

### 2.4 Checkpoint-evaluation constraints (E018 / E019 / E023)

| experiment | set (theorems × samples) | checkpoints | headline | verdict |
| --- | --- | --- | --- | --- |
| E018 (dev set) | 64 × 8 (seed1 only) | θ0/10/20/30 | θ0 65, θ10 65, θ20 64, **θ30 70** /512; θ30 − θ0 = **+0.98 pp**, CI [−2.54, +4.30], McNemar 1.0 | POSITIVE-INCONCLUSIVE |
| E019 (same dev set) | 64 × 8 | θ60 vs θ0 / θ30 | θ60c − θ0 = **+0.78 pp**, CI [−2.54, +4.30]; θ60c − θ30 = **−0.20 pp** | Case B → POSITIVE-INCONCLUSIVE; stop at 60 |
| E023 (final holdout) | 128 × 4 (fly122, step60 only) | θ0 + seed1/2/3 step60 | θ0 120; seed1 **122 (+0.39 pp)**, seed2 **113 (−1.37 pp)**, seed3 **111 (−1.76 pp)**; cross-seed mean **−0.91 pp**, 1/3 positive; all CIs straddle 0 | no stable benchmark improvement |

Constraints for Track A:

1. the strongest single point estimate anywhere (seed1 θ30, +0.98 pp) is unconfirmed and
   was measured on an *adaptively reused* dev set;
2. step60 shows no holdout gain; intermediate checkpoints beyond seed1 were **never
   evaluated** (their actor weights were pruned for seeds 2/3);
3. the E023 holdout is consumed — selection must use a **new** sealed selection set;
4. no checkpoint-pair difference is statistically established; at n=128 the paired CI is
   ≈ ±3.3 pp, so the design must target ≥ ~2 pp effects and pre-register tie-breaks
   (see §3).

### 2.5 E020-M guidance (temperature × group size) — sufficient for the next recipe decision

| condition | IGR | Z | O | trunc | candidate verify rate |
| --- | --- | --- | --- | --- | --- |
| T=0.6, n=8 | **0.406** | 0.562 | 0.031 | 0.551 | 0.244 |
| T=0.6, n=4 | 0.312 | 0.625 | 0.062 | 0.551 | 0.246 |
| T=1.0, n=8 | 0.328 | 0.625 | 0.047 | 0.395 | 0.250 |
| T=1.0, n=4 | 0.281 | 0.625 | 0.094 | 0.398 | 0.250 |

- n=8 ≥ n=4, one-directionally (+6/−0 @T=0.6, p=0.031; +3/−0 @T=1.0, p=0.25): **keep n=8**
  (already restored as recovery-ladder #1 — this paid off).
- Cooling T=0.6 raises IGR directionally but not significantly (+2/−7 @n=8, p=0.18), adds
  +16 pp truncation, and leaves the candidate **solve rate unchanged** (0.244–0.250 across
  all four cells). E020-M is therefore sufficient to conclude: **no n/temperature training
  experiment is the right next step** — it would change comparability with the three frozen
  seeds for an expected upside the diagnostic set does not support.

### 2.6 Why is the held-out gain small? (hypotheses, not conclusions)

- **H1 — effective training compute**: 60 steps × 32 sequences ≈ 6.7 M response tokens;
  weight drift ~2.4e-4 rel_L2. The recipe may simply under-move the model. (E019 30→60
  flat partially argues against a cheap "just train more" conversion — see H4.)
- **H2 — evaluation power**: ±3.3 pp paired CI at n=128 theorems × 4 means a +1 pp effect
  is unobservable at the used power. "Gain small" and "gain unmeasurable at this power"
  are not yet separated.
- **H3 — length/truncation**: ~54 % of candidates still hit the 4096 cap; the model always
  spends its full budget.
- **H4 — protocol gap**: single-turn regime (multiturn off) while the official recipe uses
  a 2-turn error-fixing mechanism at 50 % sampling — the one unrestored official baseline
  mechanism (recovery ladder #2, config audit §7). Never tested in V1; it plausibly caps
  how far reward shaping can move behavior (no feedback-conditioned repair is ever taught).
- **H5 — genuine null at this scale**: verifier RL reshapes reward-obtaining behavior
  (IGR/dynamics) without producing detectable held-out capability change in this compute
  regime.

Discriminating H1/H3/H4 would require new runs; H2 affects the interpretation of
everything and is partly addressable by a higher-powered selection evaluation (§3).

## 3. A3 preview — checkpoint inventory and selection design

Physically available candidates (verified on fly90, 2026-09-19):

| candidate | VERL checkpoint | HF export | prior evaluations |
| --- | --- | --- | --- |
| theta0 (anchor, not a candidate for `theta_RL*`) | — | `models/weights/kimina_distill_0_6b` | E018 dev set; E023 holdout |
| seed1 step10 / 20 / 30 | `runs/p3b_pilot/global_step_{10,20,30}` | `runs/p3c_models/step_{10,20,30}` | E018 dev set only |
| seed1 step60 | `runs/p3b_pilot/global_step_60` | `runs/p3c_models/step_60` | E019 dev set; E023 holdout (+0.39 pp, ns) |
| seed2 step60 | `runs/m1_seed2/global_step_60` | `runs/m1_seed2_models/step_60` | E023 holdout (−1.37 pp, ns) |
| seed3 step60 | `runs/m1_seed3/global_step_60` | `runs/m1_seed3_models/step_60` | E023 holdout (−1.76 pp, ns) |
| seed2/3 step10–30 | actor weights pruned (keep=3) | not available | — |

Selection-set requirements (to freeze at preregistration):

- **Fresh and sealed (realized)**: the V2 theorem-role registry (built 2026-09-19,
  `experiments/manifests/v2/theorem_role_registry.json`) partitions the eligible pool
  (6,729 statements = 7,620 unique minus the 763 V1-used and the 128 sealed E023 holdout)
  into B-train / B-validation / B-test / A-selection / C-joint-holdout. The **A3-primary**
  subset (512 theorems, group-aligned, order-hash frozen) is materialized as
  `v2_a001_selection_set.json`; the remainder (`A-reserve`, 162) is reserved for the
  conditional second A-track decision only.
- **Size for power**: 512 theorems × 4 samples → paired CI ≈ ±1.6–1.7 pp (target ≥ ~2 pp
  resolution).
- **One host**: all models of the comparison on fly90; never mix hosts (E023 cross-device
  audit: per-sample trajectories are not bitwise-reproducible across hosts).
- **Protocol**: strict Kimina 2.0.0, temp 1.0 / top_p 1.0 / max 4096, canonical seed
  schedule, theorem-level pairing only.
- **Selection among RL checkpoints only** (theta0 is the reference anchor, not a
  candidate — Track C needs an RL prover).
- **Pre-registered one-shot rule** with an explicit default: the current default candidate
  is **seed1-step60** (V1 recipe endpoint; the only seed with a positive point estimate on
  both dev set and holdout). A candidate replaces the default only under the pre-registered
  margin rule written before launch; no post-hoc switching, no re-draws.

## 4. Candidate next steps (maximum two)

### 4.1 Candidate 1 (accepted, no training) — `V2-A001` (preregistered)

**Checkpoint-selection evaluation on a fresh sealed selection set.**

- Question: *which physically available RL checkpoint is the most defensible `theta_RL*`
  candidate under one clean selection protocol?*
- Models: core family = seed1 {step10, 20, 30, 60} + theta0 anchor (5); recommended
  additions: seed2-step60, seed3-step60 (7 total) so the "seed1 is the default" claim is
  re-measured on equal footing.
- Compute: order of ~12–15 GPU-hours on fly90 for 7 models at the frozen 512-theorem ×
  4 set (E023 rate: 512 candidates ≈ 27 min). Purpose estimate, not a wall-time promise.
- Deliverable: canonical selection artifact(s) + the filled `V2-A001` manifest recording the
  rule application → the `theta_RL*` candidate for A4.
- Preregistration: committed before launch (`experiments/manifests/v2/V2-A001.yaml`; template
  `rl_training.template.yaml`; `track: A`, evaluation-only) including the frozen set, the
  selection rule, margin, and tie-break.
- Explicitly not: a Track C endpoint, a benchmarking claim, or a fishing exercise — one
  draw, one decision.

**2026-09-20 amendment (family leakage audit).** The audit measured the set's real
isolation: 232/512 A001 ids share an L3 source-problem family with a V1-consumed statement
(52/512 with the sealed E023 holdout; 392 effective clusters, borderline CIs ≈ 1.14x wider).
The set and the one-shot rule stay frozen. Consequences: (a) the A001 outcome is a
*selection statistic* for `theta_RL*` only — no A001 delta, even with `ci_low > 0`, is a
confirmatory RL-vs-theta0 capability claim; (b) the A4 freeze package must include the audit
and report the overlap next to the selection outcome; (c) the Track C final holdout comes
from a family-clean split (90/680 C-joint ids are family-clean today, or re-partition under
the family-granular amendment); (d) the audit completes between A001 and the A4 freeze.

### 4.2 Candidate 2 (conditional, one training run) — `V2-A002`-class

Trigger: only if (i) Candidate 1 shows every available checkpoint statistically ≤ theta0
(no candidate adopts), **and** (ii) the project decides Track A should attempt one
capability-improving recipe intervention before freezing. Otherwise skip — Track A would
freeze the best-evidenced checkpoint and document the null.

- Single intervention: **restore the official Kimina multiturn error-fixing mechanism**
  (`+data.multiturn=True`, sampling 0.5, `max_prompt_length ≥ 8192` and the dataset-side
  second-turn flow per config audit §7) — the only unrestored official baseline mechanism
  (ladder #2), addressing hypothesis H4.
- Question: *does restoring the official two-turn mechanism produce a checkpoint with a
  detectable capability gain under the same single-GPU budget?*
- Preconditions before any launch: runner/interface changes reviewed; a memory probe for
  the enlarged `max_model_len`/KV footprint; its own preregistered manifest with a go/no-go
  rule evaluated on the frozen `A-reserve` set (Δ vs theta0 and vs the incumbent
  `theta_RL*`); user confirmation and foreground monitoring per the training-run norms.
- Why not something lighter as the "one run": T=0.6 or larger-n variants contradict §2.5
  (no significant IGR gain, solve rate invariant, comparability cost); lr/horizon changes
  are outside the allowed direction list; 60→100 is barred by the E019 Case-B rule.

## 5. Deprioritized directions (recorded, not proposed)

New reward shaping · new RL algorithms · curriculum · self-training · large-scale
hyperparameter search · 500M/360M frontier · new model families · 60→100 step chasing ·
allocator-adjacent work (Track B's scope).

Also recorded: A2 (minimal recipe refinement *with training*) is **not justified yet** —
A1's specific finding is a missing *selection-evidence* surface (A3), not a defective
recipe whose fix demands training.

## 6. Decision status (2026-09-19)

Project-owner decision: A2 deferred; A3 prioritized and preregistered as `V2-A001`;
Candidate 2 stays dormant — it may be triggered only by its published conditions and
explicit owner approval, and never starts automatically. Note: fly122's pre-amendment
`V2-E001` (canonical alias `V2-B001`) is a Track B experiment, unrelated to this Track-A
numbering.

Addendum (2026-09-20): the theorem-family leakage audit completed while A001 was still
running; it is a required element of the A4 freeze package, and the A001 outcome's
positioning is selection-only (no confirmatory capability claim). No frozen set, rule or
protocol was modified by the audit.
