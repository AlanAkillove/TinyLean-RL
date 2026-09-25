# Paper contribution candidates (V3 closeout)

Status: **for owner review — not final paper text.** This document lists what the
project's completed evidence can and cannot carry into a paper. It is organised by
strength of support, not by experiment ID. Every entry is traceable to the artefacts
cited in `docs/research_evidence_matrix.md`.

Terminology follows the registry: V3-D001 = `COMPLETE / CANONICAL_GO_WITH_QUALIFICATION`;
V3-R001 = `COMPLETE / SOURCE-DRIVEN-ONLY`; V3-R002 = `NOT RUN`.

---

## A. Strong supported contributions

These are the claims the completed evidence does carry. They are stated at the
strength the evidence supports, with the qualification attached.

**A1. A source-aware prospective falsification protocol for representation-based
data-selection signals.**
The project demonstrates a concrete, preregistered evaluation pattern: (i) freeze a
retrospectively promising representation signal, (ii) run it prospectively on a fresh
frozen sample, and (iii) decompose the prospective effect into a *pooled* gate and a
*within-source* (within-synthetic) gate before any claim of generalisation is made.
Frozen Kimina representations passed the pooled prospective gate and failed the
within-synthetic gate, which changes the conclusion from "semantic predictor of RL
learning value" to "source-level distributional structure". The protocol, not the
controller, is the transferable contribution.
*Evidence:* `experiments/manifests/v3/V3-R001_results.json` (gate cells P1/P2 PASS,
S1/S2 FAIL); `docs/v3/v3_closeout.md` §2–§3.

**A2. Retrospective representation signal is real and reproducible *in-distribution*.**
Block-18 last-token `theta0` representations predict reward-informative groups above
the prevalence baseline and above a handcrafted-feature model, under nested
family-grouped cross-validation and a full-procedure bootstrap, on two nodes.
*Evidence:* AUPRC B2 0.57280 vs B1 0.42673 (fixed-OOF delta +0.14607, 95% CI
[0.05096, 0.24145]); full-procedure delta +0.10869 [−0.02504, 0.23679] (1000 reps,
fly90) replicated at +0.11118 [−0.02427, 0.23934] (500 reps, fly122); top-20
enrichment 3.46659 [3.02435, 3.92865] with 1000/1000 bootstrap reps ≥ 1.75
(`V3-D001_results.json`, `V3-D001_fullproc_bootstrap.json`,
`V3-D001_fullproc_500rep_fly122.json`).
*Qualification (mandatory, non-negotiable):* the full-procedure CI crosses zero; the
result is "canonical GO with qualification", not an unqualified win.

**A3. Pooled prospective enrichment can pass while source-generalisation fails.**
The V3-R001 attempt-2 outcome is a worked example of a failure mode that is invisible
to pooled evaluation: pooled AUPRC 0.6831 (B2) vs 0.5512 (B1), pooled gate PASS on both
criteria; within-synthetic gate FAIL on both criteria (2.9867 → 1.5185, CI lower bound
2.2123 → 0.9762). Source diagnostics show why: 41/47 analyzed groups from the
autoformalizer and 24/41 from synthetic sources contribute very different
positive rates (autoformalizer 2 positives, human 1 positive).
*Evidence:* `V3-R001_results.json` (gate, `source_diagnostics`, `mixture_diagnostics`).

**A4. A negative-in-the-right-way prospective result is publishable and was recorded
as such.**
The registry and closeout record `SOURCE-DRIVEN-ONLY` as a canonical classification
and explicitly forbid rescue (no new layers, MLP, source-balanced retraining, larger
controller, alternative sampling objective, new prospective sample, reserve access, RL
intervention). This is a methodological contribution about how to close a line
honestly and is a distinguishing feature for a venue that values preregistration.

---

## B. Secondary empirical findings

Supported by the evidence, but descriptive or single-family, and should be presented
as such.

**B1. RLVR trainability on a 0.6B Lean prover is real but modest.**
GRPO raises in-distribution generator accuracy over three seeds; the information-gain
ratio (IGR) is ~0.15 (0.158 / 0.150 / 0.150). Exact-value IGR is descriptive; the
unified picture is "nonzero but bounded trainability", not "RL solves Lean".
*Evidence:* `docs/experiment_log.md:380,419,452-457,491`; `docs/v3/data_audit.md`.

**B2. Held-out capability transfer is not demonstrated.**
Three-seed held-out differences are +0.39 / −1.37 / −1.76 pp, every CI crosses zero,
mean ≈ −0.91 pp. This is a useful, honestly-negative secondary result: on this scale,
in-distribution gains did not translate into held-out capability.
*Evidence:* `experiments/results/e023_multiseed_analysis.json`;
`experiments/manifests/e023_holdout.yaml:73-91`.

**B3. Training dynamics have a characteristic shape.**
Seed-wise peaks at iterations 21–30 / 41–50 / 1–10, read off the per-seed `ranges`
blocks rather than the summary scalar. Presented as dynamics, not as a claim about
which iteration is optimal.
*Evidence:* `experiments/results/e019_dynamics.json`, `e020_seed2_dynamics.json`,
`e022_seed3_dynamics.json`.

**B4. A reward-dead base model is worth reporting as a boundary case.**
Qwen3 Base shows near-zero trainability (IGR 0.0156 overall, 0.000 in smoke), a useful
lower boundary that contextualises the TinyLean numbers rather than standing alone.
*Evidence:* `experiments/results/e020_qwen_base_n8.json`, `e021_smoke_dynamics.json`.

**B5. Hindsight allocation has headroom that a learnable allocator did not capture.**
B003-style hindsight allocation over a cross-fitted empirical-response allocator showed
headroom that the learned allocator did not close; the 4096-token endpoint has an
expected-solves CI of [−1.00, −0.13]. Report as "headroom exists; the simple learnable
allocator did not realise it", not as "adaptive compute works".
*Evidence:* `docs/v2/b002_memo.md:69-74`, `docs/v2/b003_memo.md:9-14,61-67`,
`experiments/results/v2_b003_analysis.json`.

**B6. Family-structure leakage is measurable and large.**
The component registry tracks 7,620 statements / 1,706 merged components with
cross-role overlap. Counts here must be reported with their counting convention
(1,706 merged components vs 1,710 L3 name-families are different objects).
*Evidence:* `experiments/manifests/v2/family_component_registry.json:49-54`,
`docs/v2/family_leakage_audit.md:34,64-74`.

---

## C. Engineering / reproducibility contributions

Real work, but infrastructure- or methodology-of-execution level. Goes to
reproducibility material, not to the main scientific narrative.

**C1. A verifier lifecycle failure mode exposed by pathological Lean elaboration.**
The Lean server died under pathological elaboration of specific statements; the
project characterised the failure class and distinguished it from genuine proof
failure (INFRA_CENSORED is never forced to all-fail). This is a caution for anyone
running Lean-based RLVR at scale.
*Evidence:* `docs/v3/r001_infrastructure_amendment_cprime.md`; registry amendment C′.

**C2. Bounded recovery under a frozen protocol.**
`docker restart -t 10` → health check → nonformal cold canary, capped at
`max_recoveries_per_launch = 16`, journaled durably, with identity preflight and
`SUPERSEDED_ATTEMPT1_DIR` refusal. Attempt-2 completed after 45 recoveries across
three sessions, all successful, and the scientific result was computed only on the
complete preregistered sample.
*Evidence:* registry `V3-R001-attempt2-result` amendment; session journal
(3 entries) and recovery segment hash.

**C3. Frozen-settings and hash-pinned execution as a discipline, not a slogan.**
Frozen settings sha256 `bf069ecc…`, canonical analyzer sha256 `3b47af65…`, raw freeze
`1033b8eb…`, results sha256 `1d7128bd…`, with the analyzer executed exactly once and a
fail-closed `FrozenViolation` if a consumer runs before the raw artefact is complete.
*Evidence:* `scripts/v3_r001_analyze.py`, `experiments/manifests/v3/registry.yaml`.

**C4. Honest accounting of what was consumed.**
The 221-component `consumed_only` pool minus the 128-statement formal sample equals the
93-component sealed reserve, which was never opened. This is reproducibility bookkeeping
that makes the negative result auditable.
*Evidence:* `experiments/manifests/v3/v3_final_holdout_reserve.json`
(sha256 `b60be381…`).

---

## D. Claims explicitly ruled out

Not weakened — ruled out. Do not put any of these in a paper except as an explicit
statement of what was tested and did not hold.

**D1. "RL improves 0.6B Lean capability."** Ruled out. Held-out deltas +0.39 / −1.37 /
−1.76 pp, all CIs crossing zero, mean ≈ −0.91 pp (B2 above).

**D2. "The adaptive / learned allocator works."** Ruled out. Hindsight headroom existed
but the cross-fitted empirical-response allocator did not capture it (B5).

**D3. "A semantic controller predicts RL learning value."** Ruled out prospectively.
The within-synthetic gate failed; the pooled effect is inseparable from source-level
distributional structure. Canonical classification `SOURCE-DRIVEN-ONLY`.
*Evidence:* `V3-R001_results.json`; `docs/v3/v3_closeout.md` §2.

**D4. "A Jev-style / Jev-inspired controller intervention improves RL."** Ruled out.
The controller line is CLOSED in the registry and R002 was not launched; the
preregistered prospective falsification is the result. No rescue is permitted
(new layers, MLP, source-balanced retraining, larger controller, TinyJev/Kev,
alternative sampling objective, new prospective sample, final reserve, RL intervention).

**D5. "High pooled predictive performance implies a general semantic predictor."**
Ruled out as an inference. This is the project's most transferable cautionary result:
the pooled and within-source conclusions diverge, and only the latter survives.

---

## Usage note

If a venue requires a single headline contribution, it is **A1 + A3 + D5** — the
protocol and the divergence it exposes — with A2 as the reproduction that makes the
falsification meaningful, and the D-block as the boundary of what the project will
claim. Nothing in section B should be promoted to a headline claim.
