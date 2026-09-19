# TinyLean-RL V2 — Research Plan

**Main question**

> How can a sub-billion-parameter Lean prover solve more theorems under limited compute?

**Status: V2-0 REPOSITORY PREPARATION COMPLETE; NO V2 EXPERIMENT HAS RUN.**

_(Recorded 2026-09-19 on `main2`. V1 is frozen: see [`legacy_evidence.md`](legacy_evidence.md) and tag `v1-research-freeze-20260919`.)_

The project has two complementary pillars:

1. **Training-time**: verifier-based RL / RLVR (V1 lineage; RQ1 summarizes what is rigorously established).
2. **Inference-time**: adaptive test-time compute allocation (RQ2–RQ5; new in V2).

---

## RQ1 — RLVR under constrained compute

Reuse V1 results as-is; do **not** repackage them as new experiments. The permitted rigorous summary of V1:

- RL plumbing / trainability is established (P3-B short pilot; E019 step30→60 extension; E020/E022 seed replication).
- Positive / mixed reward groups are reliably observed (three-seed IGR 0.158 / 0.150 / 0.150).
- Multi-seed training dynamics are reproducible ("rise-then-fall" shapes, seed-dependent peak positions).
- Under the present 60-step low-compute regime, held-out capability gain is small and seed-sensitive /
  inconclusive (E023: mean Δ −0.91pp, 1/3 seeds positive, all 95% CIs straddle zero, no significant
  McNemar flips).
- **No stable benchmark improvement is claimed.**

## RQ2 — Heterogeneous inference-compute demand

Define the per-theorem compute-response curve

```text
p_i(b) = P(solve theorem i | token budget b)
```

Initial candidate budgets: `{512, 1024, 2048, 3072, 4096}`.

Goal: measure `p_i(b)` per theorem and quantify heterogeneity across theorems — this is what any
allocator would have to exploit. Curves are measured empirically; monotonicity is **not** assumed
(see [`data_contract.md`](data_contract.md) §5).

## RQ3 — Oracle allocation headroom

For a global budget `B_total`, study

```text
max  Σ_i p_i(b_i)     subject to     Σ_i b_i ≤ B_total
```

First compare **uniform allocation** vs **oracle allocation**. If the oracle-vs-uniform headroom is
small, the adaptive-allocator line may be stopped — it is not forced to continue.

## RQ4 — Lightweight adaptive allocator

Do **not** train a `theorem → optimal budget class` model. Train instead

- `f(theorem, budget) → P(success)`, or
- a multi-head variant `theorem → [p_512, p_1024, p_2048, p_3072, p_4096]`

and let a separate global optimizer choose the budget allocation.

Preferred method ladder (stop as soon as the data justifies stopping):

```text
Uniform
  → simple heuristic
  → handcrafted features + XGBoost
  → frozen prover representation + small MLP
  → (only if justified) small encoder
```

Avoid introducing a new large router at the start.

## RQ5 — RLVR × adaptive inference (main study)

Final 2×2 design:

| prover | uniform | adaptive |
| --- | --- | --- |
| Kimina Distill θ0 | A | B |
| TinyLean RLVR θRL | C | D |

Question: are training-time RL and inference-time allocation complementary?

The official Kimina-Prover-RL-0.6B may serve as an **external reference**, but does not replace the
main 2×2.

## Optional extension

Sequential / contextual-bandit allocation is an **optional extension only**; it must not be a
prerequisite for the main project line to stand.

## Phase plan (frozen)

| Phase | Scope | Status |
| --- | --- | --- |
| V2-0 | Repository / protocol freeze | repository preparation complete (no experiment run) |
| V2-1 | Prefix-vs-direct budget equivalence audit | not started |
| V2-2 | Budget-response dataset construction | not started |
| V2-3 | Uniform-vs-Oracle headroom study | not started |
| V2-4 | Lightweight allocator | not started |
| V2-5 | Distill/RLVR × Uniform/Adaptive main study | not started |
| V2-6 | External benchmark / MiniF2F final evaluation | not started |
| V2-X | Optional sequential/bandit extension | optional |

**No V2-E### manifest or result exists yet.** V2-E numbers are only assigned at formal
(pre-registered) run time; see [`../../experiments/manifests/v2/README.md`](../../experiments/manifests/v2/README.md).

## Guardrails

- No V2 experiment runs during V2-0 (repository preparation only).
- Prefix-derived budget labels are *cheap derived labels*; they may not be claimed equivalent to
  direct-budget interventions before V2-1 completes.
- No monotonic regularization in the first allocator models; monotonicity is measured, not assumed.
- Interfaces are frozen before implementation; allocator code is added only when a phase requires it.
