# Budget-Aware Evaluation of Sub-Billion Lean4 Provers: Truncation Bias, Verifier Risks, and Statistical Detectability

> **Status: side-study / preliminary draft.** This document develops the P2 calibration findings (E001–E008) as a stand-alone evaluation-methodology note (potential technical report / appendix). It is **not** the main TinyLean-RL paper — the main research plan (RL for sub-billion Lean provers) lives in [`research_plan.md`](research_plan.md), and the study record is in [`studies/evaluation_calibration.md`](studies/evaluation_calibration.md).

Draft v0.2 — data status: E001–E008 complete (8- and 32-theorem runs at 2048/4096 budgets).
All numbers below are from a single consumer laptop (RTX 4060 Laptop 8 GiB) with a local Kimina Lean Server (`projectnumina/kimina-lean-server:2.0.0`, Lean `v4.15.0`).

## Abstract

Small Lean4 provers such as Kimina-Prover-RL-0.6B are reported at ~71% Pass@32 on miniF2F — only **+2.45 pp** above their distillation base — yet such evaluations assume budgets and compute far beyond consumer hardware, and the field rarely examines what a limited budget does to the measurement itself. We study budget-aware evaluation of the Kimina Distill/RL 0.6B pair on a single 8 GiB laptop with a local Kimina Lean Server. Across a controlled series of runs (8- and 32-theorem miniF2F subsets; 2,048- and 4,096-token budgets; 32–128 verified candidates per arm), we find: (1) truncation bias is the dominant systematic effect — raising the budget lifts candidate-level Pass@1 by ~+0.13 for both checkpoints, and solved candidates stop below 43% of the ceiling; (2) run-to-run sampling noise at the 8-theorem scale (14 vs 10 candidates for the same checkpoint, identical settings) can exceed checkpoint differences, and subset choice alone flips the sign of the Distill-vs-RL comparison; (3) at 32 theorems the RL checkpoint leads by +9.4 pp candidate-level (z≈1.55, n.s.) and +3 equations theorem-level (McNemar p≈0.375) — the direction matches the official claim while confirming that a +2.45 pp gap is near the edge of statistical detectability (an independent two-proportion test would need ~5,500 theorems per arm). We additionally document verifier-side failure modes (`native_decide`-triggered REPL crashes, cold-REPL concurrency costs, proxy hijacking of local verifier traffic) with practical client-side mitigations, and distil recommendations for reproducible low-budget evaluations.

## 1 Introduction

Neural theorem proving in Lean 4 has converged on a standard evaluation protocol: sample $k$ proof candidates per theorem (for instance $k=32$) with a per-candidate token budget, verify each candidate with a Lean server, and report Pass@$k$ on miniF2F-test (244 theorems). Frontier results (e.g. Kimina-Prover Preview: pass@1 52.9%, pass@8192 80.7%) rely on budgets and compute far beyond consumer hardware, and the community's headline differences between fine-tuned checkpoints are small: Kimina-Prover-RL-0.6B reports **71.30% Pass@32**, only **+2.45 pp** above its distillation base (68.85%).

This creates three practical questions for anyone evaluating sub-billion provers on limited hardware:

1. **How does the token budget shape the measurement?** (RQ1)
2. **How do verifier-side behaviors distort it?** (RQ2)
3. **What sample size is needed to detect differences of the reported magnitude?** (RQ3)

This draft reports a controlled study of these questions using the pinned Kimina Distill/RL 0.6B pair, a local verifier, and a fully instrumented evaluation harness.

**Contributions (working list).**
- C1: A quantitative truncation study: newly solved candidates at 4,096 tokens (generating 2,166–2,780 tokens) are physically impossible under a 2,048 budget — a direct truncation-bias demonstration. Re-runs of identical configurations vary by ±4/32 candidates (seed stability), which bounds how much of the observed +0.125 Pass@1 difference can be causally attributed to budget without paired generation; we report it as observational and leave paired-generation quantification to future work.
- C2: A verifier-risk taxonomy with mitigations: `native_decide`-induced REPL crashes (HTTP 500, reproducible; at the 32-theorem scale only 1–2 of 128 candidates crash, while 9/12 and 7/11 native_decide-containing candidates simply pass — string-banning would be wrong), cold-REPL concurrency costs, and a proxy-interaction failure mode, together with concrete client-side remedies (`verify_status`, backoff+per-candidate fallback, partial persistence, `trust_env=False`).
- C3: A statistical detectability analysis: the official +2.45 pp gap requires roughly 5,500 theorems per arm under an independent comparison (~22× the miniF2F test set), and even the full 244-theorem set is near the boundary unless paired designs are used; we quantify what 8- and 32-theorem subsets can and cannot decide.

## 2 Related Work

- **Kimina-Prover Preview** (arXiv:2504.11354): reasoning-driven prover; pass@k curves up to 8192 samples; establishes the sampling-budget framing of evaluation.
- **Kimina-Prover-RL** (Numina/Kimi): open RL pipeline; the 0.6B checkpoint (+2.45 pp over distill) is our evaluation target pair; training-time validation uses $n=8$, temperature 1.0.
- **EconProver** (arXiv:2509.12603, ACL 2026): compares test-time scaling strategies by accuracy and sampling cost — strategy-level economics; we study the measurement level (budget-truncation bias and detectability).
- **Budget-Aware Evaluation of LLM Reasoning Strategies** (EMNLP 2024): budget-aware comparison of reasoning strategies in natural-language tasks; we instantiate the concern in formal proving with verifier-side instrumentation.
- **miniF2F-Lean Revisited** (NeurIPS 2025): documents formalization defects in miniF2F; this bounds how much any evaluation on subsets can claim.
- **DeepSeek-Prover-V1.5 / BFS-Prover**: context for sampling budgets and verifier infrastructure in the field.
- **Chen et al. 2021 (Codex)**: the unbiased Pass@k estimator we use.

## 3 Methodology

### 3.1 Pipeline
Prompt: Kimina chat template (system + "Think step by step in Lean 4" + formal statement). Sampling: temperature 0.6, top_p 0.95 (the official model-card inference example), $k=4$ per theorem, per-candidate budget $B \in \{2048, 4096\}$ tokens (optional third point 8192). Extraction: conservative Lean-block extractor. Verification: Kimina Lean Server `/verify`; a candidate is valid iff the response contains no `error` field and no error-severity messages (warnings allowed). Metrics: candidate-level Pass@1, unbiased Pass@$k$ (Chen et al.), theorem-level solved rate, group rates (all-zero / mixed / all-one), difficulty buckets.

### 3.2 Verifier-side instrumentation (reliability as a finding)
- `verify_status ∈ {verified, lean_error, verifier_error}` distinguishes model failures from infrastructure failures;
- one retry after a 15 s backoff, then per-candidate fallback so a crashing candidate cannot invalidate its batch;
- `.partial.json` persistence after every batch (survived two mid-run interruptions in practice);
- `httpx.trust_env = False` for local verifier traffic (a system-level proxy had produced persistent 502s while curl/container-internal calls stayed healthy);
- serialized submission: parallel `/verify` calls each spawn a cold REPL (multi-minute Mathlib load) — concurrency must be engineered, not assumed.

### 3.3 Hardware
Single RTX 4060 Laptop (8 GiB); ~45 tok/s generation throughput for the 0.6B models at 4 parallel sequences; verification is millisecond-scale on a warm REPL (~15–35 s for 32 candidates end-to-end).

## 4 Experiments (E008 pending)

### 4.1 Truncation bias and scale (E003–E007; E008 pending)
| Model | Theorems | Budget | Candidates | Pass@1 | all-zero | Solved theorems |
|---|---|---|---|---|---|---|
| Distill | 8 | 2048 | 32 | 0.3125 | 0.625 | 3/8 |
| RL | 8 | 2048 | 32 | 0.28125 | 0.625 | 3/8 |
| Distill | 8 | 4096 | 32 | 0.4375 | 0.5 | 4/8 |
| RL | 8 | 4096 | 32 | 0.40625 | 0.5 | 4/8 |
| Distill | 32 | 4096 | 128 | 0.3359 | 0.531 | 15/32 |
| RL | 32 | 4096 | 128 | 0.4297 | 0.4375 | 18/32 |
- **Checkpoint comparison (E007 vs E008, 32 theorems):** RL leads by +9.4 pp candidate-level (z≈1.55, p≈0.12) and +3 equations theorem-level (flips: RL-only {19, 25, 27, 31}, Distill-only {9}; McNemar p≈0.375). The direction matches the official +2.45 pp claim; significance does not — consistent with §5.
- **Subset sensitivity:** on the 8-theorem subset the same pair is nearly tied (Distill 14 vs RL 13, sign reversed). The sign of the comparison flips with subset choice alone.
- Budget effect (observational only): the 8-theorem set scored +0.125/+0.125 Pass@1 higher at 4096 than at 2048, but those runs are independent stochastic samples — the same configuration re-run moved by 4/32 candidates (E005 vs E007, see seed stability below), of the same magnitude. What the data *do* establish is **truncation bias**: newly solved candidates at 4096 used 2,166–2,780 tokens, which a 2,048 budget physically prevents. Quantifying the causal budget effect requires paired generation (fixed sampling seeds).
- Scale effect: on 32 theorems, 15/32 are solved by Distill at 4096 (46.9%); the second half of the subset (indices 16–31) yields only 5 solved theorems versus 10 in the first half — subset composition matters as much as scale.
- **Run-to-run variance (seed stability):** the same checkpoint on the same 8-theorem subset with identical settings scored 14/32 (E005) and 10/32 (E007, first 8 theorems) in independent sampling runs — a ±4/32 swing that dwarfs most checkpoint differences. This is a direct empirical argument for RQ3.
- Lengths: solved candidates average 1,748 tokens (43% of budget); the overall mean (3,231) is dominated by truncated attempts.

### 4.2 Difficulty stratification
Working taxonomy from 8-theorem data: easy (budget-insensitive), medium (budget-sensitive), hard (unsolved at both budgets). Recomputed on 32 theorems with category tags (algebra/numbertheory/IMO/AMC/AIME) — to appear after E008. Note: the seed-stability finding (§4.1) implies per-theorem stratification from 4-sample runs is itself noisy and should be reported with uncertainty.

### 4.3 Verifier risks
- `native_decide` on a `Finset.range 10000` product crashes the REPL (HTTP 500, reproducible; REPL process lost). RL proposed it in 3/4 candidates on `amc12_2001_p5` (Distill: 0/4 on the same theorem at 4096); at the 32-theorem scale Distill emits `native_decide` in 12/128 candidates (including one reproducible crash on theorem 3).
- Consequences for RL training: the reward side must define crash scoring (we count crashes as failures and log them).

### 4.4 Concurrency
4 concurrent `/verify` requests: 1 completes in 2.4 s, 3 time out at 300 s client-side; server later completes all four (200) after cold-REPL initialization, then closes the extra REPLs.

## 5 Statistical Detectability

(analytic, see also research_plan.md §5)
- Independent two-proportion test for +2.45 pp at $p \approx 0.7$: $n \approx 5{,}500$ theorems/arm for 80% power — beyond miniF2F's 244.
- Full-set paired (McNemar) analysis: significance requires near one-directional flips (b+c small); realistic two-way flips make +2.45 pp non-significant even at 244.
- Our observed 8-theorem candidate-level difference (3.1 pp on n=32) is far inside noise; even the +12.5 pp budget effect is only z≈1.0 under an independent test — paired designs are mandatory at these scales.

## 6 Recommendations for Resource-Constrained Evaluations

1. Report the budget–score curve, not a single number; choose the evaluation budget from the solved-length distribution (e.g. ≥ P95 of solved candidates), not from defaults.
2. Instrument the verifier: track `verifier_error` separately; never silently count infrastructure failures as model failures (or vice versa).
3. With <64 theorems, do not claim checkpoint rankings at the few-percentage-point scale; report CIs and prefer paired, same-theorem designs.
4. Pre-warm the REPL pool before batch verification; serialize reward computation in RL rollouts.

## 7 Limitations

Subset sizes (8, 32 of 244); single seed; single hardware profile; local Lean `v4.15.0` vs unknown official environment; no multi-temperature sweep.

## Appendix (to be filled)
- Commands, artifacts, and per-theorem tables for every experiment (see `docs/experiment_log.md`).
