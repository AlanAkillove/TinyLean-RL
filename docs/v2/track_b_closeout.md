# Track B Closeout — Adaptive Compute Allocation (V2-B001 -> V2-B002 -> V2-B003)

Status: **TRACK B CLOSED** (owner decision 2026-09-21, after the V2-B003
preregistered gate reading). **V2-B004 = NOT RUN - stopped by the B003 gate**;
no predictor was trained and no allocator training corpus was generated.

## 0. Final conclusion

> Large hindsight compute-allocation headroom did not translate into stable
> cross-seed theorem-level allocation headroom on the strict family-unseen hard
> subset. Even an empirical response policy with privileged access to four
> rollout outcomes per theorem produced only small absolute gains under
> constrained budgets and underperformed uniform allocation at the full-budget
> endpoint. We therefore stopped before training a deployable allocator.

Scope limits (binding):
- Applies only to the current theta0 / Promptset family-unseen hard-tail /
  <= 4096-token setting;
- does NOT claim adaptive inference is generally ineffective;
- does NOT claim the full Promptset has the same 2.1% solve rate.

## 1. Terminology correction (2026-09-21)

- Frozen artifacts and manifests are NOT rewritten. In this document and in
  future papers the B003 primary method is called the **cross-fitted
  empirical-response allocator** (equivalently: **cross-fitted
  theorem-specific response policy**) - NOT an "Oracle", because on independent
  eval seeds it can score below uniform allocation.
- "Oracle" terminology is reserved for the Hindsight / plug-in upper bounds.
- Reading note: where `b003_memo.md` says "cross-fitted Expected Oracle", read
  "cross-fitted empirical-response allocator".

## 2. B001 — budget-semantics audit (V2-B001 = raw id V2-E001)

- Canonical HF single-path determinism: 32/32 same-config reruns token-identical.
- Prefix token identity: 160/160 (token ids; 16 theorems x 2 seeds vs the 4096
  prefix).
- Prefix-derived budget labels are valid on the canonical path.
- vLLM 0.9.1 does not satisfy the trajectory-identity requirement and is
  excluded from the canonical dataset path (recorded in the B1 memo,
  `docs/v2/b1_budget_semantics.md`).

## 3. B002 — family-clean pilot + hindsight analysis

- 512 frozen family components (family-component isolation). The exclusion
  mechanism caused a strong distribution shift: synthetic 5.7% (vs 77.7% of
  the full Promptset), singleton components 93.8% (vs 13.2%), median formal
  tokens 166.5 (vs 117); component-level fail-closed exclusions consumed 95.1%
  of multi-variant statements. B002 is a strict family-unseen hard/OOD pool.
- Confirmed solve 11/512 = 2.1% after clean infra adjudication (75/75 cells
  re-verified; 20 unresolved cells kept as missing; sensitivity [11, 19]).
- Hindsight headroom is large (no-skip 86.21%, skip-allowed 98.44% on the
  allocated-cap axis) but comes mostly from triaging the rare solvable
  theorems; both are unattainable upper bounds on a single trajectory.
- The 2.1% must not be generalized to the full Promptset (source x outcome:
  synthetic 31.0%, autoformalizer 0.5%, human 0.0%).
- Pipeline calibration (diagnostic-only): 13/64 = 20.3% on a frozen E023
  subset vs the historical 23.4% => pipeline healthy.

## 4. B003 — multi-seed response + cross-fitted gate

- 192 x 8 = 1536 trajectories (4 shards, ~20h45m; merged sha256 `94f12dc4...`,
  missing=0). Pool: 9/192 theorems (4.7%) and 20/1536 trajectories (1.3%)
  ever solve; first successes only at >= 2048.
- Primary set S6 = 185/192 (9 low-coverage cells, all k=0). Fold-level
  coverage audit passed (0.11% / 0.00% of cells below 3/4 valid trials).
- Cross-fitted no-skip policy vs uniform (expected solved, N=185):
  - N*2048: 0.500 -> 2.042, absolute +1.542
  - N*3072: 1.000 -> 2.042, absolute +1.042
  - N*4096: 2.542 -> 2.042, absolute -0.500 (negative)
- Relative gains are inflated by rare-event denominators and were NOT used as
  the GO basis. The full-procedure bootstrap (re-fit + re-allocate + re-eval,
  1000 replicates) agrees closely with the conditional CI.
- 176/185 theorems show zero response at every budget: there is no learnable
  signal on this pool for a deployable predictor.
- The negative full-budget endpoint (-0.500) is the empirical reason the
  method is no longer called an "Oracle" (section 1).

## 5. Gate outcome

Preregistered cross-fitted no-skip gate (V2-B003 manifest): **NO-GO B004** —
accepted by the owner 2026-09-21. **V2-B004 = NOT RUN — stopped by the B003
gate.** No heuristic / XGBoost / MLP / small encoder was trained; no allocator
training corpus was generated. B004 is not a failed experiment but a formally
stopped one.

## 6. Research plan impact

- RQ2, RQ3: **COMPLETE** (evidence above).
- RQ4: **STOPPED BY PREREGISTERED GATE — predictor not trained**.
- Track C original 2x2 (prover x allocation) design: **superseded before
  launch** — the Adaptive arm did not pass the B003 gate, so the RL x Adaptive
  factorial study is not required. Track C will be re-designed and separately
  preregistered after Track A A4, focusing on the family-clean
  `theta0 vs theta_RL*` endpoint evaluation.
- Not permitted: new Track B experiments, re-selecting the pool, or rescuing
  the allocator by changing the distribution.

## 7. Integrity notes

- Frozen artifacts / manifests were not rewritten during closeout; the
  terminology correction lives in new documents only.
- No result was fabricated for B004.

## 8. Artifact index

| stage | key artifacts |
| --- | --- |
| B001 | `V2-E001.yaml`; `docs/v2/b1_budget_semantics.md`; registry alias V2-E001 -> V2-B001 |
| B002 | `V2-B002.yaml`; rollouts sha `523b6124...`; `v2_b002_adjudication.json`; calibration set/analysis; `v2_b002_representativeness.json`; `docs/v2/b002_memo.md` |
| B003 | `V2-B003.yaml`; eval set sha `75de2fda...`; rollouts sha `94f12dc4...`; `v2_b003_analysis.json`; `v2_b003_audit.json`; `docs/v2/b003_memo.md`; runner/analyzer/audit scripts + tests |
