# TinyLean-RL V2 — Research Plan

**Main question**

> How can a sub-billion-parameter Lean prover solve more theorems under limited compute?

**Status: V2-0 COMPLETE; TRACK A A0+A1 COMPLETE / A3 PREREGISTERED; TRACK B CLOSED (B001-B003 complete; B004 NOT RUN - stopped by the B003 gate, 2026-09-21); TRACK C REDESIGN PENDING A4.**

_(Recorded 2026-09-19 on `main2`. V1 is frozen: see [`legacy_evidence.md`](legacy_evidence.md) and tag `v1-research-freeze-20260919`.)_

_(Amendment 2026-09-21: Track B closed by the V2-B003 preregistered gate - RQ2/RQ3 COMPLETE; RQ4 STOPPED BY PREREGISTERED GATE (predictor not trained); B004 NOT RUN; the original Track C 2x2 design is superseded before launch and Track C will be re-designed after Track A A4. See [`track_b_closeout.md`](track_b_closeout.md).)_

The project runs as **two parallel tracks plus a joint final evaluation** (amendment
2026-09-19, below):

1. **Track A — RL / prover improvement** (fly90): diagnostic + minimal-recipe RL work that
   ends in a frozen final RL prover checkpoint `theta_RL*` (RQ1 is its starting evidence).
2. **Track B — adaptive compute allocator** (fly122): inference-time adaptive test-time
   compute allocation (RQ2–RQ4).
3. **Track C — joint evaluation** (fly90): the prover × allocation main study and the final
   benchmark (RQ5), started only after Track A and Track B complete.

---

## Track structure (amendment 2026-09-19)

The former linear V2-1 → V2-6 plan is reorganized into two parallel tracks plus a joint
evaluation. The RQ set and the V2-0 protocol freeze are unchanged; formal numbering now
uses the track-specific namespaces `V2-A###` / `V2-B###` / `V2-C###` (naming amendment
2026-09-19; registry: [`../../experiments/manifests/v2/registry.yaml`](../../experiments/manifests/v2/registry.yaml)),
and manifests carry a `track: A | B | C` field
(see [`../../experiments/manifests/v2/README.md`](../../experiments/manifests/v2/README.md)).

```text
Track A — RL / prover improvement      (fly90; owns theta_RL*; stages A0–A4)
Track B — adaptive compute allocator   (fly122; independent; phases V2-1…V2-4)
Track C — final joint evaluation       (fly90; only after A AND B are done; V2-5…V2-6)
```

- **Track A**: diagnose the V1 RL dynamics, refine the recipe only where evidence demands
  it, then select and freeze `theta_RL*` with an explicit selection rationale; working plan
  and A1 decision memo: [`track_a_rl_plan.md`](track_a_rl_plan.md).
- **Track B**: frozen scope unchanged — budget semantics (V2-1), budget-response dataset
  (V2-2), uniform-vs-oracle headroom (V2-3), lightweight allocator (V2-4); independent of
  Track A progress.
- **Track C**: the original 2x2 prover × allocation study (V2-5) is **superseded before launch** (2026-09-21: the Adaptive arm did not pass the B003 gate); the final benchmark (V2-6) remains, and the redesigned Track C (family-clean `theta0 vs theta_RL*` endpoint evaluation) will be separately preregistered after Track A A4.
- **Cross-track interface**: Track A hands over exactly one artifact — the frozen
  `theta_RL*` (checkpoint + manifest + hashes) at A4; no intermediate checkpoint shuttling.
  Shared interface/schema changes are allowed on both sides, but allocator training and the
  budget-response dataset stay in Track B.

## RQ1 — RLVR under constrained compute (Track A evidence base)

Reuse V1 results as-is; do **not** repackage them as new experiments. The permitted rigorous summary of V1:

- RL plumbing / trainability is established (P3-B short pilot; E019 step30→60 extension; E020/E022 seed replication).
- Positive / mixed reward groups are reliably observed (three-seed IGR 0.158 / 0.150 / 0.150).
- Multi-seed training dynamics are reproducible ("rise-then-fall" shapes, seed-dependent peak positions).
- Under the present 60-step low-compute regime, held-out capability gain is small and seed-sensitive /
  inconclusive (E023: mean Δ −0.91pp, 1/3 seeds positive, all 95% CIs straddle zero, no significant
  McNemar flips).
- **No stable benchmark improvement is claimed.**

## RQ2 — Heterogeneous inference-compute demand (Track B)

Define the per-theorem compute-response curve

```text
p_i(b) = P(solve theorem i | token budget b)
```

Initial candidate budgets: `{512, 1024, 2048, 3072, 4096}`.

Goal: measure `p_i(b)` per theorem and quantify heterogeneity across theorems — this is what any
allocator would have to exploit. Curves are measured empirically; monotonicity is **not** assumed
(see [`data_contract.md`](data_contract.md) §5).

**Status (2026-09-21): COMPLETE.** Measured on the family-clean pool (V2-B002 pilot; V2-B003 192 x 8): heterogeneity is extreme - 176/185 theorems show zero response at every budget and 9/192 theorems ever solve (first successes >= 2048). Evidence: `b002_memo.md`, `b003_memo.md`, `track_b_closeout.md`.

## RQ3 — Oracle allocation headroom (Track B)

For a global budget `B_total`, study

```text
max  Σ_i p_i(b_i)     subject to     Σ_i b_i ≤ B_total
```

First compare **uniform allocation** vs **oracle allocation**. If the oracle-vs-uniform headroom is
small, the adaptive-allocator line may be stopped — it is not forced to continue.

**Status (2026-09-21): COMPLETE.** Hindsight headroom is large (B002: no-skip 86.21% / skip-allowed 98.44% allocated-cap savings) but does NOT transfer: the cross-fitted empirical-response allocator (B003) gained only +1.5 / +1.0 expected solves at N*{2048, 3072} and scored -0.5 below uniform at N*4096 on the strict family-unseen hard subset. The oracle-vs-uniform headroom line is therefore stopped (see `track_b_closeout.md`).

## RQ4 — Lightweight adaptive allocator (Track B)

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

**Status (2026-09-21): STOPPED BY PREREGISTERED GATE — predictor not trained.** The B003 gate found no stable cross-seed theorem-level signal (176/185 zero-response; negative full-budget endpoint), so no heuristic / XGBoost / MLP / encoder was trained and no allocator training corpus was generated.

## RQ5 — RLVR × adaptive inference (main study, Track C)

Final 2×2 design:

| prover | uniform | adaptive |
| --- | --- | --- |
| Kimina Distill θ0 | A | B |
| TinyLean RLVR θRL | C | D |

Question: are training-time RL and inference-time allocation complementary?

The official Kimina-Prover-RL-0.6B may serve as an **external reference**, but does not replace the
main 2×2.

**Status (2026-09-21): 2x2 design superseded before launch** — the Adaptive arm did not pass the B003 gate, so the RL × Adaptive factorial study is not required. A redesigned Track C (family-clean `theta0 vs theta_RL*` endpoint evaluation) will be preregistered separately after Track A A4.

## Optional extension

Sequential / contextual-bandit allocation is an **optional extension only**; it must not be a
prerequisite for the main project line to stand.

## Phase plan (track-mapped)

| Track | Phase | Scope | Status |
| --- | --- | --- | --- |
| — | V2-0 | Repository / protocol freeze | complete (no experiment run) |
| A | A0 | Track structure + manifest conventions (`track:` field) | complete |
| A | A1 | V1 RL-evidence diagnosis → RL recipe decision memo | complete ([`track_a_rl_plan.md`](track_a_rl_plan.md)) |
| A | A2 | Minimal recipe refinement (new training only if A1/A3 justifies it) | gated on A3 |
| A | A3 | Checkpoint comparison on a dedicated sealed selection set | preregistered (`V2-A001`; A3-primary of the theorem-role registry) |
| A | A4 | Freeze `theta_RL*` (checkpoint + hashes + rationale) | not started |
| B | B0 | Verifier reliability guard (shared infrastructure) | complete; merged into main2 (fly122 commit) |
| B | B1 | Budget-semantics audit (raw id `V2-E001` = alias `V2-B001`) | complete (2026-09-20; canonical) |
| B | V2-1 | Prefix-vs-direct budget equivalence audit | complete via B1 (2026-09-20) |
| B | V2-2 | Budget-response dataset construction | complete via B002 + B003 (2026-09-21) |
| B | V2-3 | Uniform-vs-Oracle headroom study | complete via B002 hindsight + B003 cross-fitted (2026-09-21) |
| B | V2-4 | Lightweight allocator | NOT RUN — stopped by the B003 gate (2026-09-21) |
| C | V2-5 | Distill/RLVR × Uniform/Adaptive main study | superseded before launch (2026-09-21); Track C redesign pending A4 |
| C | V2-6 | External benchmark / MiniF2F final evaluation | pending A4 (redesigned Track C will be preregistered separately) |
| — | V2-X | Optional sequential/bandit extension | optional |

**No Track C experiment has run yet; Track A V2-A001 is interrupted-resumable (4/7 models complete; completes on fly90 per owner decision 2026-09-23); Track B is closed (B001 -> B003 complete; B004 NOT RUN — stopped by the B003 preregistered gate, 2026-09-21).** Numbers are assigned at formal (pre-registered) run time; raw
pre-amendment ids stay registered in the [registry](../../experiments/manifests/v2/registry.yaml).
Theorem-level role assignments (B-train / B-validation / B-test / A-selection /
C-joint-holdout + B1-audit-reserved) come from the frozen
[`theorem_role_registry.json`](../../experiments/manifests/v2/theorem_role_registry.json)
(exact-statement isolation only; the measured source-problem family leakage is in
[`family_leakage_audit.md`](family_leakage_audit.md), and future splits - the Track C final
holdout in particular - must use the family-component split defined there).

## Guardrails

- No V2 experiment runs during V2-0 (repository preparation only).
- Prefix-derived budget labels are *cheap derived labels*; they may not be claimed equivalent to
  direct-budget interventions before V2-1 completes.
- No monotonic regularization in the first allocator models; monotonicity is measured, not assumed.
- Interfaces are frozen before implementation; allocator code is added only when a phase requires it.
- Track A runs one formal RL job at a time on fly90; no new training without a specific
  diagnosed question (A1/A3), and every formal run is preregistered before launch.
- Checkpoint selection (A3) uses a dedicated selection set; the Track C final benchmark is
  never used to pick checkpoints, and the frozen `theta_RL*` is not swapped after A4.
- Track C's final holdout is drawn on **family components** - L3 source-family merged with
  L2 statement-skeleton and normalized nonempty natural-language equality as cross-name
  links - from a future family-granular registry amendment, never from the exact-statement
  registry (see [`family_leakage_audit.md`](family_leakage_audit.md) §5.3).
