# TinyLean-RL V2 — Documentation Index

**V2 research question**: *How can a sub-billion-parameter Lean prover solve more theorems under limited compute?*

Status: **V2 CLOSED — empirical/diagnostic freeze (2026-09-23). TRACK A: V1 RLVR trainability/dynamics evidence COMPLETE; V2-A001 TERMINATED before selection (outputs retained as historical evidence only); A2 NOT RUN; theta_RL* NOT FROZEN. TRACK B: CLOSED — B001 -> B003 COMPLETE; B004 NOT RUN (stopped by the B003 gate); raw id V2-E001 = alias V2-B001. TRACK C: NOT RUN — previous design superseded.**

> **Scope note (2026-09-23):** V2 is frozen as an empirical/diagnostic phase. No further checkpoint selection, adaptive-compute modeling, or confirmatory Track C experiment will be added to V2.

V2 runs as two parallel tracks plus a joint final evaluation (amendment 2026-09-19):

1. **Track A — training-time**: verifier-based RL / RLVR prover improvement (inherited from
   the frozen V1 lineage), ending in the frozen final checkpoint `theta_RL*` (fly90).
2. **Track B — inference-time**: adaptive test-time compute allocation (fly122).
3. **Track C — joint evaluation**: prover × allocation main study plus the final benchmark,
   started only after Track A and Track B complete (fly90).

> All of P0–P3 / E001–E024 is **V1 historical evidence**: kept, not rewritten, not renumbered
> (branch `p3-linux`, tag `v1-research-freeze-20260919`).
> New experiments use the track namespaces `V2-A###` / `V2-B###` / `V2-C###`; raw pre-amendment
> ids (`V2-E001`) and their canonical aliases are permanently registered in
> `experiments/manifests/v2/registry.yaml`.

## Contents

| Document | Purpose |
| --- | --- |
| [`research_plan.md`](research_plan.md) | V2 main question, RQ1–RQ5, Track A/B/C structure, guardrails |
| [`track_a_rl_plan.md`](track_a_rl_plan.md) | Track A (RL prover) plan + A1 RL recipe decision memo (V1 evidence synthesis, candidate next steps; inactive - see closeout) |
| [`v2_closeout.md`](v2_closeout.md) | Final V2 closeout (2026-09-23): data-supported findings only; motivation for the Jev-inspired RL-control direction |
| [`a001_termination_snapshot.md`](a001_termination_snapshot.md) | V2-A001 termination record: stop time, frozen hashes, retained artifacts; selection_outcome NONE / theta_RL* NOT FROZEN |
| [`experiment_protocol.md`](experiment_protocol.md) | Frozen protocol: theorem-level splits, decision-model metrics, allocation metrics, primary figures |
| [`data_contract.md`](data_contract.md) | Planned `budget_response` schema; prefix-truncation vs direct-budget labels; V2-1 calibration; no enforced monotonicity |
| [`legacy_evidence.md`](legacy_evidence.md) | Index of the frozen V1 evidence (links only; numbers live in the canonical docs/manifests) |
| [`b0_verifier_reliability.md`](b0_verifier_reliability.md) | Track B B0: verifier reliability audit and frozen verification policy (E024 used as forensic input only; no E024 claim) |
| [`b1_budget_semantics.md`](b1_budget_semantics.md) | Track B B1 (V2-E001): budget-semantics audit results and gate decision (canonical path PASS) |
| `experiments/manifests/v2/registry.yaml` | Canonical V2 experiment registry (track namespaces + raw-id aliases, e.g. V2-E001 = V2-B001) |
| [`../dual_server_collaboration.md`](../dual_server_collaboration.md) | Dual-server rules; V2 numbering amendment in §14 |

## Branches

- `p3-linux` — frozen V1 evidence (tag `v1-research-freeze-20260919`).
- `main2` — V2 canonical development branch (this documentation lives here).
- Dual-server collaboration rules: [`../dual_server_collaboration.md`](../dual_server_collaboration.md).

## Guardrails (summary)

- No V2 training run has happened. V2-A001 (Track A checkpoint-selection) is running on fly90; V2-E001 (Track B B1) is a completed generation-semantics audit (PASS); their manifests and raw artifacts exist under the track rules.
- Prefix-derived budget labels may not be claimed equivalent to direct-budget interventions before V2-1 completes.
- Monotonicity of `p_i(b)` is a measurement question, not an assumption.
- Interfaces are frozen before implementation; no allocator code is added during V2-0.
