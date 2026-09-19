# TinyLean-RL V2 — Documentation Index

**V2 research question**: *How can a sub-billion-parameter Lean prover solve more theorems under limited compute?*

Status: **V2-0 REPOSITORY PREPARATION COMPLETE; NO V2 EXPERIMENT HAS RUN.**

V2 has two complementary pillars:

1. **Training-time**: verifier-based RL / RLVR (inherited from the frozen V1 lineage).
2. **Inference-time**: adaptive test-time compute allocation (new in V2).

> All of P0–P3 / E001–E024 is **V1 historical evidence**: kept, not rewritten, not renumbered
> (branch `p3-linux`, tag `v1-research-freeze-20260919`).
> New experiments are numbered `V2-E001`, `V2-E002`, ... (see `experiments/manifests/v2/README.md`).

## Contents

| Document | Purpose |
| --- | --- |
| [`research_plan.md`](research_plan.md) | V2 main question, RQ1–RQ5, phase plan (V2-0 … V2-X), guardrails |
| [`experiment_protocol.md`](experiment_protocol.md) | Frozen protocol: theorem-level splits, decision-model metrics, allocation metrics, primary figures |
| [`data_contract.md`](data_contract.md) | Planned `budget_response` schema; prefix-truncation vs direct-budget labels; V2-1 calibration; no enforced monotonicity |
| [`legacy_evidence.md`](legacy_evidence.md) | Index of the frozen V1 evidence (links only; numbers live in the canonical docs/manifests) |

## Branches

- `p3-linux` — frozen V1 evidence (tag `v1-research-freeze-20260919`).
- `main2` — V2 canonical development branch (this documentation lives here).
- Dual-server collaboration rules: [`../dual_server_collaboration.md`](../dual_server_collaboration.md).

## Guardrails (summary)

- No V2 experiment has run; no V2 result, manifest or artifact exists.
- Prefix-derived budget labels may not be claimed equivalent to direct-budget interventions before V2-1 completes.
- Monotonicity of `p_i(b)` is a measurement question, not an assumption.
- Interfaces are frozen before implementation; no allocator code is added during V2-0.
