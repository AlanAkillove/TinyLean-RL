# TinyLean-RL V2 — Experiment Protocol

Frozen 2026-09-19 (V2-0) on `main2`. Applies to all new V2 experiments (`V2-A###` / `V2-B###` / `V2-C###`; naming amendment 2026-09-19, registry: `experiments/manifests/v2/registry.yaml`).
V1 protocol and records remain authoritative for V1 numbers (frozen; see [`legacy_evidence.md`](legacy_evidence.md)).

## 1. Invariants carried over from V1

- Verifier: local Kimina Lean Server (`projectnumina/kimina-lean-server:2.0.0`); a candidate counts as
  solved only with **no error-severity message and no `sorry`** (strict rule; warnings do not fail a proof).
- Every formal artifact records provenance: host, GPU, git revision, model/checkpoint revision,
  dataset/fixed-set revision, verifier image, sampling config, created_at
  (carry-over from `docs/dual_server_collaboration.md` §15).
- Raw outputs stay out of Git; manifests record metadata, hashes and result summaries.

## 2. Data splits

- **All splits are at the theorem level.** Different seeds / budgets / checkpoints of the same theorem
  must never straddle train/test.
- Near-duplicate theorem families stay within one split: normalize statements (whitespace / comments /
  binder names) and group near-duplicates before splitting; the **group**, not the raw row, is the
  split unit.
- Splits are frozen in the experiment manifest **before** the formal run; no re-splitting after seeing results.

## 3. Budget definitions (must not be conflated)

Three distinct quantities are always reported separately:

- `allocated max-token cap` — the per-theorem generation cap (e.g. 1024);
- `actually generated tokens` — decoder tokens really produced;
- `wall-clock / GPU time` — measured runtime.

It is **forbidden** to claim "same runtime" from "same token cap": equal `max_new_tokens` does not
imply equal generated tokens or equal runtime.

## 4. Decision-model metrics (allocator evaluation)

Minimum set: log loss / BCE, Brier score, calibration (reliability curve / ECE), and per-budget
prediction quality.

Prediction accuracy alone is **not** the primary end metric of the project.

## 5. Allocation metrics (primary)

Primary end metric: **solved theorems under a total compute budget `B_total`.**

Also report: total generated decoder tokens; solved theorems; tokens per solved theorem; GPU-seconds
per solved theorem; wall-clock time; budget utilization (`Σ b_i / B_total`); regret / gap to the
oracle allocation.

Primary figures:

1. solved theorems vs total generated decoder tokens;
2. solved theorems vs GPU time.

## 6. Statistics

- Paired comparisons are theorem-level (paired bootstrap CIs, win/tie/loss, exact McNemar on solved
  indicators) — same conventions as E023/E024.
- No candidate-level pseudo-independent significance tests.
- Effect sizes are reported with CIs; single-number claims without CIs are not acceptable as results.

## 7. Pre-registration and freeze discipline

- Every formal (`V2-A###` / `V2-B###` / `V2-C###`) experiment gets a manifest **before** the run
  (`experiments/manifests/v2/`; start from a template).
- Each manifest states its own freeze rule (what cannot change after launch).
- Failed / aborted runs keep their manifest with an updated status and a status note — never deleted,
  never rewritten to hide an outcome.

## 8. V2 theorem-role registry (amendment 2026-09-19)

The V2 test pool is partitioned once, deterministically, into theorem-level roles
(`experiments/manifests/v2/theorem_role_registry.json`; built by
`scripts/build_v2_theorem_roles.py`, fail-closed like the holdout builder):

```text
B-train 60% | B-validation 10% | B-test 10% | A-selection 10% | C-joint-holdout 10%
```

- Pool = the Promptset statements with no V1 consumption history (7,620 unique minus the
  763 V1-used and the 128 sealed E023 holdout = 6,729).
- Role = `sha256("<salt>|<normalized statement>")` bucket; the normalized statement is the
  split unit (near-duplicate families move together, §2). `B1-audit-reserved` statements
  (fly122 `V2-B001`) are carved out of every role.
- A-selection is split by a frozen order hash into **A3-primary** (materialized as
  `v2_a001_selection_set.json`, the V2-A001 selection set) and **A-reserve** (reserved for
  the conditional second A-track decision only).
- Experiments freeze their theorem subset from their role's partition **before** launch;
  cross-role reuse requires a new registry amendment, never an ad-hoc re-draw.
