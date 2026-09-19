# TinyLean-RL V2 — Data Contract: `budget_response`

Planned record schema and labeling rules for the budget-response dataset (RQ2/RQ3).
This contract is **design-frozen**; **no `budget_response` data has been generated yet**.

## 1. Per-rollout fields (planned minimum)

| field | notes |
| --- | --- |
| `theorem_id` | stable id (source dataset key) |
| `theorem_statement` | frozen statement text |
| `dataset` / `split` | source + theorem-level split (see experiment protocol §2) |
| `prover_checkpoint` | model label / path |
| `checkpoint_hash` / `revision` | immutable identity of the weights |
| `prompt_template_revision` | prompt + chat-template revision |
| `sampling_config` | temperature, top_p, ... |
| `seed` | per-candidate sampling seed |
| `max_new_tokens` | the cap actually requested |
| `raw_generated_text` | raw decoder output |
| `token_ids` or token-count metadata | where feasible |
| `actual_generated_length` | decoder tokens actually produced |
| `generation_runtime` | wall / GPU seconds |
| `generation_status` | ok / truncated / error |
| `lean_candidate` | assembled Lean source |
| `verify_status` | verified / lean_error / verifier_error / ... |
| `verified` | bool (strict rule) |
| `verifier_error_category` | taxonomy bucket |
| `verification_runtime` | seconds |
| `environment/server revision` | server image + env revision |
| `artifact_hashes` | sha256 of this record's artifact |

## 2. Budget label source

Every budget label carries

```text
label_source = prefix_truncation | direct_budget
```

**Never mix the two.** They are different objects; every analysis must be able to separate them.

## 3. Prefix-derived dataset (cheap labels)

Planned: one `max_new_tokens=4096` rollout per candidate; its token trajectory is truncated at
`512 / 1024 / 2048 / 3072 / 4096`; each truncation is assembled and Lean-verified independently.

This yields **cheap derived labels** — not interventional measurements.

## 4. Direct-budget calibration (V2-1)

V2-1 must actually run each budget on a small sample and check at least:

- does same-seed `direct-1024` equal the first 1024 tokens of the 4096 rollout?
- do batch size / sampler / RNG streams break prefix equivalence?
- does proof assembly change with extra tokens (e.g. fragments arriving after a complete proof)?
- does prefix-derived success approximate direct-budget success?

Until V2-1 completes, no equivalence claim between prefix-derived and direct-budget labels is permitted.

## 5. Monotonicity

`p_i(b)` monotonicity is **not assumed**. Record measured curves (with CIs); analysis may study
violations. The first allocator models receive **no monotonic regularization**.

## 6. Storage policy

Raw rollouts/artifacts remain gitignored; manifests record hashes and summaries; every field above
must be reconstructable from stored artifacts alone (fail-closed reconstruction).
