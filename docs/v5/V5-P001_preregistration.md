# V5-P001 — Offline Process-Reward Recoverability Audit

**Preregistration.** Written and committed in **Phase A**, before any historical process label
exists: at the moment of this commit the string `PROCESS_STRUCTURED_RECOVERABLE` has never been
evaluated on a historical candidate, no V5 oracle call has been made on historical data, and the
Phase-A exit condition reads **`historical process labels inspected = 0`**. Every quantity below is
`FROZEN`; the machine-readable twin is `experiments/manifests/v5/V5-P001.yaml` and
`scripts/v5_p001_spec.py` (constants), which the runner, the analyzer and the tests all read.

| | |
|---|---|
| Authority | owner directive 2026-09-26, "TinyLean-RL enters V5", §1–§34 |
| Direction | `Process-Verified RL for Tiny Lean Provers` (V5) |
| Experiment | `V5-P001`, first (and only authorized) V5 experiment |
| Question | owner §1/§13: on the existing V1 RLVR surface, how much of the outcome-level failure mass is *process-recoverable* under the paper-compatible first-error credit d1/d2? |
| Frozen manifest | `experiments/manifests/v5/V5-P001.yaml` (machine-readable twin of this document) |
| Namespace registry | `experiments/manifests/v5/registry.yaml` |
| Surface | `experiments/manifests/v5/v5_historical_surface.json` (sha256 `b3c8f472…3c5902`) |
| Oracle fixtures | `experiments/manifests/v5/v5_process_oracle_validation.json` (sha256 `9153fd2f…79a869`) |
| Oracle design | [`docs/v5/process_oracle_design.md`](process_oracle_design.md) |
| Process oracle | `scripts/v5_process_oracle.py` |
| Surface reconstruction | `scripts/v5_p001_reconstruct.py` |
| Phase-B runner | `scripts/v5_p001_process_run.py` (stages `preflight`, `process`, `freeze`, `validate`, `status`) |
| Amendments | [A](V5-P001_amendment_A.md) — two-strike wedge ⇒ censor the candidate, bounded restore, continue (owner-approved 2026-09-26, pre-outcome, execution-code only) |
| Canonical analyzer | `scripts/v5_p001_analyze.py` (exactly one run, after the freeze) |
| Tests | `tests/test_v5_process_oracle.py` |
| Run directory | `runs/v5_p001_process/` (labels, raw items, manifest, freeze, validation) |
| Result artifact | `experiments/manifests/v5/V5-P001_results.json` |
| Hosts | fly90 = coordination/analysis (CPU only); fly122 = the dedicated oracle instance |
| Training | **NOT AUTHORIZED.** `V5_R001_TRAINING_LAUNCHED: NO`; no weight update in V5-P001 |
| Generation | **NONE.** `NEW_MODEL_GENERATION: 0`; the surface is historical text only |

---

## 1. What is being tested, and what is not

V5-P001 is an **offline process-label recoverability audit** over a frozen set of *already generated*
V1 rollouts. It asks a single question:

> Among primary groups whose entire n=8 rollout batch failed at the outcome level (`ALL_FAIL`), for
> what fraction does at least one candidate admit the paper's first-error process **credit structure**
> — a locally verified tactic prefix *and* the accused tactic both mappable to a generated token —
> so that a process-supervised objective would have had a gradient-bearing, non-degenerate signal?

What V5-P001 is **not**, and may not be reported as:

* not training, RL, SFT, distillation, controller fitting or optimizer work of any kind (owner §16:
  "Do NOT update weights. Simply produce the frozen per-token advantage surface");
* not a capability, pass-rate or scaling claim about theta0 (a solved/unsolved mixture is measured as
  a *label source*, not as an achievement);
* not a re-opening of V1, V2, V3 or V4 (the sealed V3 93-component reserve is untouched:
  `SEALED_RESERVE_TOUCHED: 0`);
* not a new generation experiment: no theta0 rollout, no repair attempt, no prospective V3/V4 samples
  and no new theorem family is used as the primary surface (owner §6);
* not a redefinition of the canonical whole-proof verifier: the frozen `VerifyOutcome` semantics are
  reused verbatim; the process oracle only *reads* the elaboration info tree on top of them;
* **not a new scalar invention.** The primary credit is the paper-compatible d1 = −0.05 / d2 = −0.10
  first-error propagation; no normalized-prefix scalar is constructed, and no normalized-prefix
  surrogate may be substituted if the primary is weak (owner §2, §12).

---

## 2. Primary surface (owner §4–§6)

The surface is the historical **V1 RLVR rollout dumps only**: `seed1` (`runs/p3b_pilot/rollout_data`),
`seed2` (`runs/m1_seed2/rollout_data`), `seed3` (`runs/m1_seed3/rollout_data`), n = 8 per group.

* Grouping replicates the V3 audit exactly: `input` equality + contiguity, `\n`-split JSONL, label
  from `score_sum` (0 → `ALL_FAIL`, ≥ size → `ALL_SUCCESS`, else `MIXED`), statement joined through
  `FORMAL_BLOCK_RE` → `normalize` → `statement_id` → `component_id`.
* **Contamination policy (frozen, V3 reuse):** a candidate whose `tool_feedback` starts with
  `# System Error:` contaminates its whole group; the group is excluded from the primary surface and
  kept in the raw/all-groups table.
* Reproduced binary baseline (binary reconstruction must pass before any process work, owner §5):

| surface | groups | `ALL_FAIL` | `MIXED` | `ALL_SUCCESS` | contaminated |
|---|---|---|---|---|---|
| all groups | 720 | 603 | 110 | 7 | 34 groups / 103 candidates |
| **primary** | **686** | **575** | **104** | **7** | — |

* Primary candidates: **5488** = 2476 code-extractable + 3012 sentinel
  (`No proof found in the output.` / `Theorem statement couldn't be parsed from statement.`).
* Primary `ALL_FAIL` mass (the primary denominator): **575 groups**, of which 95 have zero code
  candidates; 1736 code candidates and 2864 sentinel candidates; per-seed 193 / 197 / 185 groups;
  `n_code` histogram `{0:95, 1:92, 2:90, 3:88, 4:52, 5:51, 6:41, 7:37, 8:29}`. These are
  pre-outcome descriptive facts of the frozen surface, not process labels.
* Provenance: dataset sha256 `68f33cee…8e1750`, registry sha256 `f5074c96…a0024`, V1 sources manifest
  sha256 `c405f5f6…e5168`; the frozen processing order (primary groups in surface order, slot order)
  hashes to `d04283eeb133f94bc5ba71ed36b758210486bfe2fa88126c1da9ab18c5da0f71`.

No theta0-only generation, no repaired sample, no V3/V4 prospective sample is part of the primary
surface; the raw/all-groups table is reported alongside, never instead.

---

## 3. Oracle, fixtures and determinism (owner §7, §21, §25, §26)

* A **dedicated** Lean server instance on fly122: container `tinylean-rl-lean-oracle-v5`,
  endpoint `http://127.0.0.1:8020`, image digest `sha256:588a2cbb…e3ffd9`, `LEAN_SERVER_MAX_REPLS=1`,
  `MAX_REPL_MEM=8G`, server timeout 120 s, client = server + 60 s slack, batch size 1, at most 2
  bounded single retries, cold canary 300 s / warm canary 120 s, at most 192 container recoveries per
  Phase-B run. The pooled production verifier is not touched.
* The request body is the historical `pred` **verbatim** — the exact string `extract_proof_from_text`
  produced — verified against the stored `pred_sha256` before submission.
* Infrastructure outcomes (`VERIFIER_TIMEOUT`, `VERIFIER_SERVER_ERROR`, `VERIFIER_UNHEALTHY`,
  `UNRESOLVED_INFRA_ERROR`) are **censored, never failures** (owner §26). A candidate-specific infra
  event is re-checked with a canary; an unhealthy instance triggers a bounded restart + cold canary;
  if the instance is unhealthy again after a recovery the run stops and the owner is informed —
  **except under [Amendment A](V5-P001_amendment_A.md) (owner-approved 2026-09-26, documented before
  any analyzable label set)**: a candidate that is still an infrastructure outcome after its bounded
  post-recovery re-submission *and* leaves the canary unhealthy again is censored
  (`PROCESS_ORACLE_INFRA`, excluded from the numerator, kept in its group's denominator), one more
  bounded recovery restores the instance, and the run continues with the next candidate; a restore
  that cannot be re-warmed still stops the run.
* **Tactic decomposition is done by Lean, not by text** (owner §7): tactics come from the infotree
  nodes whose name starts with `Lean.Parser.Tactic.`, with the pure-sequence wrappers
  (`tacticSeq`, `tacticSeq1Indented`, `tacticSeqBracketed`) skipped and exact `(name, span)` duplicates
  collapsed. Splitting by newline, semicolon text, regex or indentation alone is forbidden and unused.
* **Position convention** (calibrated in Phase A, evidence in the design doc): lines 1-based, columns
  0-based, finish exclusive, and every position is in the **frame** = submitted code minus a maximal
  leading prefix of blank/`import` lines (a comment line stops the stripping).
* 13 engineering fixtures (F01–F13) cover the goal cases: fully successful, first-tactic error,
  prefix-then-type-error, prefix-then-`unsolved goals`, unknown identifier, typeclass failure, nested
  `by`, combinator, case-style block, multi-line tactic, syntax/parser failure, term-style
  non-extractable, empty body. Frozen fixture-set sha256 `2bfb0f4f…be237fe`; each fixture passed
  expectation checks **twice** with identical derived records (owner §21: nondeterminism is a hard
  infra failure, not a tolerance).
* Phase-B re-derivation evidence: 24 candidates processed in the fly122 smoke were re-derived from
  the sources and the archived raw items with 24/24 identical record hashes, before the canonical run.

---

## 4. Statuses, blame and the credit surface (owner §2, §10, §11, §16)

`process_status` ∈ {`SUCCESS`, `PREFIX_BEARING_FAILURE`, `FIRST_TACTIC_FAILURE`,
`PARSE_OR_SYNTAX_FAILURE`, `FORMAT_NO_CODE`, `PROCESS_ORACLE_INFRA`}. Blame (the first error) uses the
frozen attribution order `contains` → `range_last` → unattributed, with a `sorry`-range fallback when
no error message attributes; the design doc pins the exact rules.

Credit (frozen, paper-compatible):

* `d1 = −0.05` for every parsed tactic **strictly before** the blamed tactic;
* `d2 = −0.10` for the blamed tactic **and everything after it** (first-error propagation);
* `success = 1` when the whole proof verifies;
* credit is written **only on the first generated token** of each tactic's span, obtained through the
  frozen tokenizer mapping of §5 — never smeared over a tactic's interior;
* when two tactics share a first token, precedence is `d2 > d1 > success`;
* nothing is fabricated: no-code / unparseable / infra candidates receive **no** tactic labels.

Sensitivity (owner §23, after the primary classification only, label-preserving by construction):
`(−0.05, −0.50)` and `(−0.10, −0.10)`, reported as credit-mass recomputations; the recovery
classification is invariant because `structured_recoverable` tests mappability, not values.

---

## 5. Token mapping (owner §9)

Chain: tactic span (frame) → character span in the submitted `pred` → character span in the generated
response → **first generated token** under the frozen `models/weights/kimina_distill_0_6b`
`Qwen2TokenizerFast` (`vocab_size = 151643`, `add_special_tokens = False`, tokenizer.json sha256
`aeb13307…d2ae4`, config sha256 `5d27a191…5dd72`).

Mapping statuses: `exact` (a token starts exactly at the tactic's first character), `contained`
(exactly one token contains it — the usual case, since BPE merges the leading indentation),
`ambiguous`, `outside_response` (the span lies in the prompt-derived formal-statement head),
`response_only` (no lattice available), `unmapped`. `exact ∪ contained` are the *mappable* statuses,
and a mappable span must additionally be **verbatim present in the response**
(`span_in_response is True`); otherwise it counts as `span_not_in_response` and credits nothing.

No tactic outside the generated response may ever receive credit.

---

## 6. Structured recoverability (owner §11, §12)

`PROCESS_STRUCTURED_RECOVERABLE(candidate)` — the primary predicate — is true iff

1. `process_status ∈ {PREFIX_BEARING_FAILURE, FIRST_TACTIC_FAILURE}`, and
2. the blamed tactic exists with index ≥ 1, and
3. at least one **locally verified** tactic strictly before the blame has a mappable first token, and
4. the blamed tactic itself has a mappable first token.

The strict inequality in (3) excludes the trivial "every parsed tactic is immediately erroneous" case.
`ANY_ACTIVE` (a failed candidate with any nonempty verified prefix) is reported **only** as a
secondary diagnostic; it is never substituted for the primary metric if the primary is weak
(owner §12). No-code, unparseable and infra candidates are never recoverable.

---

## 7. Units, denominators and estimators (owner §13)

* **Primary estimand:** `RecoveryRate = P(PROCESS_STRUCTURED_RECOVERABLE | ALL_FAIL)`, computed at the
  **primary all-fail group** level as the existence of ≥ 1 qualifying candidate in the group. Unit =
  one primary `ALL_FAIL` group. Denominator = evaluable primary `ALL_FAIL` groups, i.e. groups that
  contain at least one conclusive candidate (a group that is 100 % code-free is excluded and counted;
  a group with no code *and* no conclusive candidate cannot be a recovery host).
* **Censoring trichotomy** (frozen): `no_code` (sentinel, by construction not a Phase-B infra event),
  `censored` (Phase-B oracle infra → excluded from the numerator, kept in the denominator of its
  group), `conclusive`. Every denominator in the report is stated with respect to this trichotomy.
* **Candidate-level rate** = recoverable conclusive code candidates / code candidates in primary
  `ALL_FAIL` groups: **pre-declared secondary descriptive only**, never a gate.
* **Per-seed** estimates for `seed1/2/3` (G2).
* **Length robustness (G3):** evaluable groups sorted by `(group_max_tactics, group_key)` into four
  equal-count quartiles (zero-code groups included, with `group_max_tactics = 0`); a quartile with
  fewer than 5 groups cannot pass.
* **Uncertainty:** family/component-cluster bootstrap (cluster = `component_id`, `statement_id`
  fallback), 10 000 resamples, seed `20260926`, percentile 95 % interval, `alpha = 0.05`.
* Quantiles of credited token positions, credit conflicts, blamed-mappability and the status/failure
  decomposition are reported as descriptive tables.

---

## 8. Gates and classification (owner §22)

Evaluated in this frozen order; the first decisive step decides the label, and no gate may be rescued
by changing d1/d2, parser definitions or denominators.

| gate | requirement |
|---|---|
| **E1** | mappable-with-verbatim-span fraction among parsed tactic nodes ≥ **0.95** |
| **E2** | conclusive oracle outcomes / code-extractable primary candidates ≥ **0.90** |
| **G1** | pooled `RecoveryRate` ≥ **0.25** *and* bootstrap CI lower bound > **0.15** |
| **G2** | point estimate ≥ **0.20** in **each** seed |
| **G3** | rate > **0.15** in ≥ **3 of 4** length quartiles (each with ≥ 5 groups) |

* `PROCESS_SIGNAL_GO` iff E1, E2, G1, G2, G3 all pass.
* Otherwise, in order: `INCONCLUSIVE_PROCESS_ORACLE` (E1 or E2 fails, censored share > 0.05, or any
  seed keeps < 50 % of its all-fail groups evaluable) → `NO_PROCESS_RECOVERY` (G1 fails) →
  `PROCESS_SIGNAL_SEED_UNSTABLE` (G2 fails) → `PROCESS_SIGNAL_LENGTH_CONFOUNDED` (G3 fails).
* `PROCESS_SIGNAL_GO` authorizes **only** drafting a V5-R001 preregistration (owner §31): 
  `V5_R001_DRAFT_CREATED` may become `YES`, `V5_R001_TRAINING_LAUNCHED` stays **NO**. No training of
  any kind is launched in V5-P001, whatever the label (owner §32).

---

## 9. Phase discipline (owner §27–§30, §34)

* **Phase A** (this commit): oracle + fixtures + mapping, binary reconstruction, preregistration,
  gates, tests, lint, commit/push/sync to fly122. Exit condition: `historical process labels
  inspected = 0`.
* **Phase B**: `preflight` (re-verify every planned candidate against the frozen surface and the V1
  files; no oracle call, no label) → `process` (each primary candidate submitted **exactly once**,
  append-only labels, raw items archived per candidate, resumable, stdout restricted to
  infrastructure counters — no scientific metric is printed) → `freeze` (label hash, coverage,
  raw-item manifest, infrastructure accounting; a debug-limited run cannot freeze in the canonical
  directory) → `validate` (full re-derivation of every stored record from the sources and the archived
  raw items, hash-compared; no oracle calls).
* **Then, exactly once:** the canonical analyzer over the frozen artifacts. It refuses to run on a
  missing/mutated/failed freeze, on a debug-limited run, on a stamp mismatch, or on a plan/label
  mismatch.
* Phase B must not stop early after seeing `RecoveryRate`, and no live scientific metric may be
  printed. If any scientific metric has already been emitted, the run **stops and reports the owner**
  (`STOP_FOR_OWNER`).
* The commit hash of this preregistration and the code stamp (git HEAD + script hashes + surface and
  validation hashes) are part of every Phase-B artifact, so the analyzed bytes are exactly the frozen
  bytes.

---

## 10. Frozen artifacts and result template (owner §33, §34)

Artifacts: `docs/v5/V5-P001_preregistration.md`, `docs/v5/process_oracle_design.md`,
`docs/v5/V5-P001_amendment_A.md`, `experiments/manifests/v5/V5-P001.yaml`,
`experiments/manifests/v5/registry.yaml`,
`experiments/manifests/v5/v5_historical_surface.json`,
`experiments/manifests/v5/v5_process_oracle_validation.json`, `scripts/v5_process_oracle.py`,
`scripts/v5_p001_reconstruct.py`, `scripts/v5_p001_analyze.py` (+ the Phase-B runner
`scripts/v5_p001_process_run.py`).

`V5_P001_RESULT` reports, in this order: `process_oracle`, `historical_surface`, `primary_recovery`,
`by_seed`, `length_robustness`, `candidate_mechanism`, `token_credit`, `format_decomposition`,
`sensitivity`, `FINAL_CLASSIFICATION`, `permitted_claim`, `limitations`, `compute`, `provenance`,
`if_GO` (draft created / training launched), plus `NEW_MODEL_GENERATION: 0` and
`SEALED_RESERVE_TOUCHED: 0`.

---

## 11. Compute and safety (owner §16, §24, §25)

* No GPU generation anywhere in V5-P001; fly90 stays a coordination/archive node and does **not**
  become a model-compute node; the oracle runs on the dedicated fly122 instance only.
* The V3 93-component `SEALED` reserve, the future family-clean capability holdout and any new theorem
  family are untouched.
* Every infra failure is bounded and fail-close: timeouts are censored (missing data), never a failed
  proof; the recovery budget is finite; an unhealthy instance after recovery stops the run — except
  under [Amendment A](V5-P001_amendment_A.md)'s two-strike rule, where the *candidate* is censored
  and the instance is restored by one more bounded recovery; the restore itself remains fail-close.

## 12. Known limitations, declared before the outcome

1. One historical surface only: three GRPO seeds of a single sub-billion model; nothing here
   generalizes to another model or budget.
2. Process labels come from a single oracle pass per candidate; infra outcomes are censored, not
   failed, and a censored candidate can only shrink the numerator.
3. `RecoveryRate` is an existence predicate at group level (≥ 1 qualifying candidate), not a
   per-candidate success rate; the candidate-level rate is secondary and descriptive.
4. Group-level `ALL_FAIL` includes the 95 zero-code groups, which can never host a recovery; the
   quartile analysis is the pre-declared robustness view of that length/format confound.
5. The d1/d2 credit surface is *produced and frozen*, never applied: V5-P001 cannot and does not
   measure any training effect.
