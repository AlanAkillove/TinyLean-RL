# V5-P001 — offline process-reward recoverability audit: canonical result

Status: COMPLETE / `PROCESS_SIGNAL_GO` (all five gates pass, no failed gate)
Owner authority: V5 directive §1–§34 (2026-09-26) + Amendment A (two-strike wedge) + Analyzer
Amendment B (report-wiring repair with the pinned FreezeA lineage, 2026-09-27).
Execution HEAD `5b1c5d2` (Phase-B attempt 1, frozen as **FreezeA**); analysis HEAD
`7d0b139` (Amendment B). Canonical analyzer **productive run #1**, exactly once, on fly90.

Canonical analysis artifact: `experiments/manifests/v5/V5-P001_results.json`
(sha256 `ab46c12dbe098505ba8f3885fddb8af18ef2420dc0d6c66cc856a26e4d08ddd8`, 13 788 B).
This memo states what the frozen analyzer computed; it introduces no new endpoint and no
reinterpretation.

## 1. The question

On the existing V1 RLVR rollouts (frozen surface: 3 GRPO seeds of Kimina-Distill-0.6B theta0, 720
groups / 686 primary), what fraction of the outcome-level `ALL_FAIL` groups contain at least one
candidate that admits the paper-compatible first-error process-credit structure — verified tactics
strictly before the earliest erroneous tactic (d1 = −0.05) and the error plus everything after
(d2 = −0.10) — with the tactic → generated-token mapping established on a dedicated pinned Lean
oracle instead of text heuristics?

Primary estimand (frozen): `RecoveryRate = P(PROCESS_STRUCTURED_RECOVERABLE | ALL_FAIL)`, unit =
primary all-fail group; family/component-cluster bootstrap (10 000 resamples, seed 20260926,
percentile 95 %). All V5-P001 claims are offline process-label statements only: no training, no
model generation, no capability or gradient claim (permitted claim, verbatim in the artifact).

## 2. Execution record (Phase-B attempt 1 → FreezeA)

- Host fly122, dedicated read-only oracle `tinylean-rl-lean-oracle-v5` (pinned image digest
  `588a2cbb…`), endpoint `http://127.0.0.1:8020`, 120 s server timeout, ≤ 2 bounded single
  retries. Preflight 2026-09-26T12:48:17Z (5 488/5 488 candidates re-verified against the frozen
  surface and the V1 files, PASS); run_meta 2026-09-26T12:50:35Z; labels completed 2026-09-27
  04:37 (local, overnight); freeze 2026-09-27T03:08:22Z; structural validation
  2026-09-27T03:09:19Z.
- Plan: **5 488** candidates = 2 476 code-extractable + 3 012 sentinel (FORMAT_NO_CODE-only
  groups), each submitted **exactly once** (plan order `d04283ee…`). Attempt 0 was fail-closed
  under Amendment A and archived rename-only as `runs/v5_p001_process_attempt0_wedge/` on fly122
  (never analyzed); attempt 1 started fresh from candidate 1.
- Oracle outcomes (2 476 submissions): `lean_error` 1 935, `verified` 501, `sorry` 4,
  `verifier_timeout` 35, `verifier_server_error` 1. Infrastructure: 36 censored submissions
  (missing data, never a failed proof), 39 infra events, 2 isolated retries, 6 bounded container
  recoveries, 0 truncated partial records. `NEW_MODEL_GENERATION: 0`, `TRAINING: 0`.
- FreezeA (`v5_p001_process_freeze.json` sha256 `1d41c6a7…`): 5 488 labels (unique candidate ids
  5 488), raw manifest 2 476 items / 264 401 425 B (`e8010278…`), coverage 5 488/5 488 with 0
  missing, 0 missing raw files, 0 extra raw files, all checks passed.
- Structural validation (`v5_p001_process_validation.json` sha256 `ea895a37…`): all 5 488 records
  re-derived from the sources and the archived raw items, **0 mismatches**, PASSED — computed
  before any metric existed. Raw + labels archived hash-identical to fly90
  (`runs/v5_p001_process/`). Both hosts re-validated the FreezeA lineage by hash and count only
  after the amendment (labels 8c9f1ca6…, 5 488 lines; 2 476 raw items, 0 hash mismatches).

## 3. The one analyzer amendment (owner §1–§9 of the Amendment-B approval)

The first canonical invocation aborted **before any scientific value was computed, printed or
written** (`ANALYZER_PRE_METRIC_STRUCTURAL_ABORT / NO_SCIENTIFIC_RESULT_OBSERVED`): `build_report`
read the fixture/oracle validation manifest from the key that `load_inputs` filled with the
structural validation artifact (`KeyError: 'summary'`). Zero metrics emitted, no result artifact,
no oracle call, labels and raw bytes untouched. Amendment B (commit `7d0b139`) split the two
inputs, added per-item raw-hash verification and the structural-validation → freeze/labels chain
check, and pinned the FreezeA lineage in the stamp guard
(`analysis_amendment` block in the result: parent execution head `5b1c5d2`, parent analyzer
`f5532f50…`, analysis head `7d0b139`, analyzer `8c485eec…`). Regression evidence: T1–T4 tests, 584
passed full suite, ruff clean; T3 pins a sha256 digest over every metric-producing subtree
identical before/after the fix (`171e21c3…`). No re-freeze, no re-processing, no oracle calls, and
**no scientific computation path changed**. The canonical analyzer then ran exactly once
(productive run #1) and produced the artifact above.

## 4. Gates (evaluated in the frozen order; the first decisive step would decide)

| gate | requirement | value | verdict |
|---|---|---|---|
| **E1** mapping | ≥ 0.95 | **1.0** — 25 197/25 197 eligible tactic nodes mapped (4 exact + 25 193 contained; 0 unmapped, 0 ambiguous; 56 spans outside the response excluded) | PASS |
| **E2** oracle decision | ≥ 0.90 | **0.9855** — 2 440/2 476 conclusive (36 censored, share 0.01454) | PASS |
| **G1** pooled RecoveryRate | ≥ 0.25 and CI lower > 0.15 | **0.7217**, CI **[0.6787, 0.7645]** (415/575 groups; 384 clusters, 10 000 resamples) | PASS |
| **G2** per-seed | ≥ 0.20 in each seed | seed1 **0.7047** (136/193), seed2 **0.7056** (139/197), seed3 **0.7568** (140/185) | PASS |
| **G3** length robustness | > 0.15 in ≥ 3/4 quartiles | Q1 **0.1538** (22/143), Q2 **0.8542** (123/144), Q3 **0.9375** (135/144), Q4 **0.9375** (135/144) → **4/4** | PASS |

`failed_gates: []`; `n_quartiles_passed: 4`; `seed_min_evaluable_share: 1.0` (every seed keeps
100 % of its all-fail groups evaluable; 0 groups all-censored).

## 5. Classification and what it means

- **FINAL_CLASSIFICATION: `PROCESS_SIGNAL_GO`** — E1, E2, G1, G2, G3 all pass, so the frozen
  taxonomy takes the GO branch on the first decisive step.
- Reading, in the pre-registered vocabulary: on this frozen V1 surface the first-error d1/d2
  process-credit structure is **offline-recoverable for 72.2 % of primary all-fail groups** (G1,
  CI [0.68, 0.76]) with a stable per-seed profile (G2) and it survives the length split (G3). The
  candidate-level view (pre-declared secondary, descriptive only): 1 133/1 736 all-fail code
  candidates = 65.3 %.
- Consequence (frozen): `PROCESS_SIGNAL_GO` authorizes **only** drafting a V5-R001
  preregistration. No training, no weight update, no generation, no V5-R001 launch is authorized
  by V5-P001 (owner §31/§32). The sealed V3 93-component reserve and every future family-clean
  holdout remain untouched (`SEALED_RESERVE_TOUCHED: 0`).

## 6. Mechanism, statuses and credit surface (descriptive)

- Every recovering candidate (1 133 in 415 groups) carries status `PREFIX_BEARING_FAILURE`
  (`error` 1 120, `error+sorry` 12, `sorry` 1) — recovery is exactly the verified-prefix-then-error
  structure. Best verified prefix per recovering group: median 4, max 51; 371/415 groups have ≥ 2
  verified prefixes. `any_active_groups` 425 (the census class of groups with ≥ 1 active
  candidate).
- Status decomposition (5 488): `FORMAT_NO_CODE` 3 012, `PREFIX_BEARING_FAILURE` 1 370,
  `FIRST_TACTIC_FAILURE` 322, `PARSE_OR_SYNTAX_FAILURE` 247, `SUCCESS` 501, `PROCESS_ORACLE_INFRA`
  36.
- Credit surface mass under the canonical d1/d2: 7 929 d1 tokens, 12 269 d2 tokens, 2 984 success
  tokens, 56 credit conflicts (first d1/d2 overlap inside one candidate), blamed-tactic
  mappability 1.0; no credited span falls outside its response.
- Sensitivity (never redefining the primary): the three declared credit variants (canonical
  −0.05/−0.10, stronger-gap −0.05/−0.50, equal-penalty −0.10/−0.10) produce **label-identical**
  classifications by construction — `structured_recoverable` tests mappability, not the credit
  value; only the credit mass changes (−1 623.35 / −6 530.95 / −2 019.80 total φ).
- Length quartiles are near-saturated above the shortest bin; Q1 (0–5 max tactics, 96/143 groups
  zero-code) sits just above its 0.15 threshold at 0.1538. Reported as computed; no gate was
  rescued or reinterpreted.

## 7. Limitations (declared before the outcome, still binding)

1. One historical surface only: three GRPO seeds of a single sub-billion model; nothing here
   generalizes to another model or budget.
2. Process labels come from a single oracle pass per candidate; infra outcomes are censored, not
   failed.
3. Group-level RecoveryRate is an existence predicate, not a per-candidate rate.
4. The d1/d2 credit surface was never applied to a model in V5-P001; recoverability says nothing
   about gradient signal quality during training.

## 8. Compute and provenance

- Oracle: image digest `588a2cbb…`, 2 476 submissions, 36 censored, 6 recoveries; tokenizer
  frozen Kimina (repo `models/weights/kimina_distill_0_6b`, `tokenizer.json` `aeb13307…`).
- Frozen design hashes unchanged through execution and analysis: surface `b3c8f472…`,
  oracle-validation manifest `9153fd2f…`, fixture set `2bfb0f4f…`, plan order `d04283ee…`,
  dataset `68f33cee…`, family registry `f5074c96…`, V1 sources manifest `c405f5f6…`.
- FreezeA: labels `8c9f1ca6…` / freeze `1d41c6a7…` / run_meta `9326bf51…` / validation
  `ea895a37…` / raw manifest `e8010278…` / infra log `252e4e69…`.
- Execution code stamp: head `5b1c5d2` with analyzer `f5532f50…`; analysis code stamp: head
  `7d0b139` with analyzer `8c485eec…`; delta analyzer-only (spec `8e28edc8…`, oracle `69c8031d…`,
  reconstruct `e05e5ba9…`, runner `5c090e5c…` byte-identical on both stamps).
- Transcript of the productive run: `/tmp/v5p001_analyze_canonical_run1.out` (stdout of the single
  canonical run; the crashed pre-metric transcript is `/tmp/v5p001_analyze_canonical.out`).

## 9. Closeout statement

Recorded, committed and pushed; the canonical analyzer ran exactly once over the frozen artifacts
and this artifact is its full output. `PROCESS_SIGNAL_GO` opens exactly one door: drafting the
V5-R001 preregistration (Outcome-only RLVR vs Process-Verified RLVR), which is a draft for owner
review — no launch, no training, no weight update, no generation. Nothing here is evidence about
the sealed reserve; no V4 line is reopened; the frozen V5-P001 design is not to be revisited post
outcome.
