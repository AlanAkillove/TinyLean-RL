# V5-P001 analyzer amendment B — report-wiring repair with an immutable parent freeze

- Date: 2026-09-27 (UTC) · Branch `v5-process-verified-rl` · Host: fly90 (code, tests, lineage
  validation, canonical analysis) + fly122 (formal run artifacts)
- Authority: owner approval 2026-09-27 (“V5-P001 Analyzer Amendment B = APPROVED”), received after
  the pre-metric structural abort of the canonical analyzer described in §1. This document is the
  amendment artifact: trigger, root cause, the minimal analyzer change, the pinned parent-freeze
  lineage (execution HEAD vs analysis HEAD), the regression tests and the explicit list of
  unchanged scientific elements.
- **Scientific design changed by this amendment: NONE.** No label, no oracle output, no metric
  definition, no gate, no denominator and no threshold moved. The amendment touches only the
  canonical analyzer's *report/provenance plumbing* and its *integrity guard*.

---

## 1. Trigger and classification

```
ANALYZER_PRE_METRIC_STRUCTURAL_ABORT
NO_SCIENTIFIC_RESULT_OBSERVED
```

Phase B attempt 1 completed (5488/5488 labels, 2476 oracle submissions, 3012 sentinel candidates,
36 infra-censored, 6 recoveries, 3 two-strike censorings), the freeze **PASSED** and the structural
validation **PASSED** (5488/5488 records re-derived, 0 mismatches). The canonical analyzer's first
invocation then crashed at report-metadata assembly, before any scientific value was computed,
printed or written:

```
Traceback (most recent call last):
  File "scripts/v5_p001_analyze.py", line 740, in <module>
    raise SystemExit(main())
  File "scripts/v5_p001_analyze.py", line 729, in main
    report = build_report(inputs)
  File "scripts/v5_p001_analyze.py", line 626, in build_report
    "fixtures": inputs["validation"]["summary"],
KeyError: 'summary'
```

| fact | value |
|---|---|
| scientific metrics emitted | **0** (crash before `print_report`) |
| result artifact written | **NO** (`V5-P001_results.json` never existed) |
| labels / raw oracle outputs changed | **NO** |
| oracle calls during the abort | 0 |
| crash transcript | retained (`/tmp/v5p001_analyze_canonical.out`, 532 B, traceback only) |

The aborting invocation therefore does not count as a scientific analysis run; the repaired
analyzer's first productive run is the canonical run (owner §9).

## 2. Root cause (integration defect between two frozen functions)

`load_inputs` returned the **structural** validation artifact under the key `"validation"`, while
`build_report` treated that key as the **fixture/oracle validation manifest**:

| key access (pre-amendment) | artifact actually reached | consequence |
|---|---|---|
| `inputs["validation"]["summary"]` (line 626) | structural artifact (no `summary`) | **KeyError, crash** |
| `inputs["validation"].get("fixture_set_sha256")` (line 623) | structural artifact (no such key) | silent `None` |
| `inputs["validation"]["checks"]` (line 627) | structural artifact (correct dict by accident) | mislabeled wiring |

The unit fixtures missed it because the synthetic structural artifact in `tests` happened to carry
a `summary` key that production structural artifacts never have. No metric-producing function reads
`inputs["validation"]` at all: the crash was purely in report-metadata assembly.

## 3. The fix (report/provenance plumbing only)

`scripts/v5_p001_analyze.py`:

1. `load_inputs` now returns two distinct inputs — `"validation"` = the fixture/oracle validation
   manifest (`v5_process_oracle_validation.json`) and `"structural_validation"` = the structural
   validation artifact (plus `"stamp"` = the current code stamp).
2. `build_report` reads `"fixtures" ← inputs["validation"]["summary"]`,
   `"fixture_set_sha256" ← inputs["validation"]` (as before, now correctly) and
   `"re_derivation" ← inputs["structural_validation"]["checks"]`.
3. Provenance split: `provenance.execution_code_stamp` (FreezeA) and
   `provenance.analysis_code_stamp` (this analyzer's stamp) plus a top-level
   `analysis_amendment` block:
   `kind / parent_execution_head / parent_analyzer_sha256 / analysis_head / analyzer_sha256 /
   parent_freeze_sha256 / labels_sha256 / raw_manifest_content_hash`.

No scientific computation path changed (owner §2). The regression test suite pins this with a
digest of every metric-producing subtree (§5, T3).

## 4. Provenance: FreezeA stays the execution record

The pre-existing stamp guard required `run_meta.git_head == current git HEAD`, which an
analyzer-only amendment can never satisfy without rewriting the run metadata. Per the owner's
preferred handling, the guard — not the artifacts — was amended:

* `ANALYSIS_AMENDMENT = {kind: analyzer_code_only,
  parent_execution_head: 5b1c5d277a7d923c7aed0626f238012dfc6d9f18,
  parent_analyzer_sha256: f5532f50…}` is pinned in the analyzer.
* `verify_execution_stamp` accepts a parent stamp only if: it equals the current stamp (normal
  path), or it belongs to the pinned parent execution, and differs from the current stamp **only**
  in `git_head` and this analyzer's own script hash. All four other scripts, the historical-surface
  hash and the oracle-validation-manifest hash must be byte-identical; any other delta refuses.
* The analyzer additionally verifies, per item, that every raw oracle item still matches its
  frozen manifest hash, and that the structural validation artifact references **this** freeze
  (`freeze_sha256`) and **these** labels (`labels_sha256`).

Chain: **original Phase-B execution → FreezeA → analyzer amendment B → canonical analysis.** No
re-freeze and no no-op resume were needed: the four FreezeA artifacts are untouched, byte-identical
and permanently archived (owner §5 fallback not used).

## 5. Tests (owner §6, T1–T4)

* **T1** `test_structural_validation_fixture_is_production_shaped` — the synthetic structural
  artifact now matches the production schema (no `summary`); the pre-amendment key access raises
  `KeyError` on it, the repaired analyzer passes.
* **T2** `test_full_report_reaches_the_frozen_go_classification` — asserts source separation:
  `process_oracle.fixtures` comes from the fixture/oracle manifest, `process_oracle.re_derivation`
  from the structural artifact, `provenance.fixture_set_sha256` from the fixture manifest.
* **T3** `test_report_plumbing_fix_leaves_every_scientific_output_identical` — sha256 over all
  metric-producing report subtrees, pinned from the **pre-amendment** analyzer on the same
  synthetic records: `171e21c3b46f5031b566ba2138e128b1e7e66b9b53cf82eec76bb808fa42803b`.
  Recomputed after the fix: identical. Control: mutating one metric changes it
  (`15ce2d7de86d39f108aa77c4bb244061094b380d3137969ba1d72ecf356256a3`).
* **T4** pinned-parent immutability — the pinned lineage is accepted and recorded as an amendment;
  refused are: an unpinned execution head, a non-pinned parent analyzer hash, a moved
  scientific-script hash (e.g. the process oracle implementation), a moved historical-surface or
  oracle-validation-manifest hash, a tampered raw oracle item, and a structural validation that
  references another freeze or other labels.

## 6. Pinned hashes

| item | sha256 |
|---|---|
| `scripts/v5_p001_analyze.py` before (frozen in FreezeA) | `f5532f5019ea435030672c5ac6e8e4631b64277a941412e5d0bd56c2340bc1be` |
| `scripts/v5_p001_analyze.py` after (this amendment) | `8c485eecef92517bc49082043d790b701996ddb43ba5bd92c479da42ce393731` |
| `tests/test_v5_process_oracle.py` after (not part of the stamp) | `e1ef914af507ebb35597c2c65893023831b3992f89d320c5b15ad5b2fc3ca62d` |
| FreezeA labels (`v5_p001_process_labels.jsonl`, 5488 records) | `8c9f1ca604e2ff17ba06a79e568b8f97c79bcd5e290923cabefdeaaf7ee321ea` |
| FreezeA raw manifest (2476 items, 264 401 425 B) | `e8010278b21f8e44ad326678da5d6a12c3aeb0a72a25c22649cf1df84c003d16` |
| FreezeA freeze (`v5_p001_process_freeze.json`) | `1d41c6a7829fbccdc52b0442ff529755476ebea89f61f78b16f6b46da5cb9815` |
| FreezeA run_meta (`v5_p001_process_run_meta.json`) | `9326bf5134369528c5cf13667ca7fd0aec31512b283f9e85005a5a6eab98b89f` |
| FreezeA structural validation (`v5_p001_process_validation.json`) | `ea895a377d3b15f25ded05369c8a3b7b1b0e6cd1cdaa6e45b9b33c4ad06a8a6b` |
| FreezeA infra log (57 events) | `252e4e69c8a3d465db1f54fe6783119ac04403121f5bb636425cf30df328e303` |
| `execution_head` (original Phase-B execution commit) | `5b1c5d277a7d923c7aed0626f238012dfc6d9f18` |
| `analysis_head` (this document's commit) | recorded verbatim in the canonical result: `provenance.analysis_code_stamp.git_head` == `analysis_amendment.analysis_head` |

Unchanged stamp components (FreezeA == current, verified by the new guard): `v5_p001_spec.py`
`8e28edc8…`, `v5_process_oracle.py` `69c8031d…`, `v5_p001_reconstruct.py` `e05e5ba9…`,
`v5_p001_process_run.py` `5c090e5c…`, historical surface `b3c8f472…`, oracle-validation manifest
`9153fd2f…`. Only `git_head` and `v5_p001_analyze.py` moved between execution and analysis.

## 7. What is explicitly unchanged

Unchanged and byte-identical: the 5488 process labels; all 2476 raw oracle items and their
manifest; the freeze, run_meta, structural validation and infra-log bytes; the historical surface
and its binary baseline; the process oracle implementation and all fixtures; tactic→token mapping;
statuses, blame attribution, `PROCESS_STRUCTURED_RECOVERABLE`; the censoring trichotomy; d1 =
−0.05 / d2 = −0.10 credit surface; gates E1/E2/G1/G2/G3, thresholds, denominators, units, plan and
plan order, bootstrap (10000 resamples, seed 20260926, component-family clusters); the
classification order; Phase-A/Phase-B discipline; no live scientific metrics before the canonical
run; compute (`NEW_MODEL_GENERATION: 0`, `TRAINING: 0`); the sealed reserves.

## 8. Consequences and execution

* fly90 (analysis host, frozen mirror) runs the amended analyzer **once**; fly122 keeps the formal
  artifacts and receives the analysis result by sync. Before the run, the FreezeA lineage is
  re-validated hash-by-hash on both hosts (labels, 2476 raw items, freeze/run_meta/validation,
  surface, oracle-validation manifest) — hash and count output only, no metrics.
* Expected amended guards on the real lineage: parent stamp == FreezeA stamp (pinned
  `execution_head` + pinned parent analyzer hash); current stamp == this commit + the new analyzer
  hash; the delta is analyzer-only; labels, raw items, structural validation chain and both
  frozen manifests verify.
* Result: `experiments/manifests/v5/V5-P001_results.json`, schema as preregistered (owner §34),
  now carrying the explicit `execution_head` / `analysis_head` distinction and the
  `analysis_amendment` provenance block. The result is recorded in a separate commit; if
  `PROCESS_SIGNAL_GO`, only the V5-R001 preregistration may be drafted — no training is launched.
