# V4-P001 -- paired verifier-guided repairability probe: Stage-2 result

Status: COMPLETE / `C_NO_REPAIR_GAIN`, mechanism label `DIAGNOSTIC_NONSPECIFIC`
Owner directive: unattended night authorization, 2026-09-25 (Stage-2 codefix -> validation ->
formal Stage-2 execution -> canonical analysis -> result recording -> closeout per outcome).
Executed at commits `2422e46` (launch) / `b8011a4` + `d600905` (freeze and analysis).

Canonical analysis artifact: `experiments/manifests/v4/V4-P001_results.json`
(sha256 `5a741a83ee234c64957ecce9b17881fbc724992b5f2669ea5f76c55c52310177`).
This memo states what the frozen analyzer computed; it introduces no new endpoint.

## 1. The question

Given a Lean-level failed proof produced by frozen Kimina-Distill-0.6B theta0, does the exact
verifier diagnostic improve second-attempt proof success beyond (A) a fresh retry and (B)
self-revision without the diagnostic -- and is any such gain specific to *this* theorem's own
diagnostic (D: a real diagnostic of a *different* theorem, assigned by the frozen derangement)?

Arms (frozen): A_FRESH_RETRY, B_SELF_REVISION, C_VERIFIER_REPAIR, D_MISMATCHED_DIAGNOSTIC.
Primary endpoints: mean(C-A) (delta >= +8pp, one-sided exact McNemar p <= 0.05, bootstrap CI
lower > 0) and mean(C-B) (delta >= +5pp, same conditions). Specificity endpoint: mean(C-D),
directional rule, no effect-size threshold, reported as a mechanism label only. Arm D does not
enter the GO gate and does not replace the C-A / C-B questions.

## 2. Execution record (Stage-2 launch-1)

- Host: fly122 (RTX 3080 10 GB); frozen theta0, temperature 1.0, top_p 1.0, n=1, per-rank
  frozen seeds, frozen balanced arm schedule; repair-only, second attempt; no re-rolling.
- Launch chain: launch-0 aborted structurally (see `V4-P001_stage2_execution_amendment_B.md`);
  its 8 in-memory completions were never persisted, verified, or inspected and are permanently
  void. Launch-1 restarted at theorem rank 1 for all 512 candidates under the repaired runner
  (`scripts/v4_p001_rollout.py` sha256 `abc8249b...`).
- Run id `20260925T160617Z`; raw rows created 2026-09-25T16:08:02.574711Z .. 17:44:18.862773Z
  (row span 5776.3 s; runner-measured stage time 5811.6 s).
- Raw artifact: `runs/v4_p001/rollout/v4_p001_second_stage_raw.jsonl`
  sha256 `6597a53ff06139970ce8ed2db78d184a3a7bc81c979e164a0a6ce49b9c38f7f6`, 5,842,484 bytes,
  512 rows (128 theorems x 4 arms), 128/128 structurally complete quadruplets.
- Verifier discipline: dedicated verifier (`http://127.0.0.1:8010`, MAX_REPLS=1, sequential,
  server timeout); infrastructure is never a proof failure. 11 bounded recoveries, all
  successful (7 `verifier_server_error`, 4 `verifier_timeout`; ranks 3, 12, 34, 46, 54, 67, 81,
  112, 119, 122), sum recovery 83.8 s, sum restart 13.2 s; recovery journal 16 rows
  (5 Stage-1 + 11 Stage-2). Ceilings (192/run, 3/theorem) untouched; max per-theorem index 2.
- Compute: peak VRAM 9206 MiB (nvidia-smi sampler is the peak of record; the torch peak counter
  is reset after the engine is built and is not used). Summed per-candidate generation time
  2180.3 s; remainder of the stage time is verification, persistence and orchestration. No
  throughput or efficiency claim is made from any of these figures.
- Analyzable completeness: of the 128 nominal quadruplets, 118 are complete (92.19%); 10 are
  incomplete purely by infrastructure censoring -- 11 missing arm slots in total, rank 112
  losing two: A 1 server_error + 2 timeout (3), B 2 server_error (2), C 1 server_error + 1
  timeout (2), D 3 server_error + 1 timeout (4). Each censored arm is missing data, never a
  proof failure.

## 3. Freeze, archive, boundary

- Structural freeze `experiments/manifests/v4/V4-P001_stage2_freeze.json`
  (sha256 `9f3c72f3...`): 13/13 checks, FROZEN, computed before any endpoint existed.
- Raw + compact infra logs archived hash-identical to fly90
  (`runs/v4_p001_stage2_archive_from_fly122/`): raw `6597a53f...`, summary
  `90d21ef15c072e33e307ceae73d133ba3e9d599c0ea95e647bf78764e8a3df17`, recovery journal
  `6975bfa28f122d0d841339db13960c70e18fb567f81de48adaecf9427b8b3b0c`, verifier events
  `f5dabfda48ff026e43f1912d8fee637cd33cd3d328ae02f01028e7dcea8184e4` (1033 rows), GPU samples
  `853db4a43d898235d4706f1c53aa10e6fbe296e9179fb75f1510b0f0069eb99f`, console log
  `3fbd5483f441a383d15ac18d4c745f1706541ecc0f13e27a787725a3f0d49857` + `SHA256SUMS`.
  fly90 runs no scientific analysis; the archive is provenance only.
- Provenance re-verified inside the analyzer before any metric: 18/18 checks PASS (boundary
  validation PASS 0 failed; boundary artifacts self-hash; screening raw matches; cohort exact;
  run summary carries the committed frozen-settings / plan / schedule / seed / derangement
  hashes; 50 frozen-design checks 0 failed; sealed reserve untouched 0/0; every row carries the
  frozen model / renderer / normalizer; every candidate is a cohort rank and the frozen theorem;
  seeds are the frozen stream and do not reuse screening seeds; execution order is the frozen
  balanced schedule; prompts reproduce the frozen prompt / diagnostic / failed-proof hashes
  exactly; arm D carries the derangement donor with no byte-identical and no self diagnostics).

## 4. The one permitted analyzer correction (owner §24, Case A)

The first canonical run exited 3 (ABORT BEFORE ANALYSIS, nothing written, no scientific metric
emitted) because the provenance check `execution_order_is_the_frozen_balanced_schedule`
compared the frozen row field `position` (1-based 1..4, as the runner writes it and the
structural freeze re-asserts) against a 0-based index into the frozen arm_order; all 512 rows
were rejected. This is purely mechanical (off-by-one in a check), so exactly one code-only fix
was applied: commit `d600905` adds `+ 1` to the index, aligns the analyzer-test and
package-dry-run fixtures to the 1-based convention, and pins the 1-based range in the
fixture-consistency test so the fixtures can no longer encode the wrong basis. Analyzer
sha256 before `e20638c1...` -> after `93721be7...` (the version recorded in the results
artifact). No formula, threshold, guard, ingest or taxonomy rule was touched; the fix was made
and committed before any metric existed, and the canonical analyzer then ran exactly once.
Regression property (recorded in the commit): with the fix reverted, six analyzer tests fail.

## 5. Guards (evaluated first, in the frozen order)

| guard | rule | value | verdict |
|---|---|---|---|
| n_complete | >= 103 | 118 (10 infra-incomplete) | PASS |
| differential censoring | range <= 0.05 | rates A .0234 / B .0156 / C .0156 / D .0313 -> range 0.015625 | PASS |

Both guards pass, so the frozen taxonomy was evaluated. (Neither is a GO/NO-GO statement.)

## 6. Primary endpoints and specificity

| comparison | n pairs | delta | exact one-sided McNemar p | bootstrap CI lower | favors/against | gate |
|---|---|---|---|---|---|---|
| C - A (>= +8pp) | 18 | -1.69pp | 0.7597 | -0.0847 | 8 / 10 | FAIL (all three conditions false) |
| C - B (>= +5pp) | 16 | +6.78pp | 0.0384 | 0.0000 | 12 / 4 | FAIL (CI lower not > 0) |
| C - D (specificity) | 15 | +2.54pp | 0.3036 | -0.0424 | 9 / 6 | not demonstrated |
| B - A (descriptive) | 16 | -8.47pp | 0.9979 | -0.1525 | 3 / 13 | no gate |

Reading, in the pre-registered vocabulary: the verifier diagnostic did **not** improve
second-attempt success beyond a fresh retry (C-A is negative and nowhere near its gate); it did
show a positive nominal difference over self-revision without the diagnostic (C-B +6.78pp,
p 0.038) but the paired bootstrap CI lower bound is exactly 0.0, so the pre-registered C-B gate
is not met -- and because C-A failed, the over-retry question is closed regardless. The
C-D contrast is positive but small and not statistically distinguishable from zero, so no
theorem-specific mechanism was demonstrated.

- Classification: **C_NO_REPAIR_GAIN** ("C-A does not satisfy its three conditions, so no
  repair gain over a fresh retry").
- Mechanism label (appended, never merged): **DIAGNOSTIC_NONSPECIFIC**.
- Consequence (pre-registered): the verifier-feedback training line is closed at this design,
  as of this result. This is not a claim that no design could extract value from verification;
  it is the pre-registered disposition of this probe.

## 7. Secondary / descriptive (no gate, no claim beyond description)

- Success by arm over the 118 complete quadruplets: A 18 (15.25%), B 8 (6.78%), C 16 (13.56%),
  D 13 (11.02%). B-A is negative; self-revision without the diagnostic was the worst arm.
- Overlap (complete quadruplets): of B's 8 successes, 5 were also A-successes; of C's 16, 8
  also A-successes, 4 also B-successes; of D's 13, 7 also A, 2 also B, 7 also C.
- Transitions (all 128 nominal): B -- 8 SUCCESS, 50 SAME_ERROR_CATEGORY, 21
  DIFFERENT_ERROR_CATEGORY, 45 FORMAT_FAILURE, 1 SYNTAX_FAILURE, 3 INFRA_MISSING; C -- 16 / 35
  / 24 / 49 / 1 / 3; D -- 13 / 38 / 19 / 47 / 3 / 8. The single largest failure mode in every
  arm is a format failure of the second attempt, C included.
- By original error category (complete quadruplets): tactic_failure n=62 (A 10, B 5, C 9,
  D 5); unsolved_goals n=21 (A 5, B 1, C 5, D 3); elaboration_type_mismatch n=15 (A 1, B 1,
  C 1, D 3); unknown_identifier n=12 (A 1, B 0, C 0, D 0); typeclass_synthesis n=4 (A 1, B 1,
  C 1, D 2); other_semantic_lean_failure n=4 (all 0). Descriptive only.
- By source (complete quadruplets): synthetic n=76 -- A 18, B 8, C 13, D 10; autoformalizer
  n=31 -- 0, 0, 2, 2; human n=11 -- 0, 0, 1, 1. Do not read this as a capability statement; the
  synthesizer fractions are small and the arms were not powered for source strata.
- Generation: 55/47/50/50 of 118 rows (A/B/C/D) hit the 4096-token cap; median generated
  tokens 3882 / 3584 / 3632 / 3710; format-extraction failures 0 in every arm. Summed
  generation seconds per arm: 541.5 / 543.5 / 546.1 / 549.3.
- Infra-as-failure sensitivity (marked SENSITIVITY ONLY, never redefining the primary): with
  every censored arm read as a failure over all 128 ranks -- C-A -1.56pp (p 0.7597, 8/10,
  4 pairs carry a censored arm), C-B +6.25pp (p 0.0384, 12/4), C-D +2.34pp (p 0.3036, 9/6).
  Direction and verdicts are unchanged.

## 8. Compute and provenance

- Model: theta0 Kimina-Distill-0.6B rev `332e8a52...`, weights sha256 `34e6e630...`; run_id
  `20260925T160617Z`; frozen settings hash `f650a128c5093abfe62054b0eb62d9d20c0bda8cb07aee3eed14fa0eabfd3f39`.
- Design hashes (unchanged through execution and analysis): pool `7533c94b...`, order
  `ce329b36...`, screening seeds `2efb53d8...`, paired seeds `06ba33b8...`, arm schedule
  `9e090325...`, plan `c278707d...`, arm prompt plan `714de133...`, derangement `v4-derange-1`
  mapping `efcd8db5...`, renderer `v4-prompt-1`, normalizer `v4-diag-1`.
- Screening raw `fc8aa0a5...` untouched; sealed V3-FINAL-HOLDOUT reserve untouched
  (components 0, statements 0).
- Scripts at analysis time: analyze `93721be7...`, spec `5f6f8a0e...`, runner `abc8249b...`.

## 9. Closeout statement (owner §30, C_NO_REPAIR_GAIN)

Recorded, committed and pushed; the V4 verifier-feedback line closes with this result. No
prompt rescue, no alternative diagnostic format, no larger context, no Arm E, no different
model, no new cohort, no top-up and no reserve use; the taxonomy, thresholds, guards and
cohort are not to be revisited post outcome. V4-R001 (verifier-integrated training) is **not**
drafted -- it was a GO-only branch -- and no RL/SFT/training of any kind is authorized by this
result. No V5 design, Jev controller or adaptive-allocator work follows from this probe.

Prohibited readings that remain in force: not a training result; no AUPRC or predictive
endpoint; no throughput/efficiency claim from verifier counts; `D_INCONCLUSIVE_*` would not
have been a NO-GO; `DIAGNOSTIC_NONSPECIFIC` forbids any claim that the model understood the
theorem-specific diagnostic; nothing here is evidence about the sealed reserve.
