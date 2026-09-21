# V2-B003 Final Memo — Multi-seed Compute-Response + Cross-fitted Expected Oracle

Status: run + preregistered analysis + owner four-point audit COMPLETE.
B004 recommendation below; the decision belongs to the owner and nothing
auto-launches. No predictor was trained.

## 0. Headline

On the 185-theorem primary set the cross-fitted Expected Oracle solves
~2.0 theorems at every cap point while uniform catches up from 0 (512) to
2.5 (4096). The largest absolute gain is +1.54 expected solves at N*2048 and
turns NEGATIVE (-0.50) at N*4096; pool-level, 9/192 (4.7%) theorems and
20/1536 (1.3%) trajectories ever solve, and 176/185 theorems show zero
response. Recommendation: NO-GO for B004 (details in section 6).

## 1. Run and data integrity

- 1536/1536 records (192 theorems x K=8) in 4 parallel single-path shards;
  2026-09-20T09:29:34Z -> 2026-09-21T06:15:05Z (~20h45m); merged with
  missing=0; merged sha256
  94f12dc44cfc088647797659d7d08e58996f7b92358414b0ffa98125019b3880.
- Verifier infra outcomes: 76/7680 prefix cells (0.99%), kept as missing
  (never counted as 0 or failures).
- Coverage: primary set S6 = 185/192; 9 low-coverage cells on 7 theorems
  (ranks 45,71,92,114,126,131,173; all k=0; only rank 92 at 4096 had n=2,
  so no hidden successes beyond one cell).
- Fold-level coverage (owner audit 2): cells with fewer than 3/4 valid trials
  are 1/925 (0.11%) in fold A and 0/925 (0.00%) in fold B, far below the 5%
  threshold => gate coverage status: coverage-ok.

## 2. Solve structure (pool-level)

- theorem-level solved: 9/192 = 4.7% (any prefix); trajectory-level:
  20/1536 = 1.3%; first successes only at 2048 (4), 3072 (4), 4096 (12) -
  512 and 1024 solve nothing anywhere.
- Successes are concentrated: among the 9 solved theorems, 4 succeed in only
  1 of 8 trajectories, 1 in 2, 2 in 3, 2 in 4.
- Consistent with the B002 pool family (11/512 = 2.1% at K=1; B003 single
  trajectories 1.3%) - the reserve pool behaves as an even harder tail.

## 3. Primary method positioning (owner audit 1)

The primary method is the CROSS-FITTED EMPIRICAL RESPONSE ORACLE: it uses
each theorem's 4 fit-seed outcomes to decide the allocation and the other 4
seeds to evaluate it independently. It therefore remains a NON-DEPLOYABLE,
theorem-specific Oracle: it is not a predictor and its numbers are not the
performance any future model would achieve. Its role here is strictly to
test the cross-seed stability of the compute-response signal. The no-skip
action space (every theorem keeps the >=512 floor) is the primary space;
the skip-allowed triage space stays separate and is not part of this
headline.

## 4. Headline table (owner audit 4: absolute and relative together)

Cross-fitted no-skip Oracle at every total budget (N=185; expected solved
counts; conditional CI = analyzer bootstrap with the allocation held fixed;
full-proc = full-procedure audit bootstrap). Scope: STRICT family-unseen
hard/OOD reserve population (192 components) only - these numbers must never
be generalized to the full Promptset distribution.

| N*b   | uniform | cf-Oracle | abs gain | rel gain      | cond CI95 (abs) | full-proc CI95 (abs) | savings |
|-------|---------|-----------|----------|---------------|-----------------|----------------------|---------|
| 512   | 0.000   | 0.000     | 0.000    | n/a           | [0.0, 0.0]      | [0.0, 0.0]           | 0%      |
| 1024  | 0.000   | 2.042     | +2.042   | n/a (base 0)  | [0.50, 3.92]    | [0.50, 3.96]         | 84.53%  |
| 2048  | 0.500   | 2.042     | +1.542   | +308.3%       | [0.25, 3.17]    | [0.24, 3.13]         | 84.53%  |
| 3072  | 1.000   | 2.042     | +1.042   | +104.2%       | [0.00, 2.29]    | [0.00, 2.42]         | 84.53%  |
| 4096  | 2.542   | 2.042     | -0.500   | -19.7%        | [-1.00, -0.13]  | [-1.00, -0.13]       | 84.53%  |

Notes: the oracle value is flat at 2.042 for every cap >= 1024 (its fit-half
sees almost no responsive theorems, so extra cap cannot be used). The 84.53%
same-solved savings compare the oracle's 117,248-token allocation with the
smallest uniform budget (4096) whose expected solved covers the oracle value;
the absolute counts above are the primary reading. Rare-event rule respected:
relative gains are never reported without the absolute numbers.

## 5. Bootstrap audit (owner audit 3)

- The preregistered analyzer bootstrap holds the allocation fixed after one
  fit-half solve: it is a CONDITIONAL CI (theorem=component resampling).
- The full-procedure audit re-estimates the fit-half response, re-solves the
  knapsack and re-scores the eval half inside every one of 1000 replicates
  (fixed seed); the two CI columns above differ only marginally.
- Interpretation: allocation instability is NOT the dominant error source
  here; the uncertainty is dominated by evaluation noise on an extremely
  sparse signal. The preregistered point estimates are unchanged.

## 6. B003 -> B004 gate reading (preregistered, cross-fitted no-skip)

Preregistered bands (manifest 594b06b): (a) <10-15% relative at multiple
budget points AND small absolute gain => NO-GO; (b) >30-40% at multiple
points with substantial absolute gain => strong GO; (c) 15-30% => only
low-cost heuristic/XGBoost, no MLP.

Data: relative gains are large at N*2048 (+308%) and N*3072 (+104%) but the
absolute gains are only +1.5 and +1.0 expected solves (0.8% and 0.6% of 185
theorems), N*1024 has no relative reference (uniform solves nothing), and
N*4096 is NEGATIVE (-19.7%, CI entirely below zero). The strict band letters
therefore do not map cleanly onto this combination (large relative, tiny
absolute, sign flip at the top cap) - this is exactly the rare-event
inflation the owner audit anticipated.

READING (recommendation, owner decides): NO-GO for B004, including the
low-cost heuristic/XGBoost path. Reasons: (a) the absolute effect is
non-actionable (<= 1.5 expected solves out of 185; top point negative);
(b) the fit-half response has almost no discriminative power - the
allocation is forced to the 512 floor for ~95% of theorems and cannot use
additional cap; (c) all sizable relative numbers are rare-event artifacts of
tiny denominators; (d) structurally, 176/185 theorems show zero response at
every budget, so there is nothing for a predictor to learn on this pool.
The cross-fitted gate has done its job: it rejects the signal that the
plug-in oracle and the B002 hindsight numbers could not reject.

## 7. Compute-response analysis

- p_i(4096) distribution (k/8): 176 theorems at 0, then 4 at 1/8, 1 at 2/8,
  2 at 3/8, 2 at 4/8. Mean sensitivity 0.0138, non-zero for 9 theorems.
- Marginal gains between adjacent budgets: 512->1024 = 0.0000,
  1024->2048 = +0.0027, 2048->3072 = +0.0027, 3072->4096 = +0.0084.
- All 185 empirical curves are non-decreasing (0 non-monotonic).
- IMPORTANT (degeneracy): because every theorem has p_hat(512)=0, the
  compute sensitivity equals p_i(4096) by identity; the reported
  difficulty-vs-sensitivity Pearson/Spearman of 1.0 is therefore a
  degenerate identity, NOT evidence about the hard != compute-sensitive
  question. On this pool the two properties are structurally inseparable
  (any success IS a compute-sensitive success); the B003 design cannot
  answer that sub-question.

## 8. Limitations

- Fit-half estimates rest on 4 seeds; with success concentrated on 9
  theorems (4 of them succeeding in only 1 of 8 trajectories), the fit half
  frequently observes no response at all - the oracle is a stability test,
  not a strong estimator.
- 7 of 192 theorems were excluded from the primary set by the n>=6 coverage
  gate (all had k=0; no excluded cell except rank 92/4096 (n=2) could hide a
  success).
- Scope is the strict family-unseen hard/OOD reserve population; nothing here
  transfers to the mixed Promptset distribution.
- 76 infra cells (0.99%) are missing, not failures, consistently with B0/B2
  semantics.

## 9. Artifacts

- manifest experiments/manifests/v2/V2-B003.yaml (status -> COMPLETE)
- rollouts experiments/results/v2_b003_rollouts.jsonl (sha256 94f12dc4...)
- analysis experiments/results/v2_b003_analysis.json (preregistered gate)
- audit experiments/results/v2_b003_audit.json (four-point owner directive)
- runner/analyzer/audit: scripts/v2_b003_generate.py, v2_b003_analyze.py,
  v2_b003_audit.py (+ tests; suite 183 pass at audit commit)
