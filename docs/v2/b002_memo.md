# V2-B002 Final Memo — Family-clean Compute-Response Pilot + Hindsight Oracle Headroom

Status: **COMPLETE (closeout)** | Manifest: `experiments/manifests/v2/V2-B002.yaml` |
Owner-approved closeout plan 2026-09-20 (adjudication + calibration + representativeness audit + oracle split).
No GO/NO-GO verdict or B003 launch is issued in this memo; it presents corrected evidence and a recommendation.

## 0. Headline

The pilot ran 512 family-clean theorems (K=1 canonical trajectory each, 4-shard
parallel after the verified concurrency amendment) with a corrected solved count
of **11/512 = 2.1%**. A diagnostic replay of the identical pipeline on a frozen
E023 subset recovered the historical theta0 regime (**13/64 = 20.3%**, vs E023
theta0 record-level 23.4%), proving **the pipeline is healthy**; the low pilot
rate is a **pool property**: B002 is dominated by the *strict family-unseen hard
sub-distribution* (autoformalizer/human singletons, where theta0 solves ~0%) and
contains almost none of the easier synthetic family-variant stream.

## 1. Run and data integrity

- 512/512 records; ranks 1..512; five prefix cells each (2560 cells).
- Concurrency amendment (owner-approved, 2026-09-20): paused at 8/512, verified
  8/8 recorded trajectories reproduce token-for-token in fresh concurrent
  processes (gain 2.83x), then ran as 4 single-path shards; merged with strict
  equality checks. Raw file sha256 `523b6124e7a3d119352076e23cb62a3dca2b3f81d697cfdcdc4427dbd70da099`.
- **Infra adjudication** (`scripts/v2_b002_infra_adjudication.py`, artifact
  `experiments/results/v2_b002_adjudication.json`): 15 theorems carried >=1 infra
  cell (30 cells; "redeclaration persisted" from the Kimina REPL pool +
  timeouts). ALL 15x5 = 75 cells were re-verified in clean (restarted) column
  environments with candidates rebuilt from recorded token_ids — **75/75 sha256
  identical to raw**. Outcomes: 45 lean_error cells all confirmed; 10 infra cells
  resolved to lean_error; **20 cells remain unresolved** (7 genuine 60s REPL
  timeouts + 13 repetition of a syntax-error/redeclaration mix) and are NOT
  counted as failures (pessimistic/optimistic sensitivity below).

## 2. Corrected pattern distribution (adjudicated)

| Pattern | raw | corrected | notes |
|---|---|---|---|
| `00000` | 501 | **493 confirmed zero** | 7 `0000?` rows resolved to zero; 8 rows carry unresolved cells |
| `00001` | 2 | 2 | first success at 4096 |
| `00011` | 6 | 6 | 3072+ |
| `00111` | 3 | 3 | 2048+ |
| unresolved-any | 0 | **8 theorems** (ranks 91,115,199,379,384,450,491,510) | excluded from both sides of the counts |

**Sensitivity**: corrected solved ∈ **[11, 19]** (pessimistic: unresolved = not
solved; optimistic: every unresolved cell hypothetically solved). All 11 raw
solved theorems are outside the infra set, so **the solved count is 11 in the
pessimistic reading**; 8 hard theorems (currently all-zero or zero-with-holes)
could change state only under the optimistic bound.

## 3. easy / compute-sensitive / hopeless

- **easy (y512=1): 0/512** — no theorem succeeds at the 512 floor.
- **compute-sensitive (y512=0 and some larger budget succeeds): 11/512 (2.1%)**,
  first success: 2048 x3, 3072 x6, 4096 x2.
- **hopeless-at-4096 (y4096=0): 501 raw → 493 confirmed + 8 unresolved**.
- non-monotonic trajectories: **0/512** (every pattern is monotone).
- Source x outcome (the decisive cross): **synthetic 9/29 = 31.0%**;
  autoformalizer **2/367 = 0.5%**; human **0/116 = 0.0%**.
  Component size of solved: {1:3, 8:1, 9:2, 10:5} — the 32 multi-variant
  representatives (6.2% of the pilot) contribute **8/11** solved.

## 4. Hindsight Oracle — dual frontier (allocated max-token cap)

**PRIMARY: No-skip Oracle** (every theorem keeps the 512 floor; pure reallocation,
never triage). 512-cap bound series (cap -> solved): 262144 -> 0; 289280 -> 11;
uniform comparison: at uniform-1024 cap (524288) the no-skip oracle already
reaches all 11 solvable; uniform itself needs 4096 x 512 = 2097152.
**No-skip savings at equal solved count: 86.21%** (cap 289280 vs 2097152).

**Secondary: Skip-allowed Oracle** (0 permitted; triage + allocation upper
bound): cap 32768 -> 11 solved; **98.44%** savings. This 98.4% number depends on
assigning 0 to 501 theorems and must never be presented as a pure adaptive
budget-allocation result.

Both are **pathwise hindsight upper bounds on a single trajectory** — they know
each theorem's realized outcomes and are not achievable by any deployable policy
(including the Expected Oracle). Calibration shows the savings metric is
**distribution-dependent**: on the E023 subset (many solvable theorems) the same
oracles give 75.98% / 85.94%.

## 5. Pipeline calibration (diagnostic-only replay)

- Frozen set: 64 of E023's 128 sealed-holdout theorems (sha256(statement_id)
  ordering; synthetic 54 / human 5 / autoformalizer 5; multi-variant 57/64),
  artifact `experiments/manifests/v2/v2_b002_calibration_set.json` (sha256
  `e5ac588005b2ac9bf2291def5a6ec410c4c8b747651d758b17b0c177792a25fd`).
- Same pipeline (single-path HF, T=1.0, 4-shard), 4x16 records, merged 64/64.
- **Result: 13/64 = 20.3%** (>=1 prefix verified), patterns {00000:51, 00001:4,
  00011:3, 00111:5, 01111:1}; infra cells only 3 (all on rank 54; not solved,
  sensitivity 13..14). **All 13 solved are synthetic-source theorems.**
- Reference: E023 theta0 record-level 23.4% (116-122/512 across checkpoints),
  >=1-of-4-samples 35.9%. The calibration replay (HF single-path, 1 sample) sits
  exactly in the expected range => **pipeline and verifier chain are healthy**.
- NOT a benchmark/capability claim: diagnostic only, never used for comparisons.

## 6. Pool distribution-shift diagnosis (representativeness audit)

| metric | Full Promptset (7620) | **B002 512** | E023 128 |
|---|---|---|---|
| source synthetic | 77.7% | **5.7%** | 86.7% |
| source autoformalizer | 13.8% | **71.7%** | 6.3% |
| singleton components | 13.2% | **93.8%** | 8.6% |
| formal tokens (median) | 117 | **166.5** | 108 |
| lines (median) | 5 | **8** | 5 |

Mechanism: component-level fail-closed exclusions consumed 95.1% of all
multi-variant statements (synthetic is 99.9% multi-variant), while
autoformalizer statements are inherently singleton-dominant (72.6%) — so the
eligible pool is selectively the autoformalizer/human singleton hard tail. The
cross-check closes the loop: **theta0 solves ~31%/24%/23% of synthetic-source
theorems across three independent measurements (B002 subset, calibration, E023)
but 0.5%/0.0% of autoformalizer/human singletons.** B002's 2.1% must NOT be
generalized as theta0's overall success rate.

## 7. Recommendation: B003 (multi-seed Expected Oracle)

**Conditionally supportive of proceeding.** Rationale:

1. The core scientific question (deployable-information headroom over uniform)
   is still open: all B002 frontier numbers are hindsight and not implementable;
   Expected Oracle estimates p_i(b) from K trajectories and is the minimal
   deployable-information baseline.
2. The pilot's compute-sensitive mass is sparse (11/512) but non-degenerate, and
   the candidate-response gradient is real (0 -> 3 -> 9 -> 11 solved across
   budgets). This is exactly the regime where an expected-value allocator either
   shows reallocation value or provably does not.
3. B003's evaluation pool (192 reserve components) is the *same* family-unseen
   hard pool; conclusions must be scoped to it (no claims about the mixed
   Promptset distribution, where ~78% of mass is a different, easier regime).

**Explicit expectations to set before B003:**
- if the Expected Oracle's no-skip gain over uniform is small (e.g. <10-15% of
  allocated cap at matched solved counts), the allocator line has a low ceiling
  on this pool and a B004 decision model is not justified;
- if the gain is large (>30-40%), a B004 model has a quantified target;
- triage (skip-allowed) must be reported separately and never mixed into the
  primary number.

## 8. Open items / limitations

- 20 unresolved prefix cells on 8 theorems (sensitivity [11, 19]); never counted
  as failures. Resolvable only by heavier verification (longer timeouts).
- Calibration has 3 unresolved infra cells (rank 54); sensitivity 13..14.
- K=1 per theorem: B002 is a screening pilot, not a p_i(b) estimate (that is
  exactly what B003 adds with K=8).
- All success figures are under the frozen B1-compatible single-path HF
  semantics; vLLM is excluded; batch>1 excluded.

## 9. Artifacts

- manifest `experiments/manifests/v2/V2-B002.yaml` (status RUNNING -> COMPLETE)
- raw rollouts `experiments/results/v2_b002_rollouts.jsonl` (sha256 `523b6124...`)
- analysis `experiments/results/v2_b002_analysis.json` (dual frontier)
- adjudication `experiments/results/v2_b002_adjudication.json`
- calibration set/analysis `v2_b002_calibration_set.json`, `v2_b002_calibration_analysis.json`
- representativeness `experiments/results/v2_b002_representativeness.json`
- concurrency evidence `experiments/results/v2_b002_concurrency_check*.json`
