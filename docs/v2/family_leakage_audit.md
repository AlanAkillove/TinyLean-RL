# V2 theorem-family leakage audit (2026-09-20)

Status: **complete** · artifact: `experiments/manifests/v2/family_leakage_audit.json` ·
script: `scripts/v2_family_leakage_audit.py` · tests: `tests/test_v2_family_leakage_audit.py`.
Pure metadata (no GPU, no Lean, no training), deterministic, fail-closed. Built and run
while V2-A001 was still evaluating: it reads frozen sets only and changes nothing.

## 1. Why this audit exists

The A0/A1 protocol text promised near-duplicate family isolation (`experiment_protocol.md` §2:
"normalize statements (whitespace / comments / binder names) and group near-duplicates before
splitting"), but the realized theorem-role registry (2026-09-19) splits on the **exact**
normalized statement (per-line rstrip + whole-text strip): 6,729 eligible statements formed
6,729 groups. That guarantees **exact-statement isolation only**. The Promptset stores one
source problem as many `_v<digits>` variants - e.g. `number_theory_29735_v0001 … _v17567`
(10 members, each a different auto-generated formalization of the same natural-language
problem) - so source-problem families are the near-duplicates that matter, and this audit
measures the actual overlap.

## 2. Layers

| layer | key | notes |
| --- | --- | --- |
| L0 `exact` | `p3c_build_fixed_set.normalize` (per-line rstrip + strip) | the current split unit |
| L1 `strong` | NFKC + whitespace-collapse + strip | folds math alphanumerics (`ℕ→N`, `ℝ→R`) |
| L2 `skeleton` | L1 + first declaration name abstracted (`theorem _ …`) | catches same statement, different name |
| L3 `name-family` | theorem name minus the trailing `_v<digits>` suffix | **source-problem family** (primary) |
| L4 `nl` | NFKC + whitespace-collapse of `natural_language` | non-empty texts only |

## 3. Findings

### 3.1 The pool is mostly variant families

- 7,620 unique statements → **1,710 L3 families**; 701 multi-id families cover 6,611 ids.
- Pool (6,729): **1,568 families**; 701 multi-id families cover 5,862 ids (**87 %**).
- L2 adds little beyond L3 at the statement level: 26 duplicate groups / 56 ids over the
  dataset (23 groups in the pool); a handful are cross-family exact matches
  (e.g. `inequalities_191212_v0002` = `inequalities_230072_v3508` with different names).
  **A001 has zero L2/L1/L0 duplicates internally** - its statements are unique at those layers.
- L0/L1/L2 are unique for the pool at id level; every overlap below is family-level.

### 3.2 A001 (512) vs V1 consumption (L3, source-problem family)

| V1 source | shared families | A001 ids implicated |
| --- | --- | --- |
| E013 promptset rollouts | 10 | 13 |
| E016 n8 calibration | 7 | 10 |
| E018 dev set (P3C) | 28 | 35 |
| E020-M mechanism set | 26 | 37 |
| training dumps p3b_pilot | 60 | 77 |
| training dumps m1_seed2 | 62 | 79 |
| training dumps m1_seed3 | 78 | 99 |
| training dumps m2_qwen_smoke | 6 | 6 |
| **V1-used union** | **180** | **232 / 512 (45 %)** |
| **E023 sealed holdout** | **40** | **52 / 512 (10 %)** |
| **union** | **201** | **261 / 512 (51 %)** |

The id-level exclusion worked as designed: the L0/L1 overlap is **zero** (asserted by the
audit). The family overlap above is what the statement-level split cannot see: 45 % of the
A001 set has a `_v`-sibling that V1 trained on or evaluated.

### 3.3 Cross-role family overlap of the current partition (L3)

- **684 of 1,568 pool families span ≥ 2 roles; 5,745 of 6,729 ids (85 %) sit in them.**
- Role coverage (ids whose family spans ≥ 2 roles / role size): B-train 3,407/4,021 ·
  B-validation 607/701 · B-test 544/639 · A-selection 585/674 · C-joint-holdout 590/680 ·
  B1-audit-reserved 12/14.
- Pairwise shared families (worst): B-train × B-validation 415 · B-train × C-joint 415 ·
  B-train × B-test 403 · B-train × A-selection 392 · B-test × C-joint 236 ·
  A-selection × C-joint 226 (full matrix in the artifact).
- 70 families span **all five** buckets; distribution: 884 families in 1 role, 109 in 2,
  267 in 3, 238 in 4, 70 in 5.
- **A3-primary × A-reserve: 56 shared families** - A-reserve is not family-independent
  from the A001 set.
- B1 carve-out: family siblings of the 14 in-pool B1 statements sit in B-train (51 ids),
  B-validation (12), B-test (9), A-selection (12), C-joint-holdout (5).

### 3.4 Impact numbers for future amendments

- **A001 effective clusters**: 512 ids in **392 families** (98 families carry 218 ids).
  The ~**1.14×** CI-width factor is a *heuristic sensitivity* only (392 vs 512 clusters),
  not a formal correction: A001 executes the preregistered theorem-level bootstrap
  unchanged (see §5.1). A family-cluster bootstrap may be added after completion as a
  clearly-labelled post-hoc robustness analysis without touching the rule or the verdict.
- **Track C family-clean subset**: only **90 of 680** C-joint-holdout ids have an L3 family
  disjoint from *everything* (V1-used ∪ E023 ∪ B/A roles). If "clean" only needs to mean
  "not used for selection", the number is 361/680. A family-granular re-partition is
  required before the final evaluation can rest on a larger clean holdout.
- `source` distribution (synthetic / autoformalizer / human): A001 418/61/33 ·
  V1-used 389/216/158 · E023 111/8/9 · B-train 3,255/481/285 · C-joint 544/83/53 -
  the consumed sets skew toward autoformalizer/human provenance relative to the pool.

## 4. Decisions (no changes to any frozen set)

1. **V2-A001 stays exactly as preregistered** - no re-draw, no theorem removal, one-shot
   rule stands (`family_leakage_audit.json > freeze_guard`).
2. **Result positioning**: A001 is a *selection device* for `theta_RL*`. No A001 delta -
   even with `ci_low > 0` - is a confirmatory RL-vs-theta0 capability claim (selection bias
   on the same set, plus partial V1-family exposure). The final capability conclusion is
   reserved for a family-clean Track C holdout. The A4 freeze package must include this
   audit and report the overlap next to the selection outcome.
3. **Future registry amendments** split on the family key (L3 primary, L2 as audit layer),
   never on the raw statement; no re-split of the frozen current partition.
4. **Track B (fly122)**: derive B2 train/val/test from family-complete groups; extend the
   B1 carve-out to its family siblings (51+12+9 ids in the B pools) if the no-contact
   guarantee must hold at family level.
5. **Track C**: the final holdout is drawn on the family-component split (§5.3), not on the
   exact-statement registry; 90/680 C-joint ids are family-clean today, so the component
   split is materialized as a family-granular registry amendment before the final evaluation.

References: `experiment_protocol.md` §2/§8 (corrected 2026-09-20), `theorem_role_registry.json`
(unchanged), `track_a_rl_plan.md` §4.1 amendment.

## 5. Owner constraints (2026-09-20, post-recognition)

Recorded as canonical methodology constraints. They change neither the frozen sets nor the
running V2-A001 execution.

### 5.1 CI width is heuristic sensitivity only

The 392-vs-512 cluster arithmetic (~1.14x) is a *heuristic sensitivity note*, not a formal
CI correction. V2-A001 strictly executes the preregistered theorem-level bootstrap
(`paired_bootstrap`, 10k resamples, seed 20260917): the selection rule, margins and verdict
are computed from it, unchanged. After A001 completes, a **family-cluster bootstrap** (L3
families as resampling units) may be reported as a clearly-labelled *post-hoc robustness*
analysis only; it must never alter the selection rule or the A001 verdict.

### 5.2 A-reserve is not an independent A2 evaluation set

A-reserve shares **56 L3 families** with A3-primary, so it is no longer suitable as the
independent key-evaluation set for a future `V2-A002`. Candidate 2 stays dormant; if it is
ever triggered, its evaluation set is a **newly preregistered A2-eval set** under the
family-granular registry, family-disjoint from A001. A-reserve is **not re-drawn now**, and
no A001 frozen artifact changes. This supersedes the usage note recorded inside the frozen
`theorem_role_registry.json` (`a_selection_split.usage`); the registry file itself is
untouched.

### 5.3 Track C holdout = family-component split

The Track C final holdout is based on a **family-component split**, not the existing
exact-statement registry. Component definition (at least): merge by L3 source-family, and
add cross-name connections via L2 statement-skeleton equality and normalized nonempty
`natural_language` equality (connected components of the union of these relations). The
component split is materialized as a preregistered family-granular registry amendment
before any Track C evaluation, so that no evaluation theorem shares a component with V1
training/evaluation data or with any A/B selection pool.
