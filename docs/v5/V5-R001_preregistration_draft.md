# V5-R001 — Outcome-only RLVR vs Process-Verified RLVR: preregistration DRAFT

**Status: `DRAFT_FOR_OWNER_REVIEW` — NOT frozen, NOT approved, and NO run is authorized by this
document.** `V5_R001_DRAFT_CREATED: YES` · `V5_R001_TRAINING_LAUNCHED: NO`.

Drafting authority: `PROCESS_SIGNAL_GO` in V5-P001 (preregistration §8, registry `stop_rule`:
"PROCESS_SIGNAL_GO authorizes drafting a V5-R001 preregistration only") together with the owner's
Analyzer-Amendment-B approval message §12 ("you may automatically draft the V5-R001 (Outcome-only
RLVR vs Process-Verified RLVR) preregistration only. Do NOT launch training"). Every numeric or
structural choice below marked **OPEN** requires an owner decision; this draft is the agenda for
that decision, not a frozen design. Nothing here may be used to load a model, generate a token or
update a weight.

---

## 1. Motivation and the admissible evidence base

V5-P001 established, on the frozen V1 surface only, that the paper-compatible first-error process
credit is **constructible offline at scale**: 72.2 % of primary `ALL_FAIL` groups (415/575, CI
[0.679, 0.764]) contain at least one candidate whose generated proof consists of a verified tactic
prefix followed by a first error (`PREFIX_BEARING_FAILURE`); 1 133/1 736 all-fail code candidates
(65.3 %, secondary). The same census shows what the outcome signal alone collapses:
1 370 `PREFIX_BEARING_FAILURE` candidates are all rewarded identically as failures, while 3 012
`FORMAT_NO_CODE` candidates contain no tactic at all and are reachable by neither channel.

What V5-P001 does **not** establish, verbatim from its limitations: no training, no generation, no
capability or gradient claim; the d1/d2 credit surface was never applied to a model. The R001
question is therefore exactly the next one:

> At the same model, data, prompts and compute, does adding the audited first-error process credit
> to the RLVR reward improve held-out proof success beyond outcome-only RLVR?

## 2. Proposed design skeleton (candidate base; OPEN items flagged)

- **Model:** theta0 Kimina-Distill-0.6B (the frozen V1/V4 theta0), full-parameter GRPO-family
  training (DrGRPO, mean-only centering, no KL/entropy) — the V1 recipe family
  (`norm_adv_by_std_in_grpo=False`, n=8, temp 1.0, top_p 1.0, tb4 → 32 seq/step, max_prompt 1024,
  max_response 4096, lr 2e-6, vLLM util tuned, multiturn off).
- **Arms (the owner's two):**
  - **A — Outcome-only RLVR:** reward = frozen binary outcome (verified vs not; `sorry` = failure).
  - **B — Process-Verified RLVR:** reward = outcome + first-error process credit
    (`d1 = −0.05` verified-prefix tokens strictly before the earliest erroneous tactic,
    `d2 = −0.10` at it and after), realized on generated tokens through the **frozen** tactic →
    token mapping of the V5 process oracle. **OPEN:** credit weight λ, normalization (per-token /
    per-candidate), whether credit applies only to `PREFIX_BEARING_FAILURE` candidates, floors.
- **Everything else paired:** identical data stream, prompts, optimizer settings and seed handling
  per arm (paired seeds). **OPEN:** number of paired seed pairs.
- **Data:** **OPEN** — pool, split and evaluation groups. Constraints already frozen by policy:
  the V3-FINAL-HOLDOUT 93-component reserve and every future family-clean holdout stay sealed and
  unused; V1/V2/V3/V4 artifacts are never edited; the evaluation must be blind to the process
  channel (process metrics are diagnostics, never capability evidence).
- **Compute and host policy:** R001 training would run on fly122 only (V5 host policy; fly90 is
  coordination/archive + CPU analysis and "does NOT become a formal model-compute node").
  **HARD FEASIBILITY FLAG (must be resolved before any freeze):** fly122 carries an RTX 3080
  **10 GB**, while the V1 GRPO recipe's measured peak was ≈ 24.4 GB allocated (24 GB-class card) at
  the frozen shape. The 10 GB envelope requires a footprint study (batch/sequence sharding, vLLM
  utilization, checkpointing policy) before R001 can be frozen; **any step that performs
  optimization is training and therefore needs owner authorization of its own.**
- **Infrastructure discipline (carried over, all frozen precedents):** infrastructure outcomes are
  censored data, never failed proofs or negative reward; bounded retries/recoveries; the two-strike
  wedge precedent (V5 Amendment A); no live scientific metric during a formal run.
- **Gates and classification:** **OPEN** — to be designed before freezing with a prospective power
  analysis. Skeleton: primary = paired B−A difference in held-out proof success with a
  preregistered practical threshold and bootstrap CI lower bound; secondary = process census and
  reward-diagnostic tables; a frozen GO / NO-GO / INCONCLUSIVE taxonomy with no post-outcome rescue.
- **Reserve hygiene:** `SEALED_RESERVE_TOUCHED: 0` is the only admissible value.

## 3. Owner decisions required before this draft can become a preregistration

1. **Authorization scope:** is a bounded, nonformal foot-print study on fly122 (few optimizer
   steps, no evaluation, no scientific metric) authorized to determine the 10 GB recipe, or does
   each such step need separate approval?
2. **Budget:** total optimizer steps (or token budget) per arm; hardware-hour ceiling; wall-clock
   window; the V1 60-step precedent as a floor or ceiling?
3. **Data:** which pool feeds training, which groups form the held-out evaluation, and what makes
   them family-clean; is a new pool build required (with its own prospective freeze)?
4. **Reward shape:** λ and normalization for the process credit; the exact token-level φ; whether
   the credit is applied to all candidates or only the recoverable class.
5. **Gates:** primary endpoint definition, practical threshold, CI rule, paired-seed count and the
   power analysis before freezing.
6. **Two-arm confirm only, or A/B plus a process-only ablation** (e.g., process credit without
   outcome reward) as a mechanism arm?
7. **Freeze mechanics:** who signs the frozen preregistration (owner directive), the commit/push
   discipline, and whether fly122 must be re-preflighted first.

## 4. Prohibitions in force (unchanged until an owner directive says otherwise)

- No training, no weight update, no model generation, no R001 launch — by this draft or the
  PROCESS_SIGNAL_GO result (V5-P001 preregistration §8).
- No use of the sealed V3 93-component reserve or any family-clean capability holdout.
- No modification of any frozen V5-P001 artifact; the canonical analyzer is frozen after its
  metric-emitting run, so any analyzer change would require a new owner-approved amendment.
- No process-reward rescue path: nothing in V5-P001 or this draft authorizes changing a gate, a
  threshold or a denominator after an outcome.

## 5. Provenance

- V5-P001 canonical result `experiments/manifests/v5/V5-P001_results.json`
  (sha256 `ab46c12dbe098505ba8f3885fddb8af18ef2420dc0d6c66cc856a26e4d08ddd8`), classification
  `PROCESS_SIGNAL_GO`, analysis head `7d0b139` over FreezeA (execution head `5b1c5d2`).
- Result memo `docs/v5/V5-P001_result.md`; oracle and conventions
  `docs/v5/process_oracle_design.md`; amendments A and B.
- This draft, once committed, is a coordination artifact: it records the authorized drafting event
  and freezes nothing.
