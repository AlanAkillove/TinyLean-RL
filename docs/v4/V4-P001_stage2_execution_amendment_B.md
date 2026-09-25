# V4-P001 Stage-2 execution-code amendment B — pinned formal-statement plumbing

- Date: 2026-09-25 (UTC) · Branch `v3-jev-rl-controller` · Host: fly90 (fix + tests) → fly122
  (post-commit revalidation)
- Authority: owner directive of 2026-09-25 (sections 1–16), received after the V4-P001 Stage-2
  launch-0 structural abort. This document is the §12 artifact: trigger, launch-0 record, root cause,
  fix, the scientific-design diff, and the pinned old/new hashes.
- **Scientific design changed by this amendment: NONE.** It is a *pre-outcome execution-code
  amendment*: only execution plumbing needed to make the frozen protocol runnable was touched, and
  the module that carries the frozen design (`scripts/v4_p001_spec.py`) is byte-identical in this
  working tree. No scientific outcome exists for V4-P001 Stage 2; this amendment precedes one.

---

## 1. Trigger (owner §12)

The first `repair_one` call of the Stage-2 formal launch raised `KeyError: 'formal_statement'` at
`scripts/v4_p001_rollout.py:1654` (pre-fix runner, commit `67b68d4`):

```python
lean_source = complete_verifier_code(screening_row["formal_statement"], extracted)
```

The exception aborted the launch structurally. It fired inside the candidate loop, after the frozen
arm prompt had been rendered and the model had produced its completion.

## 2. Launch-0 record (owner §8, §9)

Classification: **`V4-P001 Stage-2 launch-0 / STRUCTURAL_ABORT / NO_SCIENTIFIC_OUTCOME`**.

Exactly **8** transient completions were generated in memory before the code exception. They were
*not* generated "before generation"; the launch reached the generation loop, then the structural
defect stopped it. Recorded facts, and nothing else:

| fact | value |
|---|---|
| `transient_completions_generated` | **8** |
| `persisted` | **0** |
| `formal_verifier_calls` | **0** |
| `scientific_labels_observed` | **0** |
| analyzer run | **NO** |

No transient completion was recovered, compared, reused or inspected for any purpose, and none
entered any decision. The formal run is `V4-P001 Stage-2 formal launch-1`: the same scientific
experiment, from theorem 1, regenerating all 512 candidates from the frozen seeds.

## 3. Root cause (owner §12)

`repair_one` read the candidate's formal statement from the **screening row**:

- the frozen Stage-1 screening-row schema (`S.SCREENING_FIELDS`, 47 fields) contains **no**
  `formal_statement` field — and must not, per owner §2 ("Do not add `formal_statement` retroactively
  to Stage-1 raw rows. Do not modify the Stage-1 raw artifact/schema");
- the canonical statement lives in the **pinned promptset parquet**, loaded as the surface mapping
  (`load_surface`), whose rows are the only place the statement text exists
  (`{statement_id, formal_statement, messages, prompt, prompt_sha256, prompt_token_count}`).

The defect was therefore pure execution plumbing: a lookup against the wrong schema. No frozen
artifact, no design element and no verdict was implicated. The prior `--stage second --dry-run`
returned before any generation and never reached `repair_one`, which is why the §19 execution-package
dry run could not have caught it — the dry run did not execute the path it was meant to rehearse.

## 4. The fix (owner §2–§7)

### 4.1 One shared assembly implementation (owner §2, §7)

`candidate_verifier_source(formal_statement, completion_text) -> (lean_source, extracted,
has_lean_block)` is the single place a candidate is assembled: `extract_proof` → `None` when nothing
Lean can be extracted → `complete_verifier_code(formal_statement, extracted)`. Both the formal stage
(`repair_one`) and the dry-run validation (`validate_candidate_assembly`) call it, so the dry run and
the formal run cannot disagree about how a candidate is built. Stage 1's own call of
`complete_verifier_code` (`screen_one`) is untouched.

### 4.2 The §3 identity guard

`frozen_formal_statement(surface, plan_row) -> str` returns the pinned statement of one frozen
theorem and fails closed (`S.FrozenViolation`) on any mismatch:

| condition | outcome |
|---|---|
| `plan_row["statement_id"]` absent from the pinned surface | fail-closed |
| pinned `formal_statement` empty or whitespace | fail-closed |
| surface `prompt_sha256` ≠ frozen plan Arm-A `prompt_sha256` (missing or different) | fail-closed |

The guard binds three frozen identities: the pinned parquet (whose bytes `load_frozen` hash-verifies),
the statement id, and the first-attempt prompt hash the plan froze in Arm A. Any mismatch stops the
run *before any candidate of that theorem is assembled or verified*.

### 4.3 Explicit plumbing, no schema cross-read (owner §2, §4)

- `repair_one(..., *, plan_row, screening_row, formal_statement, arm, ...)` takes the statement as an
  explicit keyword. The screening row keeps serving only its own schema (identity fields, category,
  failed proof, diagnostics) and is never asked for a field it does not have.
- `stage_second` resolves `formal_statements = {rank: frozen_formal_statement(surface, theorem)}` for
  all 128 ranks **before** anything is generated, aborts on the first guard violation, and then runs
  `validate_candidate_assembly` over the whole cohort, aborting with the failure list if anything
  fails. The assembly evidence is stamped into the run summary (`candidate_assembly`) and printed.
- The screening-row schema was not modified; no `formal_statement` was added to it, retroactively or
  otherwise.

### 4.4 The strengthened dry run (owner §7)

`--stage second --dry-run` now executes the same guard and the same `validate_candidate_assembly`
path the formal run executes, for all 128 frozen theorems, and reports, without a model call and
without a verifier call (`generations = 0`, `formal_verifier_calls = 0` by construction and asserted
by test):

- `surface_statements_available`, `formal_statements_nonempty`,
  `formal_statements_bound_to_the_frozen_plan` — 128 each (the §3 guard applied to the cohort);
- `failed_proofs_available`, `own_diagnostics_available`, `donor_diagnostics_available` — 128 each;
- `arm_plan_rows_complete`, `paired_seeds_valid`, `arm_orders_valid`,
  `screening_identity_fields_available`, `assembly_probes_passed` — 128 each;
- `arm_prompt_hashes_matched` — 512 (the 512 arm prompts re-rendered and matched against the frozen
  plan hashes);
- plus two cohort-level checks: `plan_ranks_are_the_frozen_cohort`,
  `screening_rows_cover_the_cohort`.

The `assembly_probes_passed` counter assembles a synthetic probe body through the real
`candidate_verifier_source` for every theorem: a source that can be assembled at all is proved for
the whole cohort. The probe is never generated, never verified and never written.

## 5. Scientific design: the diff

**NONE.** The frozen list is untouched: 128-theorem cohort; failed proofs; normalized diagnostics;
A/B/C/D definitions; the diagnostic derangement; paired seeds; the arm schedule; generation
parameters; response budget; success definition; C-A and C-B thresholds; the data guard; the
differential-censoring guard; the C-D mechanism rule; the verifier policy; the recovery limits; and
the sealed reserve. `S.FROZEN_SETTINGS` is not edited, no threshold moves, no artifact in
`experiments/` or `runs/` is rewritten, and the Stage-1 raw artifact and its schema are unchanged.
The only files this amendment touches are the runner's Stage-2 execution plumbing, the execution
package dry-run script's checks, and their tests.

## 6. Evidence (owner §5, §6, §13)

| requirement | test | what it pins |
|---|---|---|
| §5 R1 | `test_repair_one_sends_exactly_the_assembled_source_of_the_pinned_statement` | a fake verifier receives exactly `complete_verifier_code(formal_statement, extract_proof(completion))` for the pinned statement of the theorem |
| §5 R2 | `test_a_frozen_screening_row_carries_no_formal_statement_and_repair_one_needs_none` | a screening row holding exactly the frozen 47-field schema works; a poisoned `formal_statement` key is ignored, not read |
| §5 R3 | `test_the_formal_statement_guard_fails_closed_on_every_identity_mismatch` | every guard condition rejects |
| §3 | `test_a_missing_pinned_formal_statement_fails_closed_before_any_verification` | the run exits 3, the verifier is never called (`session.calls == []`), no raw artifact and no summary are written, stderr names Amendment B §3 |
| §4 | `test_every_stage_two_row_access_is_drawn_from_its_own_schema` | AST audit of every subscripted name in the module: each name's keys stay inside its own schema, no dynamic keys, and `formal_statement` is absent from `S.SCREENING_FIELDS` and from the audited `screening_row` reads |
| §6 | `test_all_four_arms_of_one_theorem_run_through_the_real_second_stage` | the real `stage_second` / `repair_one` / row assembly for one synthetic theorem (ranks 1–127 pre-written as legal censored quadruplets), all four arms, fake model + fake verifier: 0 real generations, 0 formal verifier calls |
| §7 | `test_the_second_stage_dry_run_validates_the_whole_candidate_assembly` | the 128-theorem assembly evidence above, 0 generations, 0 verifier calls, no writes |
| §7 | `test_the_candidate_source_has_a_single_implementation` | `complete_verifier_code` is called only by Stage 1 and by `candidate_verifier_source`; `candidate_verifier_source` only by `repair_one` and `validate_candidate_assembly`; `frozen_formal_statement` and `validate_candidate_assembly` only by `stage_second` |

**Regression property (owner §6).** With the pre-fix runner restored from commit `67b68d4`, exactly
the eight Amendment B tests fail (the other eleven in the file pass). Two of the failures are the
§6/§4 evidence itself: the integration fixture reproduces `KeyError: 'formal_statement'` at
`scripts/v4_p001_rollout.py:1654` — the launch-0 line, reached through `main` → `stage_second:1934` →
`repair_one` — and the schema audit reports `screening_row[...] reads ['formal_statement'], which is
not in its schema`. The amended runner passes all of them, and the runner file was restored
byte-identical afterwards (hash below).

Suite status at this commit: `tests/test_v4_p001_rollout.py` 19 passed; the V4 surface
(`tests/test_v4_p001.py` + `tests/test_v4_p001_analyze.py` + `tests/test_v4_p001_rollout.py`) 108
passed; the complete `tests/` suite passed (exit 0); `ruff check src scripts tests` clean
(ruff 0.16.7).

## 7. Validation (owner §13, §14)

On fly90 (this commit): the table above, plus the runner's own `--stage second --dry-run` executed
inside the fixtures for all 128 theorems with zero model and verifier contact.

On fly122 (post-commit, per owner §14): a fresh execution-package dry run
(`.venv/bin/python scripts/v4_p001_dry_run.py`), regenerating
`experiments/manifests/v4/V4-P001_execution_package_dry_run.json` and its registry block, requiring
the §14 evidence list — 128 frozen theorems, 512 planned candidates, 512 valid prompt hashes, 128
surface formal statements, candidate assembly validated, paired seeds valid, arm schedule valid,
verifier identity valid, canary PASS, real generations = 0, formal verifier calls = 0 — and recording
the new transcript hash. **Stage-2 relaunch remains unauthorized (owner §15).**

## 8. Hashes (old → new)

| item | pre-fix (`67b68d4`) | Amendment B |
|---|---|---|
| `scripts/v4_p001_rollout.py` | `5e506e0a695160fd813a94d3d1015bc2b89ea7e0a120e8e8a1f58d7fffdcd816` | `abc8249bdc1bb7aa0de95ddaa13b42a95d689c05160893bb074335d2978d2cc1` |
| `scripts/v4_p001_dry_run.py` | (unchanged before this commit) | `fcdb4c315940d483b6e4cec646fff3a4bf7059e7480b89d5a3a24ffe783dbf99` |
| `tests/test_v4_p001_rollout.py` | (unchanged before this commit) | `5851e3a594536230e6c3434441fed1efae58734ed2c16ddff5770d4ad7ce9f38` |
| `scripts/v4_p001_spec.py` (frozen design) | — | byte-identical, untouched |

## 9. Where each fact lives

- this amendment: `docs/v4/V4-P001_stage2_execution_amendment_B.md` (owner §12);
- runner fix + guard + shared assembly: `scripts/v4_p001_rollout.py` §"stage: second";
- execution-package dry-run revalidation: `experiments/manifests/v4/V4-P001_execution_package_dry_run.json`
  and the `execution_package` block of `experiments/manifests/v4/registry.yaml`;
- tests: `tests/test_v4_p001_rollout.py` (Amendment B section);
- the launch-0 classification and the future launch-1 identity: owner directive of 2026-09-25 §8/§9.
