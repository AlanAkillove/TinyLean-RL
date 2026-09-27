# V5-P001 count reconciliation: `3644` / `3123` / `3129` / `3012`

Status: **documentation only**, owner directive 2026-09-27 §4. This memo reconciles four
candidate counts that appear in V5-P001 and V1-adjacent material and could be mistaken for one
another. It **edits no historical result artifact, introduces no new surface, no new process
definition, no new parser heuristic and no new oracle call, and changes no classification.**
V5-P001 stays `PROCESS_SIGNAL_GO`; V3 and V4 dispositions are untouched.

Recomputed read-only on fly90 from the frozen surface
`experiments/manifests/v5/v5_historical_surface.json`
(sha256 `b3c8f472b863aaef356b97ea897e141c6206daa3b17d897aa61f64f2173c5902`), whose own provenance
and hash were verified in Phase A/B. All counts below are integer record counts, not scientific
metrics; nothing here is a new endpoint.

## 1. The four numbers

| # | definition (exactly) | universe | value |
|---|---|---|---|
| 1 | candidates whose upstream V1 record field `format_error` is not `"No error."` | full surface, 5 760 candidates | **3 644** |
| 2 | candidates whose `format_error` is the single dominant message `'Could not find a valid lean4 code block.'` | full surface, 5 760 candidates | **3 123** |
| 3 | candidates for which the **V5 frozen extraction predicate** found no provable code (`pred_kind != "code"`, i.e. `pred` is a no-proof/no-statement sentinel) | full surface, 5 760 candidates | **3 129** |
| 4 | same V5 predicate as #3 | **primary** surface (contaminated groups excluded), 5 488 candidates | **3 012** |

Numbers 1 and 2 answer *"what did the upstream V1 formatter report?"*; numbers 3 and 4 answer
*"what did V5 actually submit to the Lean oracle?"*. They are different predicates over
different fields, applied to different universes, so equality was never expected.

## 2. Two distinct predicates

**Upstream record field (`format_error`).** V1 rollout records carry a `format_error` string
produced by the *original* V1 data pipeline. `scripts/v5_p001_reconstruct.py:240-241` copies it
verbatim — `record.get("format_error", "No error.")` — and sets `has_format_error` to
`!= "No error."`. V5 never parses or re-derives this string; the surface is a read-only audit of
historical records. Group-level `n_format_error` (`:261`) is the sum of that flag.

**V5 frozen extraction predicate (`pred_kind`).** The surface also stores each candidate's
upstream `pred` field, classified by `sentinel_kind` (`scripts/v5_p001_reconstruct.py:175-180`)
against the two sentinels declared in `scripts/v5_p001_spec.py:43-45`:
`"No proof found in the output."` and `"Theorem statement couldn't be parsed from statement."`.
`pred_kind == "code"` means a code extract exists, and only such candidates are
`extractable` in the Phase-B plan (`scripts/v5_p001_process_run.py:137`) and therefore submitted
to the oracle (2 476 submissions). Sentinel candidates are labeled `FORMAT_NO_CODE` without any
oracle call. This predicate is the one that governs V5-P001's gates, and the mapping
`pred_kind == "code"` ⇔ `extractable` is exactly the boundary validated by
E1/E2 and the re-derivation check.

The two fields overlap but neither implies the other — see §4.

## 3. Universes

| universe | groups | candidates | code | sentinel |
|---|---|---|---|---|
| full frozen surface | 720 | 5 760 | 2 631 | 3 129 |
| contaminated (34 groups excluded from primary) | 34 | 272 | 155 | 117 |
| **primary** | 686 | **5 488** | **2 476** | **3 012** |
| primary `ALL_FAIL` (the G1 population) | 575 | 4 600 | 1 736 | 2 864 |

3 129 − 3 012 = 117 and 2 631 − 2 476 = 155: the contaminated 272 candidates carry both
reductions, and all 103 `infra` flags (`tool_feedback` starting with `# System Error:`) lie
inside those 34 groups (primary `infra` = 0 by construction).

## 4. Cross-tab over the full 5 760

| | `format_error == "No error."` | `format_error` = dominant message | `format_error` = another message | total |
|---|---|---|---|---|
| V5 sentinel (`pred_kind != "code"`) | 7 | 3 095 | 27 | **3 129** |
| V5 code (`pred_kind == "code"`) | 2 109 | 28 | 494 | **2 631** |
| total | 2 116 | **3 123** | 521 | 5 760 |

Identities that follow, all verified:

* `3644 = 5760 − 2116 = 3122 (sentinel with some format error) + 522 (code with some format error)`
* `3123 = 3095 (sentinel) + 28 (code)`
* `3129 = 3095 (dominant) + 27 (other message) + 7 ("No error.")`
* `3644 − 3123 = 521` candidates carry a format-error message other than the dominant one
* primary layer: `3012 = 2980 (dominant) + 25 (other) + 7 ("No error.")`, and
  `3493 = primary candidates with format_error != "No error."`
* inside the G1 population (4 600 primary all-fail candidates): 3 246 carry a format-error
  message, 2 834 of them the dominant one, and 7 sentinel candidates report `"No error."`

Full `format_error` histogram over 5 760 (sums to 5 760; 3 644 = all rows except `"No error."`):

| message | count |
|---|---|
| `Could not find a valid lean4 code block.` | 3 123 |
| `No error.` | 2 116 |
| `Tactics and Lean4 code do not match.` | 260 |
| `Tactic code contains too many lines.` | 183 |
| `Tactic block formatting error.` | 38 |
| `Lean 4 code does not start with the formal statement.` | 16 |
| `There must be more than one tactics block.` | 12 |
| `There must be exactly one think block.` | 11 |
| `Generation repeats.` | 1 |

## 5. Why the off-diagonal cells are non-empty (mechanism, not an anomaly)

**7 sentinel candidates with `"No error."` (and symmetrically 28 code candidates with the
dominant message).** `format_error` is the upstream V1 pipeline's *own* verdict on the *stored*
`pred`; the V5 predicate classifies that same `pred` string against the two sentinel literals.
The two judgments are made by different code, at different times, on the same bytes, and the
upstream pipeline's taxonomy is wider than "a code block was recovered" — hence a candidate can
be sentinel by the V5 literal test while upstream recorded no format error, and can carry the
upstream "no valid lean4 code block" message while its stored `pred` is nevertheless a concrete
extract. Neither cell is a V5 parse: both are read from historical fields.

**Why this cannot silently contaminate the V5 measurement.** V5-P001 never consumes
`format_error` as an experimental outcome. The oracle plan is built *only* from
`pred_kind == "code"` (`scripts/v5_p001_process_run.py:137, 190, 198`), and the runner's real
guard is semantic rather than label-based: for each extractable candidate it re-runs the upstream
extractor and aborts on disagreement
(`scripts/v5_p001_process_run.py:569-570`,
`raise RuntimeError(f"{entry['candidate_id']}: upstream extractor disagrees with pred")`), and
rejects an extractable candidate that carries a sentinel `pred` (`:198`). Note the scope
precisely: that agreement check covers *extractable* candidates only, which is exactly why the
7 sentinel-with-`"No error."` records survive into the surface unchanged — nothing about them was
verified against an extractor, and nothing about them needed to be, since they were never
submitted. The `"upstream_extractor_agrees": True` field in the preflight payload
(`:614`) is a literal written only when that loop completed, and its scope is the same
extractable set; it is not a claim about sentinel records.

Consequently the 2 476 oracle submissions, the E1/E2 denominators and all gate values are
functions of `pred_kind` plus the frozen oracle, and are invariant to which message the upstream
formatter happened to record.

## 6. Where each number is the right one to quote

* Planning and execution accounting for V5-P001: **3 012** sentinel / **2 476** code within the
  5 488-candidate primary plan (`n_planned 5488, n_code 2476, n_sentinel 3012`), the latter being
  the `oracle_submissions` figure. `FORMAT_NO_CODE: 3012` in
  `experiments/manifests/v5/V5-P001_results.json` is the same population labeled at analysis time,
  so the plan, the surface and the result agree without any re-derivation.
* Descriptions of the *whole* historical surface, including the 34 contaminated groups:
  **3 129** sentinel / 2 631 code of 5 760.
* Statements about what the upstream V1 formatter rejected: **3 644**, of which the dominant
  single reason is **3 123** — never as a substitute for the V5 no-code count.

## 7. Non-consequences

* No historical artifact is edited by this memo: `v5_historical_surface.json`,
  `V5-P001_results.json`, the freeze, the labels and the raw oracle items are byte-identical to
  their committed/preregistered hashes.
* No classification changes: V5-P001 remains `PROCESS_SIGNAL_GO`; V3 and V4 dispositions are
  unaffected. The count differences reconciled here were never gates, and no gate is re-evaluated.
* No new endpoint is introduced. Every figure is a record count obtainable by reading the frozen
  surface with the already-frozen reconstruction code.
* This memo does not license any re-dating, re-parsing or re-auditing of the V1 rollouts. Any
  further measurement on that surface requires its own preregistration and owner authorization.
