# V5 Process Oracle — Design, Conventions and Validation

**Scope.** This document freezes how V5-P001 turns a submitted Lean proof attempt into a
*process label*: the ordered tactic list, the first error, the per-tactic credit `d1` / `d2`,
and the tactic → generated-token mapping. It also records the infrastructure policy of the
dedicated process-oracle instance and the Phase-A validation evidence. It is the machine-readable
twin of `scripts/v5_p001_spec.py` and the prose twin of `scripts/v5_process_oracle.py`.

**It is not** a training document, a capability claim, or an RL-component design. V5-P001 performs
no training and no generation (owner directive §1, §16, §26). The oracle only *reads* historical
proof text and asks a Lean server what the elaborator did with it.

| | |
|---|---|
| Owner directive | V5-P001 (Offline Process-Reward Recoverability Audit), §2, §7–§13, §21–§23, §26, §33 |
| Frozen spec | `scripts/v5_p001_spec.py` (machine-readable) |
| Oracle implementation | `scripts/v5_process_oracle.py` |
| Surface reconstruction | `scripts/v5_p001_reconstruct.py` → `experiments/manifests/v5/v5_historical_surface.json` |
| Phase-B runner | `scripts/v5_p001_process_run.py` |
| Fixture validation | `experiments/manifests/v5/v5_process_oracle_validation.json` (fly122) |
| Dedicated instance | `tinylean-rl-lean-oracle-v5` @ `http://127.0.0.1:8020` on fly122, image digest `sha256:588a2cbb…e3ffd9` |
| Position convention | every reported line **1-based**, every column **0-based**, finish **exclusive**, in the **frame** |

---

## 1. The paper method this oracle implements

The owner §2 defines the process-credit method to be mirrored, and forbids inventing a different
primary scalar:

1. **Tactics are ordered by AST start position** — not by newline, semicolon text, regex or
   indentation (owner §7). The ordering key is the elaboration-tree source span, i.e. the
   `(start_line, start_column)` reported by Lean itself.
2. **Global outcome** `g(Y) ∈ {1, 0}` is the whole-proof verifier verdict: 1 iff the canonical
   Kimina checker accepts the submitted code with no error-severity message and no `sorry`.
3. **Earliest erroneous tactic** `j` is the first tactic (in AST order) whose elaboration failed.
4. **First-error propagation** assigns `φ = +1` on a successful proof, `φ = d1` to every tactic
   strictly before `j`, and `φ = d2` to the tactic at `j` *and everything after it*.
5. **Canonical, paper-compatible setting: `d1 = -0.05`, `d2 = -0.10`** (`D1_CANONICAL`,
   `D2_CANONICAL`). Sensitivity settings `(-0.05, -0.50)` and `(-0.10, -0.10)` are evaluated after
   the primary classification only, and are label-preserving (owner §23).

The credit is emitted **per first generated token** of each tactic span, never as a
normalized-prefix scalar and never as a model update (owner §16).

**Historical labelling pipeline (why the oracle is needed at all).** The V1 rollouts were graded
only globally: `score = acc * format`, where `acc` is the binary Lean verdict. A `score = 0`
candidate is therefore *uninformative about where the proof died* — the exact gap V5-P001 audits.
V5 does not re-derive the global label with new code: it re-submits the stored `pred` string to a
dedicated instance of the **same pinned image** so the whole-proof semantics are identical, and
then reads the infotree the production path discards.

---

## 2. Position conventions (calibrated in Phase A)

All server-reported positions — in `messages` **and** in infotree ranges — obey one convention,
calibrated with probes and then pinned by fixtures:

* line is **1-based**, column is **0-based**, `finish` is **exclusive**;
* positions are relative to the **frame** = the submitted code minus a **maximal leading prefix of
  blank-or-`import` lines**. A comment line stops the stripping. `strip_header()` replicates it
  client-side and returns `(frame, prefix_chars)`.
* `frame_offset(frame, line, column)` converts a server position to a Python character offset; a
  position outside the frame raises `PositionError` instead of being silently clamped.

Consequences that are easy to get wrong, and are therefore asserted:

* the *submitted* code (`pred`) is `normalized_formal + generated_tail`, so the tactic spans live in
  the frame, while the token spans live in the **response**; the oracle converts between the two
  with one offset chain (see §6);
* a tactic that begins at column 0 of an indented line is reported at the position of its first
  character, and a message that *ends* a sequence (e.g. `unsolved goals`) can start **after** the
  last tactic (see §5, `range_last`).

### 2.1 Replication of the historical extraction

`locate_generated_tail(response, formal_statement)` mirrors the pinned upstream
`extract_proof_from_text` **verbatim** — same `FORMAL_BLOCK_RE`, same reversed block search, same
`is_index_commented` guard, same `:=\s*by` search after the extracted theorem statement — and
additionally returns the character offset of the tail inside the response. It is checked against the
stored `pred` for every candidate of the primary surface (Phase A: 2 631 real preds over all 720
groups recompute byte-exact; 3 129 sentinels consistent; 0 unmapped statements).

---

## 3. Tactic extraction from the elaboration infotree

`flatten_infotree` walks the `response.infotree` (its root may be a dict **or** a list) and
`extract_tactics` keeps a node iff:

* `node.name` is a string starting with `Lean.Parser.Tactic.`;
* it is **not** one of the pure sequencing wrappers `tacticSeq`, `tacticSeq1Indented`,
  `tacticSeqBracketed` — those are structure, not tactics;
* its `stx.range` exists, is not `synthetic`, and both ends parse as integers.

Then the list is sorted by `(start_line, start_col, finish_line, finish_col, name)` and **exact
duplicates (same name, same span) are collapsed** — the same syntactic node can be reachable twice
in the elaboration tree, and a tactic sequence must not credit the same node twice (fixture F09).

Recorded per tactic: `i` (AST index), `name`, `start`, `finish`, `pp` (200 chars; `null`/literal
`<failed to pretty print>` for error-recovery nodes), `goals_before_n`, `goals_after_n`,
`error_hits`, `sorry_hits`, `locally_verified = (error_hits == 0 and sorry_hits == 0)`, `blamed`,
`label ∈ {success, d1, d2, null}`, and the token `mapping`.

Synthetic parser-error nodes do not produce tactic spans, so a syntax failure yields
`n_tactics = 0` (fixture F11) — that is a *parse* failure, not a zero-length prefix.

---

## 4. Process statuses (owner §10)

| status | meaning |
|---|---|
| `SUCCESS` | global outcome 1 (whole proof accepted) |
| `PREFIX_BEARING_FAILURE` | failed, first error at `j ≥ 1`, so a non-empty verified prefix exists |
| `FIRST_TACTIC_FAILURE` | failed, first blamed tactic is the first tactic (`j = 0`) |
| `PARSE_OR_SYNTAX_FAILURE` | failed, but no tactic could be blamed (no tactic spans, or no message attributed to a span) |
| `FORMAT_NO_CODE` | the historical `pred` is a sentinel: no Lean code was ever submitted — **no tactic labels of any kind are fabricated** |
| `PROCESS_ORACLE_INFRA` | the oracle could not decide (timeout / server error / transport loss): **censored, never counted as a failure** |

`failure_kind ∈ {error, sorry, error+sorry, no_error_message, null}` is a diagnostic of *which*
message class caused the failure, derived from the message ranges only.

### 4.1 Statuses are mutually exclusive by construction

The status is a pure function of `(extractable, infra, global_outcome, n_tactics, blamed)`; the
ordering above is the implementation (see `_classify_status`). Notably `blamed == 0` maps to
`FIRST_TACTIC_FAILURE` *before* any consideration of message text or `pp` quality: the fixture F11
(malformed `exact (True.intro`) pins this, with `blamed_pp_unprintable = true` recorded as a
diagnostic rather than used as a status rule.

---

## 5. First-error attribution (blame rules)

`blame_index(tactics, ranges)` attributes each message range and returns
`(blamed_index, n_unmapped, blame_kind)`; `blamed_index` is the **earliest** attributed tactic.
The rules, in order:

1. **`contains`** — the innermost tactic node whose span contains the message start. Exact rule for
   tactic-local errors (type errors, unknown identifiers, failed typeclass search).
2. **`range_last`** — when no node contains the start *and* the message carries an end position: the
   last tactic that **starts inside the message range** is blamed. This is required because Lean
   reports block-level failures such as `unsolved goals` with a range covering the failing tactic
   sequence and starting right after `by` — outside every tactic span (calibrated: F04 probe
   `pos=(1,25) endPos=(3,9)`).
3. **unattributed** — the range counts towards `n_unmapped_error_messages` and credits no tactic.

Censoring-safe extras:

* `sorry` ranges are attributed with the same rules; if an error message yields no blame but a
  `sorry` range does, the `sorry` blame stands (`failure_kind = sorry`);
* `n_unmapped_error_messages` / `n_unmapped_sorries` are recorded per candidate so a systematic
  attribution gap is *visible* in the frozen labels rather than hidden inside a fraction;
* no status is ever re-derived from message text, `pp` content, or string matching.

---

## 6. Tactic → generated-token mapping and the credit surface

The chain is: Lean span (frame) → character offset in `pred` → character offset in the generated
`response` → first generated token.

* `pred_offset = prefix_chars + frame_offset(...)`;
* if `pred_offset < len(normalized_formal)` the span lies in the prompt-derived formal head and gets
  status **`outside_response`** — it can carry no token credit (owner §9: no tactic outside the
  generated response);
* otherwise `response_offset = tail_offset + (pred_offset - len(normalized_formal))`, and the span
  text must be **verbatim present** in the response (`span_in_response`), else it is not credit-able;
* `first_token_at(lattice, response_offset)` returns the *unique* token whose character span
  contains the offset, with `exact = (token_start == response_offset)`.

Mapping statuses: `exact`, `contained`, `ambiguous`, `outside_response`, `response_only` (no lattice
available), `unmapped`. **Mappable = `exact` ∪ `contained`.**

> **Why containment and not start-equality.** The Kimina tokenizer is byte-level BPE with leading
> whitespace merged into the first token of an indented line: a tactic starting at column 2 is
> covered by a token that *starts* one or two characters earlier. Demanding `token_start ==
> response_offset` would make E1 unusable for exactly the indented proofs this audit targets. Phase A
> measured 20/20 fixture tactics mapping as `contained`, 0 as `exact`. E1 is therefore defined on
> *mappable* tactics whose span is verbatim in the response — no tokenizer merge can inflate or
> deflate it, and `n_exact` / `n_contained` are reported separately.

**Lattice identity.** `TokenMapper` loads the frozen Kimina tokenizer
(`models/weights/kimina_distill_0_6b`, `Qwen2TokenizerFast`), asserts `vocab_size == 151643` (the 26
added special tokens are counted separately as `len_with_added_tokens = 151669`), tokenizes the raw
generated response with `add_special_tokens = False`, and records `tokenizer.json` /
`tokenizer_config.json` sha256 in every validation artifact
(`aeb13307…dae4`, `5d27a191…dd72`, byte-identical on fly90 and fly122). `roundtrip_ok` is recorded
per lattice.

**Per-token credit surface (owner §16).** Credit is emitted on tactic first tokens as a
`{token_index → token_id, label}` union: `success` positions, `d1` positions, `d2` positions. When
two tactics map to the same token (a composite tactic and its first child), a frozen precedence
resolves the collision — error region (`d2`) > verified prefix (`d1`) > success — and every collision
is counted in `n_credit_conflicts` so it can be reported rather than assumed away.
`blamed_mappable` records whether the blamed tactic itself is credit-able.

**No fabricated credit.** `FORMAT_NO_CODE` candidates get the empty credit record and **no** oracle
call: the historical pipeline never submitted code for them, and no path in this oracle invents a
tactic label for a sentinel (owner §10). They still participate in denominators where the owner
defines them to (see §8).

---

## 7. Structured recoverability, the primary predicate

`structured_recoverable(record)` implements owner §11 literally:

* status is `PREFIX_BEARING_FAILURE` or `FIRST_TACTIC_FAILURE` (a real failure with a blamed tactic);
* `blamed_index ≥ 1` (strictly: the trivial case "every parsed tactic is immediately erroneous" is
  excluded);
* **and** at least one *locally verified* tactic strictly before the blame has a mappable first
  token (d1 credit exists);
* **and** the blamed tactic has a mappable first token (d2 credit exists).

Both credits must land on **mappable first tokens**; a proof whose only verified prefix is
prompt-derived (`outside_response`) or whose blamed tactic is unmappable does not count.

Two secondary diagnostics are frozen alongside it and are reported **only as diagnostics** (owner
§12 forbids silently substituting the broader metric): `any_active` (failed, `blamed_index ≥ 1`) and
`verified_prefix_count` (number of locally verified tactics before the blame).

---

## 8. Evaluation units and denominators (frozen with the preregistration)

* **Primary unit = the primary all-fail group** (one n=8 GRPO rollout of one theorem at one step,
  contamination-excluded). A group is *structured-recoverable* iff **at least one** of its candidates
  satisfies §7. `RecoveryRate = P(structured-recoverable | group label ALL_FAIL)`.
* Candidates that are `FORMAT_NO_CODE` still belong to their group and cannot make it recoverable on
  their own; a group whose only failures are parse/syntax failures cannot either.
* The **candidate-level** rate over code-extractable all-fail candidates is a **pre-declared
  secondary descriptive statistic** (same predicate, candidate denominator), never a replacement for
  the primary.
* `PROCESS_ORACLE_INFRA` candidates are **censored**: they are excluded from both the numerator and
  the denominator of the affected statistic, and the censored count is reported per group/seed.
* G3 quartiles sort primary all-fail groups by `(group_max_tactics, group_key)` and cut at equal
  group counts; a quartile's tactic-count range including ties is reported, and a quartile with
  fewer than `GATE_QUARTILE_MIN_N = 5` groups cannot pass.

Gates, bootstrap and classification taxonomy are frozen in
[`V5-P001_preregistration.md`](V5-P001_preregistration.md) / `experiments/manifests/v5/V5-P001.yaml`.

---

## 9. Oracle infrastructure policy (owner §21, §26)

A **dedicated** instance — the pooled production verifier is never touched:

| parameter | frozen value |
|---|---|
| endpoint | `http://127.0.0.1:8020` (fly122, loopback) |
| container | `tinylean-rl-lean-oracle-v5` |
| image | `projectnumina/kimina-lean-server:2.0.0`, digest pinned in `v5_p001_spec.py` |
| `LEAN_SERVER_MAX_REPLS` | `1` — one REPL, serial requests: the infotree is unambiguous and timing-sensitive tactics cannot contend for CPU |
| server timeout | 120 s (server-side budget, kills first) |
| client timeout | server timeout + 60 s slack |
| batch | 1 candidate per request |
| isolation retries | ≤ 2 bounded retries for *infrastructure* outcomes only |
| canary | 300 s cold / 120 s warm, 1 retry; a failed canary is `fail-close` |
| recovery | ≤ 192 bounded container restarts per Phase-B run, then stop and inspect |

**Semantics.**

* a *server-confirmed* timeout is final (retrying would burn the same budget and falsify the
  denominator), every other infrastructure outcome is retried in isolation up to the bound and stays
  **censored** if it persists;
* after any infrastructure event the canary gate runs; if the *server* is unhealthy the runner
  restarts the container (bounded), re-warms it and re-submits the candidate once; if the canary is
  healthy the event was candidate-specific and the candidate is censored and the run continues;
* an unhealthy server after a recovery aborts the run (`fail-close`) rather than silently censoring
  a block of candidates.

**Measured latency character.** With the REPL warm, a plainer failing proof returns in ~0.1 s while
the heavy historical candidates (polynomial case analysis, `omega`, `ring_nf`) take up to ~30 s of
CPU; the REPL is reused (`MAX_REPL_USES = -1`), so these costs are genuine elaboration time, not
re-imports. Phase B therefore runs for hours and is treated as an overnight batch with
infrastructure-only monitoring (owner §28).

---

## 10. Fixtures and determinism evidence (owner §8, §21)

`FIXTURES` holds **13** engineering fixtures, each with pinned expectations
(`global`, `status`, `n_tactics`, `blamed_index`, exact span texts, plus optional
`pp_unprintable` / `sentinel` assertions) and each derived through the *same* production path
(`extract_proof_from_text` → oracle → `derive_facts`). The set hash is
`fixture_set_sha256 = 2bfb0f4f…be237fe`.

| fixture | class | what it pins |
|---|---|---|
| F01 | fully successful tactic proof | `SUCCESS`, all tactics `success` |
| F02 | error in first tactic | `FIRST_TACTIC_FAILURE`, `j = 0` |
| F03 | valid prefix then type error | `PREFIX_BEARING_FAILURE`, `contains` blame |
| F04 | valid prefix then `unsolved goals` | `PREFIX_BEARING_FAILURE` via **`range_last`** |
| F05 | unknown identifier | blame + d1/d2 split |
| F06 | typeclass failure | failure inside a tactic span |
| F07 | nested `by` block | nested tactic nodes ordered by AST start |
| F08 | `;` combinator | composite `constructor <;> trivial` node **and** its children |
| F09 | `all_goals` case-style block | duplicate-node collapse (`duplicates_removed`) |
| F10 | multi-line tactic | span across lines, finish-exclusive |
| F11 | syntax/parser failure | `FIRST_TACTIC_FAILURE` with `blamed_pp_unprintable` |
| F12 | term-style proof | extraction **sentinel**, `FORMAT_NO_CODE`, no fabricated credit |
| F13 | empty proof body | failure with no tactic to blame |

**Phase-A evidence** (`experiments/manifests/v5/v5_process_oracle_validation.json`, produced on the
dedicated instance): 13/13 fixtures pass their semantic and span-level checks, **all deterministic
across 2 runs** (record-projection hashes identical), token diagnostics 20/20 mapped
(0 exact / 20 contained), 19 credit tokens, 0 credit conflicts.

**Phase-B re-derivation check.** `--stage validate` re-derives every stored record from the rollout
sources plus the archived raw oracle items and requires the record hash to be identical (driver
timing is projected out). The mechanism was exercised end-to-end on fly122 on a scratch cohort
(24 candidates → freeze → validate: 24/24 reproduced, 0 mismatches) and then removed.

---

## 11. Phase-B processing contract

* **Exactly once.** The runner processes the candidates of the primary surface in the frozen
  surface order (seed → step → group → slot). The order is hashed into the run meta; a resume
  re-processes nothing already stored, appends nothing twice, and refuses to continue if the code
  stamp (git HEAD + the five V5 script hashes), the surface hash, or the plan differs.
* **Append-only labels.** One canonical JSON record per candidate, `fsync`-ed per record; a partial
  trailing line from an interrupted run is truncated (and counted) instead of being parsed as data.
* **Raw archive.** One archived response item per code-extractable candidate under
  `runs/v5_p001_process/raw/`, hashed into a manifest at freeze time.
* **Infrastructure-only stdout.** The runner prints progress, submission counts, censoring counts,
  recoveries and ETA. It never computes or prints RecoveryRate, status distributions, or any other
  scientific quantity — those come from the analyzer, exactly once (owner §28–§30, §33).

---

## 12. Frozen artifacts

| artifact | producer | role |
|---|---|---|
| `experiments/manifests/v5/v5_historical_surface.json` | `v5_p001_reconstruct.py` | verified binary reconstruction of the V1 surface (720/686 groups, labels, hashes, per-candidate provenance) |
| `experiments/manifests/v5/v5_process_oracle_validation.json` | `v5_process_oracle.py fixtures` | fixture semantics + determinism on the dedicated instance |
| `runs/v5_p001_process/v5_p001_process_labels.jsonl` | `v5_p001_process_run.py --stage process` | the frozen per-candidate process labels |
| `runs/v5_p001_process/raw/**` + `…_raw_manifest.json` | same | archived oracle items + per-file sha256 |
| `runs/v5_p001_process/v5_p001_process_freeze.json` | `--stage freeze` | label hash, coverage, infrastructure accounting |
| `runs/v5_p001_process/v5_p001_process_validation.json` | `--stage validate` | full re-derivation result |
| `experiments/manifests/v5/V5-P001_results.json` | `v5_p001_analyze.py` (Phase B, once) | gates, RecoveryRate, length robustness, mechanism, sensitivity, classification |
