# V3-R001 infrastructure amendment C′ — bounded verifier recovery

- Date: 2026-09-25 (UTC) · Branch `v3-jev-rl-controller` · Host: fly122 (formal compute)
- Authority: owner directive of 2026-09-25 (sections 1–16), received after attempt-1 aborted three
  times on the shared Lean server. This document is the §10 artifact: attempt-1 status, root cause,
  scientific settings changed, infrastructure changes, unchanged surfaces, old/new configuration.
- **Scientific settings changed by this amendment: NONE.** The frozen sample, block, seeds, model,
  generation parameters, reward rule, censoring rule, gate and analysis are untouched; the single
  entry of `FROZEN_SETTINGS` that moved is verification *concurrency* (`batch_size` 4 → 1), which is
  an infrastructure execution choice recorded in §4, exactly as amendment C recorded the verifier
  execution path before it.

---

## 1. Attempt-1 record (sealed; provenance only)

`V3-R001 attempt-1: INFRASTRUCTURE_ABORT / NO_SCIENTIFIC_OUTCOME / NOT_ANALYZED` (owner instruction).

| abort | run_id | candidates generated | groups on disk | cause (verbatim from the artifact) |
|---|---|---|---|---|
| 1 | `20260925T012142Z` | 384 (ranks 1–48) | 48 | `VerifierUnhealthyError: canary probe failed; verification must stop (fail-close).` |
| 2 (resume) | `20260925T023307Z` | 0 | — | same |
| 3 (resume) | `20260925T0311xxZ` | 0 | — | same |

Preserved in `runs/v3_r001_rollout_archive_from_fly122/` (outside git): `v3_r001_raw_rollout.jsonl`
(384 rows, sha256 `4cdbab3f62f978aa1495fc99f1e5890a7071274908935e3fde84d889e2ccc405`),
`v3_r001_run_summary.aborted.chunk1-3.json`, `v3_r001_run_summary.aborted.attempt2.json`, the three
console logs and `server_lean_attempts.filtered.log` (the verifier-window server log used in §2.3).

Rules attached to this record (owner §11–§12), all mechanically enforced by the runner:

- ranks 1–48 are **never combined** with a later execution; the artifact above is immutable
  provenance and no label in it was inspected (§12). The runner refuses `runs/v3_r001/rollout` and
  any subpath of it (`S.SUPERSEDED_ATTEMPT1_DIR`), so attempt-2 cannot read, resume or append there;
- attempt-2 supersedes it entirely: fresh directory (`runs/v3_r001/rollout_attempt2`, `ATTEMPT = 2`
  stamped into every summary), from `formal_sample_rank` 1 with the frozen seeds, model and
  scientific settings unchanged (§11);
- C′ is motivated exclusively by verifier health/lifecycle evidence — the timings and message texts
  in §2, never an outcome.

## 2. The audit (owner §2): is the server-side timeout sent, and what does the server do with it?

### 2.1 What the client sends — the six required fields

| field | value | evidence |
|---|---|---|
| client timeout | `server_timeout + client_slack` = **120 + 60 = 180 s** per HTTP request | `policy.py` `VerificationSession.client_timeout` (property); passed as `timeout=` in `_batch_pass` (`policy.py:357`), `_single_pass` (`policy.py:401`), `canary_probe` (`policy.py:283`) |
| server timeout | `int(server_timeout)` = **120 s** | passed as `server_timeout=` in the same three call sites (`policy.py:358`, `402`, `284`) |
| retry count | batch pass: 1 attempt per sub-batch. Unattributed candidates: **1 + `max_single_retries` = 3** isolation attempts. A *server-confirmed* timeout is **not** retried — it is final for that candidate (`policy.py:379–382`) | `policy.py` `_single_pass` loop; `max_single_retries = 2` in `FROZEN_SETTINGS` |
| retry delay | **2.0 s** between isolation attempts (`retry_backoff_seconds`); canary backoff 5.0 s | `policy.py` `_single_pass` (`time.sleep(self.retry_backoff_seconds)`), `canary_probe` |
| request payload field | `payload["timeout"] = int(server_timeout)` — present in **every** request (single and batch) | `kimina.py:100–101` (single), `kimina.py:147–148` (batch) |
| server interpretation | `VerifyRequestBody.timeout` (server default **300 s**) → `float(timeout)` per snippet; used by `manager.prep` for the header import (`check_router.py:49`) and by `repl.send_timeout = asyncio.wait_for(send, timeout)` whose expiry logs `Lean REPL command timed out in N seconds` (`repl.py:222–231`) and destroys the REPL (`destroy_repl`, `repl.py:364`; `check_router.py:152`). `/verify` is `backward_router` → `run_checks` and **has no client-disconnect handling** (`backward_router.py:29–37`), unlike `/api/check` (`check_router.py:187–191`, which cancels and destroys on disconnect) | image source read in `runs/v3_r001_audit/kimina_server_2.0.0/` |

The frozen client therefore *always* transmits an explicit 120 s server-side budget — the `if
server_timeout is not None` guard in `kimina.py` never skips it on this path.

### 2.2 Logging evidence from the attempt-1 window

Source: `/tmp/r001_window_0925.log` on fly122 — the raw Docker log of the shared server over the
attempt-1 window `2026-09-25T01:03:56 → 03:22:13` (172,895,219 bytes); a filtered extract is in the
archive (`server_lean_attempts.filtered.log`).

| fact | value |
|---|---|
| HTTP 429 (`NoAvailableReplError`) | **0** — the pool was never exhausted |
| snippet runs (`Running snippet`) | 394 |
| distinct REPL process ids seen | 9 (pool size 16) |
| server-side timeout lines (`timed out in`) | **2 — one event**: `01:42:43.931 [442176bd] Lean REPL command timed out in 120.0 seconds` (`repl.py:227`) + its response item, followed at `01:42:43.942` by `Closing REPL 442176bd` (`repl.py:364`) |
| `Closing REPL` lines, total | 1 |

The single enforcement line carries the value **120.0** — the client's frozen budget, not the
server's 300 s default. A value that was never transmitted cannot appear in that message, so
**Case A (the timeout was not sent) is refuted**.

### 2.3 The decisive episode — Case B, sharpened

The same window, snippet `50-1` (a formal candidate id, quoted here only as a lifecycle event):

| attempt | REPL | started | response | elapsed |
|---|---|---|---|---|
| batch pass | `3062bfa7` | `02:03:47.681` | `02:12:16.801` | **509.1 s** |
| isolation retry 1 | `f6e86dbb` | `02:39:29.871` | `02:39:48.304` | 18.4 s |
| isolation retry 2 | `82ebf21a` | `03:04:09.839` | `03:04:09.995` | 0.16 s |

For 509.1 s the snippet *ran* on a reusable REPL with no `timed out` line and no `Closing REPL
3062bfa7` — the transmitted 120 s budget was not observable as a bound in that episode, and the
client had already abandoned the request at 180 s (120 + 60), leaving a server-side computation
alive on the shared pool. The two retries then behaved normally (18.4 s, then 0.16 s on a fresh
REPL), i.e. the pathology is transient REPL/loop state, not the candidate text.

Both abort summaries recorded a **green** health probe at launch (`/health` 200, canary verified in
0.11 s / 0.29 s), so every abort happened mid-run, not at preflight.

### 2.4 Determination

**Case B** (the timeout is sent; the server-side lifecycle nevertheless left a computation occupying
a reusable REPL after the client had classified the attempt). The owner's framing "sent but active
REPL lived >60 s" is confirmed and *sharpened*: not only can a computation outlive the client's
180 s bound, it can outlive the server's own transmitted budget in the states that matter. Two
consequences drive the whole amendment:

1. the frozen B0 policy behaved correctly — it detected the state (canary gate) and failed closed
   rather than writing mislabelled rows; the three aborts are the policy working;
2. the policy had **no repair path** — the only remedy in the frozen contract was "stop, restart,
   resume", and repeating the resume is exactly what owner §1 prohibits. The missing half is not a
   longer timeout (§1, §4) but a client-side capacity guarantee (§3, §5, §6).

## 3. C′ — the amendment

### 3.1 The invariant (owner §3)

> No candidate verification attempt may leave a Lean computation occupying a reusable REPL after
> that attempt is classified timed out/failed; capacity is restored before the next candidate.

### 3.2 The three mechanical changes

**(a) A dedicated verifier instance** — `infra/lean-server/r001/compose.yaml`, container
`tinylean-rl-lean-server-r001`, image pinned `projectnumina/kimina-lean-server:2.0.0`
(`sha256:588a2cbb…3ffd9`, `pull_policy: never`), published only on `127.0.0.1:8010`,
`LEAN_SERVER_MAX_REPLS=1`, `LEAN_SERVER_MAX_REPL_USES=-1` (no TTL recycling — a re-import would eat
into the frozen 120 s candidate budget), `LEAN_SERVER_MAX_REPL_MEM=8G`, `LEAN_SERVER_MAX_WAIT=60`,
16 GiB container cap with swap disabled, subnet `10.202.0.0/24`. A singleton pool turns "restore
capacity" into a bounded, known operation on a known container.

The runner refuses any other endpoint: `VerifierRecovery.verify_identity()` fails closed unless the
container exists, is running, carries the pinned image, has `LEAN_SERVER_MAX_REPLS=1`, and publishes
the endpoint's host:port. On dry-run this check is report-only (no Docker required to rehearse);
on a real run it is a preflight abort.

**(b) Per-candidate sequencing — verification concurrency 1.** `FROZEN_SETTINGS["VERIFIER"]
["batch_size"]`: 4 → 1. With a one-REPL pool, batching candidates into one request would race the
singleton and manufacture 429s; one candidate per request also means a canary can always be gated
against the *same* request path, and a mid-group fail-close no longer discards the verdicts of the
candidates already classified in the group. This is the only `FROZEN_SETTINGS` entry C′ changes
(§4).

**(c) Mandatory recovery between candidates.** `verify_candidate_with_recovery()` (the runner's only
verification path):

```
classified = session.verify([proof], [custom_id])[0]      # frozen B0 policy, batch_size = 1
if classified.outcome.is_conclusive:  next candidate      # verified / lean_error / sorry
else:                                                     # timeout / 5xx / unhealthy / unresolved
    restart the dedicated container  →  poll /health  →  nonformal cold canary
    then, and only then, the next candidate
```

- the canary after a restart runs with the **cold-start budget** (`first_timeout_s` = 600 s, frozen
  before attempt-1), because the first request against a fresh container pays the header import;
- a `VerifierUnhealthyError` no longer ends the run: the candidate is classified
  `unresolved_infra_error` (an infrastructure outcome, never a score, never a candidate failure) and
  the recovery sequence still runs; the group's label becomes `INFRA_CENSORED` by the frozen rule;
- recovery is bounded by `max_recoveries_per_run = 16` and **fail-closed**: if the identity check,
  the restart, `/health` or the canary cannot be completed, the run stops with an aborted summary,
  keeps every finished whole group, and stays resumable. A restart is **infrastructure recovery,
  not a candidate retry** (owner §6): it never adds an attempt to the candidate's frozen budget
  (1 batch + 2 isolation attempts, unchanged).

### 3.3 Rejected alternative (recorded, so the choice is visible)

Switching the verification call to `/api/check` would give server-side disconnect cleanup for free
(`check_router.py:187–191`). It is **not** used: it changes the frozen decode/taxonomy path
(different response schema, different error surface) and would be a verifier-semantics change, not
an infrastructure restoration. C′ keeps the frozen `/verify` path byte-for-byte and repairs capacity
from the client side.

### 3.4 Provenance the run writes

| artifact | content |
|---|---|
| `runs/v3_r001/rollout_attempt2/v3_r001_raw_rollout.jsonl` | candidates (unchanged schema) |
| `…/v3_r001_run_summary.json` | attempt number, `verifier_infrastructure` block (endpoint, container, image id, max_repls, concurrency, policy batch size, dedicated-instance description, recovery budget/attempted/succeeded, recovery events, verifier-event path) |
| `…/v3_r001_recovery_log.jsonl` | one durable event per recovery: trigger (rank, sample, outcome, message), restart seconds, health, canary status/latency, total seconds, ok |
| `…/v3_r001_verifier_events.jsonl` | the frozen policy's own event trail, flushed incrementally |
| `…/v3_r001_run_summary.aborted.json` | fail-close record with the same blocks plus the abort reason |

## 4. Scientific settings: the complete diff

`FROZEN_SETTINGS` sha256 **before** `978566da25f3f651aed3b8b537f8b4ebf94ecf0c0870c5cd4ebf2371e6052b3b`
→ **after** `bf069eccbc0af53768890340e77e03a46a865d05fb8bdcd7ceb2560bfe71e264`. The diff is exactly
one entry plus its note:

| entry | before | after | why |
|---|---|---|---|
| `VERIFIER.batch_size` | 4 | **1** | verification concurrency 1 on the singleton pool (§3.2b) |

Everything else in the block is byte-identical, including `server_timeout_s = 120`, `first_timeout_s
= 600`, `client_slack_s = 60`, `canary_timeout_s = 60`, `canary_retries = 1`,
`max_single_retries = 2`, both backoffs, and the recorded verification policy string. **No timeout
was lengthened and no retry count was raised** (owner §4); the machine layout, decoding parameters,
temperature, seeds, and every gate are unchanged. `VERIFIER_INFRA` is a new, purely
infrastructure-side block (endpoint, container, image, pool sizes, recovery budgets).

Unchanged and hash-pinned (all re-checked by the runner's preflight against the committed files):
sample content `0ac7134f…`, formal sample file `733a699e…`, top-20 % block `d5791fd5…`, gate
artifact `fcb700b5…`, sealed reserve `b60be381…` (SEALED, untouched), theta0
`34e6e630…`/revision `332e8a52…`, seed formula and `seed_base = 20260924`, generation
(temperature 1.0, top_p 1.0, 4096 response tokens, model len 5120, gpu util 0.85, chunk 16), label
rule `y = 1 iff 0 < n_pos < 8`, infra→`INFRA_CENSORED` censoring, both gates, and the analyzer
(D001-identical).

Limitation accepted with the change (owner §5): R001's verifier wall-clock is not comparable to
E023's or any concurrent-verifier wall-clock as an efficiency claim; R001's question is prospective
informativeness, not verifier throughput.

## 5. Owner §7 review — should a permanent INFRA_CENSORED short-circuit the rest of its group?

**Review only; nothing implemented.** A group's status is decided by the frozen rules (a candidate
whose verification ends in a timeout after the frozen retries is `verifier_timeout`, and a group with
any non-conclusive member is `INFRA_CENSORED`, `y = None`). Those censored units are excluded from
both gates; their remaining candidates' verdicts would be *unused*. Hence:

- the raw data contract does **not** require the remaining candidates of an already-censored group
  to be verified — skipping them would change no endpoint, no gate, no reported number, and would
  not touch any label rule. It is a **compute-only optimization**;
- it is **not authorized** and **not implemented**. The implemented behaviour is the frozen one:
  every remaining candidate of a penalised group is still attempted, with recovery in between;
- if the owner wants it, it needs its own pre-outcome amendment with a stated decision rule (e.g.
  "once a group is INFRA_CENSORED after the frozen retries, skip its remaining unverified
  candidates") and a mechanical guard, because it changes *which events exist*, not merely their
  cost — an audit-relevant difference, which is why it is proposed rather than assumed.

## 6. Section 8/9 validation (nonformal)

`scripts/v3_r001_stress_recovery.py` drives the production C′ code paths (`make_session()`,
`VerifierRecovery`, `verify_candidate_with_recovery()`) on the dedicated instance with **nonformal
inputs only**: the pathological input is the `interval_cases` enumeration bomb — the class E024's
chunk-7 candidates were found to carry (`docs/v2/b0_verifier_reliability.md` §2–§4, where the same
bomb was measured to time out at exactly the server budget) — and the normal inputs are the frozen
policy's own canary proof plus one second trivial proof. No V3 theorem statement, no reserve text, no
attempt-1 candidate and no model generation is involved.

The suite runs three server-confirmed timeout cycles (pathological → the server itself confirms the
timeout at the frozen budget → recovery → canary), three client-abandonment cycles reproducing
attempt-1's Case B (client abandons at 2 s while the server keeps elaborating → recovery repairs
capacity), and the §9 mixed sequence normal/pathological/normal/pathological/normal.

**Results** (full record: `runs/v3_r001/stress_cprime/` on fly122; compact artifact committed as
`experiments/manifests/v3/V3-R001_Cprime_validation.json`):

| criterion | result |
|---|---|
| T1 bounded return | **PASS** — every pathological candidate returned bounded: server-confirmed cycles `131.80 / 131.77 / 131.92 s` (frozen 120 s server budget + client slack), client-abandonment cycles `2.02 / 2.02 / 2.02 s` (their 2 s client timeout), mixed-sequence bombs `131.81 / 131.82 s`. Max 131.92 s vs the frozen `client_timeout 180 s` — the bound holds. |
| T2 no REPL leak | **PASS** — `docker top` at every checkpoint of all 6 cycles: exactly one `repl` + one `lake` process, never growing (`before / after_candidate / after_recovery` identical in all 6 cycles); post-restart the singleton is rebuilt, never leaked. |
| T3 post-timeout canary | **PASS** — all 6 post-timeout recoveries ended with a nonformal canary `verified` (`4.23 / 4.17 / 4.24 / 4.15 / 4.21 / 4.18 s`); no recovery accepted capacity without it. |
| T4 stability (3/3 + 3/3 + mixed) | **PASS** — 3/3 server-confirmed cycles, 3/3 client-abandonment cycles, mixed sequence 5/5 steps ok (normal `verified`, no recovery attempted; both bombs `verifier_timeout` → recovery → next step). 8/8 recoveries ok within the budget of 16. |
| T5 no frozen scientific setting changed | **PASS** — asserted from the live session: `server_timeout 120.0 s`, `client_timeout 180.0 s`, `batch_size 1`, `max_single_retries 2`, endpoint the dedicated `127.0.0.1:8010`; `FROZEN_SETTINGS` sha256 `bf069eccbc0af53768890340e77e03a46a865d05fb8bdcd7ceb2560bfe71e264` (= `V3-R001_Cprime.yaml#hashes.frozen_settings_after`). No timeout lengthened, no model/sample/seed/gate touched. |

Measured recovery cost (owner §13's "recovery wall-clock", artifact `#result.events`):

| trigger | `docker restart` | `/health` | canary | recovery total |
|---|---|---|---|---|
| server-confirmed timeout (REPL already destroyed server-side) | 1.13–1.21 s | ~2.07 s | ~4.2 s | 7.37–7.49 s |
| client abandonment (Lean computation still alive — attempt-1's Case B state) | 10.92–10.93 s | ~2.08 s | ~4.2 s | 17.15–17.23 s |

The restart-time differential is the direct positive evidence that the Case B state is real and that
recovery removes it: with the REPL idle the container stops in ~1.2 s, while with an abandoned Lean
computation still running the full `-t 10` grace expires before SIGKILL (10.9 s), exactly the
zombie the C′ invariant targets. Both paths end healthy and canary-verified, so the worst-case
overhead a pathological candidate adds to attempt-2 is ~132 s candidate + ~17 s recovery.

Compact artifact: `experiments/manifests/v3/V3-R001_Cprime_validation.json`, sha256
`10d6f44cb87c7918c99275d68ac7c13b36d1e7b963e007d6a3e196d4afc87e8e` (written by the harness itself,
never edited; full record incl. raw stdout `runs/v3_r001/stress_cprime/` on fly122).

## 7. Attempt-2 plan (executes only on the owner's separate launch authorization)

```
.venv/bin/python scripts/v3_r001_rollout.py --i-have-owner-launch-authorization
```

- fresh run directory `runs/v3_r001/rollout_attempt2` (`attempt: 2`); the attempt-1 directory is
  refused by construction; no row, label or verdict of attempt-1 is read, and its 48 complete groups
  are not used for any design decision (§12);
- all 128 groups from `formal_sample_rank` 1 with the frozen seeds
  `20260924 + (formal_sample_rank−1)·8 + sample_index` and the frozen model/settings of §4;
- generation is still one model call per candidate (unchanged); verification now runs one candidate
  at a time against the dedicated container with the frozen policy and the mandatory recovery;
- `_print_dry_run_summary` prints the attempt number and the whole infra block before anything runs.

## 8. Configuration hashes (old → new)

| item | attempt-1 (old) | C′ (new) |
|---|---|---|
| verifier endpoint | `http://127.0.0.1:8000` (shared, `0.0.0.0`) | `http://127.0.0.1:8010` (dedicated, loopback only) |
| compose file | `infra/lean-server/compose.yaml`, sha256 `deca3d0ed1fe4cd9fd772b711e016159fb2e6734ca9ecc2927f1790b37a93497` | `infra/lean-server/r001/compose.yaml`, sha256 `facbed7b8758aa95a5048daed4001cbb16a5f8afb81541816d5530179a4b9220` |
| container | `tinylean-rl-lean-server` (shared with everything else) | `tinylean-rl-lean-server-r001` |
| image | `projectnumina/kimina-lean-server:2.0.0` `sha256:588a2cbb…3ffd9` | same, pinned, `pull_policy: never` |
| `LEAN_SERVER_MAX_REPLS` | unset → 16 | `1` |
| `LEAN_SERVER_MAX_REPL_USES` | unset → 100 | `-1` (never) |
| recovery | none (operator-only manual restart) | mandatory, bounded (16), fail-closed, logged |
| policy batch size | 4 | 1 |
| `FROZEN_SETTINGS` sha256 | `978566da…52b3b` | `bf069ecc…71e264` |
| `scripts/v3_r001_spec.py` sha256 | `27d5b362dba1348c…` (commit 1cb9e89) | `51b3bcdb5023a48f…` |
| `scripts/v3_r001_rollout.py` sha256 | `768ca0142d1a83f4…` (commit 1cb9e89) | `a137f420452a58e3…` |
| `scripts/v3_r001_stress_recovery.py` | — | `4a59ebc8014d645f…` |

## 9. Where each fact lives

- preregistration §17 checklist gains the C′ item (owner §10): see `V3-R001_preregistration.md` §19;
- registry amendment: `experiments/manifests/v3/registry.yaml`
  (`V3-R001-infrastructure-amendment-Cprime`, append-only);
- machine-readable amendment: `experiments/manifests/v3/V3-R001_Cprime.yaml`;
- validation artifact: `experiments/manifests/v3/V3-R001_Cprime_validation.json`;
- dedicated-instance runbook: `infra/lean-server/r001/README.md`.
