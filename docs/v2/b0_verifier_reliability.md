# V2 Track B — B0: Verifier Reliability Audit and Verification Policy

- Date: 2026-09-19 (UTC)
- Worker: fly122 (RTX 3080 10GB); canonical base `main2` @ `ca804f1`
  (branch `worker/v2-allocator` = base + this work)
- Status: **B0 COMPLETE** — audit, scratch-server forensics, policy
  implementation and tests are done.
- E024 status unchanged: *PAUSED / ABORTED DUE TO VERIFIER INFRASTRUCTURE
  INCIDENT; no E024 result claim.* Nothing in B0 reruns the E024 benchmark;
  E024 artifacts appear here only as infrastructure forensic input.

## 1. Scope

E024 proved that a **client-side HTTP timeout does not terminate a
pathological server-side Lean computation**. Track B will produce thousands
of verified budget-response rollouts, so the verification path must, per
candidate:

- separate **candidate failure** from **infrastructure failure**;
- prevent a pathological candidate from stalling a dataset;
- never record infrastructure trouble silently as an ordinary model failure.

## 2. What E024 exposed (forensic summary)

From `docs/e024_status.md` and the preserved run log (sha256
`59db3fb7…465e08`, chunk-level numbers as recorded there):

- chunk 7 verification: all 16 sub-batches timed out; single-candidate
  retries also timed out; no evaluator log output for about 60 minutes;
  `/health` exceeded the 5 s cap; a trivial verification raised ReadTimeout;
  server process at ~100 % CPU; one client HTTP connection stayed open.
- remediation was a manual `docker restart`; the whole chunk became 64
  unretrievable `verifier_error` records — **no candidate-level attribution
  survived**.
- static scan of the 64 chunk-7 candidates (no Lean; forensic only): many
  carry `interval_cases` / `native_decide` / `decide` / large-`norm_num`
  proof shapes, i.e. the exact family that can monopolise a REPL
  (`interval_cases` enumeration bomb).

## 3. Server-side mechanics (Kimina Lean Server 2.0.0, read from the image)

Sources: `/root/kimina-lean-server/server/{main,manager,repl,settings}.py`
and `routers/{check,backward,health}.py` inside the pinned image;
`kimina_client` 0.2.1 for the request schema.

- `/verify` is a backward-compatible route into `run_checks`. Unlike
  `/api/check`, it **has no client-disconnect check**: abandoning a request
  never cancels the server-side work.
- the **server-side per-request timeout** is `VerifyRequestBody.timeout`
  (default **300 s**); on expiry the REPL is destroyed (SIGKILL process
  group) and the item returns
  `{"error": "Lean REPL command timed out in N seconds"}`.
- REPL pool: `max_repls` = 16 (cpu_count on this host); per-REPL memory cap
  8 GiB (`LEAN_SERVER_MAX_REPL_MEM=8G`); a request waits at most 60 s for a
  free REPL (`LEAN_SERVER_MAX_WAIT`) and then raises
  `NoAvailableReplError` → HTTP 429.
- REPLs are recycled by header match; all Track B candidates share the
  `import Mathlib` header, so they share the pool.
- snippets inside one request run concurrently (`asyncio.gather`); a REPL
  crash (empty stdout → `JSON decode error`) fails the **whole request**
  with HTTP 500.
- `"has already been declared"` (redeclaration) can leak across candidates
  on a reused REPL; V1 already treated it as a retry reason
  (`scripts/e024_minif2f_eval.py`).

## 4. Scratch-server forensics (2026-09-19, fly122)

Setup: independent container `tinylean-rl-lean-scratch` (same pinned image)
on port 8001, memory capped at 30 GiB. The formal server on port 8000 was
never touched.

| # | stimulus | observed |
|---|---|---|
| E1 | trivial proof + explicit `timeout` field | accepted; verified in 4.2 s |
| E2 | `interval_cases` bomb (500 k cases), server `timeout=60` | item timed out in exactly 60.0 s; canary verified in 4.3 s; REPL memory released |
| E3 | kernel reduction (`decide` on 2^61−1, maxRecDepth 1e6) | REPL crashed → empty stdout → HTTP 500 `JSON decode error`; server destroyed that REPL and stayed healthy (canary 4.2 s) |
| E4 | 16 concurrent interval bombs (server `timeout=30`) + 1 extra canary | all 16 returned `timed out in 30.0 seconds` within 30–40 s; the extra canary queued ~34 s behind the saturated pool and then verified; no wedge, no 429 |
| E5 | bomb with client timeout 5 s ≪ server timeout 60 s | client gone at 5 s; the server-side task kept running and was cleaned up at 60 s (zombie window = server timeout); canary unaffected |

Conclusions (load-bearing for the policy):

1. an explicit server-side timeout strictly below the client-side timeout
   bounds every pathological candidate at the server budget — demonstrated
   with all 16 REPL slots simultaneously occupied, no wedge;
2. a canary inside a saturated pool is unreliable (it queues) — canary
   probes must run with no verification in flight;
3. client abandonment leaves a server-side zombie for up to the server
   timeout — another reason to keep the server budget small;
4. REPL crashes surface as HTTP 500 (`JSON decode error`) and are
   infrastructure, not candidate failure;
5. E024's wedge combination is now explicable and policy-forbidden:
   client timeout 120 s < server default 300 s (zombies), a retry storm
   over a saturated pool, and single-candidate retries that each held
   REPLs for minutes.

## 5. Failure taxonomy (frozen for Track B)

| outcome | meaning | dataset label |
| --- | --- | --- |
| `verified` | no error-severity message and no `sorry` | solved = 1 |
| `lean_error` | genuine proof failure (incl. parse/type errors) | unsolved = 0 |
| `sorry` | proof contains `sorry` | unsolved = 0 |
| `verifier_timeout` | server-confirmed per-request timeout | unsolved = 0, flagged |
| `verifier_server_error` | 5xx / 429 / REPL crash / persisted redeclaration | excluded (no label) |
| `verifier_unhealthy` | canary probe failed | stop (fail-close) |
| `unresolved_infra_error` | response missing / unknown shape / unresolved | excluded (no label) |

`verifier_timeout` is a **candidate-level** outcome (the candidate consumed
its compute budget without a verdict) and is flagged so that B3 analyses can
run sensitivity checks; the five infrastructure outcomes may never be
written as model failures.

## 6. Policy implementation (this branch)

- `src/tinylean_rl/verifier/kimina.py` — `verify_code` / `verify_codes` gain
  an explicit `server_timeout` field (omitted by default, keeping V1
  behaviour byte-compatible).
- `src/tinylean_rl/verifier/policy.py` — frozen taxonomy +
  `VerificationSession`: batched pass → canary gate → per-candidate
  isolation with bounded retries; redeclaration retried and relabelled
  `verifier_server_error` when it persists; all events are recorded in
  `session.events` for artifact provenance.
- tests: `tests/test_verifier_policy.py` (taxonomy, degradation, fail-close,
  redeclaration) and two payload tests in `tests/test_verifier.py`; all pass
  under `ruff` clean.
- live smoke against the scratch server (4 candidates: valid / lean_error /
  bomb / sorry; `server_timeout=15`, `batch_size=2`): outcomes
  `verified / lean_error / verifier_timeout / sorry`, canary gate fired once
  after the timeout and passed; total 19.4 s.

## 7. Operational rules (binding for B1/B2 data construction)

1. Always send the explicit server-side timeout; it must be **strictly
   smaller** than the client timeout (`VerificationSession` derives
   `client_timeout = server_timeout + slack`).
2. Canary probes run **only with no verification in flight** (between
   passes); a canary inside a saturated pool queues and would be misread.
3. On canary failure: stop, restart the Lean server, re-run the prewarm
   ladder, resume — never continue writing labels from an unhealthy server.
4. In-flight candidates x batch must stay below the 16-slot REPL pool
   (Track B ships serial verification by default; any parallel variant must
   reserve at least one free slot).
5. `verifier_timeout` candidates keep label 0 **with a flag**;
   infrastructure outcomes are never labelled — they are re-run or
   quarantined with their `session.events` trail.
6. No unbounded retries anywhere in the verification path.

## 8. Residual risks and open items

- A candidate crashing the **whole container** (not a single REPL) would
  take the verifier down; the canary converts that into a detectable stop
  (fail-close) instead of silent data corruption.
- Batched requests have a 500-blast radius (one REPL crash fails the whole
  request); the isolation pass re-verifies affected candidates, and the
  canary re-gates afterwards.
- The scratch server (`tinylean-rl-lean-scratch`, port 8001) stays
  available for Track B policy experiments; only the formal server
  (port 8000) is used for B1/B2 data.
- The canary proof verifies in ~0.2-4.5 s on a warm pool; the probe budget
  (60 s server-side) tolerates a cold pool (first use of a fresh REPL
  imports Mathlib).
