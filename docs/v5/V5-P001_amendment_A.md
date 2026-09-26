# V5-P001 amendment A — two-strike wedge censoring (deterministic ultra-long elaboration)

- Date: 2026-09-26 (UTC) · Branch `v5-process-verified-rl` · Host: fly90 (code + tests) → fly122
  (archive of attempt 0, fresh preflight, relaunch)
- Authority: owner approval 2026-09-26 (“批准 A”), received after the Phase-B attempt-0 fail-close
  stop described in §2. This document is the amendment artifact: trigger, attempt-0 record, root
  cause (with the container-side evidence), the minimal execution-code change, the explicit list of
  unchanged scientific elements, and the pinned old/new hashes.
- **Scientific design changed by this amendment: NONE.** It is a pre-outcome *execution-code*
  amendment: only the abort path of one infrastructure handler was touched. No scientific outcome of
  V5-P001 exists (no `RecoveryRate`, no seed-level rate, no quartile result was computed, printed or
  inspected); the amendment precedes the first analyzable label set.

---

## 1. Scope and summary

The frozen §26 rule “an unhealthy instance after a recovery stops the run (`fail-close`)” is
predicated on such a state being *unexpected*. Attempt 0 produced a **deterministic, candidate-local**
counterexample: a proof whose elaboration (~12 min) outlives every bounded window of the frozen
protocol (client 180 s; canary probe 2 × 180 s), so on every resume the same candidate wedges the
serialized instance, triggers a recovery, wedges it again and fail-closes the run — at exactly the
same plan position, forever.

Amendment A keeps the fail-close philosophy intact and adds the missing terminal branch: after the
bounded post-recovery re-submission of a candidate *again* returns an infrastructure outcome and the
canary is *again* unhealthy, the **candidate** is censored (`PROCESS_ORACLE_INFRA`, owner §10: infra
is missing data, never a failed proof — and never a reason to abandon the remaining 4900+
candidates), the instance is restored by one more bounded recovery, and the run continues with the
next candidate. The recovery budget (≤ 192 per run) is unchanged and shared; a restore whose cold
canary fails is still a hard stop.

---

## 2. Attempt-0 record (no scientific outcome)

Classification: **`V5-P001 Phase-B attempt 0 / FAIL-CLOSE LOCK / NO SCIENTIFIC OUTCOME`**.

| fact | value |
|---|---|
| run directory | `runs/v5_p001_process/` on fly122 (now archived, §6) |
| labels written | 570 / 5488 (241 code + 329 sentinel) |
| verifier submissions | 241 (all confirmed by the archived raw items) |
| infrastructure-censored labels | **4** `PROCESS_ORACLE_INFRA` (3× `VERIFIER_TIMEOUT`, 1× `VERIFIER_SERVER_ERROR` after the bounded isolation retries; 0 transports, 0 unresolved) |
| infra events | 6 (5 candidate events before the recovery + 1 `infra_event_after_recovery`) |
| container restarts | 1 (the recovery of the wedge event) |
| scientific metrics emitted | **0** (no `RecoveryRate`, seed-wise or quartile quantity was printed or computed) |
| analyzer runs | **0** |
| poison candidate | `seed1:0019:g02:c02`, plan position **571** — absent from labels and from the raw archive, i.e. it has **no** recorded outcome of any kind |

## 3. Root cause and evidence (container-side log, UTC)

The dedicated container’s own request log shows the full cycle (all times UTC, container clock):

* **05:18:14.296** — request `v5p001-seed1:0019:g02:c02` received; the REPL begins running it.
* **05:21:14** — the client-side window (180 s) expires with no response; the server-confirmed
  timeout is final (frozen rule), the runner’s canary probe (1 retry ⇒ 2 × 180 s bound) is issued
  and queues behind the still-running proof on the single REPL (`MAX_REPLS=1`).
* **05:27:19–05:27:30** — the runner’s bounded recovery restarts the container (logged
  `recovery_start` 05:27:19, container `StartedAt` 05:27:30Z); the in-flight proof had been running
  **≈ 9 m 16 s** and every bounded window was long exhausted.
* **05:27:47.016** — the frozen one-time post-recovery re-submission is received (fresh REPL).
* **05:30:47** — the retry’s own 180 s client window expires (`infra_event_after_recovery`, logged
  by the runner); the canary probe follows.
* **05:36:52** — the run dies with `RuntimeError: oracle unhealthy again after a recovery at
  seed1:0019:g02:c02; stop and inspect`, i.e. the fail-close rule fired during the canary probe.
* **05:40:04.277** — the container finally answers the retry: **737 s (12 m 17 s)** after receipt,
  with *ordinary tactic-error* content (`tactic 'simp' failed, nested error: maximum recursion depth
  has been reached`) and an internal command time of 68.26 s. The two queued canary probes are
  answered at 05:40:04.75/.81 — the instance is healthy and idle again immediately.

Inference: the wedge is a property of **the candidate**, not of the instance. Its serialized service
time is several times the frozen bounded windows (180 s client; 2 × 180 s canary), so any resume
re-attempts position 571, wedges the instance, triggers a recovery, wedges it again and fail-closes
the run — zero label progress, two restarts per attempt. The instance itself was verified healthy
and idle afterwards (`StartedAt = 05:27:30Z` matches the logged recovery exactly).

## 4. The amended rule (only change)

Frozen text (Phase A): “if the instance is unhealthy again after a recovery the run stops and the
owner is informed.”

Amended branch in `ProcessStage.repeat_after_recovery` (`scripts/v5_p001_process_run.py`):

```python
try:
    self.client.require_healthy()
except VerifierUnhealthyError:
    # Amendment A: a proof whose elaboration outlives both the bounded
    # client window and the bounded canary window wedges the serialized
    # instance (observed: 737 s elaboration vs 180 s + 2 x 180 s budgets).
    # Owner section 26: an infrastructure outcome is censored, never a
    # failed proof and never a reason to abandon the remaining surface.
    self.infra_log.event(
        "candidate_censored_two_strikes",
        {
            "candidate_id": candidate_id,
            "outcome": result["classified"].outcome.value,
            "message": result["classified"].message[:300],
        },
    )
    self.recover(f"instance wedged again by {candidate_id}; censoring candidate")
return result
```

Semantics, in order:

1. Original submission returns an infrastructure outcome → `infra_event` → canary.
2. Canary healthy ⇒ candidate-specific infra ⇒ **censored** and continue (unchanged, frozen §26).
3. Canary unhealthy ⇒ one bounded recovery (restart + cold canary, unchanged), then the **one**
   re-submission.
4. Re-submission conclusive ⇒ the candidate’s real outcome (amended attempt 0 never reached this).
5. Re-submission infra ⇒ `infra_event_after_recovery` → canary.
6. Canary healthy ⇒ **censored** and continue (unchanged).
7. **Canary unhealthy again ⇒ `candidate_censored_two_strikes` is logged; the infrastructure result
   is recorded for the candidate (`PROCESS_ORACLE_INFRA`, `oracle.infra = true`); one more bounded
   recovery restores the instance; the run continues with the next candidate. (This branch is the
   amendment.)**
8. The restore recovery is fail-close like every other: recovery-budget exhaustion, a failed
   `docker restart`, a `/health` poll exceeding its budget or a failed cold canary all stop the run
   and report the owner.

Bookkeeping: a two-strike candidate is counted exactly like any other censored candidate — it is
excluded from numerators, kept in its group’s denominator via the frozen censoring trichotomy, and
its raw item is archived, so `stage_validate` re-derives it normally. The new event kind
`candidate_censored_two_strikes` is deliberately not named `infra_event*`, so it does not inflate
`n_infra_events`; the label-side `oracle.infra` accounting is unchanged.

## 5. What is explicitly unchanged

Unchanged and byte-identical where applicable: d1 = −0.05 / d2 = −0.10 credit and the credit
surface; statuses, blame attribution, `PROCESS_STRUCTURED_RECOVERABLE`; parser and tactic
definitions (Lean infotree, never text splitting); the censoring trichotomy; gates E1/E2/G1/G2/G3
and the classification order; denominators and units; surfaces, plan and plan order; fixtures and
determinism rules; oracle endpoint, container, image digest, timeouts, canary budgets and the
≤ 192-recovery budget; the “each candidate exactly once” rule (the re-submission is the pre-existing
frozen §26 recovery path, not a resampling); Phase-A/Phase-B discipline; the prohibition on live
scientific metrics and on early stopping; the canonical analyzer’s exactly-once rule; compute
(`NEW_MODEL_GENERATION: 0`, `TRAINING: 0`), and the sealed reserves.

## 6. Consequences and execution

* Any code edit changes the code stamp, so the attempt-0 run directory cannot (and must not) be
  continued: `runs/v5_p001_process/` was **renamed** to `runs/v5_p001_process_attempt0_wedge/` on
  fly122 — archived, never deleted, never analyzed, and pinned below by hash.
* The canonical run is a **fresh** run directory `runs/v5_p001_process/`: new preflight, then
  `--stage process` from candidate 1. The 570 attempt-0 labels are not reused (they carry the old
  stamp; mixing attempts would corrupt the freeze accounting).
* Expected cost of the amendment: ~2 recoveries per wedge candidate, bounded by the shared recovery
  budget; the observed wedge candidate’s cycle costs ≈ 20 min (180 s client + 360 s canary + restart
  + 180 s client + 360 s canary + restart) once per run.
* Attempt-0 artifacts were archived on fly122 under `runs/v5_p001_process_attempt0_wedge/` by a
  plain rename (never delete, never analyze) *before* this amendment was committed; hashes taken on
  the archived files at rename time:

| archived artifact (fly122) | size | sha256 |
|---|---|---|
| `runs/v5_p001_process_attempt0_wedge/v5_p001_process_labels.jsonl` | 2 145 794 B, 570 lines | `854f51118ad028dfee07c84263ac3f2d50de700a3c76ad59d26c16dd00492f08` |
| `runs/v5_p001_process_attempt0_wedge/v5_p001_process_infra_events.jsonl` | 1 534 B | `3e77ce66c0d968d076188d8de3b5a325346be573d8822a41e5bf56ec3d009c01` |
| `runs/v5_p001_process_attempt0_wedge/` (raw items, preflight, run meta) | 29 261 886 B total, 241 raw files (27 110 888 B under `raw/`) | archived in place under the same rename |

## 7. Pinned hashes

| item | sha256 |
|---|---|
| `scripts/v5_p001_process_run.py` before (Phase-A commit `27106df`) | `f98d810fc3e32bafe520ce7d18e1aae3093aef894a3489d4f27991cfa1f83c27` |
| `scripts/v5_p001_process_run.py` after (this amendment) | `5c090e5c6549b55af0850cce49644d34aee5707066128a9330d07d694e0c1f6c` |
| `scripts/v5_p001_spec.py`, `scripts/v5_process_oracle.py`, `scripts/v5_p001_reconstruct.py`, `scripts/v5_p001_analyze.py` | byte-identical to Phase A (stamp entries unchanged) |
| surface / oracle-validation / fixture hashes | unchanged (`b3c8f472…`, `9153fd2f…`, `2bfb0f4f…`) |

Tests: three new unit tests pin the amended policy in `tests/test_v5_process_oracle.py` —
(1) two-strike wedge ⇒ the censored infrastructure result is returned for the candidate (never a
failure), exactly one extra restore recovery runs (with its cold canary), and the event sequence is
`infra_event_after_recovery → candidate_censored_two_strikes → recovery_start → recovery_done`;
(2) a failing post-recovery retry with a healthy canary still censors with **no** extra restart
(health-restored path); (3) recovery-budget exhaustion still raises
(`budget exhausted`) — fail-close preserved.

## 8. Provenance of the decision

Owner approval (“批准 A”) of 2026-09-26 authorized exactly this: censor the wedge candidate as
`PROCESS_ORACLE_INFRA`, restore the instance with one more bounded recovery, continue the remaining
surface, and document the amendment with evidence, diff and hashes. Nothing in this amendment
touches what V5-P001 measures.
