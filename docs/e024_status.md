# E024 status: E023 retro audit (CLEAN) and theta0 paused after a Lean server wedge

- Date: 2026-09-19 (UTC)
- Host: fly122 worker (RTX 3080 10GB)
- Canonical base at pause: b4196b2b9b0765f88602976a067cfb6782710029
- Status: **PAUSED** (operator instruction); no E024 result is claimed

## 1. E023 retrospective by-prefix bug audit (CPU only)

Context: canonical review found a real evaluator bug in the shared helper
`scripts/promptset_rollout_probe.py::complete_verifier_code()`. The old code
stripped two characters from any proof body starting with `by`
(`if body.startswith("by"): body = body[2:].lstrip()`), corrupting bodies that
begin with tactic tokens such as `by_cases` or `by_contra`. Fixed in b4196b2:
strip only a standalone `by` keyword (`re.match(r"by(\s|$)", body)`).

Audit method: scan only, no Lean, no regeneration, raw artifacts untouched.

- Scan rule: `affected = extracted_proof.strip().startswith("by") and not re.match(r"^by(?:\s|$)", body)`
- Artifacts scanned (sha256 verified identical to the values recorded in
  experiments/manifests/e023_holdout.yaml):
  - e023_holdout_base.json (58d6d419...92e), 512 candidates
  - e023_holdout_seed1_fly122.json (5ed68a55...c9), 512 candidates
  - e023_holdout_seed2.json (cd350aa8...15), 512 candidates
  - e023_holdout_seed3_fly122.json (b01f3016...55), 512 candidates
- Result: **affected_total = 0**; all 2048 extracted proofs nonempty; the
  diagnostic underscore-line fingerprint check also found 0 candidates.
- Artifact: `experiments/results/e023_by_prefix_bug_audit.json` (3,382 bytes,
  sha256 cf116ee5105b0ca0d06289a7db939b4b8d75acdd7371be0f7a62ac53ebacc1c8)
- Conclusion: **E023 RETRO AUDIT: CLEAN** — E023 final statistics are
  unaffected by the boundary bug; no recomputation required.

## 2. E024 preconditions (all met before launch)

- fly122 repo synced to canonical b4196b2 via LAN fast-forward; clean tree.
- GPU guard: 0 compute processes at launch.
- Lean server restarted; health {"status":"ok"}; prewarm ladder 8/8 OK with
  verified_count = 8 at c=1, c=2 and c=4 (formal verifier path proven healthy;
  the earlier c=4 0/8 symptom did not recur).

## 3. theta0 run and Lean server wedge

- Launch: 11:28:23 UTC as systemd user unit `fly122-e024-theta0`
  (canonical defaults: 244 theorems, 4 samples, T=1.0, top_p=1.0,
  max_new_tokens 4096, seed 20260917 + idx*8 + sample, verify workers 4,
  verify batch 4; output e024_minif2f_theta0.json).
- Chunks 1-6 completed by 12:05 UTC (cumulative verified 151/384; 3 sub-batch
  retry warnings: one HTTP 500, two client timeouts, all recovered).
- During chunk 7 verification the Lean server wedged: no evaluator log output
  for about 60 minutes; /health exceeded the 5 s cap; a trivial verification
  request raised ReadTimeout; the server process held about 100 percent CPU;
  one client HTTP connection stayed open and the chunk was blocked on a single
  hanging sub-batch.
- Operational remediation 13:06 UTC: `docker restart tinylean-rl-lean-server`
  (infrastructure recovery only; no change to the theorem set, sampling
  hyperparameters, token budget or verifier semantics).
- The evaluator recovered automatically: chunk 7 completed with 0/64 newly
  verified and 64 verifier_error (the mid-flight restart invalidated the whole
  chunk), chunk 8 completed (cumulative verified 160/512). 16 new sub-batch
  warnings were emitted during the restart turbulence.
- Paused 13:16:06 UTC per operator instruction: unit stopped; GPU freed and
  verified at 0 compute processes. Run duration 107.7 minutes.

## 4. Frozen evidence (kept on fly122, not committed)

- `experiments/results/e024_minif2f_theta0.partial.json` — evaluator-written
  partial (chunks 1-8: processed_theorems 128/244, records 512/976),
  6,536,537 bytes,
  sha256 e79f3d68f44c1e8a2ea298c2625f75a60495eef446c692f93688290941f4ee95
- `experiments/results/e024_minif2f_theta0_run.log` — preserved run log
  (82,749 bytes,
  sha256 59db3fb7ef99fdd2b5babc9fdf9a3f622c111490aef3b73cb59b55b185465e08)
- Partial taxonomy (512 records): verified 160, lean_parse_error 196,
  lean_semantic_error 86, verifier_error 70. Per-chunk verified:
  30/15/37/16/26/27/0/9 of 64. No adjudication and no paired analysis were
  run; theta0 must be rerun from scratch.

## 5. Next steps (pending canonical decision)

1. Investigate the wedge: replay the chunk-7 candidate set against a scratch
   server to identify the pathological proof; consider server-side guards
   (per-request hard timeout or pool recycling between chunks) before any
   relaunch.
2. Relaunch theta0 from scratch, then adjudication, then seed1_step60 and its
   adjudication, then the paired analysis per experiments/manifests/e024_minif2f.yaml.
3. Integrate the E023 audit artifact and this status note canonically.
