# TinyLean-RL V2 — final closeout (empirical/diagnostic freeze)

Recorded 2026-09-23. V2 is closed as an empirical/diagnostic phase. No further
checkpoint selection, adaptive-compute modeling, or confirmatory Track C
experiment will be added to V2. This closeout summarizes only facts already
supported by the recorded artifacts; it adds no new capability claims.

## What V2 established (facts, with sources)

1. **RLVR trainability of the 0.6B Kimina prover is reproducible across seeds** —
   the V1 M1 seed replication (seeds 1–3 to step 60) reproduces the training
   dynamics; trainer metrics and dynamics dumps: `experiments/results/e019*`,
   `e020*`, `e022*`, `e023*`; narrative: `docs/experiment_log.md`,
   `docs/v2/legacy_evidence.md`.
2. **The informative-group ratio (~0.15) is a stable, repeatedly observed
   training phenomenon** — IGR mechanism set (`experiments/manifests/igr_mechanism_set.json`)
   and the V1 IGR statistics in the frozen V1 evidence.
3. **Held-out capability gain is small and seed-sensitive in the current
   short-run, low-compute regime** — E023 multi-seed holdout
   (`experiments/results/e023_multiseed_analysis.json` and adjudications).
4. **Qwen3 Base shows a reward-dead boundary under the same protocol** —
   M2/E021 cold-start diagnostic (`experiments/results/e021_smoke_dynamics.json`;
   `docs/seed_control_audit.md`).
5. **Verifier reliability, timeout behaviour and server-wedge failure modes are
   engineering-audited** — `docs/v2/b0_verifier_reliability.md`; incident
   records 01–05 (`.cache/v2_a001_incident0*.md`); Track B infra adjudication
   (75/75 cells re-verified sha-identical, `docs/v2/b002_memo.md`).
6. **Track B: large hindsight compute headroom did not convert into a stable
   cross-seed learnable allocation headroom** — B002 hindsight oracle upper
   bounds 86.21% / 98.44% vs the preregistered B003 cross-fitted gate flat at
   2.042 expected solved with rare-event cells; NO-GO for B004 (owner accepted).
   See `docs/v2/b002_memo.md`, `docs/v2/b003_memo.md`, `docs/v2/track_b_closeout.md`.
7. **The family-component audit reveals serious family-leakage / distribution-
   shift risk in exact-statement splits of the theorem pool** —
   `docs/v2/family_leakage_audit.md`; `experiments/manifests/v2/family_component_registry.json`.

## A001 disposition

V2-A001 (checkpoint selection) was **terminated before producing a
checkpoint-selection result** (owner decision 2026-09-23, graceful stop
21:42:35 CST). Completed finals (base/step10/step20/step30) and the
seed1_step60 partial (304/512) are retained as historical evidence only and are
not used to select a final RL checkpoint. `selection_outcome: NONE`;
`theta_RL*: NOT FROZEN`. Full record: `a001_termination_snapshot.md`.

## Direction of travel

V2 provides the empirical observations motivating a new low-cost research
direction: **predicting reward-informative RL groups with a lightweight,
calibrated decision model** (Jev-inspired decision model for RLVR control).
This closeout does not claim that the new direction works; it motivates it.
The new line proceeds outside V2 (branch `v3-jev-rl-controller`); V2 itself
stays frozen at this state.
