# V2-A001 termination snapshot (read-only)

Recorded 2026-09-23, immediately after the owner-ordered graceful stop.

> **V2-A001 was terminated before producing a checkpoint-selection result.**
> Existing completed and partial outputs are retained as historical evidence only
> and are not used to select a final RL checkpoint. This is a research-direction
> change, not an experiment failure.

## Termination

- stop command: `systemctl --user stop v2-a001` (graceful SIGTERM); the
  `v2-a001-stopwatch` wedge guard and the `v2-a001-postrun` completion watcher
  were stopped first; all three units verified inactive.
- stop timestamp: **2026-09-23 21:42:35 CST (2026-09-23T13:42:35Z)**
- termination reason: owner decision 2026-09-23 — the V2 exploration phase is
  closed; the project moves to a new research direction (Jev-inspired decision
  model for RLVR control). No checkpoint selection is wanted from V2-A001.
- canonical HEAD at termination (pre-closeout): `e2e530b634f43447cd3ff89892230c989d2da537`
- analysis state: the selection analyzer was **never executed** —
  no `experiments/results/v2_a001_analysis.json` exists, and the completion
  watcher never fired (no `.cache/v2_a001_analysis_run.log`).

## Frozen protocol references (unchanged)

- selection set: `experiments/manifests/v2/v2_a001_selection_set.json`
  sha256 `f429ddd7c628c3ce36b9ee7309c1fc68c819a0e03d20b38f5fd44112d7853e86`
- experiment manifest: `experiments/manifests/v2/V2-A001.yaml` (frozen; unmodified)
- runner: `.cache/v2_a001_resume.sh`
  sha256 `96d2d6a3a595b98ab1bcdf09edde56dd1477a99c8078134169cfe1d8054e06b6`
  (operator resume script; identical evaluator flags to `scripts/run_v2_a001.sh`)
- wedge guard: `.cache/v2_a001_stopwatch.sh`
  sha256 `cdf3339d5ce7f38e5d6ed185051739896510250d630f3f2572d4c58f893af768`
- completion watcher (inactive): `.cache/v2_a001_postrun.sh`
  sha256 `617892ad0c15542bef9f4a6e6728bbe5a20e256c01c7c51b65bef422f2772cc5`
- analyzer revision (never run): `scripts/v2_a001_analyze.py` @ `e2e530b`
- verifier: `projectnumina/kimina-lean-server:2.0.0`
  image `sha256:588a2cbbd10da509ed13f53ac136f8463fabff02dfe4eca535e7c47ae6e3ffd9`;
  memory cap 40 GiB; port 8000; container stopped 2026-09-23 (graceful,
  exit code 0) as part of the fly90 archive-only transition; restart =
  `docker start tinylean-rl-lean-server` + `scripts/prewarm_lean_server.py`.
- incident records: `.cache/v2_a001_incident01.md` … `incident03.md`
  (wedges, oomd kill, linger teardown, stopwatch evolution v2 → v4.2).

## Seven candidate model hashes (frozen; unchanged since launch)

Source: `.cache/v2_a001_model_hashes.txt` (== the record referenced by V2-A001.yaml).

| label | model.safetensors sha256 (prefix) |
| --- | --- |
| theta0 (base) | `34e6e630…` (anchor, never a candidate) |
| step10 | `987262a1…` |
| step20 | `0651e913…` |
| step30 | `0343caee…` |
| seed1_step60 | `dc6bd29d…` |
| seed2_step60 | `a3a23a6f…` |
| seed3_step60 | `0d31f947…` |

## Completed finals (retained; hashes unchanged since creation)

| model | artifact | records | verified | verifier_error | created (UTC) | artifact sha256 |
| --- | --- | --- | --- | --- | --- | --- |
| base | `experiments/results/v2_a001_base.json` | 2048 | 411/2048 | 25 | 2026-09-21T02:53:20 | `4470d0002d8592742a39b961f368ea099fbd34641e539814d14ace733a30e70c` |
| step10 | `experiments/results/v2_a001_step10.json` | 2048 | 408/2048 | 28 | 2026-09-21T06:31:36 | `fc7393d164c89b27f6e3f687c45736a670710bf8d3d45bad73df29da21620c27` |
| step20 | `experiments/results/v2_a001_step20.json` | 2048 | 399/2048 | 78 | 2026-09-21T08:42:58 | `41bbb71b948d4b84e3fdea96e0a73a03820b7bc54aa62a3226ebcf7b2dca13c4` |
| step30 | `experiments/results/v2_a001_step30.json` | 2048 | 422/2048 | 40 | 2026-09-21T13:50:04 | `53b327b1fa7d086dbccbe0e95d509fed804e707b4806e29851d97268328f676e` |

## Partial output (retained)

- seed1_step60: **304/512 theorems processed** (1216 records) at the stop;
  `experiments/results/v2_a001_seed1_step60.partial.json`,
  size 16,408,352 B, sha256
  `fee2d5a435071f65bebcb8d188b2857167eed2dfd18c4d726a5518ba1bcd9389`
  (last saved chunk 2026-09-23 21:41 CST; the in-flight chunk was discarded by
  the graceful stop).
- seed2_step60: never started (no partial, no final).
- seed3_step60: never started (no partial, no final).

## Outcome statements

- `selection_outcome: NONE`
- `theta_RL_star: NOT FROZEN`
- A2: NOT RUN; Candidate 2: NOT TRIGGERED
- Track C: NOT RUN
- No raw artifact was deleted; fly90 retains everything (zero pruning).
