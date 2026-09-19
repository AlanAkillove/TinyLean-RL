# TinyLean-RL V1 — Evidence Index (frozen 2026-09-19)

V1 is frozen as historical evidence: **P0–P3 / E001–E024** are kept, not rewritten, not renumbered
(branch `p3-linux`, annotated tag `v1-research-freeze-20260919`).

This page is an **index only** — numbers live in the canonical docs/manifests linked below;
do not restate or reinterpret them here.

| Stage | Canonical evidence |
| --- | --- |
| P0 / P1 infrastructure | [`docs/p0_status.md`](../p0_status.md); [`docs/environment.md`](../environment.md); `experiments/manifests/p0_*.yaml` |
| P2 evaluation / reward calibration | [`docs/studies/evaluation_calibration.md`](../studies/evaluation_calibration.md) |
| P2.5 RL readiness | [`docs/studies/rl_readiness.md`](../studies/rl_readiness.md); `experiments/manifests/p2_5_complete.yaml` |
| P3-0 migration / on-policy calibration | `experiments/manifests/p3_0_complete.yaml`; `experiments/manifests/p3_0_environment.yaml` |
| P3-A pipeline smoke | [`docs/experiment_log.md`](../experiment_log.md) (E015 entry) |
| P3-B learning pilot | `experiments/manifests/p3b_pilot.yaml` |
| E018 / E019 fixed-set evidence | `experiments/manifests/p3c_fixed_eval.yaml`; `experiments/manifests/m1_step60.yaml`; [`docs/experiment_log.md`](../experiment_log.md) |
| E020 / E022 seed replication | `experiments/manifests/m1_seed_replication.yaml` |
| E020-M mechanism experiment | `experiments/manifests/igr_mechanism_set.json` |
| E021 Qwen3-Base reward-dead boundary | [`docs/experiment_log.md`](../experiment_log.md); [`docs/p0_status.md`](../p0_status.md) |
| E023 final holdout | `experiments/manifests/e023_holdout.yaml`; [`docs/experiment_log.md`](../experiment_log.md) |
| E024 MiniF2F | [`docs/e024_status.md`](../e024_status.md); [`docs/e024_fly90_handoff.md`](../e024_fly90_handoff.md) |

Supporting V1 records: [`docs/research_plan.md`](../research_plan.md) and
[`docs/experiment_log.md`](../experiment_log.md) (both marked as frozen V1), plus
`docs/research-decisions.md` and the per-experiment manifests under `experiments/manifests/`.

## E024 status statement (binding)

**PAUSED / ABORTED DUE TO VERIFIER INFRASTRUCTURE INCIDENT; no E024 result claim.**

The chunk-7 verifier errors are an infrastructure artifact (Lean server wedge + mid-flight restart),
**not model failures**. E024 must not be rerun as-is; the wedge must be investigated first
(see [`docs/e024_status.md`](../e024_status.md) §5).

## Frozen untracked artifacts (kept out of Git by policy)

Independently re-hashed over LAN on 2026-09-19 (values match `docs/e024_status.md` §4):

| artifact (on fly122) | bytes | sha256 |
| --- | --- | --- |
| `experiments/results/e023_by_prefix_bug_audit.json` | 3,382 | `cf116ee5105b0ca0d06289a7db939b4b8d75acdd7371be0f7a62ac53ebacc1c8` |
| `experiments/results/e024_minif2f_theta0.partial.json` | 6,536,537 | `e79f3d68f44c1e8a2ea298c2625f75a60495eef446c692f93688290941f4ee95` |
| `experiments/results/e024_minif2f_theta0_run.log` | 82,749 | `59db3fb7ef99fdd2b5babc9fdf9a3f622c111490aef3b73cb59b55b185465e08` |

## E023 retro by-prefix audit

**CLEAN** — `affected_total = 0` across all four primary E023 artifacts; the later-fixed
`complete_verifier_code` boundary bug (fixed in `b4196b2`) does not affect any E023 statistic
(evidence: `e023_by_prefix_bug_audit.json`, see table above).

## V1 closure notes

- M2 verified-SFT intervention: preregistered only (`docs/m2_sft_intervention_plan.md`); never run.
- M3 model-size frontier: never started.
- Reserved-but-unrun V1 numbers (e.g. E025/E026) are superseded; all future experiments use `V2-E###`.
- V1 detailed status remains linked via the frozen docs above; V2 planning starts at
  [`docs/v2/research_plan.md`](research_plan.md).
