# V3 experiment manifests

V3 namespace (branch `v3-jev-rl-controller`, owner directive 2026-09-23):

- `V3-D###` — offline decision probes (existing V1 rollouts → frozen theta0
  representations → shallow calibrated heads). **No new RL training, no new rollouts.**
- `V3-R###` — RL-intervention experiments, created **only** after a `D###` passes its
  frozen GO gate **and** the owner authorizes the next stage.

Files:

| File | Role |
|---|---|
| `registry.yaml` | append-only V3 experiment registry |
| `V3-D001.yaml` | machine-readable frozen protocol for the first probe |
| `v1_rollout_sources.json` | §17 rollout-source provenance (hashes only; raw `rollout_data` is gitignored, never committed) |
| `v3_d001_folds.json` | frozen nested family-grouped CV folds (component-keyed; `folds_hash` in the manifest) |

Freeze discipline (mirrors V2): each experiment's `*_preregistration.md` +
`V3-*.yaml` are committed **before** representation extraction / model fitting and are
frozen against post-outcome change. Formal compute runs on **fly122 / RTX 3080**;
fly90 artifacts are non-formal. See `docs/v3/`.
