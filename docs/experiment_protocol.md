# Experiment protocol

## P0 boundary

P0 prepares infrastructure only. It must not include formal RL training, final benchmark claims, curriculum/frontier sampling, self-training, 500M/360M model experiments, or a large dependency upgrade to chase the latest VERL.

## Evidence-driven decisions

Record each decision in a durable entry:

```text
Decision D001
Date:
Question:
Evidence:
Decision:
Reason:
Revisit if:
```

Examples of measurements that should guide P1/P2 include proof-length distribution, model generation throughput, Lean verification throughput, formatting failure rate, all-zero/mixed/all-one reward groups, and the relationship between Pass@K and K.

## P0 completion criteria

- `uv sync` creates a project-local `.venv`.
- The fixed Kimina RL reference submodule is present and its commit is visible in Git.
- The Lean server Compose file uses `projectnumina/kimina-lean-server:2.0.0`.
- A positive and negative Lean verifier request can be tested once Docker is available.
- The model→prompt→generation→proof extraction→Lean request smoke test is implemented.
- Models, data, caches and runs remain outside Git.

## Next gate

Before P1/P2, inspect actual compatibility and measurements. Do not assume the official absolute score will reproduce. The first comparison should test whether the local evaluator preserves the qualitative relation `Kimina-RL-0.6B > Kimina-Distill-0.6B`, then use the observed proof lengths and throughput to set the pilot budget.

