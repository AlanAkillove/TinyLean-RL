# V2 experiment manifests

## Rules

- **Namespace**: new experiments use the track-specific namespaces `V2-A###` (Track A / fly90),
  `V2-B###` (Track B / fly122), `V2-C###` (Track C / fly90). The pre-amendment shared
  namespace `V2-E###` is closed for new experiments; raw pre-amendment identifiers are never
  rewritten and stay registered with their aliases in `registry.yaml`. Old V1 `E###`
  identifiers are **never reused**.
- **Track**: every manifest declares `track: A | B | C` — A = RL prover (fly90), B = adaptive
  allocator (fly122), C = joint evaluation. Track A runs follow `docs/v2/track_a_rl_plan.md`
  and must state at least: research question, starting checkpoint, training data, seed,
  training steps, n / temperature / top_p / max_response, stopping rule, checkpoint schedule,
  primary metrics, comparison checkpoint, go/no-go rule.
- A manifest is created **before** the formal run (pre-registration) and frozen per its own `freeze_rule`.
- Raw outputs remain **gitignored**; manifests record only metadata, hashes and result summaries.
- Failed / aborted experiments keep their manifest and status — **never delete or rewrite** a manifest
  to hide an outcome.
- Status values: `TEMPLATE` (skeleton only) · `PREREGISTERED` · `RUNNING` · `PAUSED` · `ABORTED` · `COMPLETE`.

## Layout

```text
experiments/manifests/v2/
├── README.md
├── registry.yaml                        # canonical experiment registry (namespaces + aliases)
├── theorem_role_registry.json           # frozen V2 theorem-role partition (coordination 2026-09-19)
├── b1_reserved_statement_ids.json       # B1-audit-reserved carve-out (theorem-role registry input)
├── templates/
│   ├── rl_training.template.yaml
│   ├── budget_response.template.yaml
│   ├── allocator.template.yaml
│   └── joint_eval.template.yaml
└── V2-A###.yaml / V2-B###.yaml / V2-C###.yaml    # created at pre-registration time (copy of a template)
```

Templates are skeletons copied at pre-registration time. They carry `status: TEMPLATE` and contain
**no run data** — no fabricated seeds, results, runtimes or artifact hashes.
