# V2 experiment manifests

## Rules

- **Namespace**: new experiments are `V2-E001`, `V2-E002`, ... . Old `E###` identifiers are **never reused**.
- A manifest is created **before** the formal run (pre-registration) and frozen per its own `freeze_rule`.
- Raw outputs remain **gitignored**; manifests record only metadata, hashes and result summaries.
- Failed / aborted experiments keep their manifest and status — **never delete or rewrite** a manifest
  to hide an outcome.
- Status values: `TEMPLATE` (skeleton only) · `PREREGISTERED` · `RUNNING` · `PAUSED` · `ABORTED` · `COMPLETE`.

## Layout

```text
experiments/manifests/v2/
├── README.md
├── templates/
│   ├── budget_response.template.yaml
│   ├── allocator.template.yaml
│   └── joint_eval.template.yaml
└── V2-E###.yaml            # created at pre-registration time (copy of a template)
```

Templates are skeletons copied at pre-registration time. They carry `status: TEMPLATE` and contain
**no run data** — no fabricated seeds, results, runtimes or artifact hashes.
