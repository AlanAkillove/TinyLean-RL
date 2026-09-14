# Reproduction record

This document is the human-readable index for external dependencies. Concrete resolved revisions should be added after the first successful download or environment installation; do not silently replace a pinned reference.

## Current P0 pins

| Component | Source | Revision/version | Role |
|---|---|---|---|
| Kimina-Prover-RL | `https://github.com/project-numina/kimina-prover-rl.git` | `e16b605e8186614c685875c9b57eb19e841b521a` | reference RL implementation |
| Kimina Lean Server | `projectnumina/kimina-lean-server` | Docker tag `2.0.0` | Lean verifier |
| Kimina Distill 0.6B | `AI-MO/Kimina-Prover-Distill-0.6B` | `332e8a5259d1bdfda19d7c7f339f30804813cd3a` | P0 model-to-Lean smoke test |
| Kimina RL 0.6B | `AI-MO/Kimina-Prover-RL-0.6B` | `43bb4da0e81cc9660057ced0025056eb031c6039` | P2 evaluation reference |
| Qwen3 Base 0.6B | `Qwen/Qwen3-0.6B-Base` | `main` until download resolves commit | later cold-start reference |
| Kimina Promptset | `AI-MO/Kimina-Prover-Promptset` | `3009c548d90160d0f5e963d72238610c6732f812` | P3 RL pilot |
| MiniF2F | `https://github.com/openai/miniF2F.git` | `4e433ff5cadff23f9911a2bb5bbab2d351ce5554` | evaluation-only benchmark |
| MiniF2F HF test | `AI-MO/minif2f_test` | `5def318348521dfc34875045a9ecddf729a2b49f` | Kimina recipe-compatible evaluation-only artifact |
| NuminaMath-LEAN | `AI-MO/NuminaMath-LEAN` | `main` until download resolves commit | later verified SFT construction |

The download scripts write machine-resolved local locks under ignored `*.local.json` files. When an asset becomes part of a reproducible experiment, copy the resolved commit/date/checksum into this document and commit that record before running the experiment.

## Reproduction rule

Record repository, revision, download date, relevant files, SHA-256 checksums where practical, runtime versions, GPU model, driver, CUDA version, command line, seed, and the exact configuration. A moving `main` reference is not sufficient for a final result.
