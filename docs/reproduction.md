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

## P3 Linux Training Stack

The P3 training experiments (P3-A/P3-B/E018) run on the Linux RTX 3090 host with the
following stack. It is pinned together with the VERL matrix (research decision D004,
`docs/research-decisions.md`) because the pinned VERL declares `vllm<=0.9.1`, which
itself requires `torch==2.7.0`.

```text
Host:      Ubuntu 24.04.4 LTS / RTX 3090 24 GB / NVIDIA driver 580.173.02
Python:    3.12.3 (uv-managed .venv)
Training stack:
  torch 2.7.0              (CUDA 12.6 wheel stack, +cu126)
  vLLM 0.9.1
  flash-attn 2.8.0.post2   (cu12 / torch2.7 / cp312 / cxx11abiTRUE wheel)
  transformers 4.53.3      (upper cap < 4.54.0: vllm 0.9.1 aimv2 AutoConfig collision)
  ray 2.48.0
  numpy 1.26.4             (verl requires numpy<2)
  kimina-client 0.2.1
VERL / Kimina: e16b605e8186614c685875c9b57eb19e841b521a (editable submodule; unmodified)
Lean verifier: projectnumina/kimina-lean-server:2.0.0 (Docker)
Dataset (P3):  AI-MO/Kimina-Prover-Promptset @ 3009c548d90160d0f5e963d72238610c6732f812
```

**Environment distinction (do not conflate):** the P0/P1/P2 inference baseline ran on
torch `2.9.0+cu128` (Windows and early Linux diagnosis); the P3 training stack above is
torch `2.7.0+cu126`. These are different environments; results from the inference
baseline must not be attributed to the P3 training stack and vice versa. The full
machine record lives in `experiments/manifests/p3_0_environment.yaml`.

## Reproduction rule

Record repository, revision, download date, relevant files, SHA-256 checksums where practical, runtime versions, GPU model, driver, CUDA version, command line, seed, and the exact configuration. A moving `main` reference is not sufficient for a final result.
