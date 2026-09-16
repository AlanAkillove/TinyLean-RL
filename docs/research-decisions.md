# Research decisions

Durable log of evidence-driven decisions. One entry per decision, newest last.

```text
Decision Dxxx
Date:
Question:
Evidence:
Decision:
Reason:
Revisit if:
```

## Decision D001

Date: 2026-09-15

Question:
Must the pinned PyTorch stack (`torch==2.9.0+cu128`) be changed for the Linux RTX 3090 host?

Evidence:
The host runs NVIDIA driver 580.173.02. `uv sync --extra inference` resolved the pinned
CUDA 12.8 wheel set and `torch.cuda.is_available()` returned `True` on the RTX 3090
(see `experiments/results/p1_generation_diagnostic_linux.json` for the first GPU run).

Decision:
Keep `torch==2.9.0+cu128` unchanged.

Reason:
The driver is far newer than the minimum required for CUDA 12.8 wheels; no upgrade is
needed and changing the pin would break baseline reproducibility.

Revisit if:
A CUDA/vLLM/VERL incompatibility appears when the P3 environment is assembled.

## Decision D002

Date: 2026-09-15

Question:
How should Hugging Face assets be downloaded on this host?

Evidence:
`https://huggingface.co` times out from this host, while `https://hf-mirror.com` answers in
under a second and served both 0.6B models (1.5 GiB each) plus three datasets without
changing the requested revisions.

Decision:
Use `HF_ENDPOINT=https://hf-mirror.com` and `HF_HUB_DISABLE_XET=1` for all downloads on
this host; keep revision pins in the manifests untouched.

Reason:
The mirror serves the exact pinned commits (`332e8a52...`, `43bb4da0...`,
`5def3183...`, `3009c548...`) recorded in `docs/reproduction.md`, so reproducibility is
preserved while making downloads feasible.

Revisit if:
The mirror starts serving different content for the pinned revisions, or direct HF access
becomes available.

## Decision D003

Date: 2026-09-15

Question:
How should the pinned Lean verifier image `projectnumina/kimina-lean-server:2.0.0` be
obtained on a host without Docker Hub access?

Evidence:
Docker Hub is unreachable. Free China mirror endpoints (docker.1ms.run, hub.rat.dev,
docker.1panel.live, docker.hlmirror.com and others) serve the image manifest quickly but
throttle the large layers (2.16 GiB + 333 MiB + 138 MiB) to roughly 10-140 KB/s or stall
entirely; their storage backends are international CDNs. Direct range probing and
multi-connection attempts did not improve throughput. GitHub release assets (a local-build
route) are throttled as well.

Decision:
Side-load the image: run `docker save` on a machine with Docker Hub access, transfer the
tar, then `docker load` on this host. The loaded image keeps the exact tag
`projectnumina/kimina-lean-server:2.0.0`; no rebuild and no tag substitution.

Reason:
Preserves the pinned verifier artifact byte-for-byte while working around the network
block. A slow background pull through a mirror stays as a fallback but is not expected to
finish in a reasonable time on its own.

Revisit if:
A reliable fast registry mirror becomes available, or a verifier behavior difference
suggests the side-loaded image is not the published one (verify by image digest
comparison with Docker Hub).

## Decision D004

Date: 2026-09-17

Question:
Can the P3-A training stack (pinned VERL @ `e16b605e`) run on the P0-pinned
`torch==2.9.0+cu128` venv, or must the pin move to the Kimina-tested matrix?

Evidence:
- The pinned VERL declares `vllm>=0.7.3,<=0.9.1` (`setup.py` `VLLM_REQUIRES`), and its
  vLLM rollout integration imports 0.9.x-era internals (`vllm.config.CompilationConfig`,
  `vllm.worker.worker_base.WorkerWrapperBase`,
  `vllm.model_executor.sampling_metadata.SamplingMetadata`).
- vllm 0.9.1 (PyPI metadata) requires `torch==2.7.0` exactly, plus torchvision 0.22.0,
  torchaudio 2.7.0 and xformers 0.0.30.
- `verl/workers/fsdp_workers.py` L287-289 forces `attn_implementation="flash_attention_2"`;
  flash-attention publishes cp312/cu12/torch2.7 wheels (2.8.0.post2) but none for torch 2.9.
- torch 2.7.0 default PyPI wheels are the CUDA 12.6 builds; driver 580.173.02 supports
  them (D001 revisit condition: "a CUDA/vLLM/VERL incompatibility appears").
- transformers must stay `<4.54.0`: vllm 0.9.1 registers an `aimv2` AutoConfig
  unconditionally, which collides with the `aimv2` mapping added upstream in
  transformers 4.54.0 (empirically verified: `import vllm` fails with 4.54.1,
  succeeds with 4.53.3).

Decision:
Move the project venv to the Kimina-tested matrix: `torch==2.7.0` (cu126 default wheels),
`vllm==0.9.1`, flash-attn 2.8.0.post2 (cp312, torch2.7, cxx11abiTRUE) and `verl` installed
editable from the pinned submodule; declared as the `training` extra in `pyproject.toml`.

Reason:
The pinned submodule's actual dependency set is the reproducibility authority for P3;
torch 2.9 was chosen for the inference-only baseline (P0/P1/P2) and cannot coexist with
`vllm<=0.9.1` in one environment.

Revisit if:
A newer pinned VERL is adopted for P3-B/M2/M3, or a P3-A failure traces back to the
torch 2.7.0 / vllm 0.9.1 pairing.
