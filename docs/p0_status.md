# P0 status report

更新时间：2026-09-15（Asia/Shanghai）。这是当前工作树的事实记录，不包含未执行的实验结果。

## 已完成

- Git repository initialized on branch `main`.
- `uv.lock` generated and project `.venv` created inside the repository.
- Kimina-Prover-RL submodule added at `e16b605e8186614c685875c9b57eb19e841b521a`.
- MiniF2F repository cloned to the ignored data directory at `4e433ff5cadff23f9911a2bb5bbab2d351ce5554`.
- MiniF2F `test.lean` and `valid.lean` each contain 244 theorem declarations in this checkout.
- Kimina Distill 0.6B downloaded to the ignored model directory at HF commit `332e8a5259d1bdfda19d7c7f339f30804813cd3a`.
- Kimina-Prover-Promptset downloaded to the ignored data directory at HF commit `3009c548d90160d0f5e963d72238610c6732f812`.
- Kimina recipe-compatible `AI-MO/minif2f_test` downloaded at HF commit `5def318348521dfc34875045a9ecddf729a2b49f`.
- Kimina RL 0.6B downloaded at HF commit `43bb4da0e81cc9660057ced0025056eb031c6039`.
- A Kimina-format 64-row train/test pilot package was materialized under ignored `data/processed/` using the pinned upstream `prepare_data.py`.
- Project cache redirection, download scripts, Docker Compose, verifier client, proof extractor and smoke tests are implemented.
- The P2 evaluator harness now supports batched verifier requests and computes Pass@1/8/32 (when the sample count permits), all-zero/mixed/all-one group rates, format failures, proof lengths and generation/verification wall time.

## Local checks

| Check | Result |
|---|---|
| `uv sync --extra inference --extra dev` | passed |
| `uv run pytest` | 4 passed |
| `uv run ruff check src scripts tests` | passed |
| `uv run python -m compileall -q src scripts tests` | passed |
| Model load | passed on CPU |
| Model generation diagnostic | 16/16 candidates generated and extracted |
| Markdown/extra-text contamination | 11/16 candidates in the diagnostic |

CUDA generation comparison (same 16 candidates, max 64 new tokens): Distill used 9.526 s with 15/16 tactic-like candidates and 8/16 Markdown-contaminated outputs; RL used 3.465 s with 16/16 tactic-like candidates and 15/16 Markdown-contaminated outputs. These are generation-only diagnostics, not Lean-verified scores.

On the first real MiniF2F theorem with the Kimina prompt format, Distill required a 2,048-token budget to emit a complete `think` block and `lean4` block; the returned proof candidate used 1,793 generated tokens. The RL checkpoint emitted a complete block in 1,301 tokens under the same ceiling. Both candidates are saved in ignored probe artifacts and remain unverified until the Linux Lean server gate is available.

The 8-theorem/128-candidate Distill run at a 512-token ceiling produced zero complete `lean4` blocks, so the initial `configs/rl/kimina_0.6b_pilot.yaml` response-length hypothesis must be revisited after verifier-backed P2 measurements.

The evaluator dry-run on one MiniF2F theorem and four GPU samples completed successfully, writing `experiments/results/p2_dry_run.json`; as expected, it reports zero verified candidates because `--dry-run` bypasses the unavailable Lean server.

Promptset inspection: 24,418 rows, 7,620 unique `statement_id` values, 16,798 duplicate rows caused by the dataset's deliberate hard-problem reweighting, and 12 missing `natural_language` values. `formal_statement` has no missing values.

The HF recipe-compatible MiniF2F artifact contains exactly 244 rows with non-null `name`, `informal_prefix` and `formal_statement` columns.

The generation diagnostic is stored locally at `experiments/results/p0_generation_diagnostic.json` and is intentionally ignored by Git. It is not a verified theorem-proving score.

## Current machine gate

- OS: Windows development environment.
- GPU: RTX 4060 Laptop, 8 GiB, driver 572.61.
- PyTorch: `2.14.0+cpu`; `torch.cuda.is_available()` is `False`.
- Docker CLI/daemon: unavailable.
- Lean/Elan/Lake: unavailable.
- WSL: command exists, but no usable Linux distribution is configured.

Therefore this machine can perform CPU model generation and repository checks, but it cannot produce an honest Lean-verified reward, Pass@K, or RL training result. `http://127.0.0.1:8000/verify` returned HTTP 502 because no local Lean server is running.

## Next execution gate

On the Linux 3090 host:

1. Clone with submodules and run `source scripts/env.sh && uv sync`.
2. Start `infra/lean-server/compose.yaml` with Docker image `projectnumina/kimina-lean-server:2.0.0`.
3. Run `uv run python scripts/verify_smoke.py`; the positive case must pass and the negative case must fail.
4. Run the model→Lean smoke test, then evaluate Distill 0.6B versus RL 0.6B before any training.
5. Only if those gates pass, revise `configs/rl/kimina_0.6b_pilot.yaml` using measured proof lengths and verifier throughput.
