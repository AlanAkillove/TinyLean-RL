# P0 status report

更新时间：2026-09-17（Asia/Shanghai）。这是当前工作树的事实记录，不包含未执行的实验结果。逐次实验过程记录见 [`experiment_log.md`](experiment_log.md)。

## Linux P3 stage（2026-09-17，服务器 RTX 3090 24 GB；P3-0 → P3-A → P3-B 完成）

- 工作分支 `p3-linux`（基线 `win`@6ef923e / tag `p2.5-win-complete`；submodule e16b605e 未动）；`win..p3-linux` = **18**、`p3-linux..win` = **0**（纯前向扩展，实测值）。
- 环境对齐 pinned VERL 矩阵（决策 D004）：torch 2.7.0+cu126、vllm 0.9.1、flash-attn 2.8.0.post2、transformers 4.53.3、ray 2.48.0、VERL editable——`experiments/manifests/p3_0_environment.yaml`。
- 迁移 gate：pytest 40 passed / GRPO reference 19 passed / ruff+compileall / doctor 23 passed / verifier 正负 gate / model→Lean smoke / reward 契约实测（kimina-client 0.2.1 ↔ server 2.0.0）全部通过（E012）。
- Promptset 校准（temp 1.0 / top_p 1.0 / n=4 / 4096，E013）：IGR 0.09375、截断 66.4%、成功轨迹未撞上限；n=8 配对校准（E016）：同 16 定理 IGR 0.00→0.0625。
- Full-FT 单步探针（E014）：exit 0、step 136.4 s、peak reserved 20.5 GB → 训练模式冻结为 FULL-FT。
- P3-A smoke（E015）：含从 `global_step_3` 的 resume 测试；mixed 组步 grad_norm 0.180。
- P3-B pilot（E017）：`run_p3_pilot.sh --steps 30 --n 8`，exit 0、4218 s；IGR 0.100→0.200→0.225、Z 0.90→0.75、score 0.059→0.175；checkpoint 10/20/30；显存峰值 23.1 GiB。
- 摘要清单：`experiments/manifests/p3_0_complete.yaml`、`experiments/manifests/p3b_pilot.yaml`；逐次记录 E012–E017。
- 下一步（待确认）：固定定理集评估 step-0/10/20/30 checkpoint，确认学习信号。

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
- Elan's pinned Lean `4.34.0-rc2` toolchain is installed under the ignored `.tools/elan/` directory; Mathlib is checked out at `9cb3970b1fb61911f7e8892dffcde5aa4a0661cc` and its Lake cache bootstrap completed.
- The native verifier now uses the exact rc2 compiler directly, batches candidates behind one Mathlib import, and terminates the full compiler process tree on timeout. It remains a fallback and is not the official Kimina evaluator.
- The P2 evaluator harness now supports batched verifier requests and computes Pass@1/8/32 (when the sample count permits), all-zero/mixed/all-one group rates, format failures, proof lengths and generation/verification wall time.
- Docker Desktop recovered after the Windows restart; the Linux daemon is up (CLI `29.8.0`) and the pinned `projectnumina/kimina-lean-server:2.0.0` container `tinylean-rl-lean-server` answers `/health` with 200 (image Lean `v4.15.0`).
- `infra/lean-server/compose.yaml` no longer forwards an empty `LEAN_SERVER_API_KEY`; the image treats a present-but-empty key as a required key (an empty string is not `None` in pydantic), so the previous unconditional variable made every `/verify` request fail with 401.
- The verifier gate passed: `scripts/verify_smoke.py` clears the positive goal set and reports `unsolved goals` for the negative case, and `scripts/smoke_test.py` completed the model → generation → extraction → Lean chain on CUDA.
- `scripts/evaluate_model.py` now parses the actual Kimina 2.0.0 response shape (`response.error` or error-severity messages mark a failure; warning-only responses are valid), stores `raw_output` in every record for audit, retries a failed verify batch once, and falls back to per-candidate verification so a crashing candidate does not invalidate healthy batch mates.
- Verification scripts reconfigure their console streams to UTF-8 so Lean goal symbols can be printed on GBK Windows consoles.
- `httpx` calls in `src/tinylean_rl/verifier/kimina.py` now set `trust_env=False`: a system-level Windows proxy had hijacked local `/verify` requests and produced persistent 502 responses while curl and in-container calls stayed healthy.
- The evaluator stores a per-record `verify_status` (`verified`/`lean_error`/`verifier_error`), retries a failed verify batch once after a 15 s backoff, falls back to per-candidate verification, and persists a `.partial.json` after every batch; long-running script loops print tqdm progress.
- The 4,096-token rerun (E005/E006) confirmed the truncation hypothesis and produced the corrected preview table below.

## Local checks

| Check | Result |
|---|---|
| `uv sync --extra inference --extra dev` | passed |
| `uv run pytest` | 8 passed |
| `uv run ruff check src scripts tests` | passed |
| `uv run python -m compileall -q src scripts tests` | passed |
| Native verifier timeout/process-tree smoke | passed |
| Lean server positive/negative gate | passed / failed as expected |
| Model→Lean smoke test | candidate verified, goals cleared |
| Audit re-run (`ruff`, `pytest`, `compileall`) | passed |
| Model load | passed on CUDA (`torch 2.9.0+cu128`) |
| Model generation diagnostic | 16/16 candidates generated and extracted |
| Markdown/extra-text contamination | 11/16 candidates in the diagnostic |

CUDA generation comparison (same 16 candidates, max 64 new tokens): Distill used 9.526 s with 15/16 tactic-like candidates and 8/16 Markdown-contaminated outputs; RL used 3.465 s with 16/16 tactic-like candidates and 15/16 Markdown-contaminated outputs. These are generation-only diagnostics, not Lean-verified scores.

On the first real MiniF2F theorem with the Kimina prompt format, Distill required a 2,048-token budget to emit a complete `think` block and `lean4` block; the returned proof candidate used 1,793 generated tokens. The RL checkpoint emitted a complete block in 1,301 tokens under the same ceiling. Both candidates are saved in ignored probe artifacts and remain unverified until the Linux Lean server gate is available.

The 8-theorem/128-candidate Distill run at a 512-token ceiling produced zero complete `lean4` blocks, so the initial `configs/rl/kimina_0.6b_pilot.yaml` response-length hypothesis must be revisited after verifier-backed P2 measurements.

Long-budget MiniF2F generation-only comparison (32 theorems × 4 samples, max 2,048 new tokens, CUDA): Distill generated 128 candidates in 2,139.156 s and produced 37 likely-Lean candidates (37 explicit `lean4` blocks), 41 complete `think` blocks, and 91 Markdown-contaminated outputs. RL generated 128 candidates in 2,104.614 s and produced 47 likely-Lean candidates (47 explicit `lean4` blocks), 52 complete `think` blocks, and 106 Markdown-contaminated outputs. These figures measure output shape only; both verified counts remain unknown until the candidates pass Lean Server.

The evaluator dry-run on one MiniF2F theorem and four GPU samples completed successfully, writing `experiments/results/p2_dry_run.json`; as expected, it reports zero verified candidates because `--dry-run` bypasses the unavailable Lean server.

Promptset inspection: 24,418 rows, 7,620 unique `statement_id` values, 16,798 duplicate rows caused by the dataset's deliberate hard-problem reweighting, and 12 missing `natural_language` values. `formal_statement` has no missing values.

The HF recipe-compatible MiniF2F artifact contains exactly 244 rows with non-null `name`, `informal_prefix` and `formal_statement` columns.

The generation diagnostics are stored locally at `experiments/results/` and are intentionally ignored by Git. They are not verified theorem-proving scores. The MiniF2F harness now reports `extraction_successes` only for explicit Lean code blocks or outputs beginning with a conservative Lean prefix; non-empty reasoning text is not counted as a candidate.

## Verified P2 preview (miniF2F, 4 samples per theorem, CUDA)

| Model | Theorems | Budget | Verified | Pass@1 | all_zero | mixed | all_one |
|---|---|---|---|---|---|---|---|
| `kimina_distill_0_6b` | 8 | 2048 | 10/32 | 0.3125 | 0.625 | 0.125 | 0.25 |
| `kimina_rl_0_6b` | 8 | 2048 | 9/32 | 0.28125 | 0.625 | 0.25 | 0.125 |
| `kimina_distill_0_6b` | 8 | 4096 | 14/32 | 0.4375 | 0.5 | 0.125 | 0.375 |
| `kimina_rl_0_6b` | 8 | 4096 | 13/32 | 0.40625 | 0.5 | 0.25 | 0.25 |
| `kimina_distill_0_6b` | 32 | 4096 | 43/128 | 0.3359 | 0.531 | 0.281 | 0.188 |
| `kimina_rl_0_6b` | 32 | 4096 | 55/128 | 0.4297 | 0.4375 | 0.3125 | 0.25 |

These numbers use the local Kimina server (Lean `v4.15.0`), not the official harness, so they are not official absolute scores. At 8 theorems the two checkpoints sit within noise of each other (Distill nominally ahead); at 32 theorems the RL checkpoint leads by +9.4 pp candidate-level (z≈1.55, not significant) and +3 equations theorem-level (McNemar p≈0.375) — the direction now matches the official +2.45 pp claim while remaining statistically undecidable at this subset size (E007 vs E008; see `docs/research_plan.md` §5). Subset choice alone can flip the sign of the comparison.

Observed constraints from the same runs:

- Truncation is the dominant systematic bias: solved candidates average ~1,700–1,750 tokens (~42% of the 4,096 budget) while failed theorem sets pin the ceiling; at 32 theorems only 15–18/32 theorems are solved at all.
- Seed stability: the same checkpoint re-run on the same 8-theorem subset moved from 14/32 to 10/32 candidates (E005 vs E007) — run-to-run sampling noise exceeds most checkpoint differences.
- `native_decide` is the verifier-side risk (reproducible HTTP 500 crashes, all on `amc12_2001_p5`/theorem 3); at 32-theorem scale the frequency is similar for both checkpoints (Distill 12/128, RL 11/128), so the earlier “RL prefers native_decide” observation is retracted.
- Concurrency: parallel `/verify` calls each spawn a cold REPL (multi-minute Mathlib load), so the P3 reward path must use serialised submission or a pre-warmed REPL pool.

## P2.5 Local RL Readiness（2026-09-16，停止条件 6/6 满足；已冻结，tag `p2.5-win-complete`）

本地 RL 前置验证完成，完整记录见 [`studies/rl_readiness.md`](studies/rl_readiness.md)，实验条目 E009–E011 见 [`experiment_log.md`](experiment_log.md)，证据清单见 `experiments/manifests/p2_5_complete.yaml`。

| W | 交付 | 关键数字 |
|---|---|---|
| W1 | Promptset 32×4 cached rollout profile | mixed 1/32、all-one 3/32（IGR 3.1%）；截断 96/128（75%）；verifier_errors 5；产物 `p2_5_promptset_profile.json` + `experiments/local_rl_batch/` |
| W2 | GRPO 移植 + rehearsal（停止条件②） | 非零 advantage ±0.5、per-sample loss ±0.386、grad_norm 2.036（有限）；全套件 27 tests 全过 |
| W3 | LoRA 单步显存探针 | r16/32 × 1024/2048 四组合全过；2048 组合 reserved 8.6–8.75 GB > 物理 8.19 GB（共享显存兜底，不可依赖）；CPU 冒烟 224/224 张量有梯度 |
| W4 | P3 config 审计 | `docs/p3_config_audit.md`；reward contract（format gating/并发/单轮/verifier_error→0/严格验证）写回 research_plan §四 |
| W5 | cold-start SFT 数据管道 | NuminaMath-LEAN 104,155→31,634；抽样验证 190/200（95%）；train 31,002 / val 632 |
| W6 | 部署命令链（Linux 目标）+ Lean server 预热 | prewarm 阶梯 c=2 达 3.93 rps（recommended_concurrency=2）；doctor/run_p3_smoke 就绪 |

结论：本机（8 GiB）完成全部前置验证；P2.5 冻结于 tag `p2.5-win-complete`。下一阶段 = **P3-0 — Linux Migration & On-Policy Calibration**（服务器 RTX 3090 24 GB）：先跑环境 gate（`bash scripts/doctor.sh`），再做 Promptset rollout 校准（temp 1.0 / n=4 / 4096）与 full-parameter 显存可行性探针；冻结 P3-A 配置后才执行 `bash scripts/run_p3_smoke.sh`（provisional runner，见 `docs/p3_linux_handoff.md`）。

## Current machine gate

- OS: Windows development environment.
- GPU: RTX 4060 Laptop, 8 GiB, driver 572.61.
- PyTorch: `2.9.0+cu128`; `torch.cuda.is_available()` is `True` (`NVIDIA GeForce RTX 4060 Laptop GPU`).
- Docker Desktop: recovered after the Windows restart. The Linux daemon is up (CLI `29.8.0`), and the pinned `projectnumina/kimina-lean-server:2.0.0` container answers `/health` with 200 (image Lean `v4.15.0`).
- Elan/Lean: available with the pinned local Lean `4.34.0-rc2` toolchain and Mathlib cache.
- WSL: Ubuntu and Docker's `docker-desktop` distributions are registered as WSL2; Docker's internal data disk was isolated for regeneration and the restart resolved the stale-socket startup failure.

The earlier Docker failure matched the Windows 11 build 26200 AF_UNIX/ReparsePoint startup issue documented in Docker Desktop feedback [#460](https://github.com/docker/desktop-feedback/issues/460) and [#536](https://github.com/docker/desktop-feedback/issues/536); the daemon started normally after the Windows restart.

This machine can now produce verifier-backed Pass@K measurements through the local Kimina server; both smoke gates and two 8×4 P2 evaluations completed. The first `/verify` request after a cold container start can take several minutes while a REPL loads Mathlib; later requests reuse the warm REPL pool and return in milliseconds. The native compiler fallback remains a diagnostic path only.

## Next execution gate

On this Windows machine (the Docker gate is now open):

**2026-09-16 update**: items 2–4 below are complete (E007/E008 32-theorem comparison plus the P2.5 work above); the next gate is P3-0 — Linux migration and on-policy calibration (`docs/p3_linux_handoff.md`); the provisional P3-A smoke (`scripts/run_p3_smoke.sh`) runs only after P3-0 freezes the config.

1. Done: the 4,096-token rerun (E005/E006) confirmed the truncation hypothesis; four hard theorems still truncate at that ceiling.
2. Scale the comparison to 32 MiniF2F theorems (and consider 8 or 32 samples for Pass@8/32) before drawing any ranking conclusion between Distill and RL — the official gap is only +2.45 pp.
3. Design the P3 reward path around the verified concurrency limit: parallel `/verify` calls each spawn a cold REPL (minutes), so use a serialised local queue or a pre-warmed REPL pool.
4. Only then revise `configs/rl/kimina_0.6b_pilot.yaml` using measured proof lengths and verifier throughput.

On the Linux 3090 host (unchanged plan):

1. Clone with submodules and run `source scripts/env.sh && uv sync`.
2. Start `infra/lean-server/compose.yaml` with Docker image `projectnumina/kimina-lean-server:2.0.0`.
3. Reproduce the same gates; record the server image's Lean version (`v4.15.0` here) as part of the environment record.
4. Treat the local Kimina-server numbers as the working baseline and the official published scores only as a qualitative reference.
