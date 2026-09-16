# TinyLean-RL

面向亚十亿参数 Lean4 定理证明器的可验证强化学习研究项目。

当前阶段：**P3-0 → P3-A → P3-B 全部完成（Linux RTX 3090 24 GB）** —— 迁移 gate、Promptset on-policy 校准（IGR 0.09375）、full-FT 显存可行性与 GRPO smoke（含 checkpoint resume）、以及 n=8 的 30 步短学习 pilot（Z_t↓、score↑、末段连续梯度）均已记录。下一步：固定定理集上的 checkpoint 确认性评估（见 [`experiments/manifests/p3b_pilot.yaml`](experiments/manifests/p3b_pilot.yaml)）。历史交接文档：[`docs/p3_linux_handoff.md`](docs/p3_linux_handoff.md)。

- 研究主线（RL for sub-billion Lean provers）：[`docs/research_plan.md`](docs/research_plan.md)
- P2 预实验结论（评估校准 / 奖励可靠性 / 奖励信息量）：[`docs/studies/evaluation_calibration.md`](docs/studies/evaluation_calibration.md)
- 逐次实验记录（E001–E017）：[`docs/experiment_log.md`](docs/experiment_log.md)
- P2.5 结论与 Linux 交接：[`docs/studies/rl_readiness.md`](docs/studies/rl_readiness.md) / [`docs/p3_linux_handoff.md`](docs/p3_linux_handoff.md)

本项目遵循“阶段门控、证据驱动”的研究路线：先建立可复现环境和最小端到端链路，再根据预实验结果决定 RL、cold-start、scaling 或 frontier sampling 的后续分支。

## 当前冻结的研究边界

- 第一 baseline：`AI-MO/Kimina-Prover-Distill-0.6B` → Kimina-Prover-RL recipe。
- Lean verifier：`projectnumina/kimina-lean-server:2.0.0`。
- 第一标准 benchmark：MiniF2F test（evaluation-only）。
- 当前只准备 Kimina 0.6B Distill、Kimina 0.6B RL 和 Qwen3 0.6B Base 的资源声明。
- RL 按阶段门控推进：P3-A/P3-B（受限算力短 pilot）已完成；未启动 P3-C / 长期训练，不实现 curriculum/frontier sampling/self-training，不开始 500M/360M 实验（M2/M3）。
- Cold-start SFT 数据已备料（NuminaMath-LEAN，`data/processed/sft_cold_start/`，prepared only、尚未训练），定位为 **M2 contingency asset**，不属于当前 P3 主线。

## 仓库结构

```text
TinyLean-RL/
├── configs/                  # 配置（P3-A/P3-B 冻结配置，见 p3_config_audit S10）
├── data/manifests/           # 数据源和版本声明
├── docs/                     # 环境、复现和实验记录规范
├── experiments/              # 实验清单与结果索引，不放大文件
├── infra/lean-server/        # Kimina Lean Server Docker Compose
├── models/manifests/         # 模型源和 revision 声明
├── scripts/                  # 环境、下载、检查和 smoke test
├── src/tinylean_rl/          # 本项目的轻量封装
├── tests/                    # 单元测试
└── third_party/              # 固定版本的上游代码
```

## 快速开始（Linux）

```bash
git clone --recurse-submodules <repository-url> TinyLean-RL
cd TinyLean-RL
source scripts/env.sh
uv sync
cp .env.example .env

docker compose -f infra/lean-server/compose.yaml up -d
bash scripts/doctor.sh
```

准备模型后，才运行：

```bash
uv run python scripts/smoke_test.py --model-key kimina_distill_0_6b
```

Windows 开发机可以使用 `scripts/env.ps1` 设置同名环境变量；Lean server 和 Kimina RL 训练仍以 Linux + NVIDIA Docker 为目标环境。

## Git 与大文件

模型权重、数据集、缓存、Docker 运行日志和实验输出均不进入 Git。外部依赖只通过 manifest、submodule 和版本记录进行追踪。每次真实环境或依赖发生变化后，先查看：

```bash
git status --short
git diff --stat
```

再决定是否提交。不要使用 `git add -f` 将权重或缓存加入仓库。

## 研究阶段

```text
P0 Preparation
  → P1 Infrastructure Validation
  → P2 Baseline Validation
  → P2.5 Local RL Readiness          (complete; tag p2.5-win-complete)
  → P3-0 Linux Migration & On-Policy Calibration   (complete; manifest p3_0_complete.yaml)
  → P3-A On-Policy GRPO Smoke Test                 (complete; E015, incl. resume test)
  → P3-B Short Learning Pilot                      (complete; E017, 30 steps @ n=8)
  → P4 Research Exploration
  → P5 Formal Study
```

P0 完成标准和暂不解决的问题见 [`docs/experiment_protocol.md`](docs/experiment_protocol.md)。

