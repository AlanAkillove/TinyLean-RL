# TinyLean-RL

面向亚十亿参数 Lean4 定理证明器的可验证强化学习研究项目。

当前阶段：**P2 Baseline Validation Complete → P2.5 Local RL Readiness**（本地 RL 前置验证进行中：Promptset 训练分布诊断、GRPO 参考实现与 rehearsal、LoRA 显存探针、P3 config 审计与云部署命令链；下一阶段为 Linux 3090 迁移 gate 与 P3-A RL plumbing smoke）。

- 研究主线（RL for sub-billion Lean provers）：[`docs/research_plan.md`](docs/research_plan.md)
- P2 预实验结论（评估校准 / 奖励可靠性 / 奖励信息量）：[`docs/studies/evaluation_calibration.md`](docs/studies/evaluation_calibration.md)
- 逐次实验记录（E001–E008）：[`docs/experiment_log.md`](docs/experiment_log.md)

本项目遵循“阶段门控、证据驱动”的研究路线：先建立可复现环境和最小端到端链路，再根据预实验结果决定 RL、cold-start、scaling 或 frontier sampling 的后续分支。

## 当前冻结的研究边界

- 第一 baseline：`AI-MO/Kimina-Prover-Distill-0.6B` → Kimina-Prover-RL recipe。
- Lean verifier：`projectnumina/kimina-lean-server:2.0.0`。
- 第一标准 benchmark：MiniF2F test（evaluation-only）。
- 当前只准备 Kimina 0.6B Distill、Kimina 0.6B RL 和 Qwen3 0.6B Base 的资源声明。
- 当前不进行正式 RL 训练，不实现 curriculum/frontier sampling/self-training，不开始 500M/360M 实验。

## 仓库结构

```text
TinyLean-RL/
├── configs/                  # 配置（当前仅保留 P0 占位）
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
  → P2.5 Local RL Readiness
  → P3 RL Pilot
  → P4 Research Exploration
  → P5 Formal Study
```

P0 完成标准和暂不解决的问题见 [`docs/experiment_protocol.md`](docs/experiment_protocol.md)。

