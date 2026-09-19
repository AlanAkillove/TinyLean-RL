# TinyLean-RL

> **Compute-efficient reinforcement learning and adaptive inference for sub-billion-parameter Lean 4 theorem provers.**

面向亚十亿参数 Lean 4 定理证明器的可验证强化学习（RLVR）与自适应推理研究项目；全部工作在单卡消费级 GPU（RTX 3090 24GB / 3080 10GB）算力约束下完成并可复现。

**Research question**

> How can a sub-billion-parameter Lean prover solve more theorems under limited compute?

- **为什么做小模型**：亚十亿参数模型让训练与推理的完整链路可以在一张消费级 GPU 上跑通；「算力受限下的学习信号」与「推理预算分配」这两个问题因此可以被干净地测量与复现。
- **Training-time（verifier-based RL / RLVR）**：以 Lean 验证器（Kimina Lean Server）作为可验证奖励来源的 GRPO 训练链路。
- **Inference-time（adaptive test-time compute）**：按定理分配生成 token 预算（compute-response curve / oracle headroom / 轻量 allocator）——V2 新方向。

## 当前状态（2026-09-19）

- **V1 证据已冻结**：annotated tag `v1-research-freeze-20260919`（branch `p3-linux`）。E001–E024 为历史证据，不删除、不改写、不重编号。
  - V1 严谨总结（不重复原始数字，详见 [V1 证据索引](docs/v2/legacy_evidence.md)）：RL 训练链路 / trainability 已建立；positive / mixed reward groups 稳定可观察；多 seed 训练动力学可复现；60-step 低算力配方下 held-out 能力增益小且 seed-sensitive / inconclusive；**未宣称稳定 benchmark improvement**。
  - E024 MiniF2F：**PAUSED / ABORTED DUE TO VERIFIER INFRASTRUCTURE INCIDENT; no E024 result claim.**（见 [docs/e024_status.md](docs/e024_status.md)）
- **V2 仓库准备完成（V2-0）；尚未运行任何 V2 实验。** **No V2 adaptive-allocation experiment has been run yet.**

## 文档地图

| 文档 | 内容 |
| --- | --- |
| [docs/v2/research_plan.md](docs/v2/research_plan.md) | V2 主问题、RQ1–RQ5、阶段计划 V2-0 … V2-X、护栏 |
| [docs/v2/experiment_protocol.md](docs/v2/experiment_protocol.md) | 冻结协议：定理级数据切分、决策模型指标、分配指标、主图 |
| [docs/v2/data_contract.md](docs/v2/data_contract.md) | `budget_response` 数据契约（prefix vs direct-budget；不预设单调性） |
| [docs/v2/legacy_evidence.md](docs/v2/legacy_evidence.md) | V1 证据索引（指向 canonical 文档 / manifests，不重述数字） |
| [docs/experiment_log.md](docs/experiment_log.md) · [docs/research_plan.md](docs/research_plan.md) | V1 全量记录（已标注冻结） |
| `experiments/manifests/v2/` | V2 manifests（命名空间 `V2-E###`；含模板与规则） |

## 复现性与协作

- 环境与依赖 pin：[docs/environment.md](docs/environment.md) · [docs/reproduction.md](docs/reproduction.md)
- 双机架构：fly90（RTX 3090）训练 + canonical controller；fly122（RTX 3080）evaluation / data worker — 规范见 [docs/dual_server_collaboration.md](docs/dual_server_collaboration.md)
- 种子控制审计：[docs/seed_control_audit.md](docs/seed_control_audit.md)
- 验证器：`projectnumina/kimina-lean-server:2.0.0`（strict validity：无 error 级消息且无 `sorry`）
- 大文件政策：模型权重、数据集、缓存、实验原始输出不进 Git；只通过 manifest + SHA256 追踪

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

Windows 开发机可用 `scripts/env.ps1`；Lean server 与训练链路仍以 Linux + NVIDIA Docker 为目标环境。

## 仓库结构

```text
TinyLean-RL/
├── configs/                  # 冻结训练配置
├── data/manifests/           # 数据源与版本声明
├── docs/                     # 研究文档（V2 文档在 docs/v2/）
├── experiments/              # manifests（含 manifests/v2/）与结果索引，不放大文件
├── infra/lean-server/        # Kimina Lean Server Docker Compose
├── models/manifests/         # 模型源与 revision 声明
├── scripts/                  # 评估 / 数据 / 环境脚本
├── src/tinylean_rl/          # 项目封装（统计 / 验证 / 推理 helpers）
├── tests/                    # 单元测试
└── third_party/              # 固定版本上游代码（pinned submodule）
```

## 分支

- `p3-linux`：**frozen V1 evidence**（tag `v1-research-freeze-20260919`）
- `main2`：**V2 canonical development**
- `main` / `win`：历史分支，保留不改写
