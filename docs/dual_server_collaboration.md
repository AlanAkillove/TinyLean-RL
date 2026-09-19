# fly90 / fly122 Dual-Server Collaboration Protocol

> 本文件是 TinyLean-RL 双机运行与 Agent 协作的 canonical 规范（normative）。
> 目标读者：任何新接手的 Qoder / Codex / 其他 Agent——**无需旧会话上下文**，只读本文件即可理解并遵守双机规则。
> 摘要与入口见 [`../README.md`](../README.md)；研究状态见 [`p0_status.md`](p0_status.md)、[`research_plan.md`](research_plan.md)。

## 0. Machine Role Table

| Host | GPU | Role | Canonical Git write | Full-FT training |
|---|---|---|---|---|
| fly90 | RTX 3090 24GB | training / controller | yes | yes |
| fly122 | RTX 3080 10GB | evaluation / data worker | no | no |

补充：fly122 可以执行 single optimizer-step FULL-FT SFT memory smoke（仅 feasibility 测量），但不承担 canonical full-FT training。

## 1. Purpose

TinyLean-RL 当前使用两台服务器并行推进研究：

```text
fly90  = training / canonical controller
fly122 = evaluation / data worker
```

设计目标：

- 提高 GPU 利用率，并行 training / evaluation；
- 不复制完整 `runs/`；
- 保持单一 canonical research history（`fly90 / p3-linux`）；
- 避免两个 Agent 同时修改同一个 Git workspace。

## 2. Hardware Inventory

### fly90

```text
hostname/alias: fly90
address: 10.3.25.90
GPU: NVIDIA GeForce RTX 3090
VRAM: 24GB
role: primary training node
```

- 保存完整历史 `runs/`；
- 负责 FULL-FT GRPO 与正式 SFT training；
- 负责 canonical manifests / docs；
- 唯一直接维护 `p3-linux`。

实测备注：fly90 本地 `hostname` 输出为 `fly`（`fly90` 为部署别名）。判断节点身份时必须结合 GPU 型号与地址，不要仅凭 hostname 字符串匹配。

凭据纪律：本文件与仓库任何文档都禁止写入 private key / PAT / password / API token；只能写“SSH deploy key is used”这类事实描述。

### fly122

```text
hostname/alias: fly122
address: 10.3.25.122
GPU: NVIDIA GeForce RTX 3080
VRAM: 10GB
RAM: 61GB
/opt: 当前审计时约 2.6TB free
```

角色：

```text
evaluation / Lean verification / data processing
statistics / checkpoint inference / post-SFT IGR evaluation
```

明确：GPU 是 **RTX 3080 10GB**——不是 RTX 3080 Ti，也不是 12GB。

## 3. Network Topology

fly122 不能从用户校园网直接访问。连接路径：

```text
user / Qoder
   ↓
fly90
   ↓ SSH (fly@10.3.25.122)
fly122
```

fly122 的公网出口经 fly90：

```text
fly122 → local SOCKS 127.0.0.1:1080 → fly90 → Internet
```

（已有 `socks-proxy-90.service`；细节属 fly122 本地运行层配置。）

LAN 实测（近似值，非 SLA）：fly90→fly122 ≈ 98.7 MiB/s；fly122→fly90 ≈ 102.6 MiB/s。按需传 3–4GB checkpoint 属合理操作。

## 4. Qoder Session Topology

两个 Qoder 会话：

```text
Session A: fly90 primary session
Session B: fly122 remote-worker session
```

但 **两个会话都物理启动在 fly90**，默认看到同一个 workspace：

```text
/home/fly/ZJQ/TinyLean-RL   ← fly90 canonical repo
```

因此 Session B：

```text
MUST treat the local fly90 workspace as READ-ONLY.
```

- Session B 禁止在本地 workspace 执行：`git switch/checkout/add/commit/merge/reset/rebase`、文件编辑、训练、evaluation、Docker、systemd 操作；
- 所有 fly122 目标操作必须显式：

```bash
ssh fly@10.3.25.122 '<command>'
```

## 5. Mandatory Host Guard

所有 fly122 重要操作之前必须先确认目标机器：

```bash
hostname
pwd
```

远程模板：

```bash
ssh fly@10.3.25.122 '
set -euo pipefail

echo "HOST=$(hostname)"
cd /home/fly/ZJQ/TinyLean-RL
pwd

...
'
```

如果 host 不是 fly122：立即停止。

同样注意：直接在 Qoder shell 里运行 `nvidia-smi` / `docker ps` / `systemctl --user ...` 看到的是 **fly90** 的状态，不能当作 fly122。

## 6. Git Ownership

canonical branch：`p3-linux`；唯一直接维护者：**fly90 session**。

fly122 使用 `worker/fly122`：

```text
fly122 worker change
    ↓ worker/fly122 commit + push origin/worker/fly122
fly90 review（git show <commit>：文件范围 / 机器特定路径泄漏 / canonical verdict / 训练协议）
    ↓ cherry-pick
tests
    ↓
p3-linux push（仅 fly90）
```

- 默认是 **review + cherry-pick**；
- 禁止 blind merge 整个 worker branch——worker branch 可能包含 host-specific helpers、intermediate commits、worker-only notes，canonical history 必须保持精炼。

### 6.1 V2 branch update（2026-09-19）

- `p3-linux` 现为 **frozen V1 evidence branch**（annotated tag `v1-research-freeze-20260919`；E001–E024 证据冻结，不再接收新实验）。
- `main2` 为 **V2 canonical development branch**（研究计划见 `docs/v2/research_plan.md`）。
- fly90 仍为 canonical controller 与两个分支的唯一直接维护者；fly122 仍为 worker，不重写 canonical history。
- 未来 worker branch 必须从**明确指定的 `main2` SHA** 分叉（handoff 中记录 base SHA），并沿用 review → cherry-pick → test → push 流程。
- artifact 传输与 raw-data policy 不变（§8–§10）。

## 7. Shared fly90 Workspace Safety

fly90 primary session 在执行以下操作前：

```text
checkout / switch / reset / merge / rebase / cherry-pick / clean / 大范围文件修改
```

必须先核验：

```bash
git status --short
git branch -vv
git rev-parse HEAD
```

- 若出现非本人产生的修改、未知 untracked files、HEAD/branch 意外变化：**STOP**；
- 禁止 `git reset --hard`、`git clean -fd`——先确定来源并向用户报告，不要自动清理。

fly122 session 对本地 fly90 workspace：**read-only**（见 §4）。禁止：git switch / add / commit、编辑文件、运行 fly90 实验、改 Docker、改 systemd。

## 8. Code vs Artifact Synchronization

- 代码同步：**只能通过 Git**。禁止 `rsync` 整个 repo 互相覆盖（任何方向）。
- artifact 传输：允许 `rsync` / `scp`。范围：

```text
HF exported checkpoints
fixed-set manifests
result JSON / analysis JSON
small logs
worker 所需数据集
```

- 传输记录：source / destination / 大小；模型 artifact 建议附 sha256 或至少目录完整性校验。

## 9. Never Mirror Full runs/

禁止把 fly90 的 `runs/` 全量同步到 fly122。

原因：

```text
>100GB、无必要、浪费存储、增加 provenance 风险
```

fly122 使用 task-scoped artifact transfer 模式：只传当前任务需要的产物。

## 10. Large Storage on fly122

fly122 的 `/opt` 为大容量磁盘，可作为：

```text
worker checkpoint staging
datasets
evaluation artifacts
```

规则：

- 不要把 `/opt/...` 等机器绝对路径写入 portable project config；
- host-specific path 只存在于本地运行层；
- repo 中用 manifest / relative label 记录逻辑来源。

## 11. Producer / Evaluator Pipeline

```text
              ┌──────────────────────────┐
              │ fly90 / RTX 3090         │
              │ Training Producer        │
              │ Canonical p3-linux       │
              └─────────────┬────────────┘
                            │
                     HF checkpoint
                       manifest
                            │
                            ▼
              ┌──────────────────────────┐
              │ fly122 / RTX 3080 10GB   │
              │ Evaluation Worker        │
              │ Lean / Data / Inference  │
              └─────────────┬────────────┘
                            │
                    result artifacts
                            │
                            ▼
              ┌──────────────────────────┐
              │ fly90 canonical analysis │
              │ docs / verdict / push    │
              └──────────────────────────┘
```

## 12. Current Default Work Split

fly90（默认负责）：

```text
Seed3 / full-FT RL
M2 verified SFT training
post-SFT GRPO
canonical analysis
canonical manifests / docs
```

fly122（默认负责）：

```text
M1 final-holdout evaluation（部分模型）
MiniF2F evaluation（部分模型）
SFT corpus filtering
Lean-heavy verification
post-SFT IGR measurement
statistics / artifact validation
```

## 13. Tasks fly122 Must NOT Run

```text
FULL-FT GRPO 60-step training
Seed3
step100
正式 M1 training
```

原因：10GB VRAM 不足以复现 fly90 的 full-FT recipe。

不得为了让任务在 fly122 上运行而引入：

```text
LoRA
CPU offload
different optimizer
different batch recipe
```

——这些会造成研究 confound。

例外：single optimizer-step FULL-FT SFT memory smoke（仅 feasibility measurement）。若 OOM：记录并停止。

## 14. Experiment Number Ownership

```text
E022 = Seed3
E023 = M1 final holdout
E024 = MiniF2F
E025 = M2 verified-SFT intervention
E026 = post-SFT GRPO trainability
```

同一实验在两机并行（如 E023）时，不得拆成 E023a / E023b 作为主编号；用 sub-run metadata 区分：

```yaml
experiment: E023
host: fly90 / fly122
model: ...
```

最终 canonical manifest 在 fly90 合并。

V2 编号（2026-09-19 起，同日命名修订）：所有新实验使用 Track-specific namespace —— Track A `V2-A###`（RL prover，fly90）、Track B `V2-B###`（adaptive allocator，fly122）、Track C `V2-C###`（joint evaluation，fly90）；既有 `E###` 编号永不复用。改名前已运行的 raw 编号 `V2-E001`（fly122 Budget Semantics Audit）**不改写、不重命名**，在 `experiments/manifests/v2/registry.yaml` 中永久登记 canonical alias **V2-B001**；`V2-E###` 对新的正式实验关闭。未执行的 V1 预留编号（E025/E026）不再使用。V2 manifests 位于 `experiments/manifests/v2/`（参见其 README、registry 与模板）；定理角色划分（B-train / B-validation / B-test / A-selection / C-joint-holdout + B1-audit-reserved）见 `theorem_role_registry.json`（coordination 2026-09-19）。

## 15. Artifact Provenance

双机产生的正式 artifact 必须记录：

```text
host
GPU
git revision
model / checkpoint revision
dataset / fixed-set revision
verifier image
sampling config
created_at
```

示例：

```yaml
host: fly122
gpu: NVIDIA GeForce RTX 3080 10GB
git_revision: ...
```

用途：论文 compute table 可重建。

## 16. Final Holdout Collaboration

默认分工（无 Seed3）：

```text
fly90 : seed1-step60
fly122: theta0, seed2-step60
```

Seed3 完成时：

```text
fly90 : seed1-step60, seed3-step60
fly122: theta0, seed2-step60
```

规则：

- 所有 worker result 回到 fly90；
- **只在 fly90** 运行 canonical 分析（`scripts/holdout_multiseed_analyze.py`），产生 canonical E023 verdict。

## 17. MiniF2F Collaboration

默认分工：

```text
fly90 : primary seed1 RL checkpoint
fly122: theta0 baseline
```

禁止 result-dependent checkpoint selection（不得从 seed2/seed3 中按结果挑“最好”的 checkpoint）。

## 18. M2 SFT Pipeline

```text
fly90 : train SFT-N → export checkpoint → transfer
fly122: mechanism-set 64×8 evaluation → IGR(N)
fly90 : apply pre-registered gate
```

支持流水并行：

```text
fly90 training next permitted rung
while
fly122 evaluates previous rung
```

前提：上一 rung 的下一步已按预注册规则允许。不得为提高 GPU 利用率绕过 gate。

## 19. Lean Server Ownership

两台机器各自运行自己的 `tinylean-rl-lean-server`。禁止跨机共用 verifier endpoint：

```text
fly90 experiments  → fly90 localhost:8000
fly122 experiments → fly122 localhost:8000
```

不要把 `localhost:8000` 理解成同一台 server。

## 20. systemd / Docker Ownership

- fly90 Qoder：只管理 fly90；
- fly122 Qoder：经 SSH 只管理 fly122；
- 禁止替对方执行 `docker restart` / `systemctl stop/start` / kill GPU processes。

## 21. Long-Running Jobs

两台机器的长任务都应使用：

```text
systemd --user
```

或项目认可的持久运行机制。不得依赖裸 SSH foreground process。

所有 retry 必须 **bounded**——禁止无限 retry。

## 22. Failure Handling

worker 失败时保留：

```text
partial artifact
logs
exit status
host state
GPU state
```

不得“修几个参数重新跑”而不记录。变更必须分类记录：

- verifier concurrency / infra batch / worker count → **infrastructure variation**；
- model / sampling / reward / training config → **experiment protocol variation**，不允许静默修改。

## 23. Canonical Source of Truth

最终项目真相只来自：

```text
fly90 / p3-linux   (frozen V1 evidence; tag v1-research-freeze-20260919)
fly90 / main2      (V2 canonical development, since 2026-09-19)
```

不是：

```text
worker/fly122
Qoder chat memory
raw ignored artifact
本地临时日志
```

worker result 必须经过 review → integration → canonicalization，才能进入正式研究结论。

## 24. New Agent Checklist

新 Agent 接手任何机器前必须先确定：

1. 我现在物理 shell 在哪台机器？
2. 我的目标节点是 fly90 还是 fly122？
3. 当前 repo 是 canonical 还是 worker？
4. 当前 branch 是什么？
5. `git status` 是否 clean？
6. 当前 HEAD 与 `origin/p3-linux` 是否一致？
7. `hostname` 输出什么？（注意：fly90 实测输出为 `fly`，结合 GPU 与地址判断）
8. GPU 是否有其他任务？
9. 当前任务属于 training 还是 evaluation？
10. 我是否有权修改 canonical `p3-linux`？

如果无法回答：

```text
STOP before modifying anything.
```
