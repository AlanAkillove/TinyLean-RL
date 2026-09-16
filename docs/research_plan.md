# TinyLean-RL 研究计划（主线）

> 定位：**亚十亿参数 Lean4 定理证明器的强化学习**（RL for sub-billion Lean provers）。
> P2 预实验结果（评估校准 / 奖励可靠性 / 奖励信息量）见 [`studies/evaluation_calibration.md`](studies/evaluation_calibration.md)——它们是 P3 的输入，不是本项目的主线目标。
> 逐次实验过程见 [`experiment_log.md`](experiment_log.md)；当前快照见 [`p0_status.md`](p0_status.md)。

## 一、项目定位与主问题

TinyLean-RL 研究「亚十亿参数 Lean4 证明器能否通过 RL 获得可验证的推理能力提升」，并定位规模与起点条件的边界。三个主问题：

1. **M1 — Compute-controlled 0.6B RL reproduction**：在受限算力（单卡 RTX 3090 24 GB 级）下复现 Kimina-Prover-RL-0.6B 的 RL 收益路径（对照其蒸馏起点；不追求官方绝对数字）。
2. **M2 — Cold-start trainability**：从更弱起点（如 `Qwen/Qwen3-0.6B-Base`，无蒸馏先验）启动 RL 的可训练性边界。
3. **M3 — Model-size frontier**：500M / 360M 规模下 RL 学习信号是否依然充分（未来比较 `IGR_{600M}` vs `IGR_{500M}` vs `IGR_{360M}`）。

核心度量：**学习动力学曲线** `IGR_t / Z_t / O_t`（定义见第五节），而非单一 benchmark 分数。

## 二、阶段状态（2026-09-16 checkpoint）

- **P0 完成**：环境、依赖 pin、验证链路实现与全部本地检查。
- **P1 完成**：验证器正/负 gate、model→Lean 端到端 smoke。
- **P2 checkpoint 完成**（E001–E008，评估协议校准与奖励可靠性预实验）：
  - Distill 0.6B：43/128 verified（32×4@4096），IGR = 28.1%；RL 0.6B：55/128，IGR = 31.3%；
  - **0.6B 起点具备充分 GRPO 学习信号**（未陷入 reward 全零状态）；
  - P3 初始输入（P3-0 校准后冻结；`max_response=4096` 为初始 rollout 预算假设）：成功长度 P95 ≈ 3,000、串行验证（验证时间占比 ~1.6%，非瓶颈）。
- **P2.5 Local RL Readiness（2026-09-16 启动，Windows RTX 4060 8 GB）**：P3 之前的本地 RL 前置验证，原则为 *test the Kimina logic, don't replace it*（不引入 TRL、不做另一套 GRPO）。范围：Promptset 训练分布诊断（32×4@4096 cached rollout）、GRPO advantage/loss 参考实现（逐式移植 pinned commit `e16b6058`）与 rehearsal、0.6B LoRA 单步显存探针、P3 config 审计（[`p3_config_audit.md`](p3_config_audit.md)）与部署命令链（Linux 目标）。
  - **停止条件**（全部满足即 complete，不再膨胀）：① Promptset 上观察到 positive/mixed reward 组；② cached rollout → GRPO advantage/loss 算通；③ backward 显存探针给出结论（成功或 OOM 均算回答）；④ `docs/p3_config_audit.md` 完成；⑤ Linux 部署启动命令链（`env.sh → compose → doctor.sh → run_p3_smoke.sh`）写好；⑥ reward contract 写回本文档与审计文档。（**2026-09-16：六条全部满足**，证据与产物见 [`studies/rl_readiness.md`](studies/rl_readiness.md) 与 `experiments/manifests/p2_5_complete.yaml`。）
  - 结果汇总见 [`studies/rl_readiness.md`](studies/rl_readiness.md)；证据落档 `experiments/manifests/p2_5_complete.yaml`。
- **P3-0 完成（2026-09-17，服务器 RTX 3090 24 GB）— Linux Migration & On-Policy Calibration**（不重跑完整 E003–E011）：
  1. Linux 环境 gate：`bash scripts/doctor.sh`（Linux / Docker / Lean server / VRAM ≥ 24 GB / VERL import）；
  2. 验证器 positive/negative gate + model→Lean smoke（Kimina Distill 0.6B）；
  3. Promptset rollout calibration（temp 1.0 / top_p 1.0 / n=4 / max_response 4096）；
  4. full-parameter 单步显存可行性探针（LoRA 仅当 full FT 不可行时作为 fallback）；
  5. 冻结真正的 P3-A 配置；记录环境基线（`nvidia-smi`、driver、系统、RAM、CPU）。
  - 结果（证据 `experiments/manifests/p3_0_complete.yaml`、E012–E014）：迁移 gate 全绿（40+19 tests、doctor 23 pass、reward 契约实测）；Promptset IGR = **0.09375**（n=4 retained，marginal ≥0.05）、截断 66.4%、成功证明未撞 4096 上限；**FULL-FT 可行**（单步 136.4 s，peak reserved 20.5 GB）；下一步 P3-A smoke（≤2–5 steps，配置已冻结）。
  - 范围、未决问题与命令序列见 [`p3_linux_handoff.md`](p3_linux_handoff.md)。

## 三、P3 计划（P3-0 → P3-A → P3-B）

### P3-0：Linux Migration & On-Policy Calibration
- 目标：环境 gate → Lean 正/负 gate 与 model→Lean smoke → Promptset rollout calibration（temp 1.0 / top_p 1.0 / n=4 / max 4096，重测 IGR / Z / O / 截断 / tokens/s / VRAM）→ full-parameter 单步显存可行性探针（LoRA 仅 fallback）→ 冻结 P3-A 配置。
- 校准判据：若 temp 1.0 下 IGR 仍过低（≲5%），恢复顺序 = ① n=8（官方 baseline 组件）② multiturn=true + `multiturn_sampling_rate=0.5`（需同步放大 `max_prompt_length`）；原则：**先恢复被 reduced-compute baseline 删除的官方机制，再引入自定义算法改进**。
- 到点即停；不重跑 E003–E011 / MiniF2F 32×4 / Windows 诊断。

### P3-A：On-Policy GRPO smoke test（RL plumbing smoke test）
- 目标：证明全链路可运行：`rollout → Lean reward → GRPO loss → optimizer step → checkpoint → resume`。
- 配置起点（由 P3-0 冻结）：Kimina-Distill-0.6B、Kimina Promptset、`n=4`（reduced-compute 起始假设；官方 n=8 是 IGR 过低时的第一恢复项）、`max_response=4096`（P3-0 初始预算假设）、小 train batch、**2–5 optimizer steps**。
- 成功判据：loss finite、gradient finite、存在 positive reward、存在 mixed group、**checkpoint 成功保存**、**成功重载 / 续训**。
- **到点即停**，不追求任何 benchmark 提升。

### P3-B：Short learning pilot（短 RL 学习 pilot）
- 前置条件：P3-A 成功。
- 目标：运行数十个 update，观察 `IGR_t / Z_t / O_t / response length / entropy / KL` 的合理变化（预期形态：`Z_t ↓`，`O_t ↑`，`IGR_t` 先稳后降）。
- 这是回答"小 compute 下 0.6B RL 能否稳定学习"的真正实验，也是 M2/M3 的基线。

## 四、奖励契约（Reward semantics，P3 前冻结）

| 情形 | reward | 记录要求 |
|---|---|---|
| Lean 验证通过 | 1 | `verify_status = verified` |
| Lean 拒绝 | 0 | `verify_status = lean_error` |
| 候选导致的确定性 verifier 崩溃（如 `native_decide` 大计算） | 0 | **`verify_status = verifier_error`，单独统计** |

- 崩溃候选不得静默合并进普通失败；worker 需可恢复、不拖累同批其他候选。
- **不做字符串禁用**（`native_decide` 多数情况正常：E007 的 12 个中含 `native_decide` 候选中 9 个通过验证）。
- 若 RL 出现对 verifier 的利用性攻击，再单独立项 mitigation。

**P2.5 W4 审计补充（依据 [`p3_config_audit.md`](p3_config_audit.md) §6/§7，2026-09-16）**：

- **format gating**：官方最终 score 为 `score = proof_rw × format_rw`（`reward/reward.py` L153），格式不合法直接 0——P3-A 沿用；本地 profile 的 `rewards.jsonl.format_ok` 即该维度对照物。
- **并发策略**：官方 `formal_rewards` 用 `max_workers=40`（L81-85），本地实测冷 REPL 并发会超时/丢结果；P3 先预热 Lean server 并限制 `reward_kwargs` 并发（默认串行小批），以 `experiments/results/lean_server_warmup.json` 的 `recommended_concurrency` 为准（W6 prewarm + doctor 检查延时）。
- **单轮声明**：官方 0.6B 开启 multiturn（数据侧第二轮回炉，`dataset.py` + `reward/error_fixing.py`），P3-A 起步**关闭**（`data.multiturn=False`），训练分布偏移记为隐藏变量（审计 §7 差异清单）；P3-0 校准若 IGR 过低，恢复顺序 = n=8 → multiturn。
- `verifier_error` 候选 → reward 0 且单独统计；worker 需可恢复、不拖累同批（沿用上表）。
- P3 起统一用严格验证口径（含 sorry 检查；`evaluate_model.py` 旧口径仅用于 E001–E008 headline 的 re-check 解释）。

## 五、关键指标定义

- `IGR_t = Pr(0 < Σ_{k=1..K} r_k < K)`（组内优势信号率，GRPO 有效梯度来源）
- `Z_t = Pr(Σ_{k=1..K} r_k = 0)`（all-zero 组率）
- `O_t = Pr(Σ_{k=1..K} r_k = K)`（all-one 组率）
- 同时记录：response length 分布、entropy、KL、format/pipeline 失败率。

## 六、Preliminary studies（已完成，作为 P3 输入；不再扩大）

| Study | 内容 | 输出位置 |
|---|---|---|
| A 评估校准 | budget → 通过率、成功长度分布 | [`studies/evaluation_calibration.md`](studies/evaluation_calibration.md) §2.1 |
| B 奖励可靠性 | verifier 行为风险与容错机制 | 同文档 §2.3 |
| C 奖励信息量 | 组率与学习信号 | 同文档 §2.2 |

边界：不跑 E009（8192）/ 16×8 / 244×32；不做 candidate-level 假配对；budget 因果结论需 paired generation（列为未来独立课题）。

## 七、版本与依赖

固定版本（模型、数据、服务端）见 [`reproduction.md`](reproduction.md)：Kimina-Prover-RL commit、Distill/RL 0.6B、Promptset、MiniF2F artifact、Kimina Lean Server `2.0.0`。
