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
  - P3 第一版配置依据：`max_response=4096`（成功长度 P95 ≈ 3,000，余量 ~25%）、串行验证（验证时间占比 ~1.6%，非瓶颈）。
- **P2 → P3 迁移 gate**（在 Linux 3090 主机上执行，不重跑完整 E003–E008）：
  1. 验证器 positive/negative gate；
  2. model→Lean smoke（Kimina Distill 0.6B）；
  3. 8 定理 × 4 样本量级校验（与 Windows 结果同量级）；
  4. throughput / peak VRAM 基线；
  5. 记录 `nvidia-smi`、driver、系统、RAM、CPU。

## 三、P3 计划（RL pilot，分两步）

### P3-A：RL plumbing smoke test
- 目标：证明全链路可运行：`rollout → Lean reward → GRPO loss → optimizer step → checkpoint → resume`。
- 配置起点：Kimina-Distill-0.6B、Kimina Promptset、`n=4`、`max_response=4096`、小 train batch、**2–5 optimizer steps**。
- 成功判据：loss finite、gradient finite、存在 positive reward、存在 mixed group、checkpoint 保存 / 重载 / 续训成功。
- **到点即停**，不追求任何 benchmark 提升。

### P3-B：短 RL learning pilot
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
