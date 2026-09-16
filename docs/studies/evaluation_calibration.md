# Preliminary Study — Evaluation Calibration & Reward Reliability（P2 预实验）

> **定位**：TinyLean-RL 的 P2 阶段预实验（E001–E008）的完整记录与结论。主线研究计划见 [`../research_plan.md`](../research_plan.md)。
> 本文件是 **side-study 材料**：若未来单独成文（budget-aware evaluation 技术报告/短文），从这里取材；当前不作为 TinyLean-RL 的主线目标。
> 三类预实验：A 评估校准（budget→通过率）、B 奖励可靠性（verifier→可靠 binary reward）、C 奖励信息量（组率→GRPO 是否可行）。

## 一、研究问题（限定在评估协议层面）

- RQ-A（校准）：token 预算如何影响观察到的通过率？P3 应选多大的 response budget？
- RQ-B（可靠性）：验证器侧行为（崩溃/并发/环境）如何污染 reward？如何检测与缓解？
- RQ-C（信息量）：当前 checkpoint 的 reward 分组是否足以支撑 GRPO？

## 二、主要发现（含 E007/E008 checkpoint 数据）

### 2.1 预算与长度——对 P3 的直接输出

| 指标 | Distill (E007) | RL (E008) |
|---|---|---|
| successful length median | 1562 | 1482 |
| successful length P90 / P95 | 2695 / 3176 | 2949 / 3038 |
| successful length max | 4041 | 4091 |
| failed length median / P90 | 4096 / 4096 | 4096 / 4096 |
| truncated_at_4096 | 77/128 | 63/128 |

- **4096 覆盖成功分布 P95 后仍有约 25% 余量**——这是 P3 第一版选 `max_response=4096` 的定量依据。
- **截断偏置存在**（2166/2780 tokens 的成功候选在 2048 下不可能完成），但 **2048→4096 的 +0.125 Pass@1 提升来自独立随机采样，不能全部归因于 budget**：同配置重跑（E005 vs E007 前 8 题）本身就有 14→10 的波动。若要量化 budget 因果效应需 paired generation（固定采样路径对比），当前不做。
- 失败候选大量撞 4096（60%/49%）不构成“需要更大预算”的证据——模型可能在错误推理轨迹上持续生成；除非有证据表明成功证明所需长度超过 4096（当前没有，P95≈3000），不应继续加大。

### 2.2 组率与学习信号——对 P3 的直接输出

| 指标（K=4） | Distill | RL |
|---|---|---|
| IGR = mixed（0<Σr<K） | **28.1%** | **31.3%** |
| all-zero（Σr=0） | 53.1% | 43.8% |
| all-one（Σr=K） | 18.8% | 25.0% |

- **0.6B checkpoint 未陷入“reward 几乎全零”**：约 28–31% 的定理组处于 mixed 状态，具备 GRPO 所需的组内优势信号。这是“0.6B 值得启动 P3”的直接证据。

### 2.3 验证器可靠性（Reward Reliability）

- `verify_status` 分类：E007 verified 43 / lean_error 84 / verifier_error 1；E008 verified 55 / lean_error 71 / verifier_error 2。
- **`native_decide` 并非必败**：E007 12 个含 `native_decide` 的候选中有 **9 个通过验证**、1 个触发 REPL 崩溃；E008 为 11 / 7 / 2。**不做字符串禁用**（那会改变证明语言语义）；崩溃候选单独记 `verify_status=verifier_error` 并计 0，不静默合并。
- 已实现容错：15 s 退避重试 → 逐候选降级 → `.partial.json` 增量保存 → `trust_env=False`；冷 REPL 并发成本与代理劫持的完整记录见 [`experiment_log.md`](../experiment_log.md) 的“并发吞吐复测”与 E005 过程事件。
- 验证时间占比：E007 生成 18,386 s / 验证 290 s ≈ **1.6%**——verifier 不是瓶颈，P3 第一版用串行验证即可；不做 scheduler/pool 优化。

### 2.4 模型对比与统计可判定性

- E007 (Distill 43/128) vs E008 (RL 55/128)：候选级 +9.4 pp（z≈1.55，n.s.）；定理级 15/32 vs 18/32（翻转 RL-only {19,25,27,31} vs Distill-only {9}，McNemar p≈0.375）。
- 小样本三证据：种子稳定性（同配置重跑 14→10）；子集反转（8 题上 Distill 领先、32 题上 RL 领先）；功效分析（独立检验需 ~5,500 定理/臂才能以 80% 功效检出 +2.45 pp——官方差距接近可判定性边界）。
- **统计方法修正**（外部审查意见，采纳）：
  - ❌ candidate-level “sample_i vs sample_i” 配对**不成立**——两个模型的第 i 个 sample 之间没有天然配对关系，不能因为编号相同就做配对检验。
  - ✅ 正确的配对单位是 **theorem**：对每道题取 $\hat p_i=c_i/K$，构造 $d_i=\hat p_i^{R}-\hat p_i^{D}$，对 $\{d_i\}$ 做 theorem-level bootstrap；或直接比较定理级 solved indicator（McNemar / 配对比例检验）。
  - 上文 McNemar 检验即为 theorem-level 形式。

## 三、文献定位

| 文献 | 内容 | 与本预实验的关系 |
|---|---|---|
| Kimina-Prover Preview（arXiv:2504.11354） | 72B；pass@1 52.9%、pass@8192 80.7%；官方评估协议 | 对照口径来源；官方预算远高于消费级硬件 |
| Kimina-Prover-RL（Numina/Kimi，2025-07） | 0.6B RL：71.30% Pass@32（base 68.85%，+2.45 pp） | 本项目的目标 pair 与官方差异量级 |
| EconProver（arXiv:2509.12603，ACL 2026） | 测试时缩放策略的成本-效率对比 | 策略层成本对比；本预实验在测量层 |
| Budget-Aware Evaluation of LLM Reasoning Strategies（EMNLP 2024） | 通用推理策略的预算感知评估 | 概念先例 |
| miniF2F-Lean Revisited（NeurIPS 2025） | miniF2F 形式化缺陷 | 数据集噪声边界 |
| Chen et al. 2021（Codex） | 无偏 Pass@k 估计 | 使用的估计量 |

## 四、若独立成文的提纲（保留，暂不推进）

1. Introduction：低预算硬件上的证明器评估；RQ-A/B/C。
2. Related Work：上表。
3. Methodology：本地 Kimina 链路与容错机制。
4. Experiments：E003–E008；预算-分数、长度分布、组率、native_decide。
5. Statistical power：本文档 §2.4 + 功效分析。
6. Recommendations：预算选择（≥ successful P95）、`verify_status` 分类记录、<64 定理不给排名结论。
7. Limitations：子集规模、单种子、本地 Lean `v4.15.0`。

## 五、边界（不再扩大的部分）

- **不再跑** E009（8192 第三点）、16×8、244×32——这些回答的是“精确复现官方排名”，不是当前主线需要的信息。
- 不做 candidate-level 配对检验；需要 budget 因果结论时用 paired generation（固定随机种子对），作为未来独立课题。
- 本预实验的输出已经足够：**P3 配置（4096、串行验证、IGR 基线）** 与 **reward 可靠性契约**。
