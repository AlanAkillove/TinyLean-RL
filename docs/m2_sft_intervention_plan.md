# M2 verified cold-start SFT intervention — 预注册计划（草案，执行前冻结）

> 背景：E021 建立了极干净的 baseline——`Qwen3-0.6B-Base` 在冻结单轮协议下
> IGR=0.0156、候选验证率较 Distill 低约 128×、5 步 GRPO smoke 全零梯度
> （reward-dead 边界）。下一步最有价值的问题不再是“Base 能否直接 RL”，而是：
>
> **需要多少 verified cold-start 监督，才能把 0.6B Base 推进 RL-trainable regime？**

## 资产与就绪状态（2026-09-18）

- NuminaMath-LEAN 原始数据：已下载（`data/raw/numinamath_lean/`，pin
  `51fa67f1f647ae1ecd81eef9f19306aa8a7b3a94`，74.8 MiB）。
- 数据集构建管道：`scripts/build_verified_sft.py`（提取 Lean 代码块、格式化为
  chat 协议、固定 seed 分割、导出 train/val parquet + prep 报告）。
  **注意**：构建过程会调用 Lean server，须在无其他 GPU/Lean 任务时执行。
- SFT trainer：pinned VERL `verl/trainer/fsdp_sft_trainer.py` + `config/sft_trainer.yaml`
  （FULL-FT，与 M1 训练模式一致；**不使用 LoRA**，与 M1 冻结决策一致）。
- 评估管道（IGR 复测）：`p3c_checkpoint_eval.py --fixed-set igr_mechanism_set.json`
  （64×8、T=1.0、同种子表；单次约 25–40 分钟）。

## 设计（执行前冻结，不得按结果更改）

1. **语料净化（在 W5 管道语义之上）**：把 SFT 语料中与以下集合的 statement 重叠项
   **过滤掉**（normalized formal statement 匹配）：
   - `igr_mechanism_set.json`（诊断集）
   - `p3c_fixed_set.json`（开发集）
   - `m1_final_holdout.json`（如已封存）
   输出 `experiments/results/m2_sft_corpus_filter.json` 记录各集合的重叠计数与过滤
   后的语料规模。若某集合重叠 > 0 而未过滤，禁止进入下一步。
2. **SFT 阶梯（“需要多少监督”）**：从固定 seed 洗牌的语料中取前缀滚动放大：
   - N ∈ **{512, 2048, 8192}** 样本（2 epochs、同一优化器/lr 组；若 8192 仍不达标
     可加 32768，需另行记录）。
   - 每个 N 训练完成后：在机制集上复测 IGR（64×8、T=1.0、同种子表）→ 得
     `IGR(N)` 曲线。
   - **停止条件（早停向上）**：首个 IGR ≥ 0.05（＝进入 WEAK 以上）的 N 即候选；
     若 IGR(N) 全程 < 0.01 → 记录 “small verified SFT 不够”（同样是有效结论）。
3. **GRPO trainability 验证（在首个 IGR≥0.05 的 N 上）**：
   - 5 步 GRPO smoke（M1 冻结 recipe）→ 判据：出现 mixed 组、非零 advantage、
     有限非零梯度、checkpoint 保存。
   - 通过则最多 **10 步 exploratory pilot**（M2-C 规则），完成后停止；**不自动
     30/60 步**。
4. **单 optimizer-step SFT 显存 smoke（先行）**：在阶梯开始前，用最小 batch 验证
   fsdp_sft_trainer 配置可跑（含 backward），记录 peak reserved；失败则先修配置
   而非改模型。
5. **记录纪律**：每一步独立提交（`feat: add M2 SFT configs` /
   `exp: record M2 SFT ladder step N` / ...）；不因中间结果选择 seed 或数据；
   语料与配置哈希写入 prep artifact。

## 成本估计（3090 单卡）

| 阶段 | 估计 |
|---|---|
| 语料构建（含 Lean 过滤） | 0.5–1 h（CPU+Lean） |
| 单步 SFT 显存 smoke | ~10 min |
| SFT 阶梯 3 档 | 0.5 / 1 / 3 h（随 N 增长；2 epochs） |
| IGR 复测 ×3 | ~1.5 h |
| GRPO smoke（+可选 10 步） | 0.5–1.5 h |
| **合计** | **约 5–8 h**（可在下一窗口执行） |

## 边界与禁令（沿用总协议）

- 不修改 GRPO/reward/采样协议；SFT 使用与 M1 相同的 prompt 协议与严格验证口径。
- 不把 SFT 语料当作 confirmatory 评估集；只作干预手段。
- 外部 GPU 任务出现即暂停（同 2026-09-18 E022 处置）。
- 本干预的最强提法上限：*“a small amount of verified cold-start supervision can
  move a tiny prover out of the verifier-RL reward-dead regime (under this
  protocol)”*——不得泛化到所有 0.6B 模型或所有协议。
