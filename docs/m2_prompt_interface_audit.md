# M2 prompt-interface audit：Kimina-Distill-0.6B vs Qwen3-0.6B-Base

> 目的：在跑 M2 cold-start RL-signal 诊断（E021）前，确认弱起点模型与 Distill
> 基线能否使用**完全相同**的 prompt 协议；记录一切接口偏差。审计日期 2026-09-17。

## 模型与 tokenizer 对比（均为本地 pinned 权重）

| 项目 | Kimina-Distill-0.6B | Qwen3-0.6B-Base |
|---|---|---|
| HF 权重 | `332e8a5259d1bdfda19d7c7f339f30804813cd3a` | `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`（2026-09-17 pin main） |
| architecture | Qwen3ForCausalLM | Qwen3ForCausalLM |
| hidden / heads / layers | 1024 / 16 / 28 | 1024 / 16 / 28 |
| tokenizer 类 | Qwen2TokenizerFast | Qwen2TokenizerFast |
| vocab size | 151,643 | 151,643（同词表） |
| bos_token_id | 151,643 | 151,643 |
| **eos_token_id（config）** | **151,645（`<|im_end|>`）** | **151,643（`<|endoftext|>`）** |
| pad_token_id | None | None |
| chat_template | 有（4,168 chars） | 有（4,116 chars，细节差异：content-string 守卫、reasoning_content 分支） |

## 渲染等价性（决定性检查）

用两个 tokenizer 分别对**同一组 messages**（`SYSTEM_PROMPT` + `USER_TEMPLATE`，即
evaluator 的 `build_prompt_text` 输入）应用 `apply_chat_template(add_generation_prompt=True)`：

```text
rendered prompts identical: True   # 逐字节一致
tail: '...\n```\n<|im_end|>\n<|im_start|>assistant\n'
```

→ **同一 prompt 协议合法**：Base 自带 chat template 且与 Distill 渲染结果完全一致
（不存在“偷偷设计更有利 prompt”的问题）。

## 记录的接口偏差（唯一一项）

- **生成终止行为**：Base 的 `eos_token_id = 151643`（`<|endoftext|>`），而 Distill 为
  151645（`<|im_end|>`）。vLLM 采样的停止集合取自模型 config，因此：
  - Base 在 assistant 位置通常**不会**以 `<|im_end|>` 终止（它未经 chat 微调），预期产出
    大量撞 4096 cap 的截断候选；
  - 该偏差**不修正**（不得为 base 定制停止词或改 prompt）——它本身就是“弱起点在固定
    协议下是否产生任何可用 reward 信号”的诊断对象。

## 预期与 gate（E021，Phase 4C/4D）

- 在 `igr_mechanism_set.json`（64 定理 × 8 样本、temp 1.0、4096）上跑 Base；
  与同集合上的 Distill 结果比较 verified rate / IGR / Z / O / format failure / truncation。
- M2 gate（预注册）：
  - `IGR < 0.01`（或几乎全零）→ **REWARD-DEAD**：不进行长 RL；转入 verified cold-start 准备
    （今晚只做数据审计 / SFT config 审计 / 单 optimizer-step 记忆冒烟，不做无人值守 SFT）。
  - `0.01 ≤ IGR < 0.05` → **WEAK**：最多做 5 步 GRPO smoke（只看 mixed group / 非零 advantage /
    finite grad / checkpoint）。
  - `IGR ≥ 0.05` → **TRAINABLE**：最多 10 步 Base GRPO exploratory pilot（M1 frozen recipe），
    完成后停止，不自动 30/60。
