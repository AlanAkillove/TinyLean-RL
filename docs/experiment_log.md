# 实验日志（Experiment log）

本文件按时间顺序记录每次实验运行的目的、设置、命令、结果与结论，只保留实际执行过的内容。
当前状态快照见 [`p0_status.md`](p0_status.md)，阶段约束与决策格式见 [`experiment_protocol.md`](experiment_protocol.md)。
大体量原始产物保存在 `experiments/results/`（不进 Git），本日志保留可核对的摘要与指向。

验收口径：候选只有被 Lean 验证器清除全部目标且无 `error` 级消息时才算通过（`warning` 不影响）；验证器为本地 Kimina Lean Server（Docker `projectnumina/kimina-lean-server:2.0.0`，镜像内 Lean `v4.15.0`）。本机数字是工作基线，不是官方绝对成绩。

---

## 2026-09-15

### E001 验证器正例/负例 gate

- 目的：确认 Kimina Lean Server 可作为奖励来源。
- 前置修复：compose 无条件传递空字符串 `LEAN_SERVER_API_KEY`，服务端（pydantic 中空串 ≠ `None`）将其视为"已配置密钥"，所有请求返回 401 `Missing API key`；改为仅在变量真正配置时传递（YAML null 形式）。
- 命令：`uv run python scripts/verify_smoke.py`
- 结果：正例（`1 + 1 = 2`）目标清零、无 error；负例（`1 + 1 = 3`）返回 `unsolved goals` error。与预期一致。
- 结论：verifier gate 通过。

### E002 model→Lean 端到端 smoke test

- 目的：验证"模型生成 → 证明抽取 → Lean 验证"完整链路在 CUDA 上可运行。
- 命令：`uv run python scripts/smoke_test.py --model-key kimina_distill_0_6b --offline`
- 结果：Distill 0.6B 生成 `example : 1 + 1 = 2 := by norm_num`；服务器判定通过（`goalsBefore: ⊢ 1 + 1 = 2` → `goalsAfter: []`），响应约 24 ms（REPL 已热）。
- 结论：端到端链路存活；后续评估可以直接进行。

### E003 P2 Distill 8×4（max 2048 tokens）

- 目的：获得首个验证器背书的 Pass@K 数据。
- 设置：8 定理 × 4 样本；temperature 0.6 / top_p 0.95；RTX 4060 Laptop 8 GiB。
- 命令：`uv run python scripts/evaluate_model.py --model-key kimina_distill_0_6b --limit 8 --samples-per-theorem 4 --max-new-tokens 2048 --offline --output experiments/results/p2_eval_distill_8x4.json`
- 首轮作废（0/32）：evaluator 的 `result_is_valid` 按服务端不存在的布尔字段解析，恒判失败。用已知有效候选的真实响应复验确认（修复前 `False`，修复后 `True`）。修复内容：按 `response.error` 与 `error` 级消息判定；新增 `raw_output` 存档以便离线审计；单元用例 7/7 通过。
- 重跑结果（完成于 2026-09-15T13:09:41Z）：**10/32**，Pass@1 0.3125；all_zero 0.625 / mixed 0.125 / all_one 0.25；生成 1250.4 s，验证 7.0 s。
- 通过分布：定理 0 `mathd_algebra_478` 2/4，定理 4 `mathd_algebra_141` 4/4，定理 5 `mathd_numbertheory_3` 4/4；其余 5 个定理全部 0/4。
- 观察：所有通过的候选生成都 <1600 tokens；未通过的定理要么撞满 2048 上限（avg_tok = 2048），要么只输出了推理文本而非 tactic。token 上限是当前主要瓶颈。
- 产物：`experiments/results/p2_eval_distill_8x4.json`。

### E004 P2 RL 8×4（max 2048 tokens）

- 设置：同 E003，模型换成 `kimina_rl_0_6b`。
- 命令：`uv run python scripts/evaluate_model.py --model-key kimina_rl_0_6b --limit 8 --samples-per-theorem 4 --max-new-tokens 2048 --offline --output experiments/results/p2_eval_rl_8x4.json`
- 过程事件：首次运行在第 4 批验证时服务器返回 500（REPL `JSON decode error`），客户端无容错导致整轮结果丢失。修复：批次失败重试 1 次，仍失败则降级为逐候选验证。重跑时在第 4 批复现同一 500（由 `native_decide` 候选触发），容错生效，流程继续完成。
- 结果（完成于 2026-09-15T13:47:46Z）：**9/32**，Pass@1 0.28125；all_zero 0.625 / mixed 0.25 / all_one 0.125；生成 1264.9 s，验证 35.8 s。
- 通过分布：定理 0 `mathd_algebra_478` 3/4，定理 4 `mathd_algebra_141` 4/4，定理 5 `mathd_numbertheory_3` 2/4；其余 5 个定理全部 0/4。
- 定理 3 `amc12_2001_p5` 重验（原批次因 500 未判定）：0/4 —— sample 0/1 干净失败；sample 2/3 生成 `native_decide` 对 `Finset.range 10000` 的乘积做原生求值，REPL 稳定崩溃（可复现），计为失败。
- 产物：`experiments/results/p2_eval_rl_8x4.json`、`experiments/results/p2_eval_rl_8x4_recheck.json`。

### E003 vs E004 对比与结论

| 模型 | Verified | Pass@1 | all_zero | mixed | all_one | 通过定理 |
|---|---|---|---|---|---|---|
| `kimina_distill_0_6b` | 10/32 | 0.3125 | 0.625 | 0.125 | 0.25 | 0, 4, 5 |
| `kimina_rl_0_6b` | 9/32 | 0.28125 | 0.625 | 0.25 | 0.125 | 0, 4, 5 |

- 此规模下 `Kimina-RL-0.6B > Kimina-Distill-0.6B` 的定性关系未复现：两者差 1 个候选，在噪声内；两个模型通过的定理集合完全相同。
- 两个模型都在同样的 5 个难题上失败，且失败均伴随 2048-token 截断；下一步提高 token 预算（≥3072）并扩大定理数（≥32）再评估排序。
- `native_decide` 是评估期的风险模式：难定理（如含大数计算）会诱导模型使用它，导致验证器侧崩溃；已通过逐候选降级验证兜底，但若未来作为训练信号需单独定义这类崩溃的记分规则。

### E005 P2 Distill 8×4（max 4096 tokens）

- 目的：验证假设“2048 token 上限抑制了模型间差异”（E003/E004 中所有失败定理均撞上限）。
- 设置：同 E003，仅 `max_new_tokens` 升到 4096。
- 命令：`uv run python scripts/evaluate_model.py --model-key kimina_distill_0_6b --limit 8 --samples-per-theorem 4 --max-new-tokens 4096 --offline --output experiments/results/p2_eval_distill_8x4_4096.json`
- 过程事件（两次中断与修复）：
  1. 首次运行第 1 批验证全部 502 Bad Gateway。根因：httpx 默认 `trust_env=True`，把本机 Windows 系统级代理设置用于本地 Lean server 请求（curl 与容器内直连均正常；`trust_env=False` 稳定复现 200）。修复：`src/tinylean_rl/verifier/kimina.py` 显式 `trust_env=False`（本地服务永不经过代理）。
  2. 修复期间为所有带长循环的脚本加入 tqdm 进度条、验证重试退避（15 s）与 `verify_status` 字段（`verified`/`lean_error`/`verifier_error`，可区分服务器故障与真实验证失败）。评估在容器重启 + REPL 预热后重跑。
- 结果（完成于 2026-09-15T15:53:46Z）：**14/32**，Pass@1 **0.4375**；all_zero 0.5 / mixed 0.125 / all_one 0.375；生成 4903.2 s，验证 26.3 s。
- 通过分布（对比 @2048）：定理 0 `mathd_algebra_478` 4/4（@2048: 2/4）；定理 1 `numbertheory_4x3m7y3neq2003` 2/4（@2048: 0/4）；定理 4 `mathd_algebra_141` 4/4；定理 5 `mathd_numbertheory_3` 4/4；定理 2/3/6/7 仍 0/4（avg_tok 3974–4096，全部撞 4096 上限）。
- 结论：
  - 预算假设成立：Pass@1 0.3125 → 0.4375（相对 +40%），all_zero 组率 0.625 → 0.5；新通过的候选（定理 1 的两个，2780/2166 tokens）在 2048 预算下必被截断。
  - 定理分成三层：简单题（0/4/5）不受预算影响；中等题（1）预算决定成败；难题（2/3/6/7）需要超过 4096 的预算（或本身就超出 Distill 0.6B 能力）。
- 产物：`experiments/results/p2_eval_distill_8x4_4096.json`。

### 并发吞吐复测（基础设施）

- 目的：测定 P3 奖励计算的并发能力；先发 4 个并发 `verify_code`（已排除代理因素）。
- 结果：仅 1 个请求成功（2.4 s），其余 3 个在 300 s 客户端超时；而串行验证稳定（E003–E005 全程正常，32 候选验证 7–36 s）。
- 结论：Kimina Lean Server 2.0.0 在本机不支持有效并发（行为机制待深入）。P3 的奖励计算必须串行提交或引入队列/多实例方案；此约束作为 P3 吞吐规划输入记录。

---

## 追加记录模板

新实验条目按时间顺序追加到本模板上方，采用以下骨架（“产物”写 `experiments/results/` 下文件名）：

```text
### E00X 标题

- 目的：
- 设置：
- 命令：
- 结果：
- 结论：
- 产物：
```

记账约定：

- 只写实际执行的内容；作废的运行也要保留，注明作废原因与修复方式。
- 关键数字来自 artifact JSON，写清楚运行时工作目录与时间（UTC）。
- 每个实验的原始输出不粘贴到本文件，只留文件名与摘要。
