# 实验日志（Experiment log）

本文件按时间顺序记录每次实验运行的目的、设置、命令、结果与结论，只保留实际执行过的内容。
当前状态快照见 [`p0_status.md`](p0_status.md)，研究主线与统计功效分析见 [`research_plan.md`](research_plan.md)，阶段约束与决策格式见 [`experiment_protocol.md`](experiment_protocol.md)。
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

### E006 P2 RL 8×4（max 4096 tokens）

- 设置：同 E005，模型换成 `kimina_rl_0_6b`。
- 命令：`uv run python scripts/evaluate_model.py --model-key kimina_rl_0_6b --limit 8 --samples-per-theorem 4 --max-new-tokens 4096 --offline --output experiments/results/p2_eval_rl_8x4_4096.json`
- 过程事件：第 4 批（定理 3）验证批次触发 500（REPL 崩溃），退避重试失败后降级逐候选；3 个候选（3-0/3-1/3-2）单独验证时同样触发 500，被记为 `verifier_error`（`verify_status` 机制首次实战）。完成后逐一重验（含 15 s 退避重试）仍全部 500——**这 3 个候选均含 `native_decide`**，稳定复现验证器崩溃，按先例计为失败。
- 结果（完成于 2026-09-15T17:38:58Z）：**13/32**，Pass@1 **0.40625**；all_zero 0.5 / mixed 0.25 / all_one 0.25；生成 5271.0 s，验证 431.9 s。
- 通过分布：定理 0 `mathd_algebra_478` 4/4；定理 1 `numbertheory_4x3m7y3neq2003` 2/4；定理 4 `mathd_algebra_141` 4/4；定理 5 `mathd_numbertheory_3` 3/4（1 个撞上限）；定理 2/3/6/7 为 0/4。
- 产物：`experiments/results/p2_eval_rl_8x4_4096.json`、`experiments/results/p2_eval_rl_8x4_4096_recheck.json`。

### E005 vs E006 对比与结论（@4096）

| 模型 | Verified | Pass@1 | all_zero | mixed | all_one |
|---|---|---|---|---|---|
| `kimina_distill_0_6b` | 14/32 | 0.4375 | 0.5 | 0.125 | 0.375 |
| `kimina_rl_0_6b` | 13/32 | 0.40625 | 0.5 | 0.25 | 0.25 |

- 两个预算下（2048/4096）Distill 均以 1 个候选之差略微领先于 RL（10 vs 9；14 vs 13）——差异在噪声内，官方 +2.45 pp 的定性关系仍未复现；需要更大样本（≥ 32 定理）与 Pass@8/32 口径才能判定。
- 预算效应在两个模型上都成立：Pass@1 Distill 0.3125 → 0.4375，RL 0.28125 → 0.40625；all_zero 组率从 0.625 降到 0.5。
- RL 在 `amc12_2001_p5` 上 4 个候选有 3 个含 `native_decide` 并稳定触发验证器崩溃（Distill 在同题为 0/4 个），RL 微调后对 `native_decide` 的倾向值得作为训练/评估侧风险点跟踪。

### 并发吞吐复测（基础设施）

- 目的：测定 P3 奖励计算的并发能力；先发 4 个并发 `verify_code`（已排除代理因素）。
- 结果：仅 1 个请求成功（2.4 s），其余 3 个在 300 s 客户端超时；服务器日志显示 4 个请求最终都返回 200，随后 3 个新建 REPL 被关闭——即并发请求各自取用独立 REPL，冷 REPL 初始化（Mathlib 加载）耗时超过客户端超时。串行验证稳定（E003–E005 全程正常，32 候选验证 7–36 s）。
-- 结论：Kimina Lean Server 2.0.0 的并发容量受限于 REPL 池冷启动成本。P3 若需并发奖励计算，必须先并行预热 N 个 REPL（或提高 `LEAN_SERVER_MAX_REPLS` 并预留初始化时间）；低风险方案是串行提交 + 本地队列。

## 2026-09-16

### E007 P2 Distill 32×4 @4096（主实验：规模扩展）

- 目的：把对比从 8 定理扩到 32 定理，为预算-分数曲线与统计可判定性分析提供主数据。
- 设置：同 E005；`--limit 32`（miniF2F 索引 0–31，含 9 道 mathd_algebra、7 道 IMO/IMO-similar、4 道 AMC、7 道数论等）。
- 命令：`uv run python scripts/evaluate_model.py --model-key kimina_distill_0_6b --limit 32 --samples-per-theorem 4 --max-new-tokens 4096 --offline --output experiments/results/p2_eval_distill_32x4_4096.json`
- 过程：运行 5.1 小时（生成 18,386 s + 验证 290 s）。1 个候选（定理 3-1，含 `native_decide`）触发 `verifier_error`，重验仍稳定 500，计为失败。
- 结果：**43/128**，Pass@1 0.3359；解出定理 **15/32**（46.9%）；all_zero 0.531 / mixed 0.281 / all_one 0.188。
- 分布：解出定理索引 [0,4,5,8,9,10,11,13,14,15,17,20,21,23,29]；前 16 题解出 10 题、后 16 题仅 5 题；12/128 候选含 `native_decide`。
- 长度：solved 候选平均 1748 tokens（预算的 43%）；全量平均 3231（大量截断）；max 4096。
- 种子稳定性观察：E007 前 8 题与 E005（同题同参数、独立采样）对比为 **14 → 10**（示例：定理 1 从 2/4 到 0/4）——同一模型同一配置的两次运行相差 4/32 个候选，直接展示了 8 定理级评估的采样噪声量级，是 RQ3（可判定性）的实测证据。
- 产物：`experiments/results/p2_eval_distill_32x4_4096.json`。

### E008 P2 RL 32×4 @4096（主实验：规模扩展）

- 设置：同 E007，模型为 `kimina_rl_0_6b`；由桥接脚本在 E007 完成后自动启动（07:06）。
- 命令：`uv run python scripts/evaluate_model.py --model-key kimina_rl_0_6b --limit 32 --samples-per-theorem 4 --max-new-tokens 4096 --offline --output experiments/results/p2_eval_rl_32x4_4096.json`
- 过程：运行约 7.5 小时。2 个候选（定理 3-0/3-1，均含 `native_decide`）触发 `verifier_error`，重验仍稳定 500，计为失败。
- 结果：**55/128**，Pass@1 **0.4297**；解出定理 **18/32**（56.3%）；all_zero 0.4375 / mixed 0.3125 / all_one 0.25。
- 分布：解出定理索引 [0,4,5,8,10,11,13,14,15,17,19,20,21,23,25,27,29,31]；难度分层 0/4=14、1-3/4=10、4/4=8。
- `native_decide`：11/128（与 Distill 的 12/128 基本一致）——E004/E006 中“RL 显著更倾向 `native_decide`”的观察在 32 定理规模下**未复现**，收回该结论。
- 产物：`experiments/results/p2_eval_rl_32x4_4096.json`、`..._recheck.json`。

### E007 vs E008 对比（32 定理 @4096，论文主表）

| 指标 | Distill | RL | 差 |
|---|---|---|---|
| 候选级 verified | 43/128 (0.3359) | 55/128 (0.4297) | +9.4 pp（z≈1.55，p≈0.12） |
| 定理级 solved | 15/32 (46.9%) | 18/32 (56.3%) | +3 题 |
| 配对翻转 | 仅 Distill 解出：{9} | 仅 RL 解出：{19,25,27,31} | McNemar p≈0.375 |
| all_zero | 0.531 | 0.4375 | −0.094 |
| 通过候选均长 | 1748 tokens | 1719 tokens | — |

- **在 32 定理规模下，RL 首次在方向上领先 Distill（+9.4 pp 候选级 / +3 定理级），与官方 +2.45 pp 方向一致；但两种统计检验均未达显著**（与预分析的“小样本不可判定”一致）。
- 子集敏感性：同一对模型在 8 定理子集上（E005/E006）几乎平手（14 vs 13，且 Distill 略高），32 定理上 RL 明显领先——**子集抽样本身能反转结论方向**，是“小样本评估不可靠”的最直接证据。
- 定理级翻转的 4 个 RL-only 题全为中低难度（索引 19/25/27/31），Distill 唯一独有题为 imo_1983_p6（索引 9）。

### P2 阶段 checkpoint 与方向决议（2026-09-16）

- 事件：外部审查（GPT）指出研究主线漂移——budget-aware evaluation 的展开正在取代 TinyLean-RL 的 RL 主线。决议**采纳并执行**：
  1. E001–E008 正式定位为 **P2 预实验**（Study A 评估校准 / B 奖励可靠性 / C 奖励信息量），收录于 [`studies/evaluation_calibration.md`](studies/evaluation_calibration.md)；
  2. **停止扩表**：不跑 E009（8192）/16×8/244×32；budget 因果结论留给未来的 paired generation 独立课题；
  3. 主线计划重写为 RL 方向（[`research_plan.md`](research_plan.md)），`paper_draft.md` 标记为 side-study；
  4. 下一阶段：Linux 3090 上的 P2→P3 迁移 gate 与 P3-A（RL plumbing smoke）。
- 统计方法修正（采纳外部审查）：candidate-level “sample_i 对 sample_i” 无天然配对、不成立；配对单位改为 theorem（`d_i = p̂_i^R − p̂_i^D` 的 bootstrap，或定理级 solved indicator 的 McNemar）。
- 统计单位修正后的复核：E007/E008 的定理级 McNemar（p≈0.375）与候选级 z≈1.55 均不显著，原结论不变。
- P3 输入（P3-0 校准后冻结；`max_response=4096` 为初始 rollout 预算假设）：成功长度 P95≈3,000、串行验证（验证时间占比 1.6%）、奖励契约（`verifier_error` 单独记录计 0）、IGR 基线 Distill 28.1% / RL 31.3%。

### E009 P2.5 W1 Promptset 32×4 cached rollout profile（停止条件①）

- 目的：在官方 Kimina-Prover-Promptset 子集上量化 GRPO 所需的组奖励信号（positive/mixed 组），并落盘 cached rollout batch 供 W2/W3 复用。
- 设置：32 个 unique statement（seed 0）× 4 候选；`max_new_tokens=4096`、`temperature=0.6`、`top_p=0.95`（E007 评估设置；官方 P3 rollout 用 temp 1.0）、batch-size 1（8 GiB 显存）。验证走本地 Kimina server（batch verify → 单候选回退）。
- 命令：`python -u scripts/promptset_rollout_probe.py --batch-size 1 --limit 32 --samples-per-theorem 4 --output experiments/results/p2_5_promptset_profile.json --batch-dir experiments/local_rl_batch`（项目 `.venv` 解释器）。
- 过程：运行 4.7 小时（生成 14,495.7 s + 验证 2,375.7 s）；batch verify 触发 21 次单候选回退（missing_item 20、redeclaration 1）。
- 结果：verified 14/128（严格格式同数）；sorry 0、format_failures 0、`verifier_errors` 5（单独计 0）；**截断 96/128 = 75%**；组率 all_zero 87.5%（28/32）、**mixed 3.125%（1/32）**、all_one 9.375%（3/32），**IGR = 0.03125**；prompt 长度（全 promptset 7,620 unique）median 231 / p95 415 / max 2,365。
- 结论：停止条件①满足（positive 3 组 + mixed 1 组）；all-zero 28/32 未触发计划中的 ≥29/32 风险线，但 IGR 显著低于 miniF2F 的 28–31%，与 75% 截断强相关——P3-0 用官方 rollout 口径（temp 1.0）时必须先复核截断率与 IGR；若仍 ≲5%，恢复顺序 = n=8 → multiturn（见 [studies/rl_readiness.md](studies/rl_readiness.md) §九）。IGR 3.1% 不等于“模型不适合 RL”。数据点：`<think>` 混入 Lean 代码（lean_error 主因）、REPL redeclaration、单候选回退路径被真实使用。
- 产物：`experiments/results/p2_5_promptset_profile.json`；`experiments/local_rl_batch/{prompts,rollouts,rewards}.jsonl + metadata.json`（artifact 创建于 2026-09-16T12:35:31Z）。

### E010 P2.5 W2 GRPO loss rehearsal（cached rollout → advantage → backward，停止条件②）

- 目的：以 pinned commit 的 GRPO 目标函数（DrGRPO mean-only 中心化 + seq-mean-token-sum-norm + 非对称 clip 0.2/0.3 + clip_ratio_c 3.0）在真实 cached rollout 上重演 advantage → loss → backward 全链路。
- 设置：CPU float32（本机 RAM 约束）；response 截断 1,024（CPU 内存）；micro-batch 1。两次运行：①链自动全零子集（`--max-theorems 8`，前 8 定理恰好全为零组）；②混合组补跑（`--theorem-indices 15,16,17,18,19`，含唯一 mixed 组 #17，2/4 verified）。
- 命令（补跑）：`.venv\Scripts\python.exe -u scripts/grpo_loss_rehearsal.py --batch-dir experiments/local_rl_batch --output experiments/results/p2_5_grpo_rehearsal_mixed.json --device cpu --theorem-indices 15,16,17,18,19 --max-response-length 1024`。
- 结果：
  - 全零子集（8 定理 / 32 候选）：advantage 全零 → loss 0、grad_norm 0（有限）；total 1,914.2 s（load 23.9 / old_logprobs 226.2 / loss_forward 517.6 / backward 1,055.2）。
  - 混合组补跑（5 定理 / 20 候选）：**advantage ∈ {−0.5, +0.5}（all_zero=false）**；per-candidate loss = [+0.386, −0.386, −0.386, +0.386]（恰为 #17 的 4 个候选）；**grad_norm = 2.036（有限）**；total 1,163.1 s（load 72.6 / old_logprobs 120.2 / loss_forward 299.2 / backward 616.0）。
- 结论：停止条件②满足——链路在真实数据上完整成立，非均匀组产生有限非零梯度。`loss.total=0` 是 mean-only 中心化下组内 Σadv=0 的数学对称性（不是失败信号）；per-candidate 项与 grad_norm 非零证明路径有效。87.5% all-zero 组率下 8 定理子集大概率无梯度信号（链首跑即命中）——IGR 是 P3 首要观察指标。
- 产物：`experiments/results/p2_5_grpo_rehearsal.json`、`experiments/results/p2_5_grpo_rehearsal_mixed.json`（链版创建于 2026-09-16T13:13:25Z；混合组补跑版 2026-09-16T14:56:12Z）。

### E011 P2.5 W3 LoRA 单步显存探针（8 GiB RTX 4060，停止条件③）

- 目的：回答“本地 8 GiB 卡能否完成一步 LoRA GRPO 反传”（成功或 OOM 都算回答）。
- 设置：float16 + gradient checkpointing + micro-batch 1；LoRA target q/k/v/o、alpha=2r、dropout 0、AdamW lr 2e-6；r16/32 × seq 1024/2048 四组合。先以 CPU bf16 冒烟（r16 × 384，smoke batch）验证代码路径。
- 命令（GPU）：链自动执行 `.venv\Scripts\python.exe -u scripts/lora_step_probe.py --batch-dir experiments/local_rl_batch --output experiments/results/p2_5_lora_step_probe.json --lora-ranks 16,32 --seq-lens 1024,2048 --device cuda`；冒烟：`--batch-dir experiments/local_rl_batch_smoke --output experiments/results/_lora_cpu_smoke.json --device cpu --dtype bfloat16 --lora-ranks 16 --seq-lens 384`。
- 结果：
  - CPU 冒烟：status=ok；**224/224 LoRA 张量全部有梯度**；forward 39.1 s / backward 998.5 s / step 1.2 s（total 1,177.5 s）；期间修复 2 个真 bug（cached batch 的 reward 字段引用、PEFT `get_base_model()` 解包）。
  - GPU 探针：**all_combinations_ok: true**（四组合全部完成 forward+backward+step）；峰值 reserved 分别为 r16@1024 4,016 MB / r32@1024 4,126 MB / r16@2048 **8,642 MB** / r32@2048 **8,752 MB**——两个 2048 组合超出物理 8,187.5 MB（peak_headroom = −564.5 MB，靠 Windows 共享显存完成）；r16@1024 单次 forward 0.17 s / backward 1.03 s。各组合候选落在 all-zero 组，loss=0、grad_norm=0（有限）——本探针验证显存与算子路径，不验证梯度信号。
- 结论：停止条件③满足——8 GiB Windows **不适合真实 P3**（2048 组合已溢出物理显存、依赖 WDDM 共享内存、吞吐不可预测）；P3 迁移 Linux RTX 3090（P3-0）。LoRA 单步路径已证明可实现，但不等于“P3 必须 LoRA”：训练模式（full-parameter 首选 / LoRA fallback）在 P3-0 显存探针后冻结。
- 产物：`experiments/results/p2_5_lora_step_probe.json`、`experiments/results/_lora_cpu_smoke.json`（GPU 探针创建于 2026-09-16T12:40:11Z；CPU 冒烟版 2026-09-16T09:15:21Z）。

### P2.5 冻结（2026-09-16）

- `win` 分支打 tag `p2.5-win-complete`；P2.5 数据、结论与 Windows 工作区冻结，本机不再新增推理 / rollout / RL 实验。
- 下一阶段：**P3-0 — Linux Migration & On-Policy Calibration**（服务器 RTX 3090 24 GB），范围与命令序列见 [`p3_linux_handoff.md`](p3_linux_handoff.md)。

---

## 2026-09-17

### E012 P3-0 Linux 迁移 gate（环境 / 测试 / 验证器 / smoke）

- 目的：确认 Windows 阶段准备的 P3 基础设施在 Linux RTX 3090 主机上真实可用（P3-0 步骤 1–3）。
- 设置：Ubuntu 24.04.4 / RTX 3090 24 GB / driver 580.173.02 / uv 0.12.13；`p3-linux` 分支自 `win`@6ef923e（tag p2.5-win-complete，submodule e16b605e）创建。
- 命令：`uv sync --all-extras` → `uv run pytest` → `ruff` → `compileall` → `doctor.sh` → `verify_smoke.py` → `prewarm_lean_server.py` → `smoke_test.py --offline`。
- 结果：
  - 环境对齐（决策 D004 及后续修复）：torch 2.9.0+cu128 → **2.7.0（cu126）** + vllm 0.9.1 + flash-attn 2.8.0.post2 + pinned VERL（editable）安装完成；transformers 上限 `<4.54`（vllm 0.9.1 的 aimv2 AutoConfig 注册冲突，实测 4.54.1 失败 / 4.53.3 成功）；conda cc shim 导致 Triton JIT 构建失败 → `CC=/usr/bin/gcc`；`from_pretrained` 改用 `torch_dtype=`（4.53 无 `dtype=` 别名）。
  - 测试：pytest **40 passed**（27 基线 + 13 strict-verifier 新增；与 Windows 的 27+19 口径一致新增） 、GRPO reference 19 passed；ruff / compileall 干净。
  - `doctor.sh`：**23 passed / 0 warnings / 0 failures**（含 vLLM/Ray/VERL import、Lean /health、验证延迟 0.1 s）。
  - 验证器正/负 gate：positive → `valid`、negative → `lean_error`（严格口径）；`/openapi.json` 在本镜像 prod 模式 404，readiness 统一 `/health`。
  - 预热（容器重启后的真实冷启动）：首请求 **3.04 s**、随后 0.07 s；c=1 13.1 rps；c=2 需新 REPL 冷启动（+3 s）→ 推荐 concurrency ≤ 2。
  - model→Lean smoke：生成 `1+1=2 := by norm_num` → 验证通过；峰值显存 1.8 GB，全程 13 s。
  - 官方 reward 路径实测（kimina_client 0.2.1 ↔ Lean server 2.0.0）：valid→1.0、type error→0、sorry→0、filtered→0——关闭 P2.5 交接的 open item。
- 结论：迁移 gate 全部通过，解锁校准与显存探针。
- 产物：`experiments/manifests/p3_0_environment.yaml`、`experiments/results/p3_0_lean_server_warmup.json`。

### E013 P3-0 Promptset rollout calibration（temp=1.0，官方口径）

- 目的：在真正训练 rollout 参数下重测 Promptset 的 IGR / 截断 / 长度 / 吞吐（P3-0 步骤 4–5）。
- 设置：Kimina-Distill-0.6B；32 unique statements（seed 0，与 P2.5 W1 相同定理集）；n=4；temp=1.0 / top_p=1.0；max_response=4096；HF generate。
- 命令：`uv run python scripts/promptset_rollout_probe.py --temperature 1.0 --top-p 1.0 --limit 32 --samples-per-theorem 4 --offline --output experiments/results/p3_0_promptset_temp1.json --batch-dir experiments/p3_0_batch`
- 结果（128 候选）：
  - verified **15/128**；truncated 85/128（66.4%）；format_failures 0；sorry 0；verifier_errors 0；
  - 组率 all_zero 26/32（0.8125）/ mixed **3/32（0.09375）** / all_one 3/32 → **IGR = 0.09375**；
  - 自然长度（剔除 pad_token_id=EOS 填充后）：verified mean 1748 / median 1525 / p95 2746 / max **3955**，**0/15 撞上限**；failed mean 3818 / median 4096（其中 85 条从未自然终止）；
  - 生成 3035 s（HF generate ≈151 有用 tok/s）、验证 131 s（≈1.0 s/候选）。
- 对照 P2.5 W1（temp 0.6，同定理集）：IGR 0.03125 → **0.09375（×3）**；截断 75% → 66.4%；verified 14 → 15。
- 结论：temp 1.0 下 IGR 进入 0.05–0.10 marginal 区间（仍 ≥5%），**n=4 retain**；4096 预算未系统性截断成功轨迹（成功最大 3955），**不上调 8192**；观测差异与采样口径变化一致，不做硬件归因。
- 产物：`experiments/results/p3_0_promptset_temp1.json`（完整记录，含 raw outputs，不入库）、`experiments/p3_0_batch/`、`experiments/results/p3_0_calibration_vram.csv`。

### E014 P3-0 full-FT 单步显存可行性探针（真实 pinned VERL trainer）

- 目的：实测 full-parameter GRPO 单步在单张 3090 24 GB 上的显存与分阶段墙钟（P3-0 步骤 6）。
- 设置：`run_p3_smoke.sh --steps 1`；full-FT（无 LoRA）；n=4；max_prompt 1024 / max_response 4096；无 KL / 无 ref worker；multiturn off；tb=4 / mini=4 / micro=2 / vLLM util=0.40 / max_num_batched_tokens=5120（按 §16 顺序从 util 0.30 / 8192 调整：0.30 下 vLLM 0.9.1 报告 KV 不足，无法服务 max_model_len=5120）。
- 命令（完整输出见 `.cache/p3_0_memprobe.log`）：`bash scripts/run_p3_smoke.sh --steps 1 data.train_batch_size=4 actor_rollout_ref.actor.ppo_mini_batch_size=4 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 actor_rollout_ref.rollout.gpu_memory_utilization=0.40 actor_rollout_ref.rollout.max_num_batched_tokens=5120 trainer.default_local_dir=runs/p3_0_memprobe`
- 结果：
  - **exit 0、无 OOM**；step 墙钟 **136.4 s**：gen 53.9 / reward（异步批次）49.4 / old_logprob 4.6 / update_actor 11.1 / save_checkpoint 17.2 / reshard 0.5；
  - 显存：torch peak allocated **19.09 GB**、reserved **20.52 GB**；nvidia-smi 采样峰值 18,095 MiB；
  - rollout：16 序列 / 67,727 tokens / ≈1,285 tok/s（vLLM）；防照口径 temp=1.0；
  - 本步 batch 无 verified 候选 → score 全 0、grad_norm 0.0（零梯度 step，属预期）；response_length mean 3930 / clip 62.5%；actor/entropy 12.67；ppo_kl 8.3e-06；
  - checkpoint：`runs/p3_0_memprobe/global_step_1/` = model 2.9 GB + optim 4.5 GB + extra_state 16 KB + fsdp_config.json + huggingface + data.pt + `latest_checkpointed_iteration.txt` → 契约四项（目录/actor/optimizer/trainer 状态/global step）齐全；
  - 观察：teardown 时一个 Ray worker 以 SIGTERM 退出（异步 reward worker 收尾），主进程正常 exit 0，不影响本步指标。
- 结论：**FULL-FT 在单张 3090 24 GB 可行**（tb4/mini4/micro2/util0.40/5120 batched）；训练模式冻结为 FULL-FT，不启用 LoRA fallback。
- 产物：`runs/p3_0_memprobe/`、`experiments/results/p3_0_memprobe_vram.csv`、`.cache/p3_0_memprobe.log`。

### E015 P3-A On-Policy GRPO smoke（3 steps + resume）

- 目的：验证全链路 on-policy：rollout → Lean reward → GRPO advantage → optimizer → checkpoint → resume（P3-A 成功标准；按协议 P3-0 完成后执行 ≤2–5 steps）。
- 设置：P3-0 冻结配置（tb4/mini4/micro2/util0.40/5120batched；n=4；max 4096；无 KL；multiturn off）；`run_p3_smoke.sh --steps 3`，checkpoint 至 `runs/p3a_smoke`（save_freq=steps）；随后 `--steps 4` 续训验证 resume。
- 结果（4 个优化步）：

| step | 墙钟 | score mean/max | advantages | grad_norm | pg_loss | alloc/reserved GB |
|---|---|---|---|---|---|---|
| 1 | 117.8 s | 0 / 0 | 0（全零组） | 0.0 | 0.0 | 19.09 / 20.52 |
| 2 | 61.5 s | 0.25 / 1.0 | 0（all-one 组） | 0.0 | 0.0 | 23.31 / 24.84 |
| 3 | 83.5 s（含 save 16.9） | 0 / 0 | 0（全零组） | 0.0 | 0.0 | 23.56 / 25.07 |
| 4（resume） | 81.7 s（含 save 14.2） | **0.0625 / 1.0** | **+0.75 / −0.25（mixed 组）** | **0.180** | **0.00256** | 23.19 / 24.75 |

  - resume 实证：日志“Found checkpoint / Resuming from …/global_step_3 / Setting global step to 3”，model/optimizer/rng/lr_scheduler 全部从 `global_step_3` 加载，进度从 3/4 开始；续训 run 总耗时 210 s。
  - 显存：nvidia-smi 采样峰值 **22,755 MiB（22.2 GiB）**；响应长度稳定（mean 3500–3930，clip 0.625–0.75），无格式崩塌；verifier 全程无不可恢复错误。
- 结论：P3-A 成功标准全部满足（非零 advantage/loss/grad；save+resume 实测通过）。观察：IGR≈0.09 的低信号密度下 4 步中 3 步零梯度——P3-B 前需先处理信号密度（恢复顺序 n=8 → multiturn），另需评估 22.2 GiB 峰值下的长跑稳定性。
- 产物：`.cache/p3a_smoke.log`、`.cache/p3a_resume.log`、`runs/p3a_smoke/`（global_step_3、global_step_4）、`experiments/results/p3a_smoke_vram.csv`、`experiments/results/p3a_resume_vram.csv`。

### E016 n=8 mini-calibration（P3-B 的 n 决策前置测量）

- 目的：按协议恢复阶梯第①项的配对测量——在相同 16 个定理上对比 n=4 vs n=8 的组信号密度（IGR），为 P3-B 的 group size 决策提供证据。
- 设置：E013 前 16 定理（seed 0 前缀性质已验证，paired）；n=8；temp 1.0 / top_p 1.0；max 4096；HF generate。
- 命令：`uv run python scripts/promptset_rollout_probe.py --temperature 1.0 --top-p 1.0 --limit 16 --samples-per-theorem 8 --offline --output experiments/results/p3_b_n8_calibration.json --batch-dir experiments/p3_b_n8_batch`
- 结果（同一 16 定理配对，n=4 数据取自 E013）：

| 指标 | n=4（E013 subset） | n=8（本次） |
|---|---|---|
| IGR | **0.000**（0 mixed / 16） | **0.0625**（1 mixed / 16） |
| all_zero / all_one | 0.8125 / 0.1875 | 0.8125 / 0.125 |
| verified | 12/64（18.8%） | 23/128（18.0%） |
| truncated | 40/64（62.5%） | 84/128（65.6%） |
| verified 自然长度 | mean 1486 / max 2210 | mean 1871 / p95 2805 / max 3729 |

  - 机制：定理 7 从 n=4 的 all-one（4/4）变为 n=8 的 **mixed（7/8）**——大 n 把近饱和组的单次失败暴露出来；定理 8/13 仍 8/8；其余 13 个定理双臂全零（难度主导）。
  - VRAM：HF 生成批次（16 序列）采样峰值 **23.5 GiB**（校准侧；训练侧受 vLLM 池上限约束）。
- 结论：n=8 在配对子集上把 IGR 从 0.00 提升到 0.0625（方向支持协议阶梯①，幅度有限）；信号密度主要受定理难度限制，n 是次阶修正。建议 P3-B 恢复官方 n=8 并监控显存余量。
- 产物：`experiments/results/p3_b_n8_calibration.json`、`experiments/p3_b_n8_batch/`、`experiments/results/p3_b_n8_calibration_vram.csv`。

---

## 2026-09-17（P3-B）

### E017 P3-B short learning pilot（n=8，30 优化步）

- 目的：运行数十个 update，观察 IGR_t / Z_t / O_t / response length / entropy / KL 的合理变化（P3-B 目标；n=8 为协议恢复阶梯①，经 E016 校准背书，用户明确批准启动）。
- 设置：冻结配置（tb4/mini4/micro2/util0.40/5120），n=8 → 32 序列/步；checkpoint 至 runs/p3b_pilot（save_freq=10，保留 3）；逐步骤 rollout dump。
- 命令：`bash scripts/run_p3_pilot.sh --steps 30 --n 8`
- 结果（30/30 步，exit 0，总耗时 4218 s；每步 4 组 × n=8，IGR 来自 rollout dump）：

| 段 | IGR | Z（全零） | O（全一） | score_mean |
|---|---|---|---|---|
| 1–10 | 0.100 | 0.900 | 0.000 | 0.0594 |
| 11–20 | 0.200 | 0.800 | 0.000 | 0.1000 |
| 21–30 | **0.225** | **0.750** | 0.025 | **0.1750** |
| 全体 | 0.175 | 0.817 | 0.008 | 0.1115 |

  - 17/30 步出现非零组；最后 10 步中 9 步含 mixed 组；**最后 5 步连续非零梯度**（grad_norm 0.08–0.19，pg_loss 非零）；
  - 每步时长 mean 136 s（78–245）；显存 nvidia-smi 峰值 **23.1 GiB** 稳定、无 OOM；verifier errors 0；
  - response_length mean 2773–3953，clip 0.31–0.69；entropy 15–40 无塌缩；format 失败 9–31/32（与 4096 截断相关，无恶化趋势）；
  - checkpoint：global_step_10 / 20 / 30（model+optim+extra_state+fsdp_config+huggingface；step_30 ≈ 7.3 GB）。
- 结论：**动力学与研究计划预测一致**（Z_t ↓、score ↑、IGR_t 仍处上升段）。注意：每步定理集不同、逐步方差大；群学习证据为聚合趋势而非单步因果——**确认性检验（同一定理集评估 step-0/10/20/30 checkpoint）列为下一步**。已按协议到点即停，未自动延长。
- 产物：`runs/p3b_pilot/`（global_step_10/20/30 + rollout_data 1–30.jsonl）、`experiments/results/p3b_pilot_vram.csv`、`.cache/p3b_pilot.log`；摘要 `experiments/manifests/p3b_pilot.yaml`。

---

## 2026-09-17（P3-C）

### E018 Fixed-set checkpoint evaluation（A smoke / B preview / C full）

- 目的：在封存固定集上配对评估 θ0/θ10/θ20/θ30，检验 P3-B 的学习信号（排除定理抽样方差；P3-B 收尾时列出的确认性检验）。
- 固定集构造：排除 E009/E013/E016/E017 已用全部 statement（150 个，`experiments/manifests/p3c_excluded_statement_ids.txt`）；可选 7470，seed 20260917 抽 64（`p3c_fixed_set.json`，生成后不得重抽）；配对单元 = 定理（严禁 candidate 级假配对）。
- 管线：`p3c_checkpoint_eval.py`（vLLM 0.9.1，temp 1.0 / top_p 1.0 / max 4096 / n=8；seed = 20260917 + idx×8 + sample；严格 Kimina 2.0.0 验证）；checkpoint 导出为与基座键完全一致的 HF 目录（311/311 键，bitwise 一致，`p3c_checkpoint_sanity.json`）。
- 命令：`uv run --no-sync python scripts/p3c_checkpoint_eval.py --checkpoint <ck> --limit 64 --samples-per-theorem 8 --chunk-theorems 16 --offline`；E018-C 顺序 θ0→θ30→θ10→θ20。
- **E018-A（smoke：4 ckpt × 2 定理 × 1 样本）**：4/4 exit 0；先暴露 summary 字段 bug（completion_tokens vs generated_tokens）并修复后通过。
- **E018-B（preview：前 16 定理 × 4 样本 × 4 ckpt）**：4/4 exit 0；verified 数 θ0 2 < θ30 5 < θ10 6 < θ20 8（/64，非确认性）。
- **E018-C（正式：64 × 8 × 4 = 2048 候选）**：4/4 exit 0；全部产物种子 0 违例（git_revision 1fb6866 / d64c31b）。

| checkpoint | verified/512 | pass@1 | pass@4 | pass@8 | IGR | all-one | trunc | verifier_error |
|---|---|---|---|---|---|---|---|---|
| θ0 | 65 | 0.1270 | 0.2350 | 0.2812 | 0.250 | 0.031 | 0.465 | 2.93% |
| θ10 | 65 | 0.1270 | 0.2275 | 0.2812 | 0.266 | 0.016 | 0.512 | 2.15% |
| θ20 | 64 | 0.1250 | 0.2435 | 0.2969 | 0.281 | 0.016 | 0.455 | 1.56% |
| θ30 | 70 | 0.1367 | 0.2562 | 0.2969 | 0.297 | 0.000 | 0.455 | 1.76% |

- 配对分析（定理级 d_i = p_i(t) − p_i(0)，bootstrap 10k / seed 20260917，`p3c_analysis.json`）：
  - θ10−θ0：+0.0000，CI [−0.0273, +0.0293]，W/T/L = 8/47/9，McNemar p=1.0；
  - θ20−θ0：−0.0020，CI [−0.0273, +0.0234]，W/T/L = 9/48/7，McNemar p=1.0；
  - θ30−θ0：**+0.0098**，CI [−0.0254, +0.0430]，W/T/L = 12/46/6，McNemar p=1.0（+4 solved / −3 lost）。
- 结论：**POSITIVE-INCONCLUSIVE**——θ30 方向为正（辅以 IGR 单调升高、all-one 组清零、pass@4/8 高于基座等次要信号），但确认性端点 95% CI 跨零、McNemar 不显著（n=64 功效有限）。措辞维持 “short-run dynamics encouraging, fixed-set confirmation inconclusive”；不称 M1 short-horizon confirmed。
- 资源：单 checkpoint verify 1721–3766 s（怪兽候选主导，生成仅 731–748 s）；生成 ~2400 tok/s 稳定；峰值显存 21.3 GiB；verified 效率 51.9–93.9/GPU·h。

**事故与修复记录（§廿四）**：

1. **验证回退风暴**（E018-C 首跑 chunk 2 耗时 17.8 min）：128 候选单次 /verify 请求超过 120 s 客户端超时后，退化为 128 条串行单验。修复（`1fb6866`）：并发子批次（8×8 workers，对齐官方 reward 路径），单个慢候选只拖累自己的子批次；其后所有 chunk 的 warn 均只回退 1–5 个子批次。
2. **主机 OOM 连锁击杀**：`interval_cases 1000..9999 <;> norm_num <;> omega` 枚举炸弹把单个 Mathlib REPL 撑到 20.4 GB RSS；07:10 UTC 内核全局 OOM 击杀容器内该 python，systemd-oomd 同期击杀用户切片进程（含评估 runner 与 tmux scope）。step20 因此中止 4 次（约 06:23 runner 被连带终止（同期 oomd 活动）/ 约 07:25 同因 / 07:34 uv-PATH 快败 / 07:35 服务端未就绪——chunk 1 全部 lean_error，产出作废）。修复（`d64c31b`）：容器 mem_limit=24 GiB；step20 运行期为与前三者可比性以 `docker update` 临时放宽到 40 GiB，跑完复原 24 GiB（均不重建容器）。
3. **容器重建代价**：07:33 应用 cap 时 `docker compose up -d` 重建容器导致镜像重跑 Mathlib 初始化（CPU 高占用约 25 min，其间所有 /verify 返回 NoAvailableReplError）——已在 compose.yaml 顶部写入警告：改限制用 `docker update`，勿重建。
4. **环境差异备案**：θ0/θ10/θ30 在长期运行的服务实例上评估；θ20 在 07:58 新初始化的实例上评估（verifier_error 率 1.6–2.9%，均在 E013 时期区间内）。若未来发现与服务器实例年龄相关的系统性行为差异，θ20 需复评。
5. step20 最终以 systemd --user 单元（`e018-step20`）运行，与终端生命周期解耦；08:05:28 启动，08:56:56 exit 0。

**E018-D Verifier-Error Adjudication（同日补做，2026-09-17 10:16–12:44 UTC）**

- 动机：θ0 与 θ30 的 verifier_error 差（15 vs 9，6 个候选）大于 θ30−θ0 的 verified 差（70 vs 65，5 个候选）；在把 +0.98pp 当作正向证据前，先清理测量噪声（原则：measurement 必须先比 effect 可靠）。
- 方法：四份 E018 JSON 中全部 `verify_status == verifier_error` 候选（43 个：15+11+8+9），用已存 `proof` 字段原样重验，**不重新采样**；单候选请求（隔离、无批内竞争）、热服务端、固定 120 s 超时（与原单验路径一致）、最多 3 次尝试（verified 或两次同型确定性结果提前停）；运行于 systemd 单元 `e018d`（支持断点续跑）。
- 环境：容器上限 40 GiB（docker update 热调）；注：此前 08:57 曾将上限恢复为 24 GiB，但正常暖 REPL 池占 26–30 GiB，降额直接杀空整个池（容器降至 0.6 GB、所有 /verify 等待 REPL 超时）——重新预热（冷 Mathlib 导入 82.4 s）后恢复；compose 已改为 40 GiB（提交 `117e1eb`）。
- 结果（43/43 全部有确定性结论，**零修正**）：

| 类别 | 数量 | 说明 |
|---|---|---|
| `deterministic_lean_failure` | 8 | 真实 Lean 拒绝（parse error、linarith 失败等，最快 0.15 s） |
| `deterministic_timeout_or_resource_exhaustion` | 35 | 在固定 120 s / 40 GiB 验证政策下可复现的候选关联资源失败（reproducible candidate-associated resource failures）：Lean 内部确定性超时（`(deterministic) timeout at isDefEq`）或两次 120 s 超时，集中于定理 #53/#40/#50 家族 |
| `verified_on_recheck` | 0 | — |
| `transient_infrastructure_failure` | 0 | — |
| `unresolved_verifier_error` | 0 | — |

  - 校正后 verified 计数与观测完全一致（65/65/64/70）；配对 Δ 不变（+0.0000 / −0.0020 / +0.0098，CI 不变）。故意的 all-errors-success 反事实敏感性边界：若把四个 checkpoint 的 43 个 verifier_error 全部“乐观”记为通过，θ30−θ0 Δ 变为 −0.00195——该区间 [−0.20pp, +0.98pp] 跨越 0，**仅作为反事实上界报告，不表示符号稳健**；可推断的只是 corrected = observed。
- 结论：E018-C 的 verifier_error 在固定 120 s / 40 GiB 策略下**全部为可复现的候选关联失败**（35 资源型 + 8 真实拒绝）——零可恢复错误（verified_on_recheck/transient/unresolved 均为 0），无需基础设施侧修正；POSITIVE-INCONCLUSIVE 判定不变，“+0.98pp 是否为测量噪声”的疑虑被排除。θ0/θ30 主对比（最一致环境）可继续承担主结论；θ20 仍标注环境偏差、降权。
- 产物：`experiments/results/e018d_verifier_error_adjudication.json`（逐候选每次尝试的 kind/status/message/duration 全记录）；脚本 `scripts/p3c_adjudicate_verifier_errors.py`。

- 产物：`experiments/results/e018_{base,step10,step20,step30}.json`、`p3c_analysis.json`、`e018a_smoke_*.json`、`e018b_preview_*.json`、`p3c_checkpoint_sanity.json`、`e018d_verifier_error_adjudication.json`；摘要 `experiments/manifests/p3c_fixed_eval.yaml`；固定集 `experiments/manifests/p3c_fixed_set.json`。
- 提交：`e8272e2`（feat pipeline）、`1fb6866`（verify fix）、`d64c31b`（infra cap）、`862aa77`（recreate 警告）、`32f039f`（exp 记录）、`117e1eb`（40 GiB cap）、`f87e282`（E018-D 脚本）。

---

## 2026-09-17/18（M1 Extension）

### E019 M1 Step30 → Step60 training extension（已完成；Case B → POSITIVE-INCONCLUSIVE）

- 目的：判断 P3-B 中观察到的正向训练动力学，在继续相同配置训练 30 个 optimizer steps 后，是否转化为更清楚的 held-out-from-pilot fixed-set capability gain（训练时长延长实验，非新算法实验）。
- 设置：完全冻结 P3-B E017 配置（GRPO/DrGRPO、`norm_adv_by_std_in_grpo=False`、FULL-FT、n=8、temp 1.0、top_p 1.0、tb4→32 seq/step、max_prompt 1024、max_response 4096、lr 2e-6、无 KL/entropy、vLLM util 0.40、multiturn off、Lean 40 GiB cap）。
- 命令：`bash scripts/run_e019.sh`（bounded resume runner；`--steps` 为**总目标 60**；VERL `resume_mode=auto` 从 `runs/p3b_pilot/global_step_30` 恢复——日志实证 `Loaded lr_scheduler from .../global_step_30`、进度条从 30/60 起步，未误跑成 90）。
- 结果（训练）：**exit 0**；13:09:12 → 14:24:56 UTC（wall **4534 s ≈ 75.6 min ≈ 1.26 GPU-h**，30 步、~140 s/步）；checkpoints `global_step_40/50/60` 已保存（10–60 全部在位）；峰值显存 allocated 24.4 GB / reserved 26.0 GB；无 OOM/NaN。良性记录：一次 Ray dashboard MetricsHead 启动 traceback（仅 UI 子进程，训练正常继续）。
- 全过程动力学（`rollout_dynamics.py` 与 `trainer_metrics.py` 双解析器交叉验证；逐 step rollout dump 1–60 完整）：

| range | IGR | Z | O | score | 非零梯度 | clip | entropy | resp_len |
|---|---|---|---|---|---|---|---|---|
| 1–10 | 0.100 | 0.900 | 0.000 | 0.0594 | 4/10 | 0.578 | 26.5 | 3598 |
| 11–20 | 0.200 | 0.800 | 0.000 | 0.1000 | 5/10 | 0.522 | 26.4 | 3519 |
| 21–30 | **0.225** | 0.750 | 0.025 | **0.1750** | 7/10 | 0.531 | 26.3 | 3510 |
| 31–40 | 0.175 | 0.825 | 0.000 | 0.0813 | 4/10 | 0.541 | 23.9 | 3572 |
| 41–50 | 0.100 | 0.900 | 0.000 | 0.0500 | 3/10 | 0.566 | 28.6 | 3626 |
| 51–60 | 0.150 | 0.850 | 0.000 | 0.0688 | 4/10 | 0.500 | 28.8 | 3523 |

  - **关键观察：训练期奖励信息量在 21–30 达到峰值（IGR 0.225 / score 0.175），31–50 明显回退（41–50 回到 0.100 / 0.050，与最初步段相当），51–60 部分恢复（0.150 / 0.0688）**。每步定理集不同、单步方差大——为聚合趋势；其是否映射到 fixed-set 能力正是 step60 评估要裁决的问题（判定规则已预冻结）。
- checkpoint sanity（§十五）：导出 bitwise 一致（311/311 keys，3.01 GB，`runs/p3c_models/step_60`）；**θ60≠θ30（rel_L2 = 1.34e-4，311/311 键变化）**、θ60≠θ0（2.35e-4）；漂移序列 θ0→θ10/20/30/60 = 0.91/1.35/1.67/**2.35e-4**。
- **step60 fixed-set 评估（完成）**：与 E018-C 完全同协议（封存 64 定理集、64×8、同种子表、40 GiB / 120 s、严格 Kimina 2.0.0）；主比较 θ60 vs 复用的 θ0 artifact。运行记录：systemd-oomd 两次击杀评估单元（14:38:49 / 15:49:35 UTC），监督器 `e019sup` 自动重启后 attempt 2 一次跑完（exit 0，16:14:18）。
- **step60 结果**：观测 verified **62/512**（pass@1 0.1211、pass@4 0.2228、pass@8 0.2656、IGR 0.250、截断 0.373）；但 `verifier_error = 32`（6.25%，均为 `Connection reset by peer` / `Server disconnected`，集中于定理 #49/50/53/55）。
- **E019-D verifier-error 复核**（严格 E018-D 协议：存证重验、单候选、暖服务、40 GiB/120 s、≤3 次）：32 个全部有确定性结论——**7 个 `verified_on_recheck`（全部来自定理 #55，其整组候选被原批次连接重置摧毁；复验 7/8 通过）**、20 个真实 Lean 拒绝、5 个候选关联资源失败、0 瞬时/0 未决。**校正计数 62 → 69/512**（pass@1 0.1348、pass@4 0.2384、pass@8 0.2812、IGR 0.266）。
- 定理级配对（校正计数，bootstrap 10k/seed 20260917）：
  - **θ60c vs θ0：+0.78pp，CI [−2.54, +4.30]，W/T/L = 8/47/9，McNemar p=1.0（+2/−2）**；
  - θ60c vs θ30：−0.20pp，CI [−3.52, +3.12]，W/T/L = 8/48/8，McNemar p=1.0（+1/−2）。
- **判定（预冻结规则）：Case B → POSITIVE-INCONCLUSIVE**——校正后小幅为正且 CI 宽跨零；θ60≈θ30（持平）。结合训练期动力学（21–30 峰值后回退），如实结论：**继续训练到 step60 既未加强也未摧毁 step30 的信号**。按规则：停止 seed1 长训练，进入 replication。
- **Phase 1B 暴露审计**（`e019_fixed_set_overlap.json`）：steps 31–60 共 960 dump 行、116 个 unique 训练 prompt、与 P3-C 固定集**交集 = 0**（无直接训练暴露）；但 P3-C 集已被用于 step30 决策（adaptive reuse），据协议降级为 **M1 development / diagnostic fixed set**；最终论文另建独立 holdout（Phase 5）。
- 运行事故记录：systemd-oomd 在用户切片内存压力下击杀评估单元两次（已由监督器自动重启）；训练本身在 tmux 下 75.6 min 一次跑完。
- 产物：`experiments/results/{e019_dynamics,e019_trainer_metrics}.json`、`runs/p3b_pilot/global_step_{40,50,60}`、`runs/p3c_models/step_60`、`experiments/results/p3c_checkpoint_sanity.json`（含 step60）；摘要 `experiments/manifests/m1_step60.yaml`（evaluation: complete；Case B）。
- 提交：`79c1f6e`（runner）、`0c736b8`（进度条）、`08a78b6`（step60 注册）、`33c0da7`（判定规则冻结）、`4a92a61`/`e1b0051`（双解析器）。

---

## 2026-09-18（M1 Seed Replication）

### E020 M1 seed2 replication（60 步完整复现，训练动力学分析）

- 目的：检验 0.6B RL 训练动力学能否在独立随机轨迹中重复（补论文最大弱点：只有 1 个训练 seed）。
- 设置/预注册：`experiments/manifests/m1_seed_replication.yaml`（运行前冻结）；seed 机制经 `docs/seed_control_audit.md` 审计——本 pinned build 唯一可靠种子为 **`+data.seed=20260918`**（dataloader 洗牌；FSDP rollout 引擎种子固定 0、不可配）。其余配置与 E019 完全相同；从 Distill 冷启动，目录 `runs/m1_seed2/`。
- Live sanity（启动后）：step-1 的 4 个定理与 seed1 **完全不同（重叠 0）**——seed plumbing 生效。
- 结果：**60/60 步 exit 0**，wall **2:18:10 ≈ 2.30 GPU-h**（~138 s/步）；checkpoints 10–60 存在，其中 40/50/60 保留 actor 权重（keep=3 使 10/20/30 的 actor 权重被清理；step30 权重损失不影响本阶段分析——dynamics 来自 rollout dumps，final holdout 只评 step60 且其权重已导出）；1 处良性 Ray-dashboard traceback（同 E019）。
- 动力学（`rollout_dynamics` 与 `trainer_metrics` 双解析器交叉验证）：

| range | seed1 IGR / score | seed2 IGR / score |
|---|---|---|
| 1–10 | 0.100 / 0.0594 | 0.150 / 0.0437 |
| 11–20 | 0.200 / 0.1000 | 0.050 / 0.0437 |
| 21–30 | 0.225 / 0.1750 | 0.175 / 0.0750 |
| 31–40 | 0.175 / 0.0813 | 0.150 / 0.0844 |
| 41–50 | 0.100 / 0.0500 | **0.250 / 0.1437** |
| 51–60 | 0.150 / 0.0688 | 0.125 / 0.0563 |
| 全体 | 0.158 / 0.0891 | 0.150 / 0.0745 |

- **结论（描述性；只有 2 个训练 seed，不做 seed 级显著性推断）**：
  - 整体量级复现：全期 IGR 0.150 vs 0.158、score 0.075 vs 0.089——量级相近；
  - **“中期进入 informative frontier、末段回落”的形态在两个 seed 均出现**（末段 IGR 低于各自峰值）；
  - 但**峰值位置是种子相关的**（seed1 在 21–30、seed2 在 41–50）——“21–30 峰值后回退”不是跨 seed 的固定模式。
- checkpoint：`runs/m1_seed2_models/step_60`（导出 bitwise 一致 311/311 keys，3.01 GB；rel_L2 vs base = 2.41e-4，与 seed1 的 2.35e-4 量级一致）；sanity `experiments/results/e020_seed2_checkpoint_sanity.json`。
- 产物：`experiments/results/{e020_seed2_dynamics,e020_seed2_trainer_metrics,e020_seed2_checkpoint_sanity}.json`、`runs/m1_seed2/`、`runs/m1_seed2_models/step_60`。

---

### E020-M Temperature × group-size mechanism calibration（Phase 3）

- 目的：解释“为何 P2.5 IGR 仅 3.1%、而 P3-0/n=8 后 RL 信号明显增加”——在固定诊断协议下量化 temperature / 组大小与 informative reward 组概率的关系。
- 集合：`igr_mechanism_set.json`（封存 64 定理，seed 20260918；排除项目全部历史已见 statement：E013/E016/P3-C/E017+E019/seed2，共 520 个）。模型固定 Distill base。
- 设计：仅生成两组 n=8（temp 1.0 / 0.6，同一 theorem/sample 种子表）；n=4 条件用每定理前 4 个样本导出，无需额外生成。候选 2×512。
- 结果（`e020_mechanism_analysis.json`）：

| condition | verified rate | IGR | Z | O | trunc | informative/1M tok |
|---|---|---|---|---|---|---|
| T=0.6 n=8 | 0.244 | **0.406** | 0.562 | 0.031 | 0.551 | 15.40 |
| T=0.6 n=4 | 0.246 | 0.312 | 0.625 | 0.062 | 0.551 | 23.75 |
| T=1.0 n=8 | 0.250 | 0.328 | 0.625 | 0.047 | 0.395 | 12.69 |
| T=1.0 n=4 | 0.250 | 0.281 | 0.625 | 0.094 | 0.398 | 21.98 |

- 定理级配对（informative 指示，newly/lost/still_inf/still_deg + exact McNemar）：
  - **n=8 vs n=4 @T=0.6：+6/−0，p=0.031（显著）**；@T=1.0：+3/−0，p=0.250——**增大 n 单向增加 informative 组、零损失**；
  - T=0.6 vs T=1.0：n=8 时 +2/−7（p=0.180）、n=4 时 +4/−6（p=0.754）——降温方向亦提高 IGR，但未达显著，且截断率 +16pp。
- **解释（按预注册纪律）**：temperature / 组大小与“固定诊断协议下获得 informative reward 组的概率”相关；**不**声称“更大 n 导致更好的最终 RL 性能”。机制观察：B 四条件的候选成功率几乎不变（0.244–0.250），变化的是分布位置（n 增大将部分 all-one 组暴露为 mixed：O 从 0.094→0.047 @T=1.0、0.062→0.031 @T=0.6）。
- 事件记录（避免重演）：t06 首次运行遭遇**容器 40 GiB 上限抖动**——seed2 的 60 步训练使 REPl 池膨胀至 15+×3.4GB，t06 验证请求在回收抖动中停滞（日志静止 54 min、0/4 chunk、无 warn、/health 无响应）。处理：停客户端 → `docker restart`（保留可写层！清空膨胀池，冷启动 114 s）→ **t06 以降低的验证并发（4×4）重跑成功**（t10 用默认 8×8 已完成；并发为基础设施旋钮，proof/timeout/validity 不变，已写入 artifacts 的 settings 字段）。教训：长训练后/重评估前应对 Lean server 做 `docker restart` 重置池。
- 产物：`experiments/results/{e020_mech_t10_n8,e020_mech_t06_n8,e020_mechanism_analysis}.json`；集合 `experiments/manifests/igr_mechanism_set.json`。

---

### E021 Qwen3-0.6B-Base cold-start diagnostic（Phase 4C/4D）

- 目的：回答 M2 第一问——比 Distill 更弱的 `Qwen3-0.6B-Base`（无 Lean/chat 先验，pin `da87bfb6`）在固定单轮 Lean RL 协议下是否还有可用 verifier reward signal。
- 设置：与 Distill 机制实验**完全同协议**（`igr_mechanism_set.json`、64×8、temp 1.0 / top_p 1.0 / max 4096、同种子表、严格 Kimina 2.0.0、prompt 逐字节一致——接口审计见 `docs/m2_prompt_interface_audit.md`；唯一记录偏差 = Base eos 为 `<|endoftext|>`，不予修正；验证并发 4×4）。命令：evaluator `--checkpoint qwen3_base`。
- 结果（`e020_qwen_base_n8.json`）：**verified = 1/512（0.2%）、Pass@1 = 0.002、IGR = 0.0156（64 组中仅 1 个 mixed）、solved≥1 = 1/64**。
- 错误构成：**lean_parse_error 464/512 = 90.6%**（产出大量非 Lean 文本）、semantic 43、format-invalid 4、verifier_error 0；截断仅 12.3%（median 长度 907 tokens——短而无效，而非撞上限）。
- 对照（同集合同协议，Distill base）：verified 128/512（25%）、IGR 0.328——**弱起点的可验证信号率相差约 100×**。
- **M2 gate 判定（预注册阈值）**：IGR 0.0156 ∈ [0.01, 0.05) → **WEAK** → 按协议执行 5 步 GRPO smoke。
- **5 步 smoke（E021-SMOKE）**：exit 0，checkpoint `global_step_5` 已保存（loss/grad 有限、entropy 31–57）；但 **5 步全部 score_mean=0、grad_norm=0、pg_loss=0；rollout IGR=0.000 / Z=1.000（20 组全零）**。
- **M2 结论**：Base 在固定单轮协议下处于 **reward-dead 边界**（IGR 0.0156、验证率较 Distill 低 ~100×、smoke 零混合组）——该协议上的 RL 不可行，cold-start 需干预（verified SFT / 不同协议）。
- 产物：`experiments/results/{e020_qwen_base_n8,e021_smoke_dynamics}.json`、`runs/m2_qwen_smoke/global_step_5`。

---

### E022 M1 seed3 replication（中止；外部 GPU 任务冲突 + oomd，按协议停止）

- 目的：预注册的第三个训练种子（+data.seed=20260919），规则与 seed2 完全相同；live sanity 已确认 seed1/seed2/seed3 的 step-1 定理集**两两不相交**（三条独立轨迹）。
- 运行记录：2026-09-18 04:42 UTC 由监督器 `m1s3sup` 启动；正常完成 steps 1–3（rollout dumps 1–3 在位，速率 ~159 s/步）。**05:02:32 UTC 被 systemd-oomd 击杀**（`oom-kill`，消耗 CPU 30min59s）。
- **根因（外部因素）**：同一时段出现**非本项目的 GPU 训练任务** `experiment/train_p2_calibration.py --model gap`（run `fullsa_p2_gap_calib_640_100`，登录会话 session-628，~04:53 UTC 启动，5 进程、占 5.8 GB 显存、82%+18%×4 CPU），违反协议“同一时间只允许一个 GPU-heavy job”的前提；用户切片内存压力叠加使 oomd 再次命中我们的大单元（本日同因击杀已多次：e019e ×2、t06 池膨胀、m1s3sup）。**未对外部任务做任何干预**。
- **决定（协议强制）**：①不重启 seed3（“同一种原因连续失败两次后不允许无限 retry”）；②**暂停后续 GPU 阶段**（final holdout 评估、MiniF2F）直至外部任务结束且主机稳定；③保留 `runs/m1_seed3/rollout_data/` 1–3.jsonl 作为中止证据。
- 恢复条件（给下一次窗口）：外部 GPU 任务结束后 → 重启 seed3（同一预注册 seed 20260919）→ 构建 M1 final holdout（构建器 `scripts/build_final_holdout.py` 已就绪，届时 seed3 语句自动纳入排除）→ holdout 评估 θ0/seed1/seed2/seed3 → MiniF2F。
- 本轮无人值守批次的完成项（不受影响）：E020 seed2 复制 ✓、E020-M 机制实验 ✓、E021 Qwen-Base 诊断与 smoke ✓。

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
