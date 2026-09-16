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
- P3 输入冻结：`max_response=4096`（成功长度 P95≈3,000）、串行验证（验证时间占比 1.6%）、奖励契约（`verifier_error` 单独记录计 0）、IGR 基线 Distill 28.1% / RL 31.3%。

### E009 P2.5 W1 Promptset 32×4 cached rollout profile（停止条件①）

- 目的：在官方 Kimina-Prover-Promptset 子集上量化 GRPO 所需的组奖励信号（positive/mixed 组），并落盘 cached rollout batch 供 W2/W3 复用。
- 设置：32 个 unique statement（seed 0）× 4 候选；`max_new_tokens=4096`、`temperature=0.6`、`top_p=0.95`（E007 评估设置；官方 P3 rollout 用 temp 1.0）、batch-size 1（8 GiB 显存）。验证走本地 Kimina server（batch verify → 单候选回退）。
- 命令：`python -u scripts/promptset_rollout_probe.py --batch-size 1 --limit 32 --samples-per-theorem 4 --output experiments/results/p2_5_promptset_profile.json --batch-dir experiments/local_rl_batch`（项目 `.venv` 解释器）。
- 过程：运行 4.7 小时（生成 14,495.7 s + 验证 2,375.7 s）；batch verify 触发 21 次单候选回退（missing_item 20、redeclaration 1）。
- 结果：verified 14/128（严格格式同数）；sorry 0、format_failures 0、`verifier_errors` 5（单独计 0）；**截断 96/128 = 75%**；组率 all_zero 87.5%（28/32）、**mixed 3.125%（1/32）**、all_one 9.375%（3/32），**IGR = 0.03125**；prompt 长度（全 promptset 7,620 unique）median 231 / p95 415 / max 2,365。
- 结论：停止条件①满足（positive 3 组 + mixed 1 组）；all-zero 28/32 未触发计划中的 ≥29/32 风险线，但 IGR 显著低于 miniF2F 的 28–31%，与 75% 截断强相关——P3 用官方 rollout 口径（temp 1.0）时必须先复核截断率与 IGR。数据点：`<think>` 混入 Lean 代码（lean_error 主因）、REPL redeclaration、单候选回退路径被真实使用。
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
- 结论：停止条件③满足——8 GiB 上单步可跑通但**无余量**（2048 组合已溢出物理显存、吞吐不可预测）；P3 正式训练按计划走 ≥24 GB 云卡。
- 产物：`experiments/results/p2_5_lora_step_probe.json`、`experiments/results/_lora_cpu_smoke.json`（GPU 探针创建于 2026-09-16T12:40:11Z；CPU 冒烟版 2026-09-16T09:15:21Z）。

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
