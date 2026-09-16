# Preliminary Study — Local RL Readiness（P2.5 阶段记录）

> **定位**：TinyLean-RL 的 P2.5 阶段（本地 RL 就绪验证，2026-09-16）的完整记录与结论。主线研究计划见 [`../research_plan.md`](../research_plan.md)；逐次实验条目见 [`../experiment_log.md`](../experiment_log.md)（E009–E011）；证据清单见 `experiments/manifests/p2_5_complete.yaml`。
> **原则**：test the Kimina logic, don't replace it —— 不引入 TRL、不自造 GRPO；实现一律以 pinned commit `e16b605e8186614c685875c9b57eb19e841b521a` 为准（GRPO 目标已移植为 `tinylean_rl.rl.grpo`，27 个参考测试锁数学）。
> **目标**：在本机（Windows + RTX 4060 8 GiB）回答“RL 的前置条件是否全部成立”，把正式 P3 RL pilot 交给云上 ≥ 24 GB 单卡。

## 一、停止条件与判定（6/6 满足）

| # | 停止条件 | 判定 | 证据 |
|---|---|---|---|
| ① | Promptset 上观察到 positive/mixed reward 组 | ✅ mixed 1/32、all-one 3/32 | E009 · `p2_5_promptset_profile.json` |
| ② | cached rollout → GRPO advantage/loss 算通 | ✅ 非零 advantage ±0.5、grad_norm 2.036 | E010 · `p2_5_grpo_rehearsal_mixed.json` |
| ③ | backward 能跑（成功或 OOM 都算回答） | ✅ 四组合全过；2048 组合溢出物理显存 | E011 · `p2_5_lora_step_probe.json` |
| ④ | `docs/p3_config_audit.md` 完成 | ✅ | 审计文档 |
| ⑤ | 云启动命令链写好（env.sh → compose → doctor.sh → run_p3_smoke.sh） | ✅ | W6 · `run_p3_smoke.sh` + prewarm |
| ⑥ | reward contract 写回 research_plan 与 audit | ✅ | `research_plan.md` §四（P2.5 W4 审计补充） |

## 二、W1 Promptset 训练分布诊断（E009）

官方 Kimina-Prover-Promptset 抽 32 个 unique statement × 4 候选（seed 0；评估口径 temp 0.6 / top_p 0.95 / max 4096），生成后逐候选经本地 Kimina server 验证，并把 cached rollout batch 落盘供 W2/W3 复用。

| 指标 | 值 |
|---|---|
| verified（严格格式同数） | 14/128（10.9%） |
| 截断（撞 4096 上限） | **96/128（75%）** |
| sorry / format_failures | 0 / 0 |
| verifier_errors | 5（单独计 0） |
| 组率 all_zero / mixed / all_one | 87.5% / **3.125%** / 9.375% |
| **IGR** | **0.03125** |
| batch verify 单候选回退 | 21 次（missing_item 20、redeclaration 1） |
| prompt 长度（全 promptset 7,620 unique） | median 231 / p95 415 / max 2,365 |

- **停止条件①成立**：positive（3 组 all-one）与 mixed（1 组）组均出现；all-zero 28/32（87.5%）未触发计划中的 ≥29/32 风险线，但显著高于 miniF2F 上的 53%/44%（E007/E008）。
- **IGR 与截断强相关**：miniF2F 上 IGR 28–31%，Promptset 上仅 3.1% —— 与 75% 截断率同一量级出现。P3 用官方 rollout 口径（temp 1.0）时必须先复核截断率与 IGR；若 IGR 持续过低，GRPO 有效组极少。
- 训练分布数据点：`<think>` 混入 Lean 代码（lean_error 主因）；REPL redeclaration；batch verify 偶发 missing item → 单候选回退路径被真实使用 21 次。

## 三、W2 GRPO 移植与重演（E010）

移植 pinned commit 的 DrGRPO 目标（`norm_adv_by_std_in_grpo=False` mean-only 中心化、`loss_agg_mode="seq-mean-token-sum-norm"`、非对称 clip 0.2/0.3、`clip_ratio_c=3.0`、无 KL、熵系数 0），先用 synthetic fixture 参考测试锁死数学，再在真实 cached rollout 上重演 advantage → loss → backward（CPU fp32，response 截断 1024）。

| 运行 | 数据 | advantage | per-candidate loss | grad_norm | 时长 |
|---|---|---|---|---|---|
| 链自动 | 前 8 定理（恰好全零组） | 全 0 | 全 0 | 0（有限） | 1,914 s |
| 混合组补跑 | 定理 15–19（含唯一 mixed 组 #17） | **±0.5（非零）** | **±0.386** | **2.036（有限）** | 1,163 s |

- **停止条件②成立**：非均匀奖励组产生非零 advantage、非零 per-sample loss 与有限非零梯度——链路在真实数据上完整成立。
- 一个容易误读的数字：混合组补跑的 `loss.total = 0` 是 mean-only 中心化下组内 Σadv=0 的数学对称性（±0.5×4 抵消），**不是失败信号**；per-candidate 项（+0.386/−0.386）与 grad_norm=2.036 才是路径有效的证据。
- 87.5% 的 all-zero 组率意味着 8 定理子集大概率无梯度信号（链的首跑即命中）——IGR 是 P3 的核心观察指标。

## 四、W3 LoRA 显存探针（E011）

问题：8 GiB RTX 4060 能否完成一步 LoRA GRPO 反传？答案：**能跑通，但没有余量**。

| 组合 | 峰值 reserved | forward / backward | 备注 |
|---|---|---|---|
| r16 @ seq 1024 | 4,016 MB | 0.17 s / 1.03 s | 舒适 |
| r32 @ seq 1024 | 4,126 MB | 0.26 s / 0.27 s | 舒适 |
| r16 @ seq 2048 | **8,642 MB** | 1.34 s / 2.46 s | **溢出物理显存 454 MB** |
| r32 @ seq 2048 | **8,752 MB** | 0.72 s / 1.73 s | **溢出物理显存 565 MB** |

- 两个 2048 组合靠 Windows 共享显存（WDDM）兜底完成（`peak_headroom = −564.5 MB`）——真实训练不可依赖：PCIe 往返、吞吐不可预测、其他进程占用时直接 OOM。
- 先行 CPU bf16 冒烟贡献了本期最有价值的排障：一次运行逮到并修复 2 个真 bug（cached batch 的 reward 字段引用、PEFT `get_base_model()` 解包），最终 **224/224 LoRA 张量全部收到梯度**（q/k/v/o × 28 层 × A/B）。
- 结论：本地只做前置验证的决策被数据支持；P3 按计划上 ≥ 24 GB 云卡（r16/r32 的 LoRA 单步路径已由本探针证明可实现）。

## 五、W4 配置审计要点（停止条件④⑥）

审计 pinned commit 的 P3 config surface，冻结以下契约（已写回 [`research_plan.md`](../research_plan.md) §四，详见 [`../p3_config_audit.md`](../p3_config_audit.md)）：

1. **format gating**：官方最终 score = `proof_rw × format_rw`（`reward/reward.py` L153），格式失败直接 0；
2. **并发策略**：官方 `max_workers=40` 仅适用于预热服务器池；本地/P3 先 prewarm，以 `lean_server_warmup.json` 的 `recommended_concurrency=2` 为准，不超放；
3. **单轮声明**：官方 0.6B 开 multiturn（数据侧第二轮回炉）；P3-v1 关闭（`data.multiturn=False`）并在论文中声明；
4. `verifier_error` 候选 → reward 0 且单独统计；
5. P3 起统一严格验证口径（含 sorry 检查）。

## 六、W5 cold-start SFT 数据管道（NuminaMath-LEAN）

为将来的 cold-start SFT / 对照分支备料（本阶段只准备，不训练）：

| 指标 | 值 |
|---|---|
| 原始 → 过滤后 | 104,155 → **31,634**（保留 statement + proof 完整行） |
| 去重 / MiniF2F 污染 | 0 / **0**（244 行对照，示例为空） |
| 抽样验证（200 条） | **190 verified（95%）**、5 lean_error、5 verifier_error、0 sorry |
| token 长度（statement） | median 150 / p95 292 / max 2,998 |
| token 长度（完整文件） | median 417 / p95 1,436 / max 5,303 |
| split | train 31,002 / val 632（2%，seed 0） |

产物：`data/processed/sft_cold_start/{train,val}.parquet` + `p2_5_sft_prep.json`（全流程 3,866 s）。

## 七、W6 云部署命令链与预热（停止条件⑤）

命令链：`source scripts/env.sh` → `uv sync --extra inference` → `docker compose -f infra/lean-server/compose.yaml up -d` → `bash scripts/doctor.sh` → `bash scripts/run_p3_smoke.sh --steps 3`（由 `experiments/manifests/p2_5_complete.yaml` 门控）。

- `scripts/prewarm_lean_server.py`：预热 + 并发阶梯测量。实测（热服务器）：预热 2 次后 c=1 中位 0.564 s / 1.96 rps，c=2 中位 0.501 s / 3.93 rps；`recommended_concurrency = 2`。
- **口径注意**：该测量发生在 W1 刚结束（服务器已被打热）之后；冷容器首请求成本另见 native_verify 记录（Mathlib 载入分钟级）。因此 `run_p3_smoke.sh` 默认先 prewarm，`--skip-prewarm` 仅在已确证服务器热的场景使用。
- `doctor.sh` 新增：GPU ≥ 24 GB 目标检查、Ray/VERL import 检查、Lean verify 延迟探针（冷 REPL 预警）、Promptset parquet 与 P2.5 manifest 检查。

## 八、P3 输入冻结（v1，与 audit 一致）

| 输入 | 值 | 依据 |
|---|---|---|
| max_response | 4,096 | E007/E008 成功长度 P95 ≈ 3,000 |
| temperature | 1.0（官方 rollout 口径；W1 用的是评估口径 0.6，P3 需复核） | audit §6 |
| verifier | 串行小批 + prewarm，concurrency ≤ 2 | W6 测量 + audit |
| multiturn | off（`data.multiturn=False`） | audit §7 |
| reward | 严格口径；`verifier_error` → 0 且单独统计 | audit §6/§7 |
| GPU | ≥ 24 GB 单卡（云） | W3：2048 组合溢出 8 GiB |
| LoRA | r16/r32，q/k/v/o，alpha=2r，dropout 0，AdamW lr 2e-6 | W3 探针 + audit |

## 九、已知风险与 P3 前置检查

1. **IGR 风险（最高）**：Promptset 上 3.1%（temp 0.6 / 4096，75% 截断）——P3 rollout 第一件事是复核截断率与 IGR；若持续 < 5%，需调整采样温度/budget 或数据子集。
2. **verifier 冷启动**：冷容器 Mathlib 载入分钟级；P3 必须先 prewarm 再 rollout。
3. **native_decide / REPL 崩溃**：`verifier_error` 容错路径已实现（重试 → 单候选回退 → partial 保存），保持单独统计。
4. **Windows 共享显存**：不可依赖（W3 的 2048 组合即例证）。
