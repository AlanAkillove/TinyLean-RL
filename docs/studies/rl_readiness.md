# Preliminary Study — Local RL Readiness（P2.5 阶段记录）

> **定位**：TinyLean-RL 的 P2.5 阶段（本地 RL 就绪验证，2026-09-16）的完整记录与结论。主线研究计划见 [`../research_plan.md`](../research_plan.md)；逐次实验条目见 [`../experiment_log.md`](../experiment_log.md)（E009–E011）；证据清单见 `experiments/manifests/p2_5_complete.yaml`。
> **原则**：test the Kimina logic, don't replace it —— 不引入 TRL、不自造 GRPO；实现一律以 pinned commit `e16b605e8186614c685875c9b57eb19e841b521a` 为准（GRPO 目标已移植为 `tinylean_rl.rl.grpo`，19 个参考测试锁数学）。
> **目标**：在本机（Windows + RTX 4060 8 GiB）回答“RL 的前置条件是否全部成立”，把正式 P3 RL pilot 交给 Linux + RTX 3090 24 GB（下阶段 **P3-0 — Linux Migration & On-Policy Calibration**，见 [`../p3_linux_handoff.md`](../p3_linux_handoff.md)）。

## 一、停止条件与判定（6/6 满足）

| # | 停止条件 | 判定 | 证据 |
|---|---|---|---|
| ① | Promptset 上观察到 positive/mixed reward 组 | ✅ mixed 1/32、all-one 3/32 | E009 · `p2_5_promptset_profile.json` |
| ② | cached rollout → GRPO advantage/loss 算通 | ✅ 非零 advantage ±0.5、grad_norm 2.036 | E010 · `p2_5_grpo_rehearsal_mixed.json` |
| ③ | backward 能跑（成功或 OOM 都算回答） | ✅ 四组合全过；2048 组合溢出物理显存 | E011 · `p2_5_lora_step_probe.json` |
| ④ | `docs/p3_config_audit.md` 完成 | ✅ | 审计文档 |
| ⑤ | 部署启动命令链写好（Linux：env.sh → compose → doctor.sh → run_p3_smoke.sh） | ✅ | W6 · `run_p3_smoke.sh` + prewarm |
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
- **IGR 与截断强相关**：miniF2F 上 IGR 28–31%，Promptset 上仅 3.1% —— 与 75% 截断率同一量级出现。**3.1% 不应读作“模型不适合 RL”**：只能解释为 *under temp=0.6 / n=4 / max_response=4096，训练分布产生稀疏信息组*；官方 rollout 口径（temp 1.0）尚未测试。P3-0 必须先复核截断率与 IGR；若 IGR 持续过低（≲5%），恢复顺序 = ① n=8（官方 baseline 组件）② multiturn=true + 0.5。
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
- 结论：**8 GiB Windows 不适合真实 P3**（2048 组合已溢出物理显存、依赖 WDDM 共享内存，吞吐不可预测）；LoRA 单步路径已由本探针证明可实现——但这不等于“P3 必须 LoRA”：训练模式首选 full-parameter，LoRA r16/r32 为 fallback，最终在 P3-0 显存可行性探针后冻结。

## 五、W4 配置审计要点（停止条件④⑥）

审计 pinned commit 的 P3 config surface，冻结以下契约（已写回 [`research_plan.md`](../research_plan.md) §四，详见 [`../p3_config_audit.md`](../p3_config_audit.md)）：

1. **format gating**：官方最终 score = `proof_rw × format_rw`（`reward/reward.py` L153），格式失败直接 0；
2. **并发策略**：官方 `max_workers=40` 仅适用于预热服务器池；本地/P3 先 prewarm，以 `lean_server_warmup.json` 的 `recommended_concurrency=2` 为准，不超放；
3. **单轮声明**：官方 0.6B 开 multiturn（数据侧第二轮回炉）；P3-A 起步关闭（`data.multiturn=False`）并在论文中声明，P3-0 校准若 IGR 过低则按 n=8 → multiturn 恢复；
4. `verifier_error` 候选 → reward 0 且单独统计；
5. P3 起统一严格验证口径（含 sorry 检查）。

## 六、W5 cold-start SFT 数据管道（NuminaMath-LEAN）

为 M2（cold-start trainability）备料的 **contingency asset**——prepared only：**未进入 P3 baseline、尚未训练**，不属于当前 P3 主线：

| 指标 | 值 |
|---|---|
| 原始 → 过滤后 | 104,155 → **31,634**（保留 statement + proof 完整行） |
| 去重 / MiniF2F 污染 | 0 / **0**（244 行对照，示例为空） |
| 抽样验证（200 条） | **190 verified（95%）**、5 lean_error、5 verifier_error、0 sorry |
| token 长度（statement） | median 150 / p95 292 / max 2,998 |
| token 长度（完整文件） | median 417 / p95 1,436 / max 5,303 |
| split | train 31,002 / val 632（2%，seed 0） |

产物：`data/processed/sft_cold_start/{train,val}.parquet` + `p2_5_sft_prep.json`（全流程 3,866 s）。

## 七、W6 部署命令链与预热（停止条件⑤）

命令链：`source scripts/env.sh` → `uv sync --extra inference` → `docker compose -f infra/lean-server/compose.yaml up -d` → `bash scripts/doctor.sh`（环境 gate）→ **P3-0 校准**（rollout temp 1.0 / full-FT 显存探针，见 [`../p3_linux_handoff.md`](../p3_linux_handoff.md)）→ 冻结 P3-A 配置 → `bash scripts/run_p3_smoke.sh --steps 3`（**provisional P3-A runner，尚未经 Linux 验证**）。整链由 `experiments/manifests/p2_5_complete.yaml` 门控。

- `scripts/prewarm_lean_server.py`：预热 + 并发阶梯测量。实测（热服务器）：预热 2 次后 c=1 中位 0.564 s / 1.96 rps，c=2 中位 0.501 s / 3.93 rps；`recommended_concurrency = 2`。
- **口径注意**：该测量发生在 W1 刚结束（服务器已被打热）之后；冷容器首请求成本另见 native_verify 记录（Mathlib 载入分钟级）。因此 `run_p3_smoke.sh` 默认先 prewarm，`--skip-prewarm` 仅在已确证服务器热的场景使用。
- `doctor.sh` 新增：GPU ≥ 24 GB 目标检查、Ray/VERL import 检查、Lean verify 延迟探针（冷 REPL 预警）、Promptset parquet 与 P2.5 manifest 检查。

## 八、P3 输入假设（P3-0 校准后冻结；与 audit 一致）

| 输入 | 值（provisional） | 依据 |
|---|---|---|
| rollout n | 4（reduced-compute 起始假设；官方 n=8 为第一恢复项） | audit §2 |
| max_response | 4,096（**P3-0 初始 rollout 预算假设**，非最终；W1@4096 截断 75%，temp 1.0 下需重测） | E007/E008 成功长度 P95 ≈ 3,000 |
| temperature | 1.0（官方 rollout 口径；W1 用的是评估口径 0.6，P3-0 校准首项） | audit §6 |
| verifier | 串行小批 + prewarm，concurrency ≤ 2 | W6 测量 + audit |
| multiturn | off（起步；IGR 过低时恢复顺序 = n=8 → multiturn=0.5） | audit §7 |
| reward | 严格口径；`verifier_error` → 0 且单独统计 | audit §6/§7 |
| 训练模式 | 首选 full-parameter（3090 显存允许时）；LoRA r16/r32（q/k/v/o，alpha=2r，dropout 0）为 fallback；P3-0 显存探针后冻结 | W3 探针 + audit |
| checkpoint | P3-A 在最后一步保存一次（`save_freq=steps`），成功重载/续训是 P3-A 判据 | `run_p3_smoke.sh` |
| GPU | ≥ 24 GB 单卡（Linux RTX 3090） | W3：2048 组合溢出 8 GiB |

## 九、已知风险与 P3 前置检查

1. **IGR 风险（最高）**：Promptset 上 3.1%（temp 0.6 / 4096，75% 截断）——P3-0 第一件事是在官方口径（temp 1.0 / top_p 1.0）下复核截断率与 IGR；若仍 < 5%：恢复顺序 = ① n=8 ② multiturn=true + `multiturn_sampling_rate=0.5`（需同步放大 `max_prompt_length` ≥ 8192）；原则：**先恢复被 reduced-compute baseline 删除的官方机制，再引入自定义算法改进**。
2. **verifier 冷启动**：冷容器 Mathlib 载入分钟级；P3 必须先 prewarm 再 rollout。
3. **native_decide / REPL 崩溃**：`verifier_error` 容错路径已实现（重试 → 单候选回退 → partial 保存），保持单独统计。
4. **Windows 共享显存**：不可依赖（W3 的 2048 组合即例证）。

## 十、冻结与 Linux 交接（2026-09-16）

- **P2.5 冻结**：`win` 分支打 tag `p2.5-win-complete`；本机不再新增推理 / rollout / RL 实验，P2.5 数据与结论冻结。
- **下一阶段**：P3-0 — Linux Migration & On-Policy Calibration（服务器 RTX 3090 24 GB）。范围、未决问题与命令序列见 [`../p3_linux_handoff.md`](../p3_linux_handoff.md)；P3-0 完成前不启动 P3-A。
- **Windows-only 隔离**：本篇与 [`../environment.md`](../environment.md) 中的 Windows 记录（GBK 控制台兼容、WDDM 共享显存、系统代理、Docker Desktop）仅为环境记录，不进入 `env.sh` / `doctor.sh` / `run_p3_smoke.sh` 的 Linux 默认路径。
