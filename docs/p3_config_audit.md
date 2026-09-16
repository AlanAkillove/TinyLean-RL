# P3 Config & Code Audit（P2.5 W4）

审计基线：`third_party/kimina-prover-rl` @ pinned commit
`e16b605e8186614c685875c9b57eb19e841b521a`（2025-08-14，"Add recipe (#44)"）。
所有 `file:line` 均指该 commit 下的仓库文件（根为 `third_party/kimina-prover-rl/`）。
官方 0.6B 参考配置：`recipe/kimina_prover_rl/kimina_prover_0.6B.sh`（下称 *recipe*）。

本文回答 8 组问题并给出本地/云 P3-A 的建议配置；未验证的部分集中在文末"未验证
假设清单"。结论用于修订 `configs/rl/kimina_0.6b_pilot.yaml`。

---

## 1. 模型加载

### 1.1 actor（训练侧）

- 加载函数：`verl/workers/fsdp_workers.py` L243 `_build_model_optimizer`。
- 初始化 dtype：`engine.py` L104 `model_dtype="fp32"` 默认；`fsdp_workers.py` L280-284
  未显式设置时 actor 用 **fp32 建模型**（注释 L269："create model in fp32. Otherwise the
  optimizer is in bf16, which is incorrect"），ref 默认 bf16（L282）。
- `AutoConfig.from_pretrained` 强制 `attn_implementation="flash_attention_2"`
  （`fsdp_workers.py` L287-289）→ **云上必须可安装 flash-attn**（版本与 torch 匹配）。
- FSDP 混精：`fsdp_workers.py` L371-381，默认 `param_dtype=bf16 / reduce_dtype=fp32 /
  buffer_dtype=fp32`；L396-417 以 `sharding_strategy`（FSDP zero3）+ `MixedPrecision`
  包装。L400-402 注释明确：**ref 强制 CPUOffload，actor 不用 CPUOffload**（"causes
  incorrect results when using grad accumulation"）。
- `gradient checkpointing`：recipe L48 `enable_gradient_checkpointing=True`。

### 1.2 rollout（vLLM 侧）

- `vllm_rollout/vllm_rollout_spmd.py` L142：`max_model_len = config.max_model_len or
  (config.prompt_length + config.response_length)`；`prompt_length/response_length`
  经 `rollout.yaml` L24-25 从 `data.max_*` 派生。
- L151：`load_format = "dummy" if config.load_format.startswith("dummy") else ...`；
  `RolloutConfig.load_format` 默认 `"dummy_dtensor"`（`workers/config/rollout.py` L141）
  → 实际传给 vLLM 的是 `"dummy"`。
- L188-192：engine kwargs 含 `max_model_len / gpu_memory_utilization /
  max_num_batched_tokens`；L203-204 `free_cache_engine=True`（默认）→ 每次 rollout 后
  `sleep(level=2)` 释放 KV cache。
- L206-220：采样参数 `n=1`（L218 注释 "already repeat in ray_trainer"），`max_tokens=
  config.response_length`（L209）。

### 1.3 actor 与 vLLM 的权重关系与同步

- **不共享权重**：actor 是 FSDP 训练模块，vLLM 是独立 engine；二者通过
  `workers/sharding_manager/fsdp_vllm.py` 拷贝 state_dict 同步。
- 同步时机：`fsdp_workers.py` L782 `with self.rollout_sharding_manager:` 包裹
  `generate_sequences`（L786）。每次 rollout 前置 `__enter__`（`fsdp_vllm.py`
  L127）收集 FSDP 参数写入 vLLM；因此 **PPO 每步 update_actor 后的下一次 rollout 会
  带上最新权重**（不是训完再同步一次）。
- 同步成本受 `base_sync_done` 控制：`fsdp_vllm.py` L122
  `base_sync_done = "dummy" not in load_format`。默认 dummy 流程会把**全量参数**
  同步进 vLLM（L139-175 收集 `state_dict()` 全量，filter `_flat_param`），即每步一步
  GPU→CPU→GPU 拷贝；`load_format=safetensors` 时 base 预载，只需增量/LoRA。
- `param_offload` / `optimizer_offload`：`fsdp_workers.py` L201-202、L725-728
  （update_actor 前 `load_fsdp_model_to_gpu/load_fsdp_optimizer`）、L755-760（之后
  offload 回 CPU）。actor 的 `fsdp_config.param_offload/optimizer_offload` 均默认
  False（`engine.py` L98-99），recipe 显式 False（L49-50）。

### 1.4 ref policy 是否加载

- `trainer/main_ppo.py` L190-195：**仅当 `algorithm.use_kl_in_reward` 或
  `actor.use_kl_loss` 为 True 才注册 RefPolicy worker**。
- recipe：`use_kl_in_reward=False`（L20/L65）且 `use_kl_loss=False`（L43）→
  **ref worker 不创建、ref 模型不加载**；`ray_trainer.py` L355
  `use_reference_policy = Role.RefPolicy in role_worker_mapping` → False，
  L913-915 的 `ref_policy_wg.init_model()` 不会执行。
- 因此 recipe L63 `ref.fsdp_config.param_offload=True` **在官方 0.6B 配置下完全不
  生效**（配置存在但 worker 不存在）。它的设计作用：ref 启用时把 ref 参数常驻 CPU、
  前向时逐层上卡（`fsdp_workers.py` L204-205 + L400-402 的 CPUOffload 强制），为
  actor 让出显存。
- 对我们的意义：P3-A 沿用无 KL 配置 → **不需要第二个模型副本，显存预算只需
  actor+optimizer+激活+vLLM**。若未来引入 KL，则 ref 显存 ≈ 0.6B·bf16（offload 模式）。

---

## 2. Rollout n=8 的依赖面

官方 n=8（recipe L56）；我们计划 n=4。

- **repeat 位置**：`ray_trainer.py` L1158-1160 为每条 prompt 生成 `uid`（uuid4）；
  L1185 `gen_batch.repeat(repeat_times=rollout.n, interleave=True)`（生成前）；
  L1221 `batch.repeat(repeat_times=rollout.n, interleave=True)`（union 前对齐）。
  interleave 保证同一 prompt 的 n 个样本相邻。
- **advantage 分组**：`ray_trainer.py` L262-273（`compute_advantage`）GRPO 分支以
  `data.non_tensor_batch["uid"]` 为 index；`core_algos.py` L261-324
  `compute_grpo_outcome_advantage`。组大小 = n。L1228 注释确认 DP 负载均衡会打乱顺序
  但不影响 advantage（基于 uid），可能影响 mini-batch 边界（loss 组成）。
- **reward manager**：`workers/reward_manager/batch.py` L84 `reward_tensor =
  zeros_like(responses)`；L89 `valid_response_lengths =
  attention_mask[:, prompt_len:].sum(-1)`；L108 `reward_tensor[i, length-1] = reward`
  （reward 落在最后一个有效 response token，`sum(dim=-1)` 即标量）。
- **整除约束（n 相关）**：
  - `fsdp_workers.py` L209-211：`ppo_mini_batch_size *= rollout.n` 后
    `//= dp_world_size`，断言 > 0。即 **`ppo_mini_batch_size` 以 prompt 数计**，
    序列数为 `ppo_mini_batch_size × n`。
  - `ray_trainer.py` L409-414：`real_train_batch_size = train_batch_size × n` 必须
    `% minimal_bsz(==n_gpus) == 0`。
  - `fsdp_workers.py` L216-229：per-rank `ppo_mini_batch_size %
    ppo_micro_batch_size_per_gpu == 0`。
- **改 n=4 的影响清单**：
  1. 每步序列数减半 → 生成时间约减半（生成是 P3-A 的时长瓶颈）。
  2. `real_train_batch_size % n_gpus == 0` 的约束更容易满足（n=4 对单卡/双卡更友好）。
  3. advantage 组内样本数 8→4：mean-only 中心化下组越小噪声越大（样本数=4 时
     单样本翻转即改变 25% 的组均值），但 DrGRPO 论文（norm by std 关）表明无需按
     std 归一，组大小影响小；IGR（mixed 率）仍是主要驱动。
  4. vLLM 单次 generate 的请求数 = train_batch_size×4 → KV cache 压力更小，
     `gpu_memory_utilization` 可以更低。
  5. 无需改 reward/advantage/mini-batch 代码路径（都不依赖 n 的具体值，只依赖整除关系）。

---

## 3. Sequence length

### 3.1 各配置的生效位置与推导

- `data.max_prompt_length`（`rl_dataset.py` L105，默认 1024）：dataset 侧
  prompt 截断/过滤长度。
- `data.max_response_length`（`_generated_ppo_trainer.yaml` L224，默认 512）：
  → 派生 `rollout.response_length`（`rollout.yaml` L25、
  `_generated_ppo_trainer.yaml` L123：`response_length: ${oc.select:data.max_response_length,512}`）。
- vLLM `max_model_len = prompt_length + response_length`（`vllm_rollout_spmd.py`
  L142），可由 `actor_rollout_ref.rollout.max_model_len` 覆盖。
- 生成截断：`vllm_rollout_spmd.py` L209 `max_tokens=config.response_length` →
  response 顶格即为截断（对应 `metric_utils.py` L154/L192 的
  `response_length/clip_ratio` 指标）。
- `actor.ppo_max_token_len_per_gpu`（`actor.py` L96，默认 16384；`actor.yaml`
  L27-30 注释建议 `n × max_prompt_length + max_response_length`）：**仅
  `use_dynamic_bsz=True` 时生效**（`dp_actor.py` L392-394 `prepare_dynamic_batch`；
  L414-417 loss 缩放）。`use_dynamic_bsz` 默认 False（`actor.py` L95），recipe 未
  覆盖 → **recipe 的 `ppo_max_token_len_per_gpu=32768`（L51）是死配置**（仍按
  `ppo_micro_batch_size_per_gpu=2` 固定切分，`dp_actor.py` L396-399）。
- `rollout.max_num_batched_tokens`（recipe L61 = 32768）：vLLM 单批 token 上限，
  与 `enable_chunked_prefill`（默认 True）联动，须 ≥ `max_model_len` 才能不报警
  （`vllm_rollout_spmd.py` L144-147）。

### 3.2 `data.truncation='error'` 的确切行为

- `rl_dataset.py` L108 默认值即 "error"；recipe L36 显式设置。
- L305-316：仅当 **prompt** token 数 > `max_prompt_length` 时按模式处理；
  `"error"` 分支在 L315-316 `raise RuntimeError`。
- L145-177：`filter_overlong_prompts=True`（默认）时数据集加载阶段先按
  `max_prompt_length` 过滤（多进程），被滤掉的 prompt 不会走到 raise。
- **response 超长不触发该错误**：response 侧只有 vLLM 的 `max_tokens` 截断，
  `data.truncation` 不作用于 response。
- 结论：**response 改为 4096 不会触发 `truncation='error'`**；风险在于
  prompt 超过 `max_prompt_length` 时（若 filter 关闭）直接报错，以及
  `max_model_len` 是否 ≥ `prompt+response`（vLLM 侧长度校验）。

### 3.3 Tensor padding 与 reward 输入

- prompts 左 pad 到 `max_prompt_length`（`rl_dataset.py` `postprocess_data`），
  responses pad 到 `response_length`；`response_mask = attention_mask[:, -
  response_length:]`（`ray_trainer.py` L196-211 `compute_response_mask`；多轮时
  工具反馈位置 mask=0，见 `vllm_rollout_spmd.py` L259-261 文档）。
- reward manager 用 `attention_mask` 求有效长度后 decode（`batch.py` L53-60）→
  **reward 输入与 padding/长度配置无耦合**；reward 函数只吃响应文本与
  `extra_info.formal_statement`（`reward/reward.py` L137-142）。

### 3.4 W1 实测 prompt 长度分布（7,620 unique，Distill tokenizer）

```json
{"count": 7620, "min": 102, "median": 231, "p95": 415, "p99": 572, "max": 2365, "mean": 251.32}
```

本地 P3 建议：`max_prompt_length=1024`（覆盖 ≥99%+；仅极少数 synthetic 长题被
`filter_overlong_prompts` 滤掉）、`max_response_length=4096`、
`max_model_len=5120`（或懒人值 8192）。现存不一致（`configs/rl/kimina_0.6b_pilot.yaml`
为 8192/3072；`docs/research_plan.md` 为 4096）在本 W4 修订中统一为 1024/4096。

---

## 4. Batch 关系与整除约束

链条（单机单/多卡）：

```text
data.train_batch_size (prompt 数, recipe 256)
  × rollout.n          → real_train_batch_size (序列数, 必须 % n_gpus == 0)   ray_trainer.py L410-414
  → 每个 DP rank 分到 real_train_batch_size / dp_size 条序列
actor.ppo_mini_batch_size (prompt 数, recipe 32)
  × rollout.n / dp_size → per-rank mini-batch 序列数 (>0, 可 % micro == 0)    fsdp_workers.py L209-229
  → gradient_accumulation = mini_序列数 / ppo_micro_batch_size_per_gpu
                                                                              dp_actor.py L396-399
```

- mini-batch 切分：`dp_actor.py` L387 `data.split(ppo_mini_batch_size)`（此时
  data 已是权限化后的序列数）；L403-466 逐 micro forward/backward；L478
  `_optimizer_step()`（grad_clip 默认 1.0，`actor.py` L217）。
- rollout/ref logprob 的 micro batch：`rollout.log_prob_micro_batch_size_per_gpu`
  （recipe L52=2；`rollout.py` L114-117，`log_prob_use_dynamic_bsz=False` 默认）。
  old_log_prob 由 actor 前向重算（`ray_trainer.py` L1248-1258），
  **不是** rollout 时记录（`calculate_log_probs=False` 默认，`rollout.py` L124）。
- `actor.validate`（`actor.py` L140-160）：`data.train_batch_size >=
  ppo_mini_batch_size`（以 prompt 数比较；注意 train_batch_size 是 prompt 数而
  mini 是我们传的 prompt 数，二者同口径）。
- 24 GB 单卡建议（P3-A，2–5 steps）：
  - `train_batch_size=8, rollout.n=4` → 32 序列/步（生成 32×≤4096 tokens）。
  - `ppo_mini_batch_size=8`（prompt）→ per-rank 32 序列；`ppo_micro_batch_size_per_gpu=4`
    → grad accumulation 8。若显存仍紧，micro=2（accum 16）或 mini=4。
  - `max_num_batched_tokens=8192~16384`、`gpu_memory_utilization=0.35~0.5`
    （vLLM 与 FSDP actor 同卡共存时保守取值）。
  - 双卡 24 GB：train_batch_size=16、n=4（64 序列 / 2 卡 = 32/rank），其余同上。

---

## 5. GRPO / DrGRPO 逐项确认（recipe）

| 项 | recipe 值 | 生效位置 |
|---|---|---|
| `adv_estimator=grpo` | L19 | `ray_trainer.py` L262-273 → `core_algos.py` L261 |
| `norm_adv_by_std_in_grpo=False` | L21 | `core_algos.py` L318-321（mean-only 中心化，DrGRPO） |
| `use_kl_in_reward=False` | L20/L65 | `ray_trainer.py` L1292-1298（不加 KL，token_level_rewards=scores） |
| `use_kl_loss=False` + `kl_loss_coef=0.0` | L43-44 | `dp_actor.py` L449-457 不进入 → loss 无 KL |
| `loss_agg_mode="seq-mean-token-sum-norm"` | L41 | `core_algos.py` L726-728：`sum(seq_losses)/loss_mask.shape[-1]`（分母=固定 response 宽度） |
| `entropy_coeff=0` | L45 | `dp_actor.py` L421-422 不算 entropy；L441-447 policy_loss=pg_loss |
| 非对称 clip `low=0.2 / high=0.3` | L46-47 | `core_algos.py` L872-877（clamp 到 [1-0.2, 1+0.3]） |
| dual-clip `clip_ratio_c=3.0` | 默认（`actor.py` L101） | `core_algos.py` L880-886（A<0 时下界裁剪） |
| `loss_mode="vanilla"` | 默认（`actor.py` L45） | `dp_actor.py` L427-439 → `compute_policy_loss_vanilla` |
| `ppo_epochs=1`, `shuffle=False` | 默认（`actor.py` L108-109） | `dp_actor.py` L390 |

- **rollout importance sampling / TIS**：在 pinned 版本全仓库检索
  `rollout_importance|tis_|truncated_importance|use_rollout_log_prob` → **无匹配**。
  `calculate_log_probs` 默认 False（`rollout.py` L124）；即使开启，
  `rollout_log_probs` 只用于 debug 指标（`ray_trainer.py` L1260-1264
  `calculate_debug_metrics`），**不影响 loss**。
- 全同组（all-same reward）advantage=0（`core_algos.py` L317-321），零梯度；
  学习信号完全来自 mixed 组 —— 与 P2.5 W1 的 IGR 诊断口径一致。
- 移植校验：`src/tinylean_rl/rl/grpo.py`（数值与 pinned 逐式一致，测试
  `tests/test_grpo_reference.py`）。

---

## 6. Reward 契约

- 组合公式：`score = proof_rw × format_rw`（`reward/reward.py` L153），
  `format_rw = (format_error == FormatError.NONE)`（L149）。
  即 **format gating**：格式不合法 → 无论 Lean 是否通过，最终 0。
- Lean 验证：`formal_rewards`（`reward/reward.py` L62-113）
  - L81-85：`client.check(snips, timeout=60, max_workers=40)`（kimina_client →
    `/api/check` 端点）。
  - L75-79：两条预过滤文案（"Theorem statement couldn't be parsed…"、"No proof
    found in the output."）**不送检**，直接记 0（L93-96 "filtered proof."）。
  - L98-107：`result.analyze()`；`SnippetStatus.valid` → 1.0/"valid proof found."；
    否则 0.0 + tool feedback（用于 multiturn）。
  - validity 含 sorry 排除（kimina_client `analyze()`），与本地
    `probe`/`promptset_rollout_probe.py` 的严格规则一致。
- verifier 崩溃/缺响应 → 0：L92-96（无 result）；异常路径由客户端超时兜底
  （`timeout=60`，超时计入 0/feedback）。
- **与本地实测的冲突与缓解**：本地环境冷启动 REPL 首个请求可能 >300 s、并发请求
  会 spawn 冷 REPL 且偶发 500（P2 结论 + `prompts` probe）。缓解：
  1. `scripts/prewarm_lean_server.py` 预热 + 串行/小并发的延时阶梯测量，固定并发策略；
  2. P3-A 起步用 `reward_kwargs` 减小 max_workers（如 4~8）或分批送检（串行小批）；
  3. 云上按 W6 runbook 先跑 doctor 检查 server 健康与延时。
- 与 `docs/research_plan.md` §四 reward 契约逐条对照：3 类 `verify_status`
  （verified / lean_error / verifier_error）与本地 probe 一致；官方额外把
  format_error 维度并入最终 score —— **本地 profile 的 `format_ok` 字段即为此对照物**
  （W1 产出 `rewards.jsonl.format_ok`）。
- 注意：P2 评估器 `scripts/evaluate_model.py` 的 `result_is_valid` 只查
  error-severity、不含 sorry 检查，与官方 `analyze()` 和本审计口径不同；
  E007/E008 headline 已完成 strict re-check（`experiments/results/
  p2_eval_rl_*_recheck.json`），P3 起统一用严格口径。

---

## 7. Multiturn（官方开启，我们 P3-v1 关闭）

- recipe：`+data.multiturn=True`、`multiturn_sampling_rate=0.5`、
  `multiturn_n_samples_in_cache=5000`（L28-30）。
- 机制（`recipe/kimina_prover_rl/kimina_prover_rl/dataset.py`）：
  - `n_turns=2`（L98）；`__getitem__` 以 `sampling_rate` 概率从
    `multi_turn_data` 缓存取样本（L175-181）。
  - `create_one_multiturn_prompt`（L240-282）：`feedback="valid proof found."`
    或 `filtered proof.` → 不加轮；feedback > `multiturn_max_feedback_length`
    (3000) → 不加轮；`turn_id >= 2` → 不加轮；否则 `prompt += [assistant
    response, user feedback]` 作为新样本，且要求新 prompt ≤ `max_prompt_length`
    （L264-267）。
  - feedback 来源：reward 计算的 `reward_extra_info["tool_feedback"]`
    （`reward.py` L106 `create_tool_message`），在 `on_batch_end` 收集进缓存
    （dataset.py L376-394；要求 `data.dataloader_num_workers=0`，recipe L32）。
- 关键理解：**multiturn 是数据侧机制**（把上一轮失败样本 + Lean 错误反馈拼成
  第二轮 prompt，作为新训练样本进入后续 batch），**不是**在同一条回复里注入 tool
  反馈的 rollout 机制。reward 仍然是逐条 response 的 `proof_rw × format_rw`。
- 我们 P3-v1 单轮的差异清单（最重要的隐藏变量）：
  1. 没有 error-fixing 第二轮的训练样本（官方 50% 采样）→ 训练分布偏向"一次成
     功/一次错误"的难度谱；
  2. 长 prompt 场景缺失（第二轮 prompt = 原 prompt + 首轮 response + 反馈，token
     数翻倍量级）→ 对 `max_prompt_length=1024` 的本地配置不兼容（第二轮基本必超），
     这本身就是不能照抄 multiturn 的原因之一；
  3. reward 路径无差异（tool_feedback 不进 score）；
  4. 官方 `data_source` 在第二轮样本上变为 `"multiturn"`（dataset.py L277）→
     wandb 指标口径差异（训练监控时注意）。
  - 结论：P3-v1 关闭 multiturn 是合理起点；若 P3-B 要复现官方曲线，需要同时放大
    `max_prompt_length`（≥8192）并恢复采样率 —— 记为 P3-A 之后的实验变量。

---

## 8. 输出表：config key → 生效位置 → 官方值 → 本地 P3 建议 → 依据

| config key | 生效位置(file:line) | 官方 0.6B 值 | 本地 P3-A 建议 | 依据 |
|---|---|---|---|---|
| `algorithm.adv_estimator` | ray_trainer.py L262-273 | grpo | grpo | §5 |
| `algorithm.norm_adv_by_std_in_grpo` | core_algos.py L318-321 | False | False | DrGRPO；W1 组分布 |
| `algorithm.use_kl_in_reward` | ray_trainer.py L1292-1298 | False | False | 无 ref 模型，显存 |
| `actor.use_kl_loss` | dp_actor.py L449-457 | False | False | 同上 |
| `actor.entropy_coeff` | dp_actor.py L421-447 | 0 | 0 | 同 recipe |
| `actor.loss_agg_mode` | core_algos.py L726-728 | seq-mean-token-sum-norm | 同 | 移植一致性 |
| `actor.clip_ratio_low/high` | core_algos.py L872-877 | 0.2 / 0.3 | 同 | recipe |
| `actor.clip_ratio_c` | core_algos.py L880-886 | 3.0(默认) | 3.0 | recipe |
| `actor.loss_mode` | dp_actor.py L427-439 | vanilla(默认) | vanilla | 单一 source |
| `actor.ppo_mini_batch_size` | fsdp_workers.py L209-229 | 32(prompt) | 8(prompt) | §4 单卡分摊 |
| `actor.ppo_micro_batch_size_per_gpu` | dp_actor.py L396-399 | 2 | 4（或 2） | 24GB 单卡 |
| `actor.ppo_epochs` / `shuffle` | actor.py L108-109 | 1 / False | 1 / False | 默认 |
| `actor.use_dynamic_bsz` | actor.py L95；dp_actor.py L392-417 | False | False | recipe 未启用 |
| `actor.ppo_max_token_len_per_gpu` | dp_actor.py L392-394 | 32768（**死配置**） | 不设/注释 | §3.1 |
| `data.train_batch_size` | ray_trainer.py L410 | 256 | 8（单卡）/16（双卡） | §4 |
| `data.max_prompt_length` | rl_dataset.py L105 | 8192 | **1024** | W1 实测分布 |
| `data.max_response_length` | rollout.yaml L25 派生 | 24576(=32768-8192) | **4096** | research_plan 统一 |
| `data.truncation` | rl_dataset.py L108, L315-316 | 'error' | 'error' | prompt 安全网 |
| `data.filter_overlong_prompts` | rl_dataset.py L145-177 | True | True | 过滤超长题 |
| `data.multiturn` 系列 | dataset.py L88-98, L175-394 | True(0.5) | **False** | §7 |
| `rollout.n` | ray_trainer.py L1185/L1221 | 8 | **4** | §2 |
| `rollout.max_model_len` | vllm_rollout_spmd.py L142 | 32768(=8192+24576) | 5120(=1024+4096) | §3 |
| `rollout.max_num_batched_tokens` | vllm_rollout_spmd.py L144-147 | 32768 | 8192 | ≥ max_model_len 保守 |
| `rollout.gpu_memory_utilization` | vllm_rollout_spmd.py L190 | 0.6 | 0.35~0.5 | 同卡共存 |
| `rollout.temperature/top_p` | rollout.py L83-85 | 1.0 / 1.0 | 1.0 / 1.0（P3 rollout） | 与 E007 0.6/0.95 的差记为 metadata |
| `rollout.calculate_log_probs` | rollout.py L124 | False | False | 无 TIS；old_log_prob 重算 |
| `rollout.free_cache_engine` | vllm_rollout_spmd.py L203-204 | True(默认) | True | 省显存 |
| `rollout.tensor_model_parallel_size` | rollout.py L102 | 1（recipe L53） | 1 | 单卡 |
| `rollout.load_format` | rollout.py L141；fsdp_vllm.py L122 | dummy_dtensor→dummy | 同 | 每步全量同步 |
| `ref.fsdp_config.param_offload` | main_ppo.py L190-195 | True（**无效**，ref 不存在） | 不设 | §1.4 |
| `reward_model.reward_manager` | workers/reward_manager/batch.py | batch | batch | tool_feedback 需要 |
| `reward_model.launch_reward_fn_async` | ray_trainer.py L1243-1246/L1284-1285 | True | True（云） | 与 rollout 重叠 |
| `custom_reward_function.reward_kwargs` | reward/reward.py L116-170 | return_dict=True | return_dict=True + 并发限制 | §6 |
| `trainer.n_gpus_per_node` | recipe L75 | 8 | 1~2 | P3-A |
| `trainer.total_epochs` / `save_freq` | recipe L77-79 | 15 / 5 | 1 微跑 / 1 | P3-A smoke |

---

## 未验证假设清单（留给云上 P3-A 验证）

1. flash-attn 与 torch/驱动版本匹配（`fsdp_workers.py` L287-289 强制
   `flash_attention_2`）；
2. vLLM 0.6 gpu_memory_utilization 与 FSDP actor 同卡共存的真实峰值显存（本地
   8 GB 无法验证；依据：recipe 8×A100 0.6 / 本地 W3 LoRA 探针作为下限参考）；
3. `dummy` load_format 每步全量同步 0.6B 的耗时占比（云上度量，若过重可改
   `load_format=safetensors` 预载 base）；
4. Lean server 在 40 workers（官方）下的稳定性：本地镜像冷 REPL/500 行为是否在
   云上同样出现（W6 prewarm + doctor 检查）；
5. n=4 与 n=8 的组方差对实际学习曲线的影响（P3-B 对照）；
6. multiturn 关闭导致的分布偏移量级（§7）；
7. Ray 在容器/多进程环境中的端口与共享内存要求（W6 runbook 覆盖）；
8. `max_num_batched_tokens=8192` 与长 prompt（~1k）+ 4k response 的吞吐折衷；
9. `data.train_batch_size=8` 时 `minimal_bsz`（=n_gpus）整除关系的多卡边界
   （单卡恒成立；双卡 32 序列 %2==0 成立）。
