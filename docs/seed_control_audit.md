# Seed control audit（E019 / M1 multi-seed）

> 目的：为 M1 seed replication（seed2/seed3）确认 pinned VERL（e16b605e）中实际控制
> 各随机源的配置字段与代码位置，避免“写了 seed=42 但没生效”。审计日期 2026-09-17。

## 有效控制点（均经代码位置核实）

| # | Hydra key | 代码位置 | 作用域 | 可复现性 |
|---|---|---|---|---|
| 1 | `+data.seed` | `verl/trainer/main_ppo.py:368` → `train_dataloader_generator.manual_seed(data_config.get("seed", 1))` | 训练 dataloader 的 shuffle 生成器（决定每一步看到哪些 prompts、顺序） | 确定性：同一 data.seed + 同一数据集 → 相同 prompt 顺序。注意该键不在 hydra struct 中，需 `+` 追加语法（已实证） |
| 2 | （不可配）FSDP rollout 引擎种子 | `vllm_rollout_spmd.py:196` → `LLM(..., seed=config.get("seed", 0))`；但 FSDP `RolloutConfig`（`verl/workers/config/rollout.py`）**无 seed 字段**（hydra 结构校验实证：`actor_rollout_ref.rollout.seed` 不在 struct；`engine.py:67` 的 `seed: int = 42` 属于 **Megatron** engine 配置） | vLLM 引擎级 RNG | **固定为 0，不可配置**（不修改下游栈的前提下）。不同 run 的采样流经不同请求顺序/数量自然发散 |
| 3 | （固定，不可配） | `verl/workers/sharding_manager/fsdp_vllm.py:116` → `get_torch_device().manual_seed(gen_dp_rank + 1000)` | 每次权重同步时重置 rollout torch RNG | 固定值，不随实验变化 |

## 未被配置的随机源（审计结论）

- `verl/workers/fsdp_workers.py` 无任何 `manual_seed/set_random_seed` 调用（grep 为空）——
  FSDP actor 路径不设进程级 RNG；actor 更新在给定 rollout batch 下无额外随机源
  （无 dropout、无随机采样），因此**训练轨迹的随机性由 #1 + #2 完全支配**。
- `verl/trainer/ppo/metric_utils.py:335` 的 `np.random.seed(seed)` 只用于指标抽样日志
  （`_maybe_log_val_generations` 等），不影响训练。

## E019（seed1）实际生效值

E019 的 runner（`scripts/run_e019.sh` / `run_p3_pilot.sh`）**未显式设置** `data.seed`，
因此 seed1 实际使用库默认值 **`data.seed = 1`**；rollout 引擎种子固定 0（不可配）。
（本审计将“默认值也是有效值”的事实记录在案；后续论文写作须按实际值描述。）

## Seed2/Seed3 预注册方案（运行前冻结，不得按结果更改）

- seed2 = **20260918** → `+data.seed=20260918`
- seed3 = **20260919** → `+data.seed=20260919`

独立性论证：dataloader 种子决定每一步的 prompt 序列（已由 live sanity 实证与 seed1 不同），
而 rollout 采样流随请求序列/状态发散——两条训练轨迹从第 1 步起即独立；唯一不可控的是
vLLM 引擎的初始种子（固定 0）。这是本 pinned build 在不修改下游栈前提下的最大可控程度。

其它一切配置与 E019 完全相同（frozen P3-B recipe）；从
`Kimina-Prover-Distill-0.6B` 冷启动，独立目录 `runs/m1_seed2/`、`runs/m1_seed3/`。

## Live sanity（启动后必须执行）

1. 训练开始后取 `runs/m1_seed2/rollout_data/1.jsonl`，提取 prompts 的 formal statement，
   与 seed1 的 `runs/p3b_pilot/rollout_data/1.jsonl` 比较：**statement 序列必须不同**。
2. 抽查生成文本：seed2 的 step-1 输出窗口与 seed1 不同（同一 prompt 采样流不同）。
3. 若两者完全相同 → seed plumbing 无效，**立即停止**（无值守停止条件之一）。
