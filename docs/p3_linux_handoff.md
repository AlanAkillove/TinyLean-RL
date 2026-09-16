# P3-0 Linux 交接（RTX 3090 24 GB）

> 本文是 `win` 分支（Windows P2.5 收尾）与 Linux P3-0 之间的交接单。原则：**不重跑已完成实验；先在 Linux 完成环境 gate 与 on-policy 校准，再冻结 P3-A 配置**。

## Source

- branch: `win`
- tag: `p2.5-win-complete`
- Kimina-Prover-RL submodule: `e16b605e8186614c685875c9b57eb19e841b521a`（Windows 收尾阶段不升级）
- 结论与证据：`experiments/manifests/p2_5_complete.yaml`、[`studies/rl_readiness.md`](studies/rl_readiness.md)（E009–E011）

## Linux next branch

建议新开分支：

```bash
git checkout win && git checkout -b p3-linux
```

## P3-0 goals（按顺序）

1. Linux 环境 gate（`bash scripts/doctor.sh`：Linux / Docker / Lean server / VRAM ≥ 24 GB / VERL import）；
2. Lean positive/negative test（`scripts/verify_smoke.py`）；
3. model → Lean smoke（Kimina Distill 0.6B）；
4. Promptset rollout calibration @ `temperature=1.0` / `top_p=1.0` / `n=4` / `max_response=4096`；
5. 测 IGR / Z / O / 截断率 / tokens/s / VRAM；
6. full-parameter 单步显存可行性探针（方法在 P3-0 内决定；可参考 `scripts/lora_step_probe.py` 的结构）；
7. 仅当 full FT 不可行才启用 LoRA（r16/r32 fallback）；
8. 冻结 P3-A 配置（回写 `configs/rl/kimina_0.6b_pilot.yaml` 与 [`p3_config_audit.md`](p3_config_audit.md) §9）。

校准判据：若 temp 1.0 下 IGR 仍过低（≲5%），恢复顺序 = ① n=8（官方 baseline 组件）② `multiturn=true` + `multiturn_sampling_rate=0.5`（需同步放大 `max_prompt_length`）。原则：**先恢复被 reduced-compute baseline 删除的官方机制，再引入自定义算法改进**。

## Explicitly NOT required

- 不重跑 E003–E011；
- 不重跑 MiniF2F 32×4；
- 不重做 Windows 诊断（GBK / 共享显存 / 代理 / Docker Desktop 均与 Linux 无关）；
- 不启动长 RL 训练（P3-B 之前不做）。

## Known open questions

- Promptset IGR under `temperature=1.0`；
- n=4 vs n=8；
- full FT memory feasibility；
- multiturn necessity；
- real GRPO step wall time；
- checkpoint save/resume path。

## Exact command sequence（Linux handoff）

```bash
# 0) clone + branch
git clone --recurse-submodules <repository-url> TinyLean-RL
cd TinyLean-RL
git fetch origin
git checkout win            # tag p2.5-win-complete
git checkout -b p3-linux

# 1) environment gate（P3-0 步骤 1）
source scripts/env.sh
uv sync --extra inference --extra verifier --extra data
cp .env.example .env
docker compose -f infra/lean-server/compose.yaml up -d
bash scripts/doctor.sh

# 2) verifier + model smoke（P3-0 步骤 2–3）
uv run python scripts/verify_smoke.py         # positive/negative gate
uv run python scripts/smoke_test.py --model-key kimina_distill_0_6b

# 3) rollout calibration（P3-0 步骤 4–5；官方 rollout 口径 temp 1.0）
uv run python scripts/promptset_rollout_probe.py \
    --temperature 1.0 --top-p 1.0 \
    --limit 32 --samples-per-theorem 4 \
    --output experiments/results/p3_0_promptset_temp1.json \
    --batch-dir experiments/p3_0_batch

# 4) full-FT memory probe -> freeze the P3-A config（P3-0 步骤 6–8）

# 5) P3-A smoke（仅在配置冻结后；provisional runner）
bash scripts/run_p3_smoke.sh --dry            # 先检查生成命令，不执行
bash scripts/run_p3_smoke.sh                  # 2–5 steps；最后一步保存 checkpoint
```

备注：

- `scripts/run_p3_smoke.sh` 是 **provisional P3-A runner**——未在 Linux 验证、训练模式（full-parameter vs LoRA）未冻结，必须在 P3-0 完成后才可运行；全部待验证参数见 [`p3_config_audit.md`](p3_config_audit.md) §9。
- 任何 rollout 前先 prewarm Lean server（冷容器首请求为分钟级，见 `scripts/prewarm_lean_server.py`）。
- Windows-only 工作区（GBK / 共享显存 / 代理）记录在 [`environment.md`](environment.md) 与 [`studies/rl_readiness.md`](studies/rl_readiness.md)，不进入 Linux 默认脚本路径。
