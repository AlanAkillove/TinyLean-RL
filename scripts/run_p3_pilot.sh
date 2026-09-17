#!/usr/bin/env bash
# P3-B short learning pilot runner (frozen P3-A config; docs/p3_config_audit.md S10).
#
# Runs the pinned VERL trainer for a bounded number of optimizer steps on the
# frozen single-GPU configuration, with periodic checkpoints and per-step
# rollout dumps so the IGR_t / Z_t / O_t learning dynamics can be computed
# after the run. The runner is strictly bounded by --steps and never
# auto-scales: the protocol forbids silently extending a running experiment
# (P3-C stays undefined until P3-B completes and is reviewed).
#
# --steps is a TOTAL target. VERL's resume_mode=auto (the default) restores the
# latest global_step_* checkpoint in --dir when one exists and trains only up
# to this total: E015 resumed global_step_3 with --steps 4; E019 resumes
# global_step_30 with --steps 60 and must end at global_step_60 (not 90).
#
# Frozen config (docs/p3_config_audit.md S10, validated by E014/E015):
#   tb 4 prompts x n -> 16 (n=4) or 32 (n=8) sequences, mini 4, micro 2,
#   max_prompt 1024, max_response 4096, vLLM util 0.40, max_num_batched_tokens
#   5120, DrGRPO without KL, multiturn off. --n 8 restores the official
#   baseline group size (recovery ladder #1); the default keeps the calibrated
#   n=4.
#
# Command chain: source scripts/env.sh -> (gates) -> prepare_data (idempotent)
# -> prewarm -> main_ppo. Checkpoints under --dir (default runs/p3b_pilot),
# rollout dumps under <dir>/rollout_data.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/env.sh" >/dev/null

# Resolve the project virtualenv explicitly (env.sh does not activate it).
PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  if [[ -x "$ROOT/.venv/bin/python" ]]; then
    PYTHON="$ROOT/.venv/bin/python"
  else
    PYTHON="python3"
  fi
fi

STEPS=30
N=4
SAVE_FREQ="${P3_PILOT_SAVE_FREQ:-10}"
LOCAL_DIR="${P3_PILOT_DIR:-$ROOT/runs/p3b_pilot}"
DRY=0
SKIP_PREWARM=0
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage: bash scripts/run_p3_pilot.sh [--steps N] [--n K] [--save-freq M]
                                    [--dir PATH] [--skip-prewarm] [--dry]
                                    [hydra overrides...]

  --steps N        Optimizer steps for the pilot (1..500, default 30). The run
                   stops at N and is never extended automatically.
  --n K            Rollout group size (1..16, default 4). --n 8 restores the
                   official baseline group size (tb 4 prompts -> 32 sequences).
  --save-freq M    Checkpoint every M steps (default 10; keep the last 3).
  --dir PATH       Checkpoint/output directory (default runs/p3b_pilot).
  --skip-prewarm   Skip the Lean server warm-up / latency ladder.
  --dry            Print the command chain without executing anything.

Any other key=value argument is forwarded to the VERL trainer as a hydra override.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --steps) STEPS="${2:?--steps requires a value}"; shift 2 ;;
    --n) N="${2:?--n requires a value}"; shift 2 ;;
    --save-freq) SAVE_FREQ="${2:?--save-freq requires a value}"; shift 2 ;;
    --dir) LOCAL_DIR="${2:?--dir requires a value}"; shift 2 ;;
    --skip-prewarm) SKIP_PREWARM=1; shift ;;
    --dry) DRY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --*) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

if ! [[ "$STEPS" =~ ^[0-9]+$ ]] || (( STEPS < 1 || STEPS > 500 )); then
  echo "P3-B pilot: --steps must be an integer in 1..500 (got '$STEPS')." >&2
  exit 2
fi
if ! [[ "$N" =~ ^[0-9]+$ ]] || (( N < 1 || N > 16 )); then
  echo "P3-B pilot: --n must be an integer in 1..16 (got '$N')." >&2
  exit 2
fi
if ! [[ "$SAVE_FREQ" =~ ^[0-9]+$ ]] || (( SAVE_FREQ < 1 )); then
  echo "P3-B pilot: --save-freq must be a positive integer (got '$SAVE_FREQ')." >&2
  exit 2
fi

fail() { echo "P3-B pilot blocked: $1" >&2; exit 2; }

# --- Paths -----------------------------------------------------------------
RECIPE_DIR="$ROOT/third_party/kimina-prover-rl/recipe/kimina_prover_rl"
PROMPT_SETS_DIR="$ROOT/data/processed/p3_promptset"
PROMPT_SET_NAME="AI-MO/Kimina-Prover-Promptset"
TRAIN_PARQUET="$PROMPT_SETS_DIR/prompt_sets/$PROMPT_SET_NAME/train.parquet"
TEST_PARQUET="$PROMPT_SETS_DIR/prompt_sets/$PROMPT_SET_NAME/test.parquet"
MODEL_PATH="$TINYLEAN_MODEL_ROOT/kimina_distill_0_6b"
P25_GATE_FILE="$ROOT/experiments/manifests/p2_5_complete.yaml"
P30_MANIFEST="$ROOT/experiments/manifests/p3_0_complete.yaml"
WARMUP_JSON="$ROOT/experiments/results/lean_server_warmup_p3.json"
ROLLOUT_DUMP_DIR="$LOCAL_DIR/rollout_data"

# --- Commands ---------------------------------------------------------------
PREPARE_CMD=(
  "$PYTHON" "$RECIPE_DIR/prepare_data.py"
  --train-dataset "$PROMPT_SET_NAME"
  --test-dataset "AI-MO/minif2f_test"
  --path "$PROMPT_SETS_DIR"
)

PREWARM_CMD=(
  "$PYTHON" "$ROOT/scripts/prewarm_lean_server.py"
  --warmup-requests 2
  --measure-requests 2
  --concurrency 1,2
  --output "$WARMUP_JSON"
)

MAIN_PPO_CMD=(
  "$PYTHON" -m verl.trainer.main_ppo
  algorithm.adv_estimator=grpo
  algorithm.use_kl_in_reward=False
  algorithm.norm_adv_by_std_in_grpo=False
  data.train_files="[$TRAIN_PARQUET]"
  data.val_files="[$TEST_PARQUET]"
  data.train_batch_size=4
  data.max_prompt_length=1024
  data.max_response_length=4096
  +data.return_extra_info=True
  +data.multiturn=False
  data.return_raw_chat=True
  data.dataloader_num_workers=0
  data.filter_overlong_prompts=True
  data.custom_cls.path="$RECIPE_DIR/kimina_prover_rl/dataset.py"
  data.custom_cls.name=NuminaRLDataset
  data.truncation=error
  actor_rollout_ref.model.path="$MODEL_PATH"
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  actor_rollout_ref.actor.optim.lr=2e-6
  actor_rollout_ref.actor.ppo_mini_batch_size=4
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2
  actor_rollout_ref.actor.loss_agg_mode="seq-mean-token-sum-norm"
  actor_rollout_ref.actor.use_kl_loss=False
  actor_rollout_ref.actor.kl_loss_coef=0.0
  actor_rollout_ref.actor.entropy_coeff=0
  actor_rollout_ref.actor.clip_ratio_low=0.2
  actor_rollout_ref.actor.clip_ratio_high=0.3
  actor_rollout_ref.actor.fsdp_config.param_offload=False
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=5120
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  actor_rollout_ref.rollout.gpu_memory_utilization=0.40
  actor_rollout_ref.rollout.n="$N"
  actor_rollout_ref.rollout.max_num_batched_tokens=5120
  actor_rollout_ref.rollout.max_model_len=5120
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4
  reward_model.reward_manager=batch
  reward_model.launch_reward_fn_async=True
  custom_reward_function.path="$RECIPE_DIR/kimina_prover_rl/reward/reward.py"
  custom_reward_function.name=reward
  +custom_reward_function.reward_kwargs.return_dict=True
  trainer.critic_warmup=0
  # The pinned recipe's NuminaRLDataset.on_batch_end calls wandb.log() after
  # every step, so an initialized wandb run must exist (env.sh defaults
  # WANDB_MODE=offline).
  trainer.logger='["console","wandb"]'
  trainer.project_name='kimina-prover-p3b'
  trainer.experiment_name='p3b-pilot'
  trainer.n_gpus_per_node=1
  trainer.nnodes=1
  trainer.save_freq="$SAVE_FREQ"
  trainer.max_actor_ckpt_to_keep=3
  trainer.test_freq=-1
  trainer.val_before_train=False
  trainer.total_epochs=1
  trainer.total_training_steps="$STEPS"
  trainer.default_local_dir="$LOCAL_DIR"
  trainer.rollout_data_dir="$ROLLOUT_DUMP_DIR"
)

print_cmd() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

if (( DRY )); then
  echo "DRY RUN - commands that would execute (gates not evaluated):"
  echo "[1/3] dataset preparation"
  print_cmd "${PREPARE_CMD[@]}"
  echo "[2/3] Lean server warm-up"
  print_cmd "${PREWARM_CMD[@]}"
  echo "[3/3] pinned VERL trainer (P3-B pilot: $STEPS steps, n=$N, 1 GPU)"
  print_cmd "${MAIN_PPO_CMD[@]}" "${EXTRA_ARGS[@]}"
  exit 0
fi

# --- Gates ------------------------------------------------------------------
[[ "$(uname -s)" == "Linux" ]] || fail "requires Linux (detected $(uname -s))"
if ! command -v docker >/dev/null 2>&1 || ! docker info >/dev/null 2>&1; then
  fail "Docker daemon unavailable; the Kimina Lean Server must run in Docker."
fi
[[ -d "$MODEL_PATH" ]] || fail "model missing: $MODEL_PATH"
[[ -f "$ROOT/data/raw/kimina_promptset/data/train-00000-of-00001.parquet" ]] \
  || fail "promptset missing: data/raw/kimina_promptset/data/train-00000-of-00001.parquet"
[[ -f "$P25_GATE_FILE" ]] \
  || fail "P2.5 completion manifest missing: $P25_GATE_FILE"
[[ -f "$P30_MANIFEST" ]] \
  || fail "P3-0 completion manifest missing: $P30_MANIFEST (run the P3-0 chain first)"
[[ -f "$RECIPE_DIR/kimina_prover_0.6B.sh" ]] \
  || fail "pinned recipe missing; run: git submodule update --init --recursive"
if ! curl --silent --show-error --fail --max-time 5 "$LEAN_SERVER_API_URL/health" >/dev/null 2>&1; then
  fail "Lean server unreachable at $LEAN_SERVER_API_URL (docker compose -f infra/lean-server/compose.yaml up -d)"
fi
if ! "$PYTHON" -c 'import verl' >/dev/null 2>&1; then
  fail "VERL is not importable with '$PYTHON'; install the pinned submodule (docs/environment.md runbook)"
fi

# dataset.py / reward.py import the kimina_prover_rl package by absolute path.
export PYTHONPATH="$RECIPE_DIR${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$LOCAL_DIR" "$ROLLOUT_DUMP_DIR"

echo "[P3 runner] steps=$STEPS n=$N save_freq=$SAVE_FREQ dir=$LOCAL_DIR"
echo "  sequences/step: $((4 * N)) (tb 4 prompts x n=$N)"
echo "  watch: live 'Training Progress' bar (%, step, ETA) and IGR_t / Z_t / O_t"
echo "  (from rollout dumps), score mean, response length, clip ratio, entropy,"
echo "  ppo_kl, grad_norm, step time, peak VRAM. On resume the bar starts at the"
echo "  checkpoint's global step. Pace reference: P3-B averaged ~140 s/step"
echo "  (30 steps = 4218 s), so ~30 resumed steps project to roughly 70-120 min."

echo "[1/3] Dataset preparation (idempotent)"
if [[ -f "$TRAIN_PARQUET" && -f "$TEST_PARQUET" ]]; then
  echo "  cached: $TRAIN_PARQUET"
else
  "${PREPARE_CMD[@]}"
fi

echo "[2/3] Lean server warm-up and latency ladder"
if (( SKIP_PREWARM )); then
  echo "  skipped (--skip-prewarm)"
else
  "${PREWARM_CMD[@]}"
fi

echo "[3/3] Launching pinned VERL trainer (P3-B pilot: $STEPS steps, n=$N, 1 GPU)"
echo "  frozen config: docs/p3_config_audit.md S10, configs/rl/kimina_0.6b_pilot.yaml"
"${MAIN_PPO_CMD[@]}" "${EXTRA_ARGS[@]}"
