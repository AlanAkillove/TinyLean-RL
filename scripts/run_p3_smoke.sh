#!/usr/bin/env bash
# Provisional P3-A on-policy smoke for the pinned Kimina-Prover-RL trainer (P2.5 W6).
#
# STATUS: provisional P3-A runner - NOT yet Linux-validated, NOT yet final
# training-mode validated (full-parameter vs LoRA is frozen during P3-0).
# Run it only after P3-0 (Linux migration & on-policy calibration) has frozen
# the P3-A config; see docs/p3_linux_handoff.md.
#
# Command chain (Linux, single GPU 24 GB, see docs/environment.md):
#   source scripts/env.sh
#   uv sync --extra inference            # plus the pinned VERL install (runbook)
#   docker compose -f infra/lean-server/compose.yaml up -d
#   bash scripts/doctor.sh               # environment gate
#   (P3-0: Promptset rollout calibration at temp 1.0 + full-FT memory probe)
#   bash scripts/run_p3_smoke.sh --steps 3
#
# Gates: Linux, Docker daemon + reachable Lean server, 0.6B model directory,
# Promptset parquet, and the P2.5 completion manifest
# (experiments/manifests/p2_5_complete.yaml).  The audited overrides come from
# docs/p3_config_audit.md and configs/rl/kimina_0.6b_pilot.yaml (n=4, single
# GPU, 2-5 optimizer steps, no KL -> no reference worker).
# Checkpoint contract (P3-A success criteria): save once at the final step
# (save_freq = steps) so save/reload can be validated; P3_SMOKE_SAVE_FREQ
# overrides.  `--dry` prints the full command chain without executing anything.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/env.sh" >/dev/null

# Resolve the project virtualenv explicitly: on Linux `python3` is the system
# interpreter, which does not carry the VERL/vLLM training stack (env.sh only
# exports paths; it does not activate the venv).
PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  if [[ -x "$ROOT/.venv/bin/python" ]]; then
    PYTHON="$ROOT/.venv/bin/python"
  else
    PYTHON="python3"
  fi
fi

STEPS=3
DRY=0
SKIP_PREWARM=0
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage: bash scripts/run_p3_smoke.sh [--steps N] [--skip-prewarm] [--dry] [hydra overrides...]

  --steps N        Optimizer steps (1 = P3-0 full-FT memory probe; 2..5 = P3-A smoke; default 3).
  --skip-prewarm   Skip the Lean server warm-up / latency ladder.
  --dry            Print the command chain without executing anything.

Any other `key=value` argument is forwarded to the VERL trainer as a hydra
override (e.g. actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2), which
is how the P3-0 memory probe walks the OOM adjustment order.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --steps) STEPS="${2:?--steps requires a value}"; shift 2 ;;
    --skip-prewarm) SKIP_PREWARM=1; shift ;;
    --dry) DRY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --*) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

# --steps 1 is reserved for the P3-0 full-FT memory probe (docs/p3_linux_handoff.md):
# a single optimizer step answers the feasibility question, so the 2..5 P3-A gate
# must not require running two steps before the memory question is settled.
if ! [[ "$STEPS" =~ ^[0-9]+$ ]] || (( STEPS < 1 || STEPS > 5 )); then
  echo "P3-A smoke: --steps must be an integer in 1..5 (1 = P3-0 memory probe; 2..5 = P3-A smoke) (got '$STEPS')." >&2
  exit 2
fi

# P3-A checkpoint contract: save once at the final smoke step (save_freq = steps)
# so the save -> reload/resume path can actually be validated in one run.
SAVE_FREQ="${P3_SMOKE_SAVE_FREQ:-$STEPS}"

fail() { echo "P3-A smoke blocked: $1" >&2; exit 2; }

# --- Paths -----------------------------------------------------------------
RECIPE_DIR="$ROOT/third_party/kimina-prover-rl/recipe/kimina_prover_rl"
PROMPT_SETS_DIR="$ROOT/data/processed/p3_promptset"
PROMPT_SET_NAME="AI-MO/Kimina-Prover-Promptset"
TRAIN_PARQUET="$PROMPT_SETS_DIR/prompt_sets/$PROMPT_SET_NAME/train.parquet"
TEST_PARQUET="$PROMPT_SETS_DIR/prompt_sets/$PROMPT_SET_NAME/test.parquet"
MODEL_PATH="$TINYLEAN_MODEL_ROOT/kimina_distill_0_6b"
GATE_FILE="$ROOT/experiments/manifests/p2_5_complete.yaml"
WARMUP_JSON="$ROOT/experiments/results/lean_server_warmup_p3.json"

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

# Audited P3-A overrides (docs/p3_config_audit.md, configs/rl/kimina_0.6b_pilot.yaml):
# - n=4 instead of 8 (reduced-compute starting hypothesis; n=8 is the first
#   recovery lever if P3-0 calibration shows IGR below ~5%);
# - train_batch_size 8 prompts -> 32 sequences;
# - max_prompt_length 1024 (W1: p99 of 7,620 prompts = 572) + 4096 response
#   (P3-0 initial rollout-budget hypothesis);
# - DrGRPO: mean-only advantage, seq-mean-token-sum-norm, asymmetric clip,
#   no KL loss and no KL in reward -> VERL skips the reference policy worker;
# - multiturn disabled for the reduced-compute start (audit S7 records the
#   deviation);
# - single GPU, human-scale step count; checkpoint saved once at the final
#   step (trainer.save_freq = steps) to validate the save/reload path.
MAIN_PPO_CMD=(
  "$PYTHON" -m verl.trainer.main_ppo
  algorithm.adv_estimator=grpo
  algorithm.use_kl_in_reward=False
  algorithm.norm_adv_by_std_in_grpo=False
  data.train_files="[$TRAIN_PARQUET]"
  data.val_files="[$TEST_PARQUET]"
  data.train_batch_size=8
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
  actor_rollout_ref.actor.ppo_mini_batch_size=8
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4
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
  actor_rollout_ref.rollout.n=4
  actor_rollout_ref.rollout.max_num_batched_tokens=8192
  actor_rollout_ref.rollout.max_model_len=5120
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4
  reward_model.reward_manager=batch
  reward_model.launch_reward_fn_async=True
  custom_reward_function.path="$RECIPE_DIR/kimina_prover_rl/reward/reward.py"
  custom_reward_function.name=reward
  +custom_reward_function.reward_kwargs.return_dict=True
  trainer.critic_warmup=0
  trainer.logger='["console"]'
  trainer.project_name='kimina-prover-p3a'
  trainer.experiment_name='p3a-smoke'
  trainer.n_gpus_per_node=1
  trainer.nnodes=1
  trainer.save_freq="$SAVE_FREQ"
  trainer.test_freq=-1
  trainer.val_before_train=False
  trainer.total_epochs=1
  trainer.total_training_steps="$STEPS"
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
  echo "[3/3] pinned VERL trainer (P3-A smoke: $STEPS steps, n=4, 1 GPU)"
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
[[ -f "$GATE_FILE" ]] \
  || fail "P2.5 completion manifest missing: $GATE_FILE (see docs/studies/rl_readiness.md)"
[[ -f "$RECIPE_DIR/kimina_prover_0.6B.sh" ]] \
  || fail "pinned recipe missing; run: git submodule update --init --recursive"
if ! curl --silent --show-error --fail --max-time 5 "$LEAN_SERVER_API_URL/health" >/dev/null 2>&1; then
  fail "Lean server unreachable at $LEAN_SERVER_API_URL (docker compose -f infra/lean-server/compose.yaml up -d); the 2.0.0 image disables /openapi.json in prod mode, so readiness is probed via /health"
fi
if ! "$PYTHON" -c 'import verl' >/dev/null 2>&1; then
  fail "VERL is not importable with '$PYTHON'; install the pinned submodule (docs/environment.md runbook)"
fi

# dataset.py / reward.py import the kimina_prover_rl package by absolute path.
export PYTHONPATH="$RECIPE_DIR${PYTHONPATH:+:$PYTHONPATH}"

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

echo "[3/3] Launching pinned VERL trainer (P3-A smoke: $STEPS steps, n=4, 1 GPU)"
echo "  overrides: docs/p3_config_audit.md, configs/rl/kimina_0.6b_pilot.yaml"
"${MAIN_PPO_CMD[@]}" "${EXTRA_ARGS[@]}"
