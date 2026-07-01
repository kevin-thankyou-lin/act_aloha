#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-/home/linke/miniforge3/envs/aloha/bin/python}"
ROBOMIMIC_ROOT="${ROBOMIMIC_ROOT:-/home/linke/Projects/robomimic-SAIL-awe-aloha-20260630}"
DATA_ROOT="${DATA_ROOT:-/tmp/act_awe_policy_data}"
OUT_ROOT="${OUT_ROOT:-/tmp/dp_sail_aloha_20260630}"
EPOCHS="${EPOCHS:-1000}"
ROLLOUTS="${ROLLOUTS:-50}"

export PYTHONPATH="$ROBOMIMIC_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=0

mkdir -p "$OUT_ROOT/train" "$OUT_ROOT/eval"

common_args() {
  local task="$1"
  printf '%s\n' \
    --task_name "$task" \
    --policy_class Diffusion \
    --batch_size 8 \
    --seed 0 \
    --num_epochs "$EPOCHS" \
    --lr 1e-4 \
    --chunk_size 32 \
    --precision_key awe_precisions \
    --num_precision_heads 3 \
    --precision_weight 0.1 \
    --num_inference_steps 10
}

train_task() {
  local task="$1"
  local ckpt_dir="$OUT_ROOT/train/$task"
  mkdir -p "$ckpt_dir"
  if [[ -s "$ckpt_dir/policy_best.ckpt" ]]; then
    echo "[DP-SAIL] training already complete: $task"
    return
  fi
  mapfile -t args < <(common_args "$task")
  "$PYTHON" imitate_episodes.py \
    "${args[@]}" \
    --ckpt_dir "$ckpt_dir" \
    --dataset_dir "$DATA_ROOT/$task" \
    --num_episodes_override 50 \
    2>&1 | tee "$OUT_ROOT/train/${task}.log"
}

eval_task() {
  local task="$1"
  local ckpt_dir="$OUT_ROOT/train/$task"
  mapfile -t args < <(common_args "$task")
  for head in 0 1 2; do
    for seed in 0 1 2; do
      local log="$OUT_ROOT/eval/${task}__awe_h${head}_s${seed}.log"
      if grep -q '^\[SAILAWE\]' "$log" 2>/dev/null; then
        echo "[DP-SAIL] evaluation already complete: $task head=$head seed=$seed"
        continue
      fi
      ACT_NUM_ROLLOUTS="$ROLLOUTS" ACT_EVAL_SEED="$seed" AWE_HEAD_INDEX="$head" \
        "$PYTHON" imitate_episodes.py \
          --eval \
          "${args[@]}" \
          --ckpt_dir "$ckpt_dir" \
          2>&1 | tee "$log"
    done
  done
}

echo "[DP-SAIL] start $(date -Is) epochs=$EPOCHS rollouts=$ROLLOUTS"
for task in sim_transfer_cube_scripted sim_insertion_scripted; do
  train_task "$task"
done
for task in sim_transfer_cube_scripted sim_insertion_scripted; do
  eval_task "$task"
done
echo "[DP-SAIL] complete $(date -Is)"
