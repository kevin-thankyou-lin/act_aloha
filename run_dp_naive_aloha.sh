#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

TASK="${TASK:?set TASK to an ALOHA simulation task}"
PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/mnt/amlfs-04/home/linke/dp_naive_aloha_h100_e60_20260630/data}"
OUT_ROOT="${OUT_ROOT:-/mnt/amlfs-04/home/linke/dp_naive_aloha_h100_e60_20260630/runs}"
EPOCHS="${EPOCHS:-1000}"
ROLLOUTS="${ROLLOUTS:-50}"
PRED_HORIZON="${PRED_HORIZON:-100}"
EXEC_HORIZON="${EXEC_HORIZON:-60}"

export PYTHONUNBUFFERED=1
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export AWE_DYNAMIC=0
export ACT_SPEED=1

RUN_DIR="$OUT_ROOT/$TASK"
CKPT_DIR="$RUN_DIR/checkpoints"
mkdir -p "$CKPT_DIR" "$RUN_DIR/eval"

common_args=(
  --task_name "$TASK"
  --policy_class Diffusion
  --batch_size 8
  --seed 0
  --num_epochs "$EPOCHS"
  --lr 1e-4
  --chunk_size "$PRED_HORIZON"
  --num_inference_steps 10
)

echo "[DP-NAIVE] start $(date -Is) task=$TASK predict=$PRED_HORIZON execute=$EXEC_HORIZON epochs=$EPOCHS"
if [[ ! -s "$CKPT_DIR/policy_best.ckpt" ]]; then
  "$PYTHON" imitate_episodes.py \
    "${common_args[@]}" \
    --ckpt_dir "$CKPT_DIR" \
    --dataset_dir "$DATA_ROOT/$TASK" \
    --num_episodes_override 50 \
    2>&1 | tee "$RUN_DIR/train.log"
else
  echo "[DP-NAIVE] training already complete: $TASK"
fi

for seed in 0 1 2; do
  log="$RUN_DIR/eval/seed_${seed}.log"
  if grep -q '^\[SPEEDBASE\]' "$log" 2>/dev/null; then
    echo "[DP-NAIVE] evaluation already complete: $TASK seed=$seed"
    continue
  fi
  ACT_EXEC_HORIZON="$EXEC_HORIZON" ACT_NUM_ROLLOUTS="$ROLLOUTS" ACT_EVAL_SEED="$seed" \
    "$PYTHON" imitate_episodes.py \
      --eval \
      "${common_args[@]}" \
      --ckpt_dir "$CKPT_DIR" \
      2>&1 | tee "$log"
done
echo "[DP-NAIVE] complete $(date -Is) task=$TASK"
