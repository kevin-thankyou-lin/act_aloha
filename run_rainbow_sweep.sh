#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export LEARNER="${LEARNER:-rainbow}"
export K_STACK="${K_STACK:-1}"
export SPEED_CHUNK_LEN="${SPEED_CHUNK_LEN:-60}"
export MIN_SPEED="${MIN_SPEED:-1}"
export MAX_SPEED="${MAX_SPEED:-5}"
export GAMMA="${GAMMA:-0.99}"
export FLAT_DISCOUNT="${FLAT_DISCOUNT:-1}"
export ST_PER="${ST_PER:-1}"

if [[ -n "${SPEEDTUNING_RL_ROOT:-}" ]]; then
  export PYTHONPATH="${SPEEDTUNING_RL_ROOT}:${PYTHONPATH:-}"
fi

train_eps="${TRAIN_EPISODES:-300}"
eval_eps="${EVAL_EPISODES:-30}"
python_bin="${PYTHON:-python}"
out_root="${OUT_ROOT:-speedpol_logs/rainbow_sweep_$(date -u +%Y%m%dT%H%M%SZ)}"
task_specs="${TASK_SPECS:-sim_transfer_cube_scripted:ckpt/sim_transfer_cube_scripted sim_insertion_scripted:ckpt/sim_insertion_5k}"
arms="${SWEEP_ARMS:-plain_a005:0.005:0.0 plain_a01:0.01:0.0 mono_a01:0.01:0.5 mono_a02:0.02:0.5}"

mkdir -p "$out_root"
summary="$out_root/summary.txt"
: > "$summary"

echo "rainbow_sweep_start=$(date -u)" | tee -a "$summary"
echo "out_root=$out_root" | tee -a "$summary"
echo "task_specs=$task_specs" | tee -a "$summary"
echo "arms=$arms" | tee -a "$summary"
echo "python=$python_bin" | tee -a "$summary"
echo "train_eps=$train_eps eval_eps=$eval_eps speeds=${MIN_SPEED}-${MAX_SPEED} chunk=${SPEED_CHUNK_LEN} kstack=${K_STACK}" | tee -a "$summary"

for spec in $task_specs; do
  task="${spec%%:*}"
  ckpt_dir="${spec#*:}"
  test -s "$ckpt_dir/policy_best.ckpt"
  test -s "$ckpt_dir/dataset_stats.pkl"

  for arm in $arms; do
    IFS=: read -r name alpha mono_lambda <<< "$arm"
    run_dir="$out_root/${task}_${name}"
    mkdir -p "$run_dir"
    speed_ckpt="$run_dir/speed_policy.pt"
    log="$run_dir/run.log"
    : > "$log"

    echo "=== task=$task arm=$name alpha=$alpha mono=$mono_lambda $(date -u) ===" | tee -a "$log" "$summary"
    env \
      TASK="$task" \
      BASE_CKPT_DIR="$ckpt_dir" \
      MODE=train \
      ALPHA="$alpha" \
      MONO_LAMBDA="$mono_lambda" \
      SPEED_CKPT="$speed_ckpt" \
      NUM_EPISODES="$train_eps" \
      "$python_bin" aloha_speed.py 2>&1 | tee -a "$log"

    env \
      TASK="$task" \
      BASE_CKPT_DIR="$ckpt_dir" \
      MODE=eval \
      ALPHA="$alpha" \
      MONO_LAMBDA="$mono_lambda" \
      SPEED_CKPT="$speed_ckpt" \
      NUM_EPISODES="$eval_eps" \
      "$python_bin" aloha_speed.py 2>&1 | tee -a "$log"

    grep "\\[ALOHA-SPEED\\]" "$log" | tail -2 | sed "s|^|$task $name |" | tee -a "$summary"
  done
done

echo "rainbow_sweep_done=$(date -u)" | tee -a "$summary"
cat "$summary"
