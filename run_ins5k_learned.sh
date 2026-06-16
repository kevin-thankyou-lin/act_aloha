#!/bin/bash
set -e
cd /home/linke/Projects/act_aloha
source /home/linke/miniforge3/etc/profile.d/conda.sh
conda activate aloha
unset PYTHONPATH
export MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0
CK=ckpt/sim_insertion_5k; T=sim_insertion_scripted
run_arm () {  # $1=name  $2=extra env assignments
  echo "=== INSERTION-5k learned: $1 ==="
  env TASK=$T BASE_CKPT_DIR=$CK MODE=train GAMMA=0.99 SPEED_CKPT=$CK/sp_$1.pt NUM_EPISODES=300 MIN_SPEED=1 MAX_SPEED=8 $2 python aloha_speed.py
  env TASK=$T BASE_CKPT_DIR=$CK MODE=eval  SPEED_CKPT=$CK/sp_$1.pt NUM_EPISODES=30 MIN_SPEED=1 MAX_SPEED=8 $2 python aloha_speed.py
}
run_arm plain_a0005 "ALPHA=0.005"
run_arm mono_a001   "ALPHA=0.01 MONO_LAMBDA=0.5"
run_arm adv_a001    "ALPHA=0.01 ADV_BOUND_LAMBDA=0.1"
echo "INS5K LEARNED DONE"
