#!/bin/bash
set -e
cd /home/linke/Projects/act_aloha
source /home/linke/miniforge3/etc/profile.d/conda.sh
conda activate aloha
unset PYTHONPATH
export MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0
run_arm () {  # $1=ckptdir $2=task $3=name $4=extra env
  echo "=== $2 $3 ==="
  env TASK=$2 BASE_CKPT_DIR=$1 MODE=train GAMMA=0.99 SPEED_CKPT=$1/sp_$3.pt NUM_EPISODES=300 MIN_SPEED=1 MAX_SPEED=8 $4 python aloha_speed.py
  env TASK=$2 BASE_CKPT_DIR=$1 MODE=eval  SPEED_CKPT=$1/sp_$3.pt NUM_EPISODES=30 MIN_SPEED=1 MAX_SPEED=8 $4 python aloha_speed.py
}
# insertion (cliff) - the key test
run_arm ckpt/sim_insertion_5k sim_insertion_scripted advt_a15 "ALPHA=0.01 ADV_BOUND_LAMBDA=1.0 ADV_DISC_ANCHOR=0.15"
run_arm ckpt/sim_insertion_5k sim_insertion_scripted advt_a05 "ALPHA=0.01 ADV_BOUND_LAMBDA=1.0 ADV_DISC_ANCHOR=0.05"
# transfer-cube (cliff)
run_arm ckpt/sim_transfer_cube_scripted sim_transfer_cube_scripted advt_a15 "ALPHA=0.01 ADV_BOUND_LAMBDA=1.0 ADV_DISC_ANCHOR=0.15"
echo "ADVTIGHT LOCAL DONE"
