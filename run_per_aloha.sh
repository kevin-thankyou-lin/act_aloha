#!/bin/bash
# ALOHA SpeedTuning with Rainbow PER + paper-faithful per-decision gamma=0.99.
# Single-env MuJoCo harness (no multi-env), so #2 N/A here; #3 PER + faithful discount.
cd /home/linke/Projects/act_aloha
source /home/linke/miniforge3/etc/profile.d/conda.sh
conda activate aloha
unset PYTHONPATH
set -o pipefail
export MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0
export ST_PER=1 FLAT_DISCOUNT=1 GAMMA=0.99 BETA=1.0 MIN_SPEED=1 MAX_SPEED=8
LOG=/home/linke/Projects/act_aloha/per_aloha.log
echo "=== ALOHA PER faithful start $(date) ===" | tee $LOG
run_arm () {  # $1=ckptdir $2=task $3=alpha $4=tag
  echo "### TRAIN $2 $4 alpha=$3 $(date)" | tee -a $LOG
  env TASK=$2 BASE_CKPT_DIR=$1 MODE=train ALPHA=$3 SPEED_CKPT=$1/sp_$4.pt NUM_EPISODES=300 \
    python aloha_speed.py 2>&1 | grep -E "ALOHA-SPEED|aloha-speed|ep [0-9]+/|Error|Traceback" | tee -a $LOG
  echo "### EVAL  $2 $4 alpha=$3 $(date)" | tee -a $LOG
  env TASK=$2 BASE_CKPT_DIR=$1 MODE=eval ALPHA=$3 SPEED_CKPT=$1/sp_$4.pt NUM_EPISODES=50 \
    python aloha_speed.py 2>&1 | grep -E "ALOHA-SPEED|Error|Traceback" | tee -a $LOG
}
run_arm ckpt/sim_transfer_cube_scripted sim_transfer_cube_scripted 0.02 per_a002
run_arm ckpt/sim_transfer_cube_scripted sim_transfer_cube_scripted 0.04 per_a004
run_arm ckpt/sim_insertion_5k           sim_insertion_scripted      0.01 per_a001
run_arm ckpt/sim_insertion_5k           sim_insertion_scripted      0.02 per_a002
run_arm ckpt/sim_insertion_5k           sim_insertion_scripted      0.04 per_a004
echo "=== ALOHA PER DONE $(date) ===" | tee -a $LOG
