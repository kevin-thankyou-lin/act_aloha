#!/bin/bash
cd /home/linke/Projects/act_aloha
source /home/linke/miniforge3/etc/profile.d/conda.sh
conda activate aloha
unset PYTHONPATH
export MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0
export TASK=sim_transfer_cube_scripted BASE_CKPT_DIR=ckpt/sim_transfer_cube_scripted
export LAPS=2 ALPHA=0.01 HARVEST_ATTEMPTS=50 DISTILL_EPOCHS=400 RUN_TAG=fw1
python aloha_flywheel.py
echo "FLYWHEEL SCRIPT DONE rc=$?"
