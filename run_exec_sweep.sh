#!/bin/bash
# Insertion exec-horizon sweep: replan every K steps at native fidelity (speed=1, no
# interpolation), decoupled from speed. Probes whether the weak insertion base is due
# to running 100 actions open-loop.
cd /home/linke/Projects/act_aloha
source /home/linke/miniforge3/etc/profile.d/conda.sh
conda activate aloha
unset PYTHONPATH
export MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0
CK=ckpt/sim_insertion_scripted; T=sim_insertion_scripted
OUT=$CK/exec_horizon_sweep.txt; : > "$OUT"
for K in 100 50 40 25 16; do
  echo "===== exec_horizon=$K (native, no interp, speed=1) =====" | tee -a "$OUT"
  ACT_EXEC_HORIZON=$K ACT_SPEED=1 ACT_NUM_ROLLOUTS=30 python imitate_episodes.py --task_name $T --ckpt_dir $CK \
    --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 \
    --dim_feedforward 3200 --num_epochs 2000 --lr 1e-5 --seed 0 --eval 2>&1 | grep SPEEDBASE | tee -a "$OUT"
done
echo "EXEC_SWEEP_DONE" | tee -a "$OUT"
