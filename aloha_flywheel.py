"""Self-distillation speed flywheel (policy-independent accelerator + distill loop).

Each lap:
  1. CONTROLLER  train a general speed controller (SpeedPolicy: qpos-framestack +
     chunk-embed -> speed) on the CURRENT ACT policy.                [aloha_speed.run]
  2. HARVEST     run ACT + controller, record the SUCCESSFUL *fast* rollouts as new
     HDF5 demos (the actual executed, accelerated trajectories — they reach success
     in fewer steps than the base policy).
  3. DISTILL     fine-tune ACT on (original demos + harvested fast demos) so the
     policy internalizes the fast behavior.                          [imitate_episodes]
  4. EVAL        greedy controller on the new ACT -> SR + steps-to-success.

A faster policy lets the controller find the next compressible margin -> the Pareto
frontier marches outward lap over lap. Policy-independent: only needs predict-chunk /
execute / observe-success; the controller uses generic features, the distill is plain BC.

Run (aloha env, GPU): unset PYTHONPATH; MUJOCO_GL=egl python aloha_flywheel.py
Env: TASK BASE_CKPT_DIR DATA_DIR LAPS ALPHA HARVEST_ATTEMPTS DISTILL_EPOCHS
"""
import os
import subprocess
import time

import h5py
import numpy as np
import torch

from constants import SIM_TASK_CONFIGS
from utils import sample_box_pose, set_seed
from imitate_episodes import get_image
from sim_env import make_sim_env, BOX_POSE
from aloha_speed import build_act, SpeedPolicy, chunk_embed, CHUNK, CAMERAS, STATE_DIM, MAX_T

TASK = os.environ.get('TASK', 'sim_transfer_cube_scripted')
LAPS = int(os.environ.get('LAPS', '3'))
ALPHA = float(os.environ.get('ALPHA', '0.01'))
HARVEST_ATTEMPTS = int(os.environ.get('HARVEST_ATTEMPTS', '60'))
DISTILL_EPOCHS = int(os.environ.get('DISTILL_EPOCHS', '600'))
MIN_SPEED, MAX_SPEED = 1, 8
RUN = os.environ.get('RUN_TAG', time.strftime('%m%d_%H%M%S'))
ROOT = f'flywheel_runs/{TASK}_{RUN}'
BASE_DATA = SIM_TASK_CONFIGS[TASK]['dataset_dir']


def harvest(ckpt_dir, controller, lap_dir, attempts):
    """Roll out ACT + controller; save successful fast episodes as HDF5 demos.
    Returns (n_saved, list of step-lengths)."""
    os.makedirs(lap_dir, exist_ok=True)
    policy, pre, post = build_act(ckpt_dir)
    env = make_sim_env(TASK); maxr = env.task.max_reward
    saved, lens = 0, []
    for ep in range(attempts):
        if 'insertion' in TASK:
            from utils import sample_insertion_pose
            BOX_POSE[0] = np.concatenate(sample_insertion_pose())
        else:
            BOX_POSE[0] = sample_box_pose()
        ts = env.reset()
        frames = []
        traj = {'qpos': [], 'qvel': [], 'action': [], 'img': {c: [] for c in CAMERAS}}
        t, success, chunk = 0, False, None
        while t < MAX_T and not success:
            obs = ts.observation
            qn = pre(np.array(obs['qpos'], np.float32)).astype(np.float32)
            qt = torch.from_numpy(qn).float().cuda().unsqueeze(0)
            img = get_image(ts, CAMERAS)
            with torch.inference_mode():
                chunk = policy(qt, img)[0].cpu().numpy()
            frames.append(qn)
            while len(frames) < controller.k_stack:
                frames.insert(0, qn)
            frames = frames[-controller.k_stack:]
            feat = np.concatenate(frames + [chunk_embed(chunk)]).astype(np.float32)
            v = controller.select(feat)
            L = int(np.ceil(CHUNK / v))
            for j in range(L):
                if t >= MAX_T or success:
                    break
                src = min(j * v, CHUNK - 1)
                lo = int(np.floor(src)); hi = min(lo + 1, CHUNK - 1); w = src - lo
                a = chunk[lo] * (1 - w) + chunk[hi] * w
                cmd = post(a)
                # record the env-space command + the obs that preceded it
                traj['qpos'].append(np.array(ts.observation['qpos'], np.float64))
                traj['qvel'].append(np.array(ts.observation['qvel'], np.float64))
                for c in CAMERAS:
                    traj['img'][c].append(np.array(ts.observation['images'][c], np.uint8))
                traj['action'].append(np.array(cmd, np.float64))
                ts = env.step(cmd); t += 1
                if ts.reward == maxr:
                    success = True
        if success:
            _write_demo(lap_dir, saved, traj)
            saved += 1; lens.append(t)
    return saved, lens


def _pad(arr, T):
    """Pad a list to length T by repeating the last element (hold-at-success).
    ACT's get_norm_stats stacks all episodes -> they must be equal length (=MAX_T)."""
    a = list(arr)
    while len(a) < T:
        a.append(a[-1])
    return np.asarray(a[:T])


def _write_demo(lap_dir, idx, traj):
    # pad the (short) fast-success trajectory to MAX_T so all demos are equal length
    for k in ('qpos', 'qvel', 'action'):
        traj[k] = _pad(traj[k], MAX_T)
    for c in CAMERAS:
        traj['img'][c] = _pad(traj['img'][c], MAX_T)
    T = len(traj['action'])
    path = os.path.join(lap_dir, f'episode_{idx}.hdf5')
    with h5py.File(path, 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
        root.attrs['sim'] = True
        obs = root.create_group('observations'); image = obs.create_group('images')
        for c in CAMERAS:
            image.create_dataset(c, (T, 480, 640, 3), dtype='uint8', chunks=(1, 480, 640, 3))
            root[f'/observations/images/{c}'][...] = np.asarray(traj['img'][c])
        obs.create_dataset('qpos', (T, 14)); root['/observations/qpos'][...] = np.asarray(traj['qpos'])
        obs.create_dataset('qvel', (T, 14)); root['/observations/qvel'][...] = np.asarray(traj['qvel'])
        root.create_dataset('action', (T, 14)); root['/action'][...] = np.asarray(traj['action'])


def build_lap_dataset(harvest_dir, n_harvest, lap_data_dir):
    """Symlink original demos + harvested fast demos into one dir, renumbered."""
    os.makedirs(lap_data_dir, exist_ok=True)
    n_base = SIM_TASK_CONFIGS[TASK]['num_episodes']
    i = 0
    for e in range(n_base):
        src = os.path.abspath(os.path.join(BASE_DATA, f'episode_{e}.hdf5'))
        if os.path.exists(src):
            dst = os.path.join(lap_data_dir, f'episode_{i}.hdf5')
            if not os.path.exists(dst):
                os.symlink(src, dst)
            i += 1
    for h in range(n_harvest):
        src = os.path.abspath(os.path.join(harvest_dir, f'episode_{h}.hdf5'))
        dst = os.path.join(lap_data_dir, f'episode_{i}.hdf5')
        if not os.path.exists(dst):
            os.symlink(src, dst)
        i += 1
    return i


def distill(lap_data_dir, n_eps, in_ckpt_dir, out_ckpt_dir):
    """Fine-tune ACT (warm-started from the current policy) on the combined dataset
    (orig + harvested fast demos). Warm-start matters: from-scratch needs ~2000 epochs;
    fine-tuning the good base on the fast-enriched data converges in a few hundred."""
    os.makedirs(out_ckpt_dir, exist_ok=True)
    cmd = ['python', 'imitate_episodes.py', '--task_name', f'_flywheel_{TASK}',
           '--ckpt_dir', out_ckpt_dir, '--policy_class', 'ACT', '--kl_weight', '10',
           '--chunk_size', str(CHUNK), '--hidden_dim', '512', '--batch_size', '8',
           '--dim_feedforward', '3200', '--num_epochs', str(DISTILL_EPOCHS),
           '--lr', '1e-5', '--seed', '0']
    env = dict(os.environ, FLYWHEEL_DATASET_DIR=lap_data_dir, FLYWHEEL_NUM_EPISODES=str(n_eps),
               FLYWHEEL_INIT_CKPT=os.path.join(in_ckpt_dir, 'policy_best.ckpt'))
    subprocess.run(cmd, check=True, env=env)


def evaluate(ckpt_dir, controller, n=20):
    policy, pre, post = build_act(ckpt_dir)
    env = make_sim_env(TASK); maxr = env.task.max_reward
    SR, S2S, SPD = [], [], []
    for ep in range(n):
        if 'insertion' in TASK:
            from utils import sample_insertion_pose
            BOX_POSE[0] = np.concatenate(sample_insertion_pose())
        else:
            BOX_POSE[0] = sample_box_pose()
        ts = env.reset(); frames = []
        t, success, speeds = 0, False, []
        while t < MAX_T and not success:
            obs = ts.observation
            qn = pre(np.array(obs['qpos'], np.float32)).astype(np.float32)
            qt = torch.from_numpy(qn).float().cuda().unsqueeze(0)
            img = get_image(ts, CAMERAS)
            with torch.inference_mode():
                chunk = policy(qt, img)[0].cpu().numpy()
            frames.append(qn)
            while len(frames) < controller.k_stack:
                frames.insert(0, qn)
            frames = frames[-controller.k_stack:]
            feat = np.concatenate(frames + [chunk_embed(chunk)]).astype(np.float32)
            v = controller.select(feat); speeds.append(v)
            L = int(np.ceil(CHUNK / v))
            for j in range(L):
                if t >= MAX_T or success:
                    break
                src = min(j * v, CHUNK - 1)
                lo = int(np.floor(src)); hi = min(lo + 1, CHUNK - 1); w = src - lo
                a = chunk[lo] * (1 - w) + chunk[hi] * w
                ts = env.step(post(a)); t += 1
                if ts.reward == maxr:
                    success = True
        SR.append(1.0 if success else 0.0); SPD.append(float(np.mean(speeds)))
        if success:
            S2S.append(t)
    return np.mean(SR), np.mean(SPD), (np.mean(S2S) if S2S else -1)


def main():
    set_seed(1000)
    os.makedirs(ROOT, exist_ok=True)
    ckpt = os.environ.get('BASE_CKPT_DIR', f'ckpt/{TASK}')
    log = open(os.path.join(ROOT, 'flywheel.log'), 'a')

    def emit(m):
        print(m); log.write(m + '\n'); log.flush()

    emit(f'=== FLYWHEEL {TASK} laps={LAPS} alpha={ALPHA} base={ckpt} {time.ctime()} ===')
    for lap in range(LAPS):
        emit(f'--- LAP {lap}  policy={ckpt}  {time.ctime()} ---')
        # 1. controller on current policy (reuse aloha_speed training via subprocess)
        cpath = os.path.join(ROOT, f'controller_lap{lap}.pt')
        # mono prior on the controller: plain alpha collapses to max speed on cliff
        # tasks -> harvest/eval over-speed -> 0 SR. mono caps speed -> sane fast demos.
        subprocess.run(['python', 'aloha_speed.py'], check=True, env=dict(
            os.environ, MODE='train', ALPHA=str(ALPHA), SPEED_CKPT=cpath,
            NUM_EPISODES='200', TASK=TASK, BASE_CKPT_DIR=ckpt, GAMMA='0.99',
            MONO_LAMBDA=os.environ.get('CTRL_MONO_LAMBDA', '0.5')))
        from aloha_speed import SpeedPolicy as SP
        feat_dim = 2 * STATE_DIM + 3 * STATE_DIM
        controller = SP(feat_dim, MAX_SPEED - MIN_SPEED + 1, MIN_SPEED, ALPHA, 1.0,
                        train=False, k_stack=2, state_dim=STATE_DIM)
        controller.load(cpath)
        # 2. harvest fast successes
        hdir = os.path.join(ROOT, f'harvest_lap{lap}')
        n, lens = harvest(ckpt, controller, hdir, HARVEST_ATTEMPTS)
        emit(f'  harvested {n}/{HARVEST_ATTEMPTS} fast successes, mean_len={np.mean(lens) if lens else -1:.0f}')
        if n < 5:
            emit('  too few harvested; stopping flywheel'); break
        # 3. distill into a new ACT
        ldata = os.path.join(ROOT, f'data_lap{lap}')
        n_eps = build_lap_dataset(hdir, n, ldata)
        out_ckpt = os.path.join(ROOT, f'ckpt_lap{lap}')
        distill(ldata, n_eps, ckpt, out_ckpt)
        ckpt = out_ckpt
        # 4. eval new policy
        sr, spd, s2s = evaluate(ckpt, controller, n=20)
        emit(f'  LAP {lap} RESULT: SR={sr:.3f} mean_speed={spd:.2f} steps_to_success={s2s:.0f}')
    emit(f'=== FLYWHEEL DONE {time.ctime()} ===')


if __name__ == '__main__':
    main()
