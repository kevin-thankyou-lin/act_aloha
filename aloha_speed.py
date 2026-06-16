"""SpeedTuning learned speed policy on ALOHA + ACT (second testbed).

Per chunk: ACT predicts a 100-action chunk; a small dueling double-Q speed policy
picks a discrete speed v from [frame-stack(qpos) ⊕ chunk-embed], the chunk is
accelerated by linear interpolation to ceil(100/v) steps, executed, and the policy
is trained by RL on reward alpha*v^beta + success (per-chunk, sparse success).
Uses the shared speedtuning_rl.SpeedQLearner core (dueling + double-Q, deferred
gamma**L discount). alpha FIXED per run.

Env vars: MODE(train|eval) ALPHA BETA SPEED_CKPT NUM_EPISODES MIN_SPEED MAX_SPEED.
"""
import collections
import os
import pickle

import numpy as np
import torch
import torch.nn as nn

from constants import SIM_TASK_CONFIGS
from utils import sample_box_pose, sample_insertion_pose, set_seed
from imitate_episodes import make_policy, get_image
from sim_env import make_sim_env, BOX_POSE

from speedtuning_rl import SpeedQLearner, margin_loss

TASK = os.environ.get('TASK', 'sim_transfer_cube_scripted')
CKPT_DIR = os.environ.get('BASE_CKPT_DIR', f'ckpt/{TASK}')
CHUNK = 100
CAMERAS = ['top']
STATE_DIM = 14
MAX_T = SIM_TASK_CONFIGS[TASK]['episode_len'] if TASK in SIM_TASK_CONFIGS else 400


class SpeedPolicy:
    """Thin adapter over speedtuning_rl.SpeedQLearner for the ALOHA testbed.

    State = frame-stack(qpos) + chunk-embed; action = discrete speed. The shared core
    owns the replay buffer, dueling double-Q net, eps-greedy, and the deferred
    gamma**L TD update (L = ceil(CHUNK/v) passed via observe). A delicacy prior
    (qpos -> slow/fast) optionally adds a DQfD margin pinning the slow action (index 0
    = min_speed) on delicate states, mirroring the YAM prior."""

    def __init__(self, feat_dim, n_actions, min_speed, alpha, beta, train,
                 gamma=0.9, lr=3e-4, eps_start=0.5, eps_end=0.05,
                 eps_decay=800, bs=128, learn_start=256, target_sync=200, device='cuda',
                 delicacy_path='', margin=1.0, margin_lambda=0.0, k_stack=2, state_dim=14,
                 mono_lambda=0.0, mono_tau=0.4):
        self.min_speed, self.alpha, self.beta = min_speed, alpha, beta
        self.k_stack, self.state_dim, self.device = k_stack, state_dim, device
        self.margin, self.margin_lambda = margin, margin_lambda
        self.learner = SpeedQLearner(
            n_actions, gamma=gamma, dueling=True, double_q=True, hidden=128, lr=lr,
            eps_start=eps_start, eps_end=eps_end, eps_decay_steps=eps_decay,
            buffer_size=50000, batch_size=bs, learn_start=learn_start,
            target_sync=target_sync, device=device, train=train,
            mono_lambda=mono_lambda, mono_tau=mono_tau)
        self.learner.ensure_built(feat_dim)
        self.deli_net = self.deli_mu = self.deli_sd = None
        if delicacy_path and os.path.exists(delicacy_path) and margin_lambda > 0:
            dn = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)).to(device)
            ck = torch.load(delicacy_path, map_location=device)
            dn.load_state_dict(ck['state_dict']); dn.eval()
            self.deli_net = dn
            self.deli_mu = torch.tensor(ck['mu'], dtype=torch.float32, device=device)
            self.deli_sd = torch.tensor(ck['sd'], dtype=torch.float32, device=device)
            self.learner.margin_provider = self._margin_provider
            print(f'[aloha-speed] +prior: delicacy margin (lambda={margin_lambda}) from {delicacy_path}')

    @property
    def eps(self):
        return self.learner.epsilon

    def select(self, feat):
        return self.learner.select(feat[None, :]) + self.min_speed

    def observe(self, reward, done, chunk_len=None):
        self.learner.observe(reward, done, chunk_len=chunk_len)

    def _margin_provider(self, qnet):
        # DQfD margin: pin the slow action (idx 0 = min_speed) on delicate states drawn
        # from replay. qpos = last state_dim dims of the frame-stack block.
        if self.learner.buffer.size < max(64, self.learner.batch_size):
            return None
        s = self.learner.buffer.sample(self.learner.batch_size)[0]
        s = torch.from_numpy(s).to(self.device)
        qpos = s[:, (self.k_stack - 1) * self.state_dim: self.k_stack * self.state_dim]
        with torch.no_grad():
            deli = (self.deli_net((qpos - self.deli_mu) / self.deli_sd).squeeze(1) > 0)
        if not deli.any():
            return None
        astar = torch.zeros(int(deli.sum()), dtype=torch.long, device=self.device)
        return self.margin_lambda * margin_loss(qnet, s[deli], astar, self.margin)

    @property
    def last_loss(self):
        return self.learner.last_loss

    def save(self, p):
        torch.save({'q': self.learner.qnet.state_dict(), 'min_speed': self.min_speed}, p)

    def load(self, p):
        ck = torch.load(p, map_location=self.device)
        self.learner.qnet.load_state_dict(ck['q']); self.learner.sync_target()


def chunk_embed(ch):  # ch: [T, D] normalized actions
    return np.concatenate([ch.mean(0), ch.std(0), ch[-1] - ch[0]]).astype(np.float32)


def build_act(ckpt_dir=None):
    import sys
    ckpt_dir = ckpt_dir or CKPT_DIR
    # detr's build parses sys.argv -> give it a valid ACT command (must match the trained arch)
    sys.argv = ['aloha_speed.py', '--ckpt_dir', ckpt_dir, '--policy_class', 'ACT',
                '--task_name', TASK, '--seed', '0', '--num_epochs', '1', '--lr', '1e-5',
                '--kl_weight', '10', '--chunk_size', str(CHUNK), '--hidden_dim', '512',
                '--dim_feedforward', '3200', '--batch_size', '8']
    cfg = {'lr': 1e-5, 'num_queries': CHUNK, 'kl_weight': 10, 'hidden_dim': 512,
           'dim_feedforward': 3200, 'lr_backbone': 1e-5, 'backbone': 'resnet18',
           'enc_layers': 4, 'dec_layers': 7, 'nheads': 8, 'camera_names': CAMERAS}
    policy = make_policy('ACT', cfg)
    policy.load_state_dict(torch.load(os.path.join(ckpt_dir, 'policy_best.ckpt')))
    policy.cuda().eval()
    with open(os.path.join(ckpt_dir, 'dataset_stats.pkl'), 'rb') as f:
        st = pickle.load(f)
    pre = lambda q: (q - st['qpos_mean']) / st['qpos_std']
    post = lambda a: a * st['action_std'] + st['action_mean']
    return policy, pre, post


def run(alpha, beta, train, speed_ckpt, num_episodes, min_speed, max_speed, k_stack=2,
        delicacy_path='', margin_lambda=0.0, gamma=0.99, mono_lambda=0.0):
    set_seed(1000)
    policy, pre, post = build_act()
    env = make_sim_env(TASK); env_max_reward = env.task.max_reward
    n_actions = max_speed - min_speed + 1
    feat_dim = k_stack * STATE_DIM + 3 * STATE_DIM
    sp = SpeedPolicy(feat_dim, n_actions, min_speed, alpha, beta, train, gamma=gamma, k_stack=k_stack,
                     state_dim=STATE_DIM, delicacy_path=delicacy_path, margin_lambda=margin_lambda,
                     mono_lambda=mono_lambda)
    if speed_ckpt and os.path.exists(speed_ckpt) and not train:
        sp.load(speed_ckpt); print(f'loaded speed policy {speed_ckpt}')
    SR, S2S, SPD = [], [], []
    for ep in range(num_episodes):
        BOX_POSE[0] = np.concatenate(sample_insertion_pose()) if 'insertion' in TASK else sample_box_pose()
        ts = env.reset()
        frames = collections.deque(maxlen=k_stack)
        t, success, s2s, speeds = 0, False, None, []
        while t < MAX_T and not success:
            obs = ts.observation
            qn = pre(np.array(obs['qpos'], np.float32)).astype(np.float32)
            qpos_t = torch.from_numpy(qn).float().cuda().unsqueeze(0)
            img = get_image(ts, CAMERAS)
            with torch.inference_mode():
                chunk = policy(qpos_t, img)[0].cpu().numpy()  # [CHUNK, D] normalized
            frames.append(qn)
            while len(frames) < k_stack:
                frames.appendleft(qn)
            feat = np.concatenate(list(frames) + [chunk_embed(chunk)]).astype(np.float32)
            v = sp.select(feat); speeds.append(v)
            L = int(np.ceil(CHUNK / v))
            for j in range(L):
                if t >= MAX_T or success:
                    break
                src = min(j * v, CHUNK - 1)
                lo = int(np.floor(src)); hi = min(lo + 1, CHUNK - 1); frac = src - lo
                a = chunk[lo] * (1 - frac) + chunk[hi] * frac
                ts = env.step(post(a)); t += 1
                if ts.reward == env_max_reward:
                    success, s2s = True, t
            done = success or t >= MAX_T
            r = alpha * (v ** beta) + (1.0 if success else 0.0)
            sp.observe(r, done, chunk_len=L)
        SR.append(1.0 if success else 0.0); SPD.append(float(np.mean(speeds)))
        if success:
            S2S.append(s2s)
        if train and speed_ckpt and (ep + 1) % 50 == 0:
            sp.save(speed_ckpt)
        if (ep + 1) % 25 == 0:
            print(f'  ep {ep+1}/{num_episodes} SR(last25)={np.mean(SR[-25:]):.2f} '
                  f'meanspeed={np.mean(SPD[-25:]):.2f} eps={sp.eps:.2f} loss={sp.last_loss}')
    if train and speed_ckpt:
        sp.save(speed_ckpt)
    mode = 'train' if train else 'eval'
    s2s_m = float(np.mean(S2S)) if S2S else -1.0
    print(f'[ALOHA-SPEED] mode={mode} alpha={alpha} SR={np.mean(SR):.3f} '
          f'mean_speed={np.mean(SPD):.2f} s2s={s2s_m:.1f} n={num_episodes}')


if __name__ == '__main__':
    MODE = os.environ.get('MODE', 'train')
    run(alpha=float(os.environ.get('ALPHA', '0.02')),
        beta=float(os.environ.get('BETA', '1.0')),
        train=(MODE == 'train'),
        speed_ckpt=os.environ.get('SPEED_CKPT', 'ckpt/sim_transfer_cube_scripted/speedpol.pt'),
        num_episodes=int(os.environ.get('NUM_EPISODES', '400' if MODE == 'train' else '30')),
        min_speed=int(os.environ.get('MIN_SPEED', '1')),
        max_speed=int(os.environ.get('MAX_SPEED', '8')),
        delicacy_path=os.environ.get('DELICACY', ''),
        margin_lambda=float(os.environ.get('MARGIN_LAMBDA', '0')),
        gamma=float(os.environ.get('GAMMA', '0.99')),
        mono_lambda=float(os.environ.get('MONO_LAMBDA', '0')))
