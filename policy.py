import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
from torchvision.models import ResNet18_Weights, resnet18
from diffusers import DDIMScheduler
from robomimic.algo.diffusion_policy import ConditionalUnet1D

from detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer
import IPython
e = IPython.embed

class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        self.precision_only = args_override.get('precision_only', False)
        self.precision_weight = args_override.get('precision_weight', 1.0)
        print(f'KL Weight {self.kl_weight}')

    def __call__(self, qpos, image, actions=None, is_pad=None, precisions=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if precisions is not None:
            precisions = precisions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]
            _, _, precision_hat, _ = self.model(qpos, image, env_state)
            mask = ~is_pad.unsqueeze(-1)
            precision_loss = F.binary_cross_entropy_with_logits(
                precision_hat, precisions, reduction='none'
            )
            precision_loss = (precision_loss * mask).sum() / mask.sum() / precisions.shape[-1]
            precision_accuracy = (
                ((precision_hat >= 0) == (precisions >= 0.5)) * mask
            ).sum() / mask.sum() / precisions.shape[-1]
            return {
                'precision_bce': precision_loss,
                'precision_accuracy': precision_accuracy,
                'loss': self.precision_weight * precision_loss,
            }
        if actions is not None: # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            a_hat, is_pad_hat, _, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            return loss_dict
        else: # inference time
            a_hat, _, precision_hat, (_, _) = self.model(qpos, image, env_state)
            return (a_hat, precision_hat) if precision_hat is not None else a_hat

    def configure_optimizers(self):
        return self.optimizer

    def train(self, mode=True):
        super().train(mode)
        if self.precision_only:
            # The frozen action policy is an inference-time feature extractor.
            # Keep its BatchNorm and dropout behavior identical during head training.
            self.model.eval()
            self.model.precision_head.train(mode)
        return self


class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model # decoder
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None # TODO
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict['mse'] = mse
            loss_dict['loss'] = loss_dict['mse']
            return loss_dict
        else: # inference time
            a_hat = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


class DiffusionPolicy(nn.Module):
    """Naive image-conditioned Diffusion Policy action predictor."""

    def __init__(self, args_override):
        super().__init__()
        self.num_queries = args_override['num_queries']
        self.num_inference_steps = args_override.get('num_inference_steps', 10)

        self.image_encoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.image_encoder.fc = nn.Identity()
        self.qpos_encoder = nn.Sequential(
            nn.Linear(14, 128),
            nn.Mish(),
            nn.Linear(128, 128),
        )
        cond_dim = 512 + 128
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=14,
            global_cond_dim=cond_dim,
            down_dims=[128, 256, 512],
        )
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=100,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=False,
            prediction_type='epsilon',
        )
        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=args_override['lr'], weight_decay=1e-6
        )

    def encode_obs(self, qpos, image):
        batch_size, num_cameras = image.shape[:2]
        image = image.flatten(0, 1)
        image = F.interpolate(image, size=(240, 320), mode='bilinear', align_corners=False)
        image = transforms.functional.normalize(
            image,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        image_features = self.image_encoder(image).reshape(batch_size, num_cameras, -1).mean(1)
        return torch.cat([image_features, self.qpos_encoder(qpos)], dim=-1)

    def __call__(self, qpos, image, actions=None, is_pad=None, precisions=None):
        obs_cond = self.encode_obs(qpos, image)
        if actions is not None:
            actions = actions[:, :self.num_queries]
            valid = ~is_pad[:, :self.num_queries]
            noise = torch.randn_like(actions)
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (actions.shape[0],),
                device=actions.device,
            )
            noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)
            noise_pred = self.noise_pred_net(noisy_actions, timesteps, global_cond=obs_cond)
            l2 = ((noise_pred - noise).square() * valid.unsqueeze(-1)).sum()
            l2 = l2 / valid.sum().clamp_min(1) / actions.shape[-1]
            return {'l2': l2, 'loss': l2}

        actions = torch.randn(
            qpos.shape[0], self.num_queries, 14, device=qpos.device, dtype=qpos.dtype
        )
        self.noise_scheduler.set_timesteps(self.num_inference_steps, device=qpos.device)
        for timestep in self.noise_scheduler.timesteps:
            noise_pred = self.noise_pred_net(actions, timestep, global_cond=obs_cond)
            actions = self.noise_scheduler.step(noise_pred, timestep, actions).prev_sample
        return actions

    def configure_optimizers(self):
        return self.optimizer

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
