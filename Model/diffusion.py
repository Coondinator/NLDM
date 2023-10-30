import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .ema import EMA
from .argument import Arguments
from functools import partial
from argparse import ArgumentParser

class NLDM(nn.Module):

    def __init__(self, config_file, model, betas):
        super().__init__()
        args = Arguments('./Model', filename=config_file)
        self.seed = args.seed
        self.latent_size = args.latent_size
        self.model = model
        self.use_ema = args.use_ema

        if self.use_ema:
            self.ema_decay = args.ema_decay
            self.ema_start = args.ema_start
            self.ema_update_rate = args.ema_update_rate

        if args.loss_type not in ["l1", "l2"]:
            raise ValueError("__init__() got unknown loss type")
        self.loss_type = args.loss_type

        self.step = 0
        #self.latent_chanel = args.latent_channel
        self.num_timesteps = len(betas)

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)  # 累乘

        to_torch = partial(torch.tensor, dtype=torch.float32)  # partial?

        self.register_buffer("beta", to_torch(betas))  # variable in buffer is fixed
        self.register_buffer("alphas", to_torch(alphas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))

        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1 - alphas_cumprod)))
        self.register_buffer("reciprocal_sqrt_alphas", to_torch(np.sqrt(1 / alphas)))

        self.register_buffer("remove_noise_coeff", to_torch(betas / np.sqrt(1 - alphas_cumprod)))
        self.register_buffer("sigma", to_torch(np.sqrt(betas)))

    def forward(self, latent, y=None):
        b, w = latent.shape # b:batch, w: 2560

        device = latent.device
        if w != self.latent_size:
            raise ValueError("latent length does not match diffusion parameters")
        torch.manual_seed(self.seed)
        self.seed += 1
        t = torch.randint(0, self.num_timesteps, (b,), device=device)  #randomly generate t between[0, num_timesteps) with batch numbers

        return self.get_losses(latent, t, y)

    def get_losses(self, latent, t, y):
        torch.manual_seed(self.seed)
        self.seed += 1

        noise = torch.randn_like(latent)  # generateing a noise with latent‘s shape
        # batch, 1, 2560
        perturbed_latent = self.add_noise(latent, t, noise)  # add noise

        # batch, 1, 2560
        estimated_noise = self.model(perturbed_latent, t, y)  # estimate noise

        # self.sample(estimated_noise.shape[0],estimated_noise.device)
        # 256, 1, 2560

        if self.loss_type == "l1":
            loss = F.l1_loss(estimated_noise, noise)
        elif self.loss_type == "l2":
            loss = F.mse_loss(estimated_noise, noise)

        return loss

    def add_noise(self, latent, t, noise):
        out = extract(self.sqrt_alphas_cumprod, t, latent.shape) * latent + extract(self.sqrt_one_minus_alphas_cumprod,
                                                                                    t, latent.shape) * noise
        return out

    def remove_noise(self, latent, t, y):
        output = (latent - extract(self.remove_noise_coeff, t, latent.shape) * self.model(latent, t, y)) * extract(self.reciprocal_sqrt_alpha, t, latent.shape)
        return output

    @torch.no_grad()
    def sample(self, batch_size, device, y=None, use_ema=False):
        if y is not None and batch_size != len(y):
            raise ValueError("sample batch size different from length of given y")

        x = torch.randn(batch_size, self.img_channels, *self.img_size, device=device)

        for t in range(self.num_timesteps - 1, -1, -1):
            t_batch = torch.tensor([t], device=device).repeat(batch_size)
            x = self.remove_noise(x, t_batch, y, use_ema)

            if t > 0:
                x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)

        return x.cpu().detach()

    @torch.no_grad()
    def sample_diffusion_sequence(self, batch_size, device, y=None, use_ema=True):
        if y is not None and batch_size != len(y):
            raise ValueError("sample batch size different from length of given y")

        x = torch.randn(batch_size, self.img_channels, *self.img_size, device=device)
        diffusion_sequence = [x.cpu().detach()]

        for t in range(self.num_timesteps - 1, -1, -1):
            t_batch = torch.tensor([t], device=device).repeat(batch_size)
            x = self.remove_noise(x, t_batch, y, use_ema)

            if t > 0:
                x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)

            diffusion_sequence.append(x.cpu().detach())

        return diffusion_sequence


def generate_linear_schedule(T, low, high):
    print("generate_linear_schedule")
    beta = np.linspace(low * 1000 / T, high * 1000 / T, T)
    return beta

def extract(a, t, latent_shape):
    """
    extract element from a, and reshape to (b,1,1,1,1...) 1's number is len(x_shape)-1
    a : sqrt_alphas_cumprod [1000]
    t : time_step
    latent_shape : latent shape

    Example:
        extract(self.sqrt_alphas_cumprod, t, x.shape) * x +
        extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * noise
    """
    b, *_ = t.shape  # b : batch size
    out = a.gather(-1, t)

    return out.reshape(b, *((1,) * (len(latent_shape) - 1)))  # 第一个*是展开的操作



