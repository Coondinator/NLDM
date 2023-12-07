from ExPIL import process_ExPIL, process_single_ExPIL, ExPIL
from torch.utils.data import dataset, Dataset, DataLoader
import wandb
import numpy as np
import math
from inspect import isfunction
from functools import partial

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from einops import rearrange

import torch
from torch import nn, einsum
import torch.nn.functional as F

from test_transformer.transformer import TransformerModel
from Model_MLP.denoiser import MLP
from Model_Transformer.denoiser import MldDenoiser


device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(114514)
torch.cuda.manual_seed(114514)
np.random.seed(114514)

def process_data():
    data_path = 'ExPIL_new'
    z, *_ = process_ExPIL(data_path)
    return z

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    return betas_for_alpha_bar(
        timesteps,
        lambda t: math.cos((t + s) / (1 + s) * math.pi / 2) ** 2,
    )

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return np.linspace(beta_start, beta_end, timesteps)

def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = _extract_into_tensor(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = _extract_into_tensor(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

def  _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


@torch.no_grad()
def q_posterior_mean_variance(x_start, x_t, t):
    # q(x_{t-1} | x_t, x_0)

    assert x_start.shape == x_t.shape
    posterior_mean = (
        _extract_into_tensor(posterior_mean_coef1, t, x_t.shape) * x_start
        + _extract_into_tensor(posterior_mean_coef2, t, x_t.shape) * x_t
    )

    return posterior_mean


@torch.no_grad()
def p_mean_variance(model, x, t):
    model_output = model(x, t)
    x_start = model_output
    mean = q_posterior_mean_variance(x_start, x, t)
    return mean, x_start

@torch.no_grad()
def p_sample_x_start(model, x, t, t_index):
    model_mean, pred_x_start = p_mean_variance(model, x, t)
    if t_index == 0:
        return {'sample':model_mean, 'pred_x_start':pred_x_start}
    else:
        posterior_variance_t = _extract_into_tensor(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        return {'sample':model_mean + torch.sqrt(posterior_variance_t) * noise, 'pred_x_start':pred_x_start}

def p_sample_loop_x_start(model, shape):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []

    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):

        out = p_sample_x_start(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
        img = out['sample']
        imgs.append(img.cpu().numpy())
    return img, imgs

@torch.no_grad()
def p_sample(model, x, t, t_index):
    betas_t = _extract_into_tensor(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = _extract_into_tensor(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = _extract_into_tensor(sqrt_recip_alphas, t, x.shape)

    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = _extract_into_tensor(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise

    # Algorithm 2 but save all images:


@torch.no_grad()
def p_sample_loop(model, shape):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []

    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):

        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
        imgs.append(img.cpu().numpy())
    return img, imgs

def p_losses(denoise_model, x_start, t, noise=None, loss_type="l2", train_mode='noise'):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
    if train_mode == 'noise':
        predicted_noise = denoise_model(x_noisy, t)
        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()
    elif train_mode == 'x_start':
        pred_x_start = denoise_model(x_noisy, t)
        if loss_type == 'l1':
            loss = F.l1_loss(x_start, pred_x_start)
        elif loss_type == 'l2':
            loss = F.mse_loss(x_start, pred_x_start)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(x_start, pred_x_start)
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()

    return loss


def basic_train(model, train_loader, test_dataloader, n_epoch, save_path, loss_type="l2", lrate=1e-3, train_mode='noise'):

    optim = torch.optim.Adam(model.parameters(), lr=lrate)
    model.train()
    wandb.init(
        # set the wandb project where this run will be logged
        project="new_expi",
        group="mlp_normalization",
        job_type="train",
        # track hyperparameters and run metadata
        config={
            "learning_rate": lrate,
            "architecture": "ldm",
            "dataset": "ExPIL",
            "iteration": 50000,
        }
    )
    for ep in range(n_epoch):
        print(f'epoch {ep}')
        acc_train_loss = 0
        # linearly decay learning rate
        optim.param_groups[0]['lr'] = lrate * (1 - ep / n_epoch)

        pbar = tqdm(train_loader)
        for x in pbar:  # x: images
            optim.zero_grad()
            x = x.to(device)

            # perturb data
            noise = torch.randn_like(x)
            t = torch.randint(0, timesteps, (x.shape[0],)).to(device)
            loss = p_losses(model, x, t, noise, loss_type, train_mode)
            acc_train_loss += loss.item()
            loss.backward()

            optim.step()

        acc_val_loss = 0
        with torch.no_grad():
            model.eval()
            for test_data in test_dataloader:
                test_data = test_data.to(device)
                val_loss = p_losses(model, test_data, t, noise, loss_type, train_mode)
                acc_val_loss += val_loss.item()
                #print('test_loss:', test_loss.item())

        acc_train_loss /= len(train_loader)
        acc_val_loss /= len(test_dataloader)

        wandb.log({"acc_train_loss": acc_train_loss, "test_loss": acc_val_loss})
        print('[{:03d}/{}] acc_train_loss: {:.4f}\t test_loss: {:.4f}'.format(
                ep, n_epoch, acc_train_loss, acc_val_loss))

    torch.save(model.state_dict(), save_path)
    wandb.finish()



if __name__ == '__main__':
    train_data = np.array(process_data())
    print(train_data.shape)
    data_mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)
    print(data_mean.shape)
    train_data = (train_data - data_mean) / std
    dataset_size = train_data.shape[0]
    print(train_data.shape)
    train_data = ExPIL(train_data)

    timesteps = 200
    batch_size = 32
    lr = 1e-4

    # define beta schedule
    betas = linear_beta_schedule(timesteps=timesteps)

    # define alphas
    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
    alphas_cumprod_next = np.append(alphas_cumprod[1:], 0.0)

    # calculations for diffusion q(x_t | x_{t-1}) and others
    sqrt_recip_alphas = np.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)
    log_one_minus_alphas_cumprod = np.log(1.0 - alphas_cumprod)
    sqrt_recip_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod)
    sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod - 1)

    # calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = (betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))
    posterior_mean_coef1 = (
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    )
    posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev)* np.sqrt(alphas)/ (1.0 - alphas_cumprod)
    )

    d_model = 2560  # embedding dimension
    d_hid = 2560  # dimension of the feedforward network model in ``nn.TransformerEncoder``
    nlayers = 2  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
    nhead = 2  # number of heads in ``nn.MultiheadAttention``
    dropout = 0.2  # dropout probability

    model = MLP(d_model,d_hid,nlayers,dropout=dropout).to(device)

    remainder = dataset_size % batch_size

    # 舍弃余数样本
    new_dataset_size = dataset_size - remainder
    dataset, _ = torch.utils.data.random_split(train_data, [new_dataset_size, remainder])

    train_set, test_set = torch.utils.data.random_split(dataset, [224, 32])
    train_dataloader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)
    basic_train(model, train_dataloader, test_dataloader, 1000, 'Model_MLP/Model_x_start_4.pt', lrate=lr, train_mode='x_start')
    model.load_state_dict(torch.load('Model_MLP/Model_x_start_4.pt'))
    model.eval()
    result = torch.rand([1,1,2560]).to(device)



    sample, _ = p_sample_loop_x_start(model, result.shape)
    sample = sample.detach().cpu().numpy()
    sample = sample * std + data_mean
    print(sample[...,2550:])
    np.save('Model_MLP/sample_x_start_4.npy', sample)



















