from typing import Dict, Tuple
from tqdm import tqdm
import torchvision
import torch
import wandb
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from Model_Test.denoiser import MLP
from Model_Test.denoiser2 import ContextUnet
from ExPIL import process_ExPIL, ExPIL

# diffusion hyperparameters
timesteps = 500
beta1 = 1e-4
beta2 = 0.02

# network hyperparameters
device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
n_feat = 128 # 64 hidden dimension feature
n_cfeat = 5 # context vector is of size 5
height = 28 # 16x16 image

# training hyperparameters
batch_size = 32
n_epoch = 64
lrate=1e-3

b_t = (beta2 - beta1) * torch.linspace(0, 1, timesteps + 1, device=device) + beta1
a_t = 1 - b_t
ab_t = torch.cumsum(a_t.log(), dim=0).exp()
print(ab_t.shape)
ab_t[0] = 1

time_emb_dim = 10

mlp = ContextUnet(in_channels=1, n_feat=n_feat, n_cfeat=n_cfeat, height=height).to(device)

data_path = '/home/leo/Project/Datasets/ExPIL'
z, *_ = process_ExPIL(data_path)
ExPIL_dataset = ExPIL(z=np.array(z))
train_size = int(0.8*len(ExPIL_dataset))
print('train_size:', train_size)
train2 = torch.ones((2, 2560))
#train_size = 80
test_size = len(ExPIL_dataset)-train_size

train_data, test_data = torch.utils.data.random_split(ExPIL_dataset, [train_size, test_size])
#train_dataloader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
train_dataloader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=64, shuffle=True)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# 下载训练集
dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

wandb.init(
    # set the wandb project where this run will be logged
    project="WN_Model",

    # track hyperparameters and run metadata
    config={
        "learning_rate": 0.0001,
        "architecture": "MLP_LDM",
        "dataset": "ExPIL",
        "Epoch": 32,
    }
)

# helper function: perturbs an image to a specified noise level
def perturb_input(x, t, noise):
    return ab_t.sqrt()[t, None, None, None] * x + (1 - ab_t[t, None, None, None]) * noise

# dataloader

# training without context code
optimizer = torch.optim.AdamW(mlp.parameters(), lr=lrate)
# set into train mode
mlp.train()
print('dataloader_shape:', len(train_dataloader))
for ep in range(n_epoch):

    # linearly decay learning rate
    optimizer.param_groups[0]['lr'] = lrate * (1 - ep / n_epoch)

    pbar = tqdm(dataloader, mininterval=2)
    total_loss = 0

    for x,_ in pbar:  # x: images

        #x = x.view(batch_size,-1)

        optimizer.zero_grad()
        x = x.to(device)
        # perturb data
        noise = torch.randn_like(x)
        t = torch.randint(1, timesteps + 1, (x.shape[0],)).to(device)


        x_pert = perturb_input(x, t, noise)
        #print(x_pert.shape)

        # use network to recover noise
        pred_noise = mlp(x_pert, t / timesteps)

        # loss is mean squared error between the predicted and true noise
        loss = F.mse_loss(pred_noise, noise)
        loss.backward()
        total_loss += loss

        optimizer.step()
    train_loss = total_loss / len(dataloader)

    wandb.log({"train_loss": train_loss})
    print('[{:03d}/{}] train_loss: {:.4f}'.format(
            ep, n_epoch, train_loss))


    # save model periodically

