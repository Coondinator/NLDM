import os
import math
import numpy as np
import torch
import torch.nn as nn

class PositionalEmbedding(nn.Module):
    __doc__ = r"""Computes a positional embedding of timesteps.

    Input:
        x:tensor of shape (N)
    Output:
        tensor of shape (N, dim)
    Args:
        dim (int): embedding dimension
        scale (float): linear scale to be applied to timesteps. Default: 1.0
    """

    def __init__(self, dim, scale=1.0):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.scale = scale

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim-1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = torch.outer(x * self.scale, emb)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class HiddenLayer(nn.Module):
    def __init__(self, latent_dim, time_dim):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_features=latent_dim+time_dim, out_features=latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3)
        )

    def forward(self, x, time_emb):
        x = torch.cat((x, time_emb), dim=1)
        x = self.layer(x)
        return x

class DownSample(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=input_dim//2),
            nn.BatchNorm1d(input_dim//2),
            nn.ReLU(),
            nn.Dropout(p=0.3)
        )

    def forward(self, x):
        x = self.layer(x)
        return x

class UpSample(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, input_dim*2),
            nn.BatchNorm1d(input_dim*2),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.layer(x)
        return x

class MLP(nn.Module):
    def __init__(self, base_channels, time_emb_dim, layer_num):
        super().__init__()
        self.time_emb_layer = nn.Sequential(
            PositionalEmbedding(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, 10),
        ) if time_emb_dim is not None else None
        '''
        self.time_bias = nn.Sequential(nn.SiLU,
            nn.Linear(time_emb_dim, out_channels)) if time_emb_dim is not None else None
        '''
        self.input_layer = nn.Linear(in_features=2570, out_features=2560)
        self.activation0 = nn.ReLU()
        self.hidden_layer = nn.ModuleList()

        for i in range(layer_num):
          self.hidden_layer.append(HiddenLayer(latent_dim=2560, time_dim=10))

        self.output_layer = nn.Linear(in_features=2560, out_features=2560)
        self.activation = nn.Sigmoid()

    def forward(self, x, time, class_emb=None):

        if self.time_emb_layer is not None:
            if time is None:
                raise ValueError("time conditioning was specified but time is not passed")
            time_emb = self.time_emb_layer(time)  #[batch, 2560]

        else:
            time_emb = None

        #x += time_emb
        x = torch.cat((x, time_emb), dim=1)
        #print('x.shape:', x.shape)
        x = self.activation0(self.input_layer(x))
        skip = []
        skip.append(x)

        '''
        if self.time_bias is not None:
            if time_emb is None:
                raise ValueError("time conditioning was specified but time_emb is not passed")
            x += self.time_bias(time_emb)[:, :, None, None]
        '''

        #print(x.shape)
        #print(time_emb.shape)
        for index, layer in enumerate(self.hidden_layer):
            if (index == 0):
                x = layer(x, time_emb)
            else:
                residual = skip.pop()
                x = x + residual
                x = layer(x, time_emb)

            skip.append(x)

        x = self.activation(self.output_layer(x))

        return x




