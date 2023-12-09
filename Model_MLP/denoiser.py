import os
import math
import numpy as np
import torch
import torch.nn as nn

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class HiddenLayer(nn.Module):
    def __init__(self, latent_dim, dropout):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=latent_dim),
            nn.GELU(),
            nn.Dropout(p=dropout)
        )
        self.norm2 = nn.LayerNorm(latent_dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(latent_dim, 3 * latent_dim, bias=True)
        )

    def forward(self, x, t):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t).chunk(3, dim=1)
        x = x + gate_mlp.unsqueeze(1) * self.layer(modulate(self.norm2(x), shift_mlp, scale_mlp))

        return x

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, d_model):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, d_model, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class MLP(nn.Module):

    def __init__(self, in_dim, h_dim, layer_n, dropout):
        super().__init__()

        self.input_layer = nn.Linear(in_dim, h_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)
        self.t_embedder = TimestepEmbedder(h_dim)
        self.hidden_layers = nn.ModuleList([
            HiddenLayer(h_dim, dropout=dropout) for _ in range(layer_n)
        ])
        self.final_layer = FinalLayer(h_dim, in_dim)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.input_layer.bias.data.zero_()
        self.input_layer.weight.data.uniform_(-initrange, initrange)
        for block in self.hidden_layers:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)


    def forward(self, x, t):

        x = self.dropout(self.activation(self.input_layer(x)))
        t = self.t_embedder(t)  # (N, D)  # (N, D)
        for block in self.hidden_layers:
            x = block(x, t)  # (N, T, D)
        x = self.final_layer(x,t)
        
        return x
      
      
class MLP_old(nn.Module):
       def __init__(self, latent_dim, positional_num, time_emb_dim, layer_num):
        super().__init__()
        self.time_emb_layer = nn.Sequential(
            PositionalEmbedding(positional_num),
            nn.Linear(positional_num, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, 10),
        ) if time_emb_dim is not None else None
        '''
        self.time_bias = nn.Sequential(nn.SiLU,
            nn.Linear(time_emb_dim, out_channels)) if time_emb_dim is not None else None
        '''
        self.input_layer = nn.Linear(in_features=(latent_dim+time_emb_dim), out_features=latent_dim)
        self.activation0 = nn.ReLU()
        self.hidden_layer = nn.ModuleList()

        for i in range(layer_num):
          self.hidden_layer.append(HiddenLayer(latent_dim=latent_dim, time_dim=time_emb_dim))

        self.output_layer = nn.Linear(in_features=latent_dim, out_features=latent_dim)
        self.activation = nn.Sigmoid()

    def forward(self, x, time, class_emb=None):

        if self.time_emb_layer is not None:
            if time is None:
                raise ValueError("time conditioning was specified but time is not passed")
            time_emb = self.time_emb_layer(time)  #[batch, latent_dim]

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



