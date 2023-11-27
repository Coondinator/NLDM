import torch
import torch.nn as nn
from .embedding import (TimestepEmbedding,
                        Timesteps)
from .cross_attention import (SkipTransformerEncoder,
                              TransformerDecoder,
                              TransformerDecoderLayer,
                              TransformerEncoder,
                              TransformerEncoderLayer)
from .position_encoding import build_position_encoding


class MldDenoiser(nn.Module):

    def __init__(self,
                 skip_connection,
                 nfeats: int = 263,
                 condition: str = "none",
                 latent_dim: list = [1, 1280],
                 ff_size: int = 1024,
                 num_layers: int = 7,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 normalize_before: bool = False,
                 activation: str = "gelu",
                 flip_sin_to_cos: bool = True,
                 return_intermediate_dec: bool = False,
                 position_embedding: str = "learned",
                 arch: str = "trans_enc",
                 freq_shift: int = 0,
                 guidance_scale: float = 7.5,
                 guidance_uncondp: float = 0.1,
                 text_encoded_dim: int = 768,
                 nclasses: int = 10,
                 **kwargs) -> None:

        super().__init__()

        self.latent_dim = latent_dim[-1]
        self.text_encoded_dim = text_encoded_dim
        self.condition = condition
        self.abl_plus = False
        self.ablation_skip_connection = skip_connection
        # self.diffusion_only = ablation.VAE_TYPE == "no"
        self.arch = arch
        # self.pe_type = ablation.DIFF_PE_TYPE

        # emb proj
        self.time_proj = Timesteps(text_encoded_dim, flip_sin_to_cos,
                                   freq_shift)
        self.time_embedding = TimestepEmbedding(text_encoded_dim,
                                                self.latent_dim)

        self.query_pos = build_position_encoding(
            self.latent_dim, position_embedding=position_embedding)
        '''
        if self.condition in ["text", "text_uncond"]:
            # text condition
            # project time from text_encoded_dim to latent_dim
            self.time_proj = Timesteps(text_encoded_dim, flip_sin_to_cos,
                                       freq_shift)
            self.time_embedding = TimestepEmbedding(text_encoded_dim,
                                                    self.latent_dim)
            # project time+text to latent_dim
            if text_encoded_dim != self.latent_dim:
                # todo 10.24 debug why relu
                self.emb_proj = nn.Sequential(
                    nn.ReLU(), nn.Linear(text_encoded_dim, self.latent_dim))
        elif self.condition in ['action']:
            self.time_proj = Timesteps(self.latent_dim, flip_sin_to_cos,
                                       freq_shift)
            self.time_embedding = TimestepEmbedding(self.latent_dim,
                                                    self.latent_dim)
            self.emb_proj = EmbedAction(nclasses,
                                        self.latent_dim,
                                        guidance_scale=guidance_scale,
                                        guidance_uncodp=guidance_uncondp)
        else:
            raise TypeError(f"condition type {self.condition} not supported")

        if self.pe_type == "actor":
            self.query_pos = PositionalEncoding(self.latent_dim, dropout)
            self.mem_pos = PositionalEncoding(self.latent_dim, dropout)
        elif self.pe_type == "mld":
            self.query_pos = build_position_encoding(
                self.latent_dim, position_embedding=position_embedding)
            self.mem_pos = build_position_encoding(
                self.latent_dim, position_embedding=position_embedding)
        else:
            raise ValueError("Not Support PE type")
        '''
        if self.arch == "trans_enc":
            if self.ablation_skip_connection:
                # use DETR transformer
                encoder_layer = TransformerEncoderLayer(
                    self.latent_dim,
                    num_heads,
                    ff_size,
                    dropout,
                    activation,
                    normalize_before,
                )
                encoder_norm = nn.LayerNorm(self.latent_dim)
                self.encoder = SkipTransformerEncoder(encoder_layer,
                                                      num_layers, encoder_norm)
            else:
                # use torch transformer
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=self.latent_dim,
                    nhead=num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation)
                self.encoder = nn.TransformerEncoder(encoder_layer,
                                                     num_layers=num_layers)
        elif self.arch == "trans_dec":
            decoder_layer = TransformerDecoderLayer(
                self.latent_dim,
                num_heads,
                ff_size,
                dropout,
                activation,
                normalize_before,
            )
            decoder_norm = nn.LayerNorm(self.latent_dim)
            self.decoder = TransformerDecoder(
                decoder_layer,
                num_layers,
                decoder_norm,
                return_intermediate=return_intermediate_dec,
            )
        else:
            raise ValueError(f"Not supported architechure{self.arch}!")

    # def forward(self, latent, timestep, encoder_hidden_states=None, lengths=None,  **kwargs):
    def forward(self, latent, timestep, encoder_hidden_states=None, **kwargs):

        # print('shape:', latent.dim())
        latent = latent.permute(1, 0, 2)
        # 1. time embedding
        timesteps = timestep.expand(latent.shape[1]).clone()

        time_emb = self.time_proj(timesteps)
        time_emb = time_emb.to(dtype=latent.dtype)
        # [1, bs, latent_dim] <= [bs, latent_dim]
        time_emb = self.time_embedding(time_emb).unsqueeze(0)

        # 2. condition + time embedding
        if self.condition == "none":
            emb_latent = time_emb

        elif self.condition in ["text", "text_uncond"]:
            # text_emb [seq_len, batch_size, text_encoded_dim] <= [batch_size, seq_len, text_encoded_dim]
            encoder_hidden_states = encoder_hidden_states.permute(1, 0, 2)
            text_emb = encoder_hidden_states  # [num_words, bs, latent_dim]
            # textembedding projection
            if self.text_encoded_dim != self.latent_dim:
                # [1 or 2, bs, latent_dim] <= [1 or 2, bs, text_encoded_dim]
                text_emb_latent = self.emb_proj(text_emb)
            else:
                text_emb_latent = text_emb
            if self.abl_plus:
                emb_latent = time_emb + text_emb_latent
            else:
                emb_latent = torch.cat((time_emb, text_emb_latent), 0)
        else:
            raise TypeError(f"condition type {self.condition} not supported")

        if self.arch == "trans_enc":

            xseq = torch.cat((latent, emb_latent), axis=0)

            # if self.ablation_skip_connection:
            #     xseq = self.query_pos(xseq)
            #     tokens = self.encoder(xseq)
            # else:
            #     # adding the timestep embed
            #     # [seqlen+1, bs, d]
            #     # todo change to query_pos_decoder
            xseq = self.query_pos(xseq)
            tokens = self.encoder(xseq)

            sample = tokens[:latent.shape[0]]

        elif self.arch == "trans_dec":
            # tgt    - [1 or 5 or 10, bs, latent_dim]
            # memory - [token_num, bs, latent_dim]
            latent = self.query_pos(latent)
            emb_latent = self.mem_pos(emb_latent)
            latent = self.decoder(tgt=latent, memory=emb_latent).squeeze(0)

        else:
            raise TypeError("{self.arch} is not supoorted")

        # 5. [batch_size, latent_dim[0], latent_dim[1]] <= [latent_dim[0], batch_size, latent_dim[1]]
        #sample = sample.permute(1, 0, 2)

        return (sample,)
