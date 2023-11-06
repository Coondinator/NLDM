import torch
import numpy as np
from Model.argument import Arguments
from Model.diffusion import NLDM, generate_linear_schedule
from Model.denoiser import MLP
from torch.utils.data import dataset, Dataset, DataLoader

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

torch.set_default_dtype(torch.float32)
torch.manual_seed(0)
def get_NLDM():
    config_name = 'NLDM_config.yaml'
    args = Arguments('./Model', filename=config_name)
    betas = generate_linear_schedule(T=1000, low=1e-4, high=0.02)
    time_emb_dim = 128

    mlp = MLP(base_channels=2560, time_emb_dim=time_emb_dim, layer_num=8)
    diffusion = NLDM(config_file=config_name, model=mlp, betas=betas).to(device)

    return diffusion

def sample(load, save):
    model = get_NLDM()
    model.load_state_dict(torch.load(load_path))
    model.eval()
    batch_size = 1
    sample = model.sample(batch_size, device=device)

    return


if __name__ == '__main__':
    save_path = './Save/'
    load_path = './Model/Model.pt'
    sample(load_path, save_path)