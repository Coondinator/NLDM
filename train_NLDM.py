import argparse
import datetime

import torch
import numpy as np
from tqdm import tqdm
from ExPIL import process_ExPIL, ExPIL
from Model.argument import Arguments
from Model.diffusion import NLDM, generate_linear_schedule
from Model.denoiser import MLP
from torch.utils.data import dataset, Dataset, DataLoader


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


torch.set_default_dtype(torch.float32)
torch.manual_seed(0)

data_path = '/home/leo/Project/Datasets/ExPIL'
z, *_ = process_ExPIL(data_path)
ExPIL_dataset = ExPIL(z=np.array(z))
train_size = int(0.8*len(ExPIL_dataset))
test_size = len(ExPIL_dataset)-train_size

train_data, test_data = torch.utils.data.random_split(ExPIL_dataset, [train_size, test_size])
train_dataloader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=64, shuffle=True)

learning_rate = 0.0001
iteration = 1000

def get_NLDM():
    config_name = 'NLDM_config.yaml'
    args = Arguments('./Model', filename=config_name)
    betas = generate_linear_schedule(T=1000, low=1e-4, high=0.02)
    time_emb_dim = 128

    mlp = MLP(base_channels=2560, time_emb_dim=time_emb_dim, layer_num=8)
    diffusion = NLDM(config_file=config_name, model=mlp, betas=betas).to(device)

    return diffusion

def basic_train(model, save_path):
    acc_train_loss = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for i in range(iteration):

        model.train()
        for train_data in train_dataloader:
            train_data = train_data.to(device)
            train_loss = model(train_data)
            acc_train_loss += train_loss.item()
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        test_loss = 0
        with torch.no_grad():
            model.eval()
            for test_data in test_dataloader:
                test_data = test_data.to(device)
                test_loss = model(test_data)
                test_loss += test_loss.item()

            #sample = model.sample(batch_size=1, device=device)
        test_loss /= len(test_dataloader)
        acc_train_loss /= len(train_dataloader)

        print('[{:03d}/{}] acc_train_loss: {:.4f}\t test_loss: {:.4f}'.format(
                i, iteration, acc_train_loss, test_loss))

    torch.save(model.state_dict(), save_path)


if __name__ == '__main__':

    save_path = './Model/Model.pt'
    model = get_NLDM()
    basic_train(model, save_path)

