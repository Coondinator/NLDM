import wandb
import torch
import numpy as np

from ExPIL import process_ExPIL, process_single_ExPIL, ExPIL
from Model_MLP.argument import Arguments
from Model_MLP.diffusion import NLDM, generate_linear_schedule
from Model_MLP.denoiser import MLP
from torch.utils.data import dataset, Dataset, DataLoader

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

torch.set_default_dtype(torch.float32)
torch.manual_seed(0)

data_path = '/home/leo/Project/Datasets/ExPIL'
z, *_ = process_single_ExPIL(data_path)
ExPIL_dataset = ExPIL(latent=np.array(z))
train_size = int(0.8*len(ExPIL_dataset))
test_size = len(ExPIL_dataset)-train_size

train_data, test_data = torch.utils.data.random_split(ExPIL_dataset, [train_size, test_size])
ones_datat = torch.ones((2, 2560))

train_dataloader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=64, shuffle=True)

learning_rate = 0.0001
iteration = 2000

def get_NLDM():
    config_name = 'NLDM_config.yaml'
    args = Arguments('Model_MLP', filename=config_name)
    betas = generate_linear_schedule(T=1000, low=1e-4, high=0.02)

    mlp = MLP(positional_num=args.positional_num, time_emb_dim=args.time_emb_dim, layer_num=args.layer_num)
    diffusion = NLDM(config_file=config_name, model=mlp, betas=betas).to(device)

    return diffusion

def basic_train(model, save_path):

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    wandb.init(
        # set the wandb project where this run will be logged
        project="single--test",

        # track hyperparameters and run metadata
        config={
            "learning_rate": 0.0001,
            "architecture": "ldm",
            "dataset": "ExPIL",
            "iteration": 50000,
        }
    )

    for i in range(iteration):
        acc_train_loss = 0
        model.train()
        for train_data in train_dataloader:

            train_data = train_data.to(device)
            train_loss = model(train_data)
            print('train_loss:', train_loss.item())

            acc_train_loss += train_loss.item()
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            #model.update_ema()

        total_test_loss = 0
        with torch.no_grad():
            model.eval()
            for test_data in test_dataloader:
                test_data = test_data.to(device)
                test_loss = model(test_data)
                total_test_loss += test_loss.item()
                #print('test_loss:', test_loss.item())

        acc_train_loss /= len(train_dataloader)
        total_test_loss /= len(test_dataloader)

        wandb.log({"acc_train_loss": acc_train_loss, "test_loss": total_test_loss})
        print('[{:03d}/{}] acc_train_loss: {:.4f}\t test_loss: {:.4f}'.format(
                i, iteration, acc_train_loss, total_test_loss))

    torch.save(model.state_dict(), save_path)
    wandb.finish()


if __name__ == '__main__':

    save_path = 'Model_MLP/Model.pt'
    model = get_NLDM()
    basic_train(model, save_path)

