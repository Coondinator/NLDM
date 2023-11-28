import wandb
import torch
import numpy as np
from ExPIL import process_ExPIL, process_single_ExPIL, ExPIL, create_MultiVariationData
from Model_Transformer.argument import Arguments
from Model_Transformer.diffusion import LDM
from torch.utils.data import dataset, Dataset, DataLoader

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

torch.set_default_dtype(torch.float32)
torch.manual_seed(0)

data_path = '/home/leo/Project/Datasets/ExPIL'
z, *_ = process_single_ExPIL(data_path)
ExPIL_dataset = ExPIL(latent=np.array(z))
train_size = int(0.8 * len(ExPIL_dataset))
test_size = len(ExPIL_dataset) - train_size

train_data, test_data = torch.utils.data.random_split(ExPIL_dataset, [train_size, test_size])
ones_data = torch.ones((2, 1, 1280))
multi_variation_data = create_MultiVariationData(200, 1280, unsqueeze=1)
print(multi_variation_data.shape)

train_dataloader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=32, shuffle=True)

learning_rate = 0.0025
iteration = 2000


def get_LDM():
    config_name = 'LDM_config.yaml'
    args = Arguments('Model_Transformer', filename=config_name)

    diffusion = LDM(args=args).to(device)

    return diffusion


def basic_train(model, save_path):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 400, 0.8, verbose=False)

    wandb.init(
        # set the wandb project where this run will be logged
        project="data--test",

        # track hyperparameters and run metadata
        config={
            "learning_rate": 0.0025,
            "architecture": "transformer_ldm",
            "prediction": "latent",
            "dataset": "ExPIL",
            "iteration": 2000,
        }
    )

    for i in range(iteration):
        acc_train_loss = 0
        model.train()

        for train_data in train_dataloader:
            train_data = train_data.to(device)
            train_loss = model(train_data)
            # print('train_loss:', train_loss.item())

            acc_train_loss += train_loss.item()
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            # scheduler.step()
        '''
        iterator = iter(train_dataloader)
        example = next(iterator)
        train_data = example.to(device)
        train_loss = model(train_data)
        
        #print('train_loss:', train_loss.item())

        acc_train_loss += train_loss.item()
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        '''
        total_test_loss = 0
        with torch.no_grad():
            model.eval()
            for test_data in test_dataloader:
                test_data = test_data.to(device)
                test_loss = model(test_data)
                total_test_loss += test_loss.item()
                # print('test_loss:', test_loss.item())

        acc_train_loss /= len(train_dataloader)
        total_test_loss /= len(test_dataloader)

        wandb.log({"acc_train_loss": acc_train_loss, "test_loss": total_test_loss})
        print('[{:03d}/{}] acc_train_loss: {:.4f}\t test_loss: {:.4f}'.format(
            i, iteration, acc_train_loss, total_test_loss))

    torch.save(model.state_dict(), save_path)
    wandb.finish()


if __name__ == '__main__':
    save_path = 'Model_Transformer/Model.pt'
    model = get_LDM()
    basic_train(model, save_path)
