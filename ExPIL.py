import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.distributions.multivariate_normal import MultivariateNormal


def create_MultiVariationData(batch, dim, unsqueeze=None):
    # Create a tensor data with multivariate gaussian distribution [bacth, (1), dim]

    # 定义均值和协方差矩阵（1000维）
    mean = torch.zeros(dim)
    covariance_matrix = torch.eye(dim)  # 假设协方差矩阵是单位矩阵

    # 创建多维高斯分布
    multivariate_normal = MultivariateNormal(loc=mean, covariance_matrix=covariance_matrix)

    data = multivariate_normal.sample((batch,))

    if unsqueeze:
        data = data.unsqueeze(unsqueeze)
    return data


def process_ExPIL(root_path):
    ExPIL_zpair = []
    ExPIL_trans = []
    ExPIL_index = []
    ExPIL_frames = []

    if os.path.exists(root_path):
        for root, dirs, files in os.walk(root_path):
            for file in files:
                if file.split('.')[1] == 'npz':
                    file_path = os.path.join(root, file)
                    data = np.load(file_path)
                    temp_zpair = np.concatenate((data['zlm'], data['zgm'], data['zlf'], data['z_gf']), axis=1)
                    temp_trans = np.concatenate((data['trans_m'], data['trans_f']), axis=0)
                    temp_index = data['index']
                    temp_frame = data['frame']
                    ExPIL_zpair.append(temp_zpair)
                    ExPIL_trans.append(temp_trans)
                    ExPIL_index.append(temp_index)
                    ExPIL_frames.append(temp_frame)

    return ExPIL_zpair, ExPIL_trans, ExPIL_index, ExPIL_frames


def process_single_ExPIL(root_path):
    ExPIL_zpair = []
    ExPIL_trans = []
    ExPIL_index = []
    ExPIL_frames = []

    if os.path.exists(root_path):
        for root, dirs, files in os.walk(root_path):
            for file in files:
                if file.split('.')[1] == 'npz':
                    file_path = os.path.join(root, file)
                    data = np.load(file_path)
                    temp_zpair = np.concatenate((data['zlm'], data['zgm']), axis=1)
                    temp_trans = data['trans_m']
                    temp_index = data['index']
                    temp_frame = data['frame']
                    ExPIL_zpair.append(temp_zpair)
                    ExPIL_trans.append(temp_trans)
                    ExPIL_index.append(temp_index)
                    ExPIL_frames.append(temp_frame)

    return ExPIL_zpair, ExPIL_trans, ExPIL_index, ExPIL_frames


class ExPIL(Dataset):
    def __init__(self, latent, extra_channel=True, trans=None, index=None, frames=None):
        if extra_channel:
            # MLP denoiser input shape [batch, latent_dim]
            self.z = torch.from_numpy(latent)
        else:
            # Transformer denoiser input shape [batch, 1, latent_dim]
            self.z = torch.squeeze(torch.from_numpy(latent))
        '''
        self.genre = None
        self.index = torch.from_numpy(index)
        self.frames = torch.from_numpy(frames)
        self.trans = torch.from_numpy(trans)
        '''

    def __len__(self):
        return len(self.z)

    def __getitem__(self, item):
        return self.z[item]


if __name__ == '__main__':
    data_path = '/home/leo/Project/Datasets/ExPIL'
    z, *_ = process_ExPIL(data_path)
    print(z[0].shape)

    # data = ExPIL(z=z)
