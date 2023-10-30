import os
import numpy as np
import torch
from torch.utils.data import Dataset

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



class ExPIL(Dataset):
    def __init__(self, z, trans=None, index=None, frames=None):
        self.z = torch.squeeze(torch.from_numpy(z))
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

    #data = ExPIL(z=z)