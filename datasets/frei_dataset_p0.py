import os
import json
import torch
from torch.utils.data import Dataset
import numpy as np


class FreiDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        def json_load(p):
            msg = 'File does not exists: %s' % p
            assert os.path.exists(p), msg

            with open(p, 'r') as fi:
                d = json.load(fi)
            return d

        k_path = os.path.join(root_dir, 'training_K.json')
        mano_path = os.path.join(root_dir, 'training_mano.json')
        xyz_path = os.path.join(root_dir, 'training_xyz.json')

        K_list = json_load(k_path)
        mano_list = json_load(mano_path)
        xyz_list = json_load(xyz_path)

        assert len(K_list) == len(mano_list), 'Size mismatch.'
        assert len(K_list) == len(xyz_list), 'Size mismatch.'

        self.db_data_anno = list(zip(K_list, mano_list, xyz_list))
        self.transform = transform

    def __len__(self):
        return len(self.db_data_anno)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        K, mano, xyz = self.db_data_anno[idx]
        uv = self.projectPoints(xyz, K).flatten()
        mano = np.array(mano).flatten()[:58]

        sample = {'uv': uv, 'mano': mano}

        if self.transform:
            sample = self.transform(sample)

        return sample

    @staticmethod
    def projectPoints(xyz, K):
        """ Project 3D coordinates into image space. """
        xyz = np.array(xyz)
        K = np.array(K)
        uv = np.matmul(K, xyz.T).T
        return uv[:, :2] / uv[:, -1:]
