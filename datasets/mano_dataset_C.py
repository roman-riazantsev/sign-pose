import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from manopth.manolayer import ManoLayer
from utils.fh_utils import json_load
from sklearn.neighbors import KernelDensity


class ManoDatasetC(Dataset):
    def __init__(self, base_path, transform, train_indices):
        self.transform = transform

        mano_path = os.path.join(base_path, '%s_mano.json' % 'training')
        mano_list = json_load(mano_path)
        mano_array = np.array(mano_list).squeeze(1)
        mano_poses = mano_array[..., :51]

        mano_poses = mano_poses[train_indices]

        self.kde = KernelDensity(bandwidth=0.15, kernel='gaussian')
        self.kde.fit(mano_poses)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.mano_layer = ManoLayer(
            mano_root='mano/models', use_pca=False, ncomps=45, flat_hand_mean=False)

        self.mano_layer.to(self.device)

    def __len__(self):
        return 32560

    def __getitem__(self, idx):
        sample = self.kde.sample()
        pose = sample[..., :48]
        shape_start = sample[..., 48:]
        shape = np.ones([1, 10])
        shape[..., :3] = shape_start

        x = {
            'p': pose,
            's': shape
        }
        x = self.transform(x)

        hand_verts, hand_joints = self.mano_layer(x['p'], x['s'])
        batch_size = hand_joints.shape[0]
        hand_joints = hand_joints.reshape([batch_size, 63])

        sample = {
            'hand_joints': torch.squeeze(hand_joints),
            'hand_verts': torch.squeeze(hand_verts),
            'poses': torch.squeeze(x['p']),
            'shapes': torch.squeeze(x['s'])
        }

        return sample
