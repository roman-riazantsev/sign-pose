from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from fh_utils import load_db_annotation, read_img, read_msk, projectPoints, split_theta


class FreiHandDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, base_path, version, transform=None):
        """
        Args:
            base_path (string): Path to where the FreiHAND dataset is located.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.db_data_anno = load_db_annotation(base_path, 'training')
        self.base_path = base_path
        self.version = version
        self.transform = transform

    def __len__(self):
        return len(self.db_data_anno)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = read_img(idx, self.base_path, 'training', self.version)
        msk = read_msk(idx, self.base_path)

        K, mano, xyz = self.db_data_anno[idx]
        K, mano, xyz = [np.array(x) for x in [K, mano, xyz]]
        uv = projectPoints(xyz, K)

        poses, shapes, uv_root, scale = split_theta(mano)

        sample = {
            'img': img,
            'msk': msk,
            'uv': uv,
            'poses': poses,
            'shapes': shapes
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
