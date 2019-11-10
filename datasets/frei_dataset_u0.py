from __future__ import print_function, division

import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

from utils.fh_utils import load_db_annotation, read_img, read_msk, projectPoints, split_theta


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

        y_2d = self.get_y_2d(uv, msk)

        poses, shapes, uv_root, scale = split_theta(mano)

        sample = {
            'img': img,
            'msk': msk,
            'uv': uv,
            'y_2d': y_2d,
            'poses': poses[0],
            'shapes': shapes[0],
            'uv_root': uv_root[0],
            'scale': scale[0],
            'xyz': xyz,
            'mano': mano,
            'K': K
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    @staticmethod
    def get_y_2d(uv, msk):
        y_2d = np.zeros([224, 224, 22])
        for idx, point in enumerate(uv):
            y, x = uv[idx].astype(np.int)
            x = 223 if x > 223 else x
            y = 223 if y > 223 else y
            z = idx
            y_2d[x, y, z] = 1

        filter = np.ones([5, 5])
        y_2d = cv2.filter2D(y_2d, -1, filter)

        points = np.sum(y_2d, axis=2)
        points[points > 1] = 1
        background = np.ones([224, 224]) - points

        y_2d[..., -1] = background

        return y_2d
