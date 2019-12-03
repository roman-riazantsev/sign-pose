from __future__ import print_function, division

import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

from utils.fh_utils import load_db_annotation, read_img, read_msk, projectPoints, split_theta


class FreiHandDatasetS2(Dataset):
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

        K, _, xyz = self.db_data_anno[idx]
        K, xyz = [np.array(x) for x in [K, xyz]]
        uv = projectPoints(xyz, K)
        y_2d = self.get_y_2d(uv)

        depth_info = self.get_depth_info(xyz)

        X = np.concatenate([img, y_2d], axis=2)

        sample = {
            'X': X,
            'depth_info': depth_info,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    @staticmethod
    def get_y_2d(uv):
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

    @staticmethod
    def get_depth_info(xyz):
        depth = xyz[..., 2].copy()
        depth = (depth - depth.min())
        max_depth = depth.max()
        depth_normed = depth / max_depth
        depth_info = np.append(depth_normed, max_depth)
        return depth_info
