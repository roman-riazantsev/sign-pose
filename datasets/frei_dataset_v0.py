from __future__ import print_function, division

import os

import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

from utils.fh_utils import load_db_annotation, read_img, read_msk, projectPoints, split_theta, json_load


class FreiHandVideoDataset(Dataset):
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

        self.xyz_array = self.get_xyz_array()
        self.xyz_dict = self.array_to_dict(self.xyz_array)

    def __len__(self):
        return len(self.db_data_anno)

    def __getitem__(self, idx):
        frame_0 = self.get_sample(idx)
        frame_1_idx = self.get_idx_of_frame_1(idx)
        frame_1 = self.get_sample(frame_1_idx)

        return frame_0, frame_1

    def get_sample(self, idx):
        img, msk = self.get_img_and_mask(idx)
        K, mano, xyz, uv = self.get_vector_annotations(idx)

        y_2d = self.get_y_2d(uv)

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

    def get_img_and_mask(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = read_img(idx, self.base_path, 'training', self.version)
        msk = read_msk(idx, self.base_path)
        return img, msk

    def get_vector_annotations(self, idx):
        K, mano, xyz = self.db_data_anno[idx]
        K, mano, xyz = [np.array(x) for x in [K, mano, xyz]]
        uv = projectPoints(xyz, K)

        return K, mano, xyz, uv

    def get_idx_of_frame_1(self, frame_0_idx):
        reference_xyz = self.xyz_array[frame_0_idx]

        idx_of_nearest_xyz, _ = self.get_closest_tensor(
            all_tensors=self.xyz_dict,
            reference_tensor=reference_xyz
        )
        return idx_of_nearest_xyz

    def get_closest_tensor(self, all_tensors, reference_tensor):
        def get_distance(idx_tensor_pair):
            _, tensor_ = idx_tensor_pair
            distance = np.abs(tensor_ - reference_tensor).sum()
            return distance

        return min(all_tensors.items(), key=get_distance)

    def get_xyz_array(self):
        xyz_path = os.path.join(self.base_path, '%s_xyz.json' % 'training')
        xyz_list = json_load(xyz_path)
        xyz_array = np.array(xyz_list).reshape(32560, 63)
        return xyz_array

    @staticmethod
    def array_to_dict(arr):
        key_val_pairs = enumerate(arr)
        d = dict(key_val_pairs)
        return d

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
