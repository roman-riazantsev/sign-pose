from __future__ import print_function, division

import os
import random
from collections import Iterable

import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

from utils.fh_utils import load_db_annotation, read_img, read_msk, projectPoints, split_theta, sample_version, json_load


class FreiDatasetB(Dataset):
    def __init__(self, config, excluded_indices=None, transform=None, is_vid=False):
        self.__dict__.update(config)
        self.is_vid = is_vid
        self.db_data_anno = load_db_annotation(self.dataset_path, self.subset_name)
        self.transform = transform
        self.xyz_array = self.get_xyz_array()
        self.xyz_dict = self.array_to_dict(self.xyz_array)
        self.xyz_dict = self.remove_keys(self.xyz_dict, excluded_indices)

    def __len__(self):
        return len(self.db_data_anno)

    def get_sample(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.background_name == 'random':
            background_name = random.choice(['gs', 'hom', 'sample', 'auto'])
            background_type = getattr(sample_version, background_name)
            img = read_img(idx, self.dataset_path, self.subset_name, background_type)
        else:
            background_type = getattr(sample_version, self.background_name)
            img = read_img(idx, self.dataset_path, self.subset_name, background_type)

        K, _, xyz = self.db_data_anno[idx]
        K, xyz = [np.array(x) for x in [K, xyz]]
        uv = projectPoints(xyz, K)
        depth = self.get_depth_info(xyz)

        uv_maps = self.get_uv_maps(uv)

        sample = {
            'img': img,
            'uv_maps': uv_maps,
            'depth': depth
        }

        return sample

    def __getitem__(self, idx):
        frame_0 = self.get_sample(idx)
        if self.is_vid:
            frame_1_idx = self.get_idx_of_frame_1(idx)
            frame_1 = self.get_sample(frame_1_idx)

            sample = {}
            f0i, f0uv, f1i, f1uv = frame_0['img'], frame_0['uv_maps'], frame_1['img'], frame_1['uv_maps']
            sample['X'] = np.concatenate([f0i, f0uv, f1i, f1uv], axis=2)
            sample['prev_depth'] = frame_0['depth']
            sample['Y'] = frame_1['depth']
            output = sample
        else:
            sample = {}
            f0i, f0uv = frame_0['img'], frame_0['uv_maps']
            sample['X'] = np.concatenate([f0i, f0uv], axis=2)
            sample['Y'] = frame_0['depth']
            output = sample

        if self.transform:
            sample = self.transform(output)

        return sample

    def get_idx_of_frame_1(self, frame_0_idx):
        reference_xyz = self.xyz_array[frame_0_idx]
        all_tensors = self.remove_keys(self.xyz_dict, frame_0_idx)

        idx_of_nearest_xyz, _ = self.get_closest_tensor(
            all_tensors=all_tensors,
            reference_tensor=reference_xyz
        )
        return idx_of_nearest_xyz

    def get_closest_tensor(self, all_tensors, reference_tensor):
        def get_distance(idx_tensor_pair):
            _, tensor_ = idx_tensor_pair
            distance = np.abs(tensor_ - reference_tensor).sum()
            return distance

        return min(all_tensors.items(), key=get_distance)

    @staticmethod
    def get_uv_maps(uv):
        uv_maps = np.zeros([224, 224, 22])
        for idx, point in enumerate(uv):
            y, x = uv[idx].astype(np.int)
            x = 223 if x > 223 else x
            y = 223 if y > 223 else y
            z = idx
            uv_maps[x, y, z] = 1

        label_size = np.ones([5, 5])
        uv_maps = cv2.filter2D(uv_maps, -1, label_size)

        points = np.sum(uv_maps, axis=2)
        points[points > 1] = 1
        background = np.ones([224, 224]) - points

        uv_maps[..., -1] = background

        return uv_maps

    def get_xyz_array(self):
        xyz_path = os.path.join(self.dataset_path, '%s_xyz.json' % 'training')
        xyz_list = json_load(xyz_path)
        xyz_array = np.array(xyz_list).reshape(32560, 63)
        return xyz_array

    @staticmethod
    def array_to_dict(arr):
        key_val_pairs = enumerate(arr)
        d = dict(key_val_pairs)
        return d

    @staticmethod
    def remove_keys(d, keys):
        dict_copy = dict(d)
        if keys is not None:
            if isinstance(keys, Iterable):
                for k in keys:
                    dict_copy.pop(k, None)
            else:
                del dict_copy[keys]
        return dict_copy

    @staticmethod
    def get_depth_info(xyz):
        depth = xyz[..., 2].copy()
        depth = (depth - depth.min())
        max_depth = depth.max()
        depth_normed = depth / max_depth
        depth_info = np.append(depth_normed, max_depth)
        return depth_info
