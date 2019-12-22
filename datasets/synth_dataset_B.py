from __future__ import print_function, division

import os
import random
from collections import Iterable
import skimage.io as io

import cv2
import torch
import numpy as np
from skimage import color
from skimage.transform import resize
from torch.utils.data import Dataset

from utils.fh_utils import load_db_annotation, read_img, read_msk, projectPoints, split_theta, sample_version, json_load


class SynthDatasetB(Dataset):
    def __init__(self, config, transform=None, is_vid=False, augment=False):
        self.__dict__.update(config)
        self.augment = augment
        self.is_vid = is_vid
        self.transform = transform

        xyz_uv_path = '/'.join([self.dataset_path, 'xyz_uv.npz'])
        xyz_uv = np.load(xyz_uv_path)

        backgrounds_path = '/'.join([self.dataset_path, 'backgrounds.npz'])
        self.backgrounds_paths = np.load(backgrounds_path)['arr_0']

        self.rgb_paths = self.get_rgb_paths(self.dataset_path)
        self.xyz_array = xyz_uv['arr_0'].reshape(-1, 63)
        self.xyz_dict = self.array_to_dict(self.xyz_array)

        self.uv_array = xyz_uv['arr_1']

    def __len__(self):
        return len(self.xyz_array)

    def get_sample(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = self.load_img(idx)
        padded_img = np.ones([640, 640, 3])
        padded_img[..., 0] *= 14
        padded_img[..., 1] *= 255
        padded_img[..., 2] *= 14
        padded_img[81:561] = img

        if self.augment:
            padded_img = self.add_background(padded_img)
        image_resized = resize(padded_img, (224, 224), anti_aliasing=True)
        image_resized = cv2.cvtColor(image_resized.astype('float32'), cv2.COLOR_BGR2RGB)
        image_resized = image_resized / 255.

        uv = self.uv_array[idx].copy()
        uv = self.pad_and_resize_uv(uv)

        xyz = self.xyz_array[idx]
        depth = self.get_depth_info(xyz)

        uv_maps = self.get_uv_maps(uv)

        sample = {
            'img': image_resized,
            'uv': uv,
            'uv_maps': uv_maps,
            'depth': depth
        }

        return sample

    def load_img(self, idx):
        img_rgb_path = self.rgb_paths[idx]
        img = cv2.imread(img_rgb_path)
        # img = io.imread(img_rgb_path)
        # img = color.rgba2rgb(img)
        # img = img / 255.
        return img

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
            output = self.transform(output)

        return output

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
    def get_rgb_paths(base_path):
        rgb_paths = []
        for filename in os.listdir(base_path):
            if filename.endswith("_color.png"):
                full_path = '/'.join([base_path, filename])
                rgb_paths.append(full_path)

        rgb_paths.sort()
        return rgb_paths

    @staticmethod
    def pad_and_resize_uv(uv):
        uv[:, 1] += 80.
        uv[:, 0] *= 224. / 640.
        uv[:, 1] *= 224. / 640.
        return uv

    @staticmethod
    def get_depth_info(xyz):
        xyz = xyz.reshape([21, 3])
        depth = xyz[..., 2].copy()
        depth = (depth - depth.min())
        max_depth = depth.max()
        depth_normed = depth / max_depth
        max_depth = (((max_depth - 421.53959010000005) / 158.125) * 0.15754850605821397) + 0.02699158462784934
        depth_info = np.append(depth_normed, max_depth)
        return depth_info

    def add_background(self, img):
        random_path = random.choice(self.backgrounds_paths)
        bg = cv2.imread(random_path)
        bg = self.randomCrop(bg)

        bg = cv2.resize(bg, (640, 640))

        img[img == [14, 255, 14]] = bg[img == [14, 255, 14]]

        return img

    @staticmethod
    def randomCrop(img, width=480, height=480):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width]
        return img
