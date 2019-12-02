import os
import numpy as np
from utils.utils import to_numpy
from utils.fh_utils import *

from utils.fh_utils import load_db_annotation


class FramesGenerator:
    def __init__(self, base_path, version):
        self.base_path = base_path
        self.version = version

    def order_frames(self, annotation, init_idx, length):
        if annotation == 'xyz':
            anno_array = self.get_xyz_array()
        elif annotation == 'mano':
            anno_array = self.get_mano_array()
        else:
            xyz_array = self.get_xyz_array()
            K_array = self.get_K_array()
            anno_array = self.get_uv_array(xyz_array, K_array)

        anno_dict = self.array_to_dict(anno_array)
        # print(anno_dict[0])
        indeces = self.get_frames_from_dict(anno_dict, init_idx, length)
        return indeces

    def get_xyz_array(self):
        xyz_path = os.path.join(self.base_path, '%s_xyz.json' % self.version)
        xyz_list = json_load(xyz_path)
        xyz_array = np.array(xyz_list).reshape(32560, 63)
        return xyz_array

    def get_mano_array(self):
        mano_path = os.path.join(self.base_path, '%s_mano.json' % self.version)
        mano_list = json_load(mano_path)
        mano_array = np.array(mano_list).squeeze(1)
        mano_poses = mano_array[..., :48]
        return mano_poses

    def get_K_array(self):
        K_path = os.path.join(self.base_path, '%s_K.json' % self.version)
        K_list = json_load(K_path)
        K_array = np.array(K_list)
        return K_array

    def get_uv_array(self, xyz_array, K_array):
        uv_list = []

        xyz_array = xyz_array.reshape([-1, 3, 3])
        for idx, (xyz, K) in enumerate(zip(xyz_array, K_array)):
            uv = projectPoints(xyz, K)
            uv_list.append(uv)

        uv_array = np.array(uv_list)
        return uv_array

    @staticmethod
    def drop_element(vec_dict, key):
        new_dict = dict(vec_dict)
        val = new_dict[key]
        del new_dict[key]
        return val, new_dict

    @staticmethod
    def get_nearest(vector, vec_dict):
        def get_distance(dict_item, vector=vector):
            _, val = dict_item
            distance = np.abs(val - vector).sum()
            return distance

        return min(vec_dict.items(), key=get_distance)

    def get_frames_from_dict(self, vec_dict, init_idx, length):
        list_of_idxs = []
        cur_val, vec_dict = self.drop_element(vec_dict, init_idx)
        list_of_idxs.append(init_idx)

        for i in range(length):
            idx_of_nearest, nearest_vector = self.get_nearest(cur_val, vec_dict)
            list_of_idxs.append(idx_of_nearest)
            cur_val, vec_dict = self.drop_element(vec_dict, idx_of_nearest)

        return list_of_idxs

    @staticmethod
    def array_to_dict(arr):
        key_val_pairs = enumerate(arr)
        d = dict(key_val_pairs)
        return d
