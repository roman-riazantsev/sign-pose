import os
import numpy as np

from utils.fh_utils import json_load
import torch
from manopth.manolayer import ManoLayer
import pickle
from manopth import demo
from utils.utils import to_numpy


def get_poses_and_shapes(mano_path):
    mano_list = json_load(mano_path)

    mano_array = np.array(mano_list).squeeze(1)
    poses = mano_array[..., :48]
    shapes = mano_array[..., 48:58]

    return poses, shapes


def xyz_from_mano(poses, shapes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Select number of principal components for pose space
    ncomps = 45

    # Initialize MANO layer
    mano_layer = ManoLayer(
        mano_root='mano/models', use_pca=False, ncomps=ncomps, flat_hand_mean=False)
    mano_layer.to(device)

    poses = torch.from_numpy(poses).float().to(device)
    shapes = torch.from_numpy(shapes).float().to(device)

    # Forward pass through MANO layer
    hand_verts, hand_joints = mano_layer(poses, shapes)

    return hand_verts, hand_joints


def main():
    base_path = '../../Datasets/FreiHAND_pub_v1'
    mano_path = os.path.join(base_path, '%s_mano.json' % 'training')

    poses, shapes = get_poses_and_shapes(mano_path)

    hand_verts, hand_joints = xyz_from_mano(poses, shapes)

    mano_and_xyz = {
        'poses': poses,
        'shapes': shapes,
        'hand_verts': hand_verts,
        'hand_joints': hand_joints
    }


    f = open("analysis/mano_and_xyz.pkl", "wb")
    pickle.dump(mano_and_xyz, f)
    f.close()

    # manp_xyz_array = to_numpy(hand_joints)

    # np.save('analysis/mano_xyz.npy', manp_xyz_array)


main()
