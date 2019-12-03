import os
from pprint import pprint
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.mano_xyz_dataset_mae0 import ManoXYZDataset
from manopth import demo
from manopth.manolayer import ManoLayer
from models.model_mae0 import ModelMAE0
from trainers.trainer_mae0 import TrainerMAE0
from transforms.np_to_tensor import NpToTensor

from configs.config_mae0 import CONFIG_MAE0
from models.model_m0 import ModelM0
from trainers.trainer_m0 import TrainerM0
from utils.utils import to_numpy


def load_state(model, save_path):
    if os.path.exists(save_path):
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        return model


if __name__ == '__main__':
    build_id = CONFIG_MAE0['build_id']
    save_path = f'results/mae/{build_id}/{build_id}.pt'

    dataset = ManoXYZDataset(
        CONFIG_MAE0['dataset_path'],
        transform=NpToTensor()
    )

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True
    )

    model = ModelMAE0()
    model = load_state(model, save_path)

    sample = next(iter(dataloader))
    xyz = sample['hand_joints']
    poses = sample['poses']
    shapes = sample['shapes']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    _, xyz_, poses_, shapes_ = model(xyz)

    shapes = torch.zeros([2, 10], dtype=torch.float32).to(device)

    shapes[:, :3] = shapes_[:, :3]

    # print(poses_[0], shapes_[0])

    # Select number of principal components for pose space
    ncomps = 45

    # Initialize MANO layer
    mano_layer = ManoLayer(
        mano_root='mano/models', use_pca=False, ncomps=ncomps, flat_hand_mean=False).to(device)


    hand_verts, hand_joints = mano_layer(poses_, shapes)
    hand_verts = hand_verts.cpu().detach()
    hand_joints = hand_joints.cpu().detach()
    demo.display_hand({
        'verts': hand_verts,
        'joints': hand_joints
    },
        mano_faces=mano_layer.th_faces)
