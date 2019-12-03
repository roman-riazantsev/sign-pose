import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from configs.config_u0 import CONFIG_U0
from datasets.frei_dataset_u0 import FreiHandDataset
from manopth.manolayer import ManoLayer
from models.model_u0 import ModelU0
from transforms.to_tensor_u0 import ToTensor
from utils.utils import to_numpy, plot_hand
from manopth import demo

build_id = CONFIG_U0['build_id']

dataset = FreiHandDataset(
    CONFIG_U0['dataset_path'],
    CONFIG_U0['data_version'],
    transform=ToTensor()
)

dataloader = DataLoader(dataset)
sample = next(iter(dataloader))
random_pose = sample['poses']
random_shape = sample['shapes']

print(random_pose.shape)
# Initialize MANO layer
mano_layer = ManoLayer(
    mano_root='mano/models', use_pca=False, ncomps=48, flat_hand_mean=False)

# Forward pass through MANO layer
hand_verts, hand_joints = mano_layer(random_pose, random_shape)
demo.display_hand({
    'verts': hand_verts,
    'joints': hand_joints
},
    mano_faces=mano_layer.th_faces)

print(hand_joints[0].shape)
print("------")
print(sample['xyz'][0])
