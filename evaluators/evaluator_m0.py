import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from configs.config_m0 import CONFIG_M0
from datasets.frei_dataset_u0 import FreiHandDataset
from manopth.manolayer import ManoLayer
from models.model_m0 import ModelM0
from models.model_u0 import ModelU0
from transforms.to_tensor_u0 import ToTensor
from utils.fh_utils import projectPoints
from utils.utils import to_numpy, plot_hand
from manopth import demo

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
build_id = CONFIG_M0['build_id']

dataset = FreiHandDataset(
    CONFIG_M0['dataset_path'],
    CONFIG_M0['data_version'],
    transform=ToTensor()
)

dataloader = DataLoader(dataset, shuffle=True)
sample = next(iter(dataloader))
random_pose = sample['poses']
random_shape = sample['shapes']

print(random_pose.shape)
# Initialize MANO layer
mano_layer = ManoLayer(
    mano_root='mano/models', use_pca=False, ncomps=48, flat_hand_mean=False)

mano_layer.to(device)
# Forward pass through MANO layer
hand_verts, hand_joints = mano_layer(random_pose, random_shape)
# demo.display_hand({
#     'verts': hand_verts,
#     'joints': hand_joints
# },
#     mano_faces=mano_layer.th_faces)

uv_root = sample['uv_root']
scale = sample['scale']
hand_joints = hand_joints.reshape([1, -1])
x = torch.cat((hand_joints, uv_root, scale), 1)
x = torch.cat((x, x), 1)

# NETWORK WAY
model = ModelM0()

checkpoint = torch.load(f'results/{build_id}.pt', map_location=torch.device(device))
model.load_state_dict(checkpoint['model_state_dict'])

g_step = checkpoint['g_step'] + 1
running_loss = checkpoint['running_loss']

print(f'Model "{build_id}" loaded. g_step: {g_step}; running_loss: {running_loss}')

model.to(device)
model.eval()

# x = hand_joints[0].reshape([1, -1])
xyz_ = model(x)
xyz_ = to_numpy(xyz_).reshape([21, 3])
# # print(xyz_)
#
# y = sample['xyz']
K = sample['K']
img = sample['img']
# # print("yb", y)
# # y= y.reshape([CONFIG_M0['batch_size'], -1])
# # print("ya", y.reshape([21, 3]))
#
# y = to_numpy(y)
# # print(y.shape)


# MATRIX WAY

# x = [[1.0692e-03, -2.9279e-04, 1.8849e-04, 3.7177e-01],
#      [-2.8060e-05, 8.8870e-04, 9.8314e-05, 1.8859e-01],
#      [-3.1334e-05, -2.9735e-04, 1.1492e-03, 9.4405e-01],
#      [-2.1028e-02, -1.3706e-01, 5.3836e-01, 9.8845e-01]]
# xyz_ = to_numpy(hand_joints)[0]
# xyz_ones = np.ones([21, 4])
# xyz_ones[..., :-1] = xyz_
# xyz_ = xyz_ones.dot(x)
# xyz_ = xyz_[..., :-1]


K = to_numpy(K)[0]
uv = projectPoints(xyz_, K)
# print(K)
# print(uv)
img = to_numpy(img)
img = img[0].transpose((1, 2, 0))
#
fig, ax = plt.subplots()
ax.imshow(img)
plot_hand(ax, uv, order='uv')
plt.show()
