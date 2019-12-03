import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from configs.config_m0 import CONFIG_M0
from datasets.frei_dataset_u0 import FreiHandDataset
from manopth.manolayer import ManoLayer
from transforms.to_tensor_u0 import ToTensor
from utils.fh_utils import projectPoints
from utils.utils import to_numpy, plot_hand, recover_root, get_focal_pp
from manopth import demo

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
build_id = CONFIG_M0['build_id']

dataset = FreiHandDataset(
    CONFIG_M0['dataset_path'],
    CONFIG_M0['data_version'],
    transform=ToTensor()
)

# dataloader = DataLoader(dataset, shuffle=False)
sample = dataset[124]
# random_pose = sample['poses'][0]
# random_shape = sample['shapes'][0]
#
# mano_layer = ManoLayer(
#     mano_root='mano/models', use_pca=False, ncomps=48, flat_hand_mean=False)
#
# mano_layer.to(device)
# hand_verts, hand_joints = mano_layer(random_pose, random_shape)
#
# hand_joints = to_numpy(hand_joints)[0]
# print(hand_joints)
# xyz_ones = np.ones([21, 4])
# xyz_ones[..., :-1] = hand_joints
# x = np.array([[-5.6334e-05, 0.0000e+00, 0.0000e+00, -2.6049e-04],
#               [0.0000e+00, -5.6334e-05, 0.0000e+00, -1.3024e-04],
#               [0.0000e+00, 0.0000e+00, -5.6334e-05, -4.0667e-04],
#               [0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00]])
# hand_joints = xyz_ones.dot(x)
# hand_joints = np.array([[8.2529e-02, -1.4945e-01, 5.6315e-01, 1],
#                         [6.6335e-02, -1.7623e-01, 5.8878e-01, 1],
#                         [4.4146e-02, -1.8020e-01, 6.1253e-01, 1],
#                         [3.3096e-02, -1.6868e-01, 6.3744e-01, 1],
#                         [2.7584e-03, -1.5482e-01, 6.5776e-01, 1],
#                         [4.3642e-02, -1.5129e-01, 6.5138e-01, 1],
#                         [2.6643e-02, -1.4592e-01, 6.8187e-01, 1],
#                         [1.2250e-02, -1.4310e-01, 7.0081e-01, 1],
#                         [3.6873e-03, -1.3518e-01, 7.2599e-01, 1],
#                         [3.5133e-02, -1.2599e-01, 6.4969e-01, 1],
#                         [1.7186e-02, -1.1599e-01, 6.7677e-01, 1],
#                         [1.9427e-03, -1.0687e-01, 6.9442e-01, 1],
#                         [-5.4359e-03, -9.3021e-02, 7.1830e-01, 1],
#                         [3.0754e-02, -1.0840e-01, 6.2777e-01, 1],
#                         [1.4456e-02, -1.0251e-01, 6.5401e-01, 1],
#                         [-1.7585e-03, -9.1628e-02, 6.7300e-01, 1],
#                         [-1.3311e-02, -8.3420e-02, 6.9725e-01, 1],
#                         [2.5966e-02, -9.8290e-02, 6.0717e-01, 1],
#                         [1.3304e-02, -8.8364e-02, 6.2325e-01, 1],
#                         [-7.6417e-07, -7.7943e-02, 6.3471e-01, 1],
#                         [-7.6329e-03, -6.4999e-02, 6.5365e-01, 1]])
# hand_joints = hand_joints[..., :-1]

uv_root = sample['uv_root']
scale = sample['scale']

# focal, pp = get_focal_pp(K)
# xyz_root = recover_root(uv_root, scale, focal, pp)

# pose_by_root(xyz_root[0], poses[0], shapes[0])


K = sample['K']
print("K", K)
img = sample['img']
print(img.shape)
# hand_joints[9] = xyz_root
xyz = to_numpy(sample['xyz'])
# K = to_numpy(K)

K = [[-2.4566, -1.7960, 141.4863],
     [2.2762, 10.7719, 111.2687],
     [0.0000, 0.0000, 1.0000]]
uv = projectPoints(xyz, K)

# uv = [[81.6912, 136.1244],
#       [93.0584, 109.9410],
#       [100.1987, 91.8695],
#       [113.6654, 81.9669],
#       [119.1983, 70.1710],
#       [139.4812, 104.4189],
#       [160.4549, 100.6390],
#       [157.9915, 94.5050],
#       [148.5879, 87.4507],
#       [143.7224, 117.3830],
#       [160.1517, 116.7967],
#       [150.6411, 107.7159],
#       [138.0303, 96.6124],
#       [133.7154, 130.7154],
#       [140.2368, 122.3100],
#       [132.6299, 113.7178],
#       [123.9726, 100.2617],
#       [123.5582, 139.2142],
#       [119.5465, 130.4882],
#       [114.5742, 123.0273],
#       [109.8051, 112.0727]]

img = to_numpy(img)
img = img.transpose((1, 2, 0))

fig, ax = plt.subplots()
ax.imshow(img)
print(img.shape)
plot_hand(ax, uv, order='uv')
plt.show()